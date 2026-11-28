/*
 * Copyright © 2020 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "helpers.h"

class AvailabilityVisibility : public spirv_test {};

TEST_F(AvailabilityVisibility, opload_vis)
{
   /*
               OpCapability Shader
               OpCapability VulkanMemoryModel
               OpCapability VulkanMemoryModelDeviceScope
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical Vulkan
               OpEntryPoint GLCompute %4 "main" %9
               OpExecutionMode %4 LocalSize 1 1 1
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 Block
               OpDecorate %9 DescriptorSet 0
               OpDecorate %9 Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
  %_struct_7 = OpTypeStruct %uint
%_ptr_StorageBuffer__struct_7 = OpTypePointer StorageBuffer %_struct_7
          %9 = OpVariable %_ptr_StorageBuffer__struct_7 StorageBuffer
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
     %device = OpConstant %int 1
          %4 = OpFunction %void None %3
          %5 = OpLabel
         %13 = OpAccessChain %_ptr_StorageBuffer_uint %9 %int_0
         %14 = OpLoad %uint %13 NonPrivatePointer|MakePointerVisible %device
               OpStore %13 %14
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010500, 0x00070000, 0x00000010, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x000014e1, 0x00020011, 0x000014e2, 0x0006000b,
      0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
      0x00000000, 0x00000003, 0x0006000f, 0x00000005, 0x00000002, 0x6e69616d,
      0x00000000, 0x00000003, 0x00060010, 0x00000002, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00050048, 0x00000004, 0x00000000, 0x00000023,
      0x00000000, 0x00030047, 0x00000004, 0x00000002, 0x00040047, 0x00000003,
      0x00000022, 0x00000000, 0x00040047, 0x00000003, 0x00000021, 0x00000000,
      0x00020013, 0x00000005, 0x00030021, 0x00000006, 0x00000005, 0x00040015,
      0x00000007, 0x00000020, 0x00000000, 0x0003001e, 0x00000004, 0x00000007,
      0x00040020, 0x00000008, 0x0000000c, 0x00000004, 0x0004003b, 0x00000008,
      0x00000003, 0x0000000c, 0x00040015, 0x00000009, 0x00000020, 0x00000001,
      0x0004002b, 0x00000009, 0x0000000a, 0x00000000, 0x00040020, 0x0000000b,
      0x0000000c, 0x00000007, 0x0004002b, 0x00000009, 0x0000000c, 0x00000001,
      0x00050036, 0x00000005, 0x00000002, 0x00000000, 0x00000006, 0x000200f8,
      0x0000000d, 0x00050041, 0x0000000b, 0x0000000e, 0x00000003, 0x0000000a,
      0x0006003d, 0x00000007, 0x0000000f, 0x0000000e, 0x00000030, 0x0000000c,
      0x0003003e, 0x0000000e, 0x0000000f, 0x000100fd, 0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);

   nir_intrinsic_instr *intrinsic = find_intrinsic(nir_intrinsic_barrier, 0);
   ASSERT_NE(intrinsic, nullptr);

   EXPECT_EQ(nir_intrinsic_memory_semantics(intrinsic), NIR_MEMORY_MAKE_VISIBLE);
   EXPECT_NE(nir_intrinsic_memory_modes(intrinsic) & nir_var_mem_ssbo, 0);
   EXPECT_EQ(nir_intrinsic_memory_scope(intrinsic), SCOPE_DEVICE);
   EXPECT_EQ(nir_intrinsic_execution_scope(intrinsic), SCOPE_NONE);
}

TEST_F(AvailabilityVisibility, opstore_avail)
{
   /*
               OpCapability Shader
               OpCapability VulkanMemoryModel
               OpCapability VulkanMemoryModelDeviceScope
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical Vulkan
               OpEntryPoint GLCompute %4 "main" %9
               OpExecutionMode %4 LocalSize 1 1 1
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 Block
               OpDecorate %9 DescriptorSet 0
               OpDecorate %9 Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
  %_struct_7 = OpTypeStruct %uint
%_ptr_StorageBuffer__struct_7 = OpTypePointer StorageBuffer %_struct_7
          %9 = OpVariable %_ptr_StorageBuffer__struct_7 StorageBuffer
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
     %device = OpConstant %int 1
          %4 = OpFunction %void None %3
          %5 = OpLabel
         %13 = OpAccessChain %_ptr_StorageBuffer_uint %9 %int_0
         %14 = OpLoad %uint %13
               OpStore %13 %14 NonPrivatePointer|MakePointerAvailable %device
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010500, 0x00070000, 0x00000010, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x000014e1, 0x00020011, 0x000014e2, 0x0006000b,
      0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
      0x00000000, 0x00000003, 0x0006000f, 0x00000005, 0x00000002, 0x6e69616d,
      0x00000000, 0x00000003, 0x00060010, 0x00000002, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00050048, 0x00000004, 0x00000000, 0x00000023,
      0x00000000, 0x00030047, 0x00000004, 0x00000002, 0x00040047, 0x00000003,
      0x00000022, 0x00000000, 0x00040047, 0x00000003, 0x00000021, 0x00000000,
      0x00020013, 0x00000005, 0x00030021, 0x00000006, 0x00000005, 0x00040015,
      0x00000007, 0x00000020, 0x00000000, 0x0003001e, 0x00000004, 0x00000007,
      0x00040020, 0x00000008, 0x0000000c, 0x00000004, 0x0004003b, 0x00000008,
      0x00000003, 0x0000000c, 0x00040015, 0x00000009, 0x00000020, 0x00000001,
      0x0004002b, 0x00000009, 0x0000000a, 0x00000000, 0x00040020, 0x0000000b,
      0x0000000c, 0x00000007, 0x0004002b, 0x00000009, 0x0000000c, 0x00000001,
      0x00050036, 0x00000005, 0x00000002, 0x00000000, 0x00000006, 0x000200f8,
      0x0000000d, 0x00050041, 0x0000000b, 0x0000000e, 0x00000003, 0x0000000a,
      0x0004003d, 0x00000007, 0x0000000f, 0x0000000e, 0x0005003e, 0x0000000e,
      0x0000000f, 0x00000028, 0x0000000c, 0x000100fd, 0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);

   nir_intrinsic_instr *intrinsic = find_intrinsic(nir_intrinsic_barrier, 0);
   ASSERT_NE(intrinsic, nullptr);

   EXPECT_EQ(nir_intrinsic_memory_semantics(intrinsic), NIR_MEMORY_MAKE_AVAILABLE);
   EXPECT_NE(nir_intrinsic_memory_modes(intrinsic) & nir_var_mem_ssbo, 0);
   EXPECT_EQ(nir_intrinsic_memory_scope(intrinsic), SCOPE_DEVICE);
   EXPECT_EQ(nir_intrinsic_execution_scope(intrinsic), SCOPE_NONE);
}

/* Position of the first intrinsic of the given op in the entry point, or -1. */
static int
intrinsic_position(nir_shader *shader, nir_intrinsic_op op)
{
   int pos = 0;
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == op)
            return pos;
         pos++;
      }
   }
   return -1;
}

TEST_F(AvailabilityVisibility, opimageread_vis)
{
   /*
               OpCapability Shader
               OpCapability VulkanMemoryModel
               OpCapability VulkanMemoryModelDeviceScope
               OpMemoryModel Logical Vulkan
               OpEntryPoint GLCompute %main "main" %img
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %img DescriptorSet 0
               OpDecorate %img Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %v2int = OpTypeVector %int 2
     %v4uint = OpTypeVector %uint 4
     %imgtype = OpTypeImage %uint 2D 0 0 0 2 R32ui
        %ptr = OpTypePointer UniformConstant %imgtype
        %img = OpVariable %ptr UniformConstant
      %int_0 = OpConstant %int 0
      %coord = OpConstantComposite %v2int %int_0 %int_0
     %device = OpConstant %int 1
      %zero = OpConstantNull %v4uint
       %main = OpFunction %void None %fn
          %l = OpLabel
         %im = OpLoad %imgtype %img
          %v = OpImageRead %v4uint %im %coord MakeTexelVisible|NonPrivateTexel %device
               OpImageWrite %im %coord %v
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010500, 0x00070000, 0x00000012, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x000014e1, 0x00020011, 0x000014e2, 0x0003000e,
      0x00000000, 0x00000003, 0x0006000f, 0x00000005, 0x00000001, 0x6e69616d,
      0x00000000, 0x00000002, 0x00060010, 0x00000001, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00040047, 0x00000002, 0x00000022, 0x00000000,
      0x00040047, 0x00000002, 0x00000021, 0x00000000, 0x00020013, 0x00000003,
      0x00030021, 0x00000004, 0x00000003, 0x00040015, 0x00000005, 0x00000020,
      0x00000000, 0x00040015, 0x00000006, 0x00000020, 0x00000001, 0x00040017,
      0x00000007, 0x00000006, 0x00000002, 0x00040017, 0x00000008, 0x00000005,
      0x00000004, 0x00090019, 0x00000009, 0x00000005, 0x00000001, 0x00000000,
      0x00000000, 0x00000000, 0x00000002, 0x00000021, 0x00040020, 0x0000000a,
      0x00000000, 0x00000009, 0x0004003b, 0x0000000a, 0x00000002, 0x00000000,
      0x0004002b, 0x00000006, 0x0000000b, 0x00000000, 0x0005002c, 0x00000007,
      0x0000000c, 0x0000000b, 0x0000000b, 0x0004002b, 0x00000006, 0x0000000d,
      0x00000001, 0x0003002e, 0x00000008, 0x0000000e, 0x00050036, 0x00000003,
      0x00000001, 0x00000000, 0x00000004, 0x000200f8, 0x0000000f, 0x0004003d,
      0x00000009, 0x00000010, 0x00000002, 0x00070062, 0x00000008, 0x00000011,
      0x00000010, 0x0000000c, 0x00000600, 0x0000000d, 0x00040063, 0x00000010,
      0x0000000c, 0x00000011, 0x000100fd, 0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);

   nir_intrinsic_instr *intrinsic = find_intrinsic(nir_intrinsic_barrier, 0);
   ASSERT_NE(intrinsic, nullptr);

   EXPECT_EQ(nir_intrinsic_memory_semantics(intrinsic), NIR_MEMORY_MAKE_VISIBLE);
   EXPECT_NE(nir_intrinsic_memory_modes(intrinsic) & nir_var_image, 0);
   EXPECT_EQ(nir_intrinsic_memory_scope(intrinsic), SCOPE_DEVICE);
   EXPECT_EQ(nir_intrinsic_execution_scope(intrinsic), SCOPE_NONE);

   /* MakeTexelVisible makes the texel visible before it is read. */
   int barrier = intrinsic_position(shader, nir_intrinsic_barrier);
   int load = intrinsic_position(shader, nir_intrinsic_image_deref_load);
   ASSERT_GE(barrier, 0);
   ASSERT_GE(load, 0);
   EXPECT_LT(barrier, load);
}

TEST_F(AvailabilityVisibility, opimagewrite_avail)
{
   /*
               OpCapability Shader
               OpCapability VulkanMemoryModel
               OpCapability VulkanMemoryModelDeviceScope
               OpMemoryModel Logical Vulkan
               OpEntryPoint GLCompute %main "main" %img
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %img DescriptorSet 0
               OpDecorate %img Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %v2int = OpTypeVector %int 2
     %v4uint = OpTypeVector %uint 4
     %imgtype = OpTypeImage %uint 2D 0 0 0 2 R32ui
        %ptr = OpTypePointer UniformConstant %imgtype
        %img = OpVariable %ptr UniformConstant
      %int_0 = OpConstant %int 0
      %coord = OpConstantComposite %v2int %int_0 %int_0
     %device = OpConstant %int 1
      %zero = OpConstantNull %v4uint
       %main = OpFunction %void None %fn
          %l = OpLabel
         %im = OpLoad %imgtype %img
          %v = OpImageRead %v4uint %im %coord
               OpImageWrite %im %coord %v MakeTexelAvailable|NonPrivateTexel %device
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010500, 0x00070000, 0x00000012, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x000014e1, 0x00020011, 0x000014e2, 0x0003000e,
      0x00000000, 0x00000003, 0x0006000f, 0x00000005, 0x00000001, 0x6e69616d,
      0x00000000, 0x00000002, 0x00060010, 0x00000001, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00040047, 0x00000002, 0x00000022, 0x00000000,
      0x00040047, 0x00000002, 0x00000021, 0x00000000, 0x00020013, 0x00000003,
      0x00030021, 0x00000004, 0x00000003, 0x00040015, 0x00000005, 0x00000020,
      0x00000000, 0x00040015, 0x00000006, 0x00000020, 0x00000001, 0x00040017,
      0x00000007, 0x00000006, 0x00000002, 0x00040017, 0x00000008, 0x00000005,
      0x00000004, 0x00090019, 0x00000009, 0x00000005, 0x00000001, 0x00000000,
      0x00000000, 0x00000000, 0x00000002, 0x00000021, 0x00040020, 0x0000000a,
      0x00000000, 0x00000009, 0x0004003b, 0x0000000a, 0x00000002, 0x00000000,
      0x0004002b, 0x00000006, 0x0000000b, 0x00000000, 0x0005002c, 0x00000007,
      0x0000000c, 0x0000000b, 0x0000000b, 0x0004002b, 0x00000006, 0x0000000d,
      0x00000001, 0x0003002e, 0x00000008, 0x0000000e, 0x00050036, 0x00000003,
      0x00000001, 0x00000000, 0x00000004, 0x000200f8, 0x0000000f, 0x0004003d,
      0x00000009, 0x00000010, 0x00000002, 0x00050062, 0x00000008, 0x00000011,
      0x00000010, 0x0000000c, 0x00060063, 0x00000010, 0x0000000c, 0x00000011,
      0x00000500, 0x0000000d, 0x000100fd, 0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);

   nir_intrinsic_instr *intrinsic = find_intrinsic(nir_intrinsic_barrier, 0);
   ASSERT_NE(intrinsic, nullptr);

   EXPECT_EQ(nir_intrinsic_memory_semantics(intrinsic), NIR_MEMORY_MAKE_AVAILABLE);
   EXPECT_NE(nir_intrinsic_memory_modes(intrinsic) & nir_var_image, 0);
   EXPECT_EQ(nir_intrinsic_memory_scope(intrinsic), SCOPE_DEVICE);
   EXPECT_EQ(nir_intrinsic_execution_scope(intrinsic), SCOPE_NONE);

   /* MakeTexelAvailable makes the texel available after it is written. */
   int barrier = intrinsic_position(shader, nir_intrinsic_barrier);
   int store = intrinsic_position(shader, nir_intrinsic_image_deref_store);
   ASSERT_GE(barrier, 0);
   ASSERT_GE(store, 0);
   EXPECT_GT(barrier, store);
}

TEST_F(AvailabilityVisibility, opcopymemory_visavail_both_combined)
{
   /*
               OpCapability Shader
               OpCapability VulkanMemoryModel
               OpCapability VulkanMemoryModelDeviceScope
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical Vulkan
               OpEntryPoint GLCompute %4 "main" %9
               OpExecutionMode %4 LocalSize 1 1 1
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 Block
               OpDecorate %9 DescriptorSet 0
               OpDecorate %9 Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
  %_struct_7 = OpTypeStruct %uint
%_ptr_StorageBuffer__struct_7 = OpTypePointer StorageBuffer %_struct_7
          %9 = OpVariable %_ptr_StorageBuffer__struct_7 StorageBuffer
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
     %device = OpConstant %int 1
  %workgroup = OpConstant %int 2
          %4 = OpFunction %void None %3
          %5 = OpLabel
         %13 = OpAccessChain %_ptr_StorageBuffer_uint %9 %int_0
               OpCopyMemory %13 %13 NonPrivatePointer|MakePointerAvailable|MakePointerVisible %device %workgroup
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010500, 0x00070000, 0x00000010, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x000014e1, 0x00020011, 0x000014e2, 0x0006000b,
      0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
      0x00000000, 0x00000003, 0x0006000f, 0x00000005, 0x00000002, 0x6e69616d,
      0x00000000, 0x00000003, 0x00060010, 0x00000002, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00050048, 0x00000004, 0x00000000, 0x00000023,
      0x00000000, 0x00030047, 0x00000004, 0x00000002, 0x00040047, 0x00000003,
      0x00000022, 0x00000000, 0x00040047, 0x00000003, 0x00000021, 0x00000000,
      0x00020013, 0x00000005, 0x00030021, 0x00000006, 0x00000005, 0x00040015,
      0x00000007, 0x00000020, 0x00000000, 0x0003001e, 0x00000004, 0x00000007,
      0x00040020, 0x00000008, 0x0000000c, 0x00000004, 0x0004003b, 0x00000008,
      0x00000003, 0x0000000c, 0x00040015, 0x00000009, 0x00000020, 0x00000001,
      0x0004002b, 0x00000009, 0x0000000a, 0x00000000, 0x00040020, 0x0000000b,
      0x0000000c, 0x00000007, 0x0004002b, 0x00000009, 0x0000000c, 0x00000001,
      0x0004002b, 0x00000009, 0x0000000d, 0x00000002, 0x00050036, 0x00000005,
      0x00000002, 0x00000000, 0x00000006, 0x000200f8, 0x0000000e, 0x00050041,
      0x0000000b, 0x0000000f, 0x00000003, 0x0000000a, 0x0006003f, 0x0000000f,
      0x0000000f, 0x00000038, 0x0000000c, 0x0000000d, 0x000100fd, 0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);

   nir_intrinsic_instr *first = find_intrinsic(nir_intrinsic_barrier, 0);
   nir_intrinsic_instr *second = find_intrinsic(nir_intrinsic_barrier, 1);
   ASSERT_NE(first, nullptr);
   ASSERT_NE(second, nullptr);

   EXPECT_EQ(nir_intrinsic_memory_semantics(first), NIR_MEMORY_MAKE_VISIBLE);
   EXPECT_NE(nir_intrinsic_memory_modes(first) & nir_var_mem_ssbo, 0);
   EXPECT_EQ(nir_intrinsic_memory_scope(first), SCOPE_WORKGROUP);
   EXPECT_EQ(nir_intrinsic_execution_scope(first), SCOPE_NONE);

   EXPECT_EQ(nir_intrinsic_memory_semantics(second), NIR_MEMORY_MAKE_AVAILABLE);
   EXPECT_NE(nir_intrinsic_memory_modes(second) & nir_var_mem_ssbo, 0);
   EXPECT_EQ(nir_intrinsic_memory_scope(second), SCOPE_DEVICE);
   EXPECT_EQ(nir_intrinsic_execution_scope(first), SCOPE_NONE);
}

TEST_F(AvailabilityVisibility, opcopymemory_visavail_both_separate)
{
   /*
               OpCapability Shader
               OpCapability VulkanMemoryModel
               OpCapability VulkanMemoryModelDeviceScope
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical Vulkan
               OpEntryPoint GLCompute %4 "main" %9
               OpExecutionMode %4 LocalSize 1 1 1
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 Block
               OpDecorate %9 DescriptorSet 0
               OpDecorate %9 Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
  %_struct_7 = OpTypeStruct %uint
%_ptr_StorageBuffer__struct_7 = OpTypePointer StorageBuffer %_struct_7
          %9 = OpVariable %_ptr_StorageBuffer__struct_7 StorageBuffer
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
     %device = OpConstant %int 1
  %workgroup = OpConstant %int 2
          %4 = OpFunction %void None %3
          %5 = OpLabel
         %13 = OpAccessChain %_ptr_StorageBuffer_uint %9 %int_0
               OpCopyMemory %13 %13 NonPrivatePointer|MakePointerAvailable %device NonPrivatePointer|MakePointerVisible %workgroup
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010500, 0x00070000, 0x00000010, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x000014e1, 0x00020011, 0x000014e2, 0x0006000b,
      0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
      0x00000000, 0x00000003, 0x0006000f, 0x00000005, 0x00000002, 0x6e69616d,
      0x00000000, 0x00000003, 0x00060010, 0x00000002, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00050048, 0x00000004, 0x00000000, 0x00000023,
      0x00000000, 0x00030047, 0x00000004, 0x00000002, 0x00040047, 0x00000003,
      0x00000022, 0x00000000, 0x00040047, 0x00000003, 0x00000021, 0x00000000,
      0x00020013, 0x00000005, 0x00030021, 0x00000006, 0x00000005, 0x00040015,
      0x00000007, 0x00000020, 0x00000000, 0x0003001e, 0x00000004, 0x00000007,
      0x00040020, 0x00000008, 0x0000000c, 0x00000004, 0x0004003b, 0x00000008,
      0x00000003, 0x0000000c, 0x00040015, 0x00000009, 0x00000020, 0x00000001,
      0x0004002b, 0x00000009, 0x0000000a, 0x00000000, 0x00040020, 0x0000000b,
      0x0000000c, 0x00000007, 0x0004002b, 0x00000009, 0x0000000c, 0x00000001,
      0x0004002b, 0x00000009, 0x0000000d, 0x00000002, 0x00050036, 0x00000005,
      0x00000002, 0x00000000, 0x00000006, 0x000200f8, 0x0000000e, 0x00050041,
      0x0000000b, 0x0000000f, 0x00000003, 0x0000000a, 0x0007003f, 0x0000000f,
      0x0000000f, 0x00000028, 0x0000000c, 0x00000030, 0x0000000d, 0x000100fd,
      0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);

   nir_intrinsic_instr *first = find_intrinsic(nir_intrinsic_barrier, 0);
   nir_intrinsic_instr *second = find_intrinsic(nir_intrinsic_barrier, 1);
   ASSERT_NE(first, nullptr);
   ASSERT_NE(second, nullptr);

   EXPECT_EQ(nir_intrinsic_memory_semantics(first), NIR_MEMORY_MAKE_VISIBLE);
   EXPECT_NE(nir_intrinsic_memory_modes(first) & nir_var_mem_ssbo, 0);
   EXPECT_EQ(nir_intrinsic_memory_scope(first), SCOPE_WORKGROUP);
   EXPECT_EQ(nir_intrinsic_execution_scope(first), SCOPE_NONE);

   EXPECT_EQ(nir_intrinsic_memory_semantics(second), NIR_MEMORY_MAKE_AVAILABLE);
   EXPECT_NE(nir_intrinsic_memory_modes(second) & nir_var_mem_ssbo, 0);
   EXPECT_EQ(nir_intrinsic_memory_scope(second), SCOPE_DEVICE);
   EXPECT_EQ(nir_intrinsic_execution_scope(second), SCOPE_NONE);
}

TEST_F(AvailabilityVisibility, opcopymemory_avail)
{
   /*
               OpCapability Shader
               OpCapability VulkanMemoryModel
               OpCapability VulkanMemoryModelDeviceScope
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical Vulkan
               OpEntryPoint GLCompute %4 "main" %9
               OpExecutionMode %4 LocalSize 1 1 1
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 Block
               OpDecorate %9 DescriptorSet 0
               OpDecorate %9 Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
  %_struct_7 = OpTypeStruct %uint
%_ptr_StorageBuffer__struct_7 = OpTypePointer StorageBuffer %_struct_7
          %9 = OpVariable %_ptr_StorageBuffer__struct_7 StorageBuffer
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
     %device = OpConstant %int 1
          %4 = OpFunction %void None %3
          %5 = OpLabel
         %13 = OpAccessChain %_ptr_StorageBuffer_uint %9 %int_0
               OpCopyMemory %13 %13 NonPrivatePointer|MakePointerAvailable %device
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010500, 0x00070000, 0x0000000f, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x000014e1, 0x00020011, 0x000014e2, 0x0006000b,
      0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
      0x00000000, 0x00000003, 0x0006000f, 0x00000005, 0x00000002, 0x6e69616d,
      0x00000000, 0x00000003, 0x00060010, 0x00000002, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00050048, 0x00000004, 0x00000000, 0x00000023,
      0x00000000, 0x00030047, 0x00000004, 0x00000002, 0x00040047, 0x00000003,
      0x00000022, 0x00000000, 0x00040047, 0x00000003, 0x00000021, 0x00000000,
      0x00020013, 0x00000005, 0x00030021, 0x00000006, 0x00000005, 0x00040015,
      0x00000007, 0x00000020, 0x00000000, 0x0003001e, 0x00000004, 0x00000007,
      0x00040020, 0x00000008, 0x0000000c, 0x00000004, 0x0004003b, 0x00000008,
      0x00000003, 0x0000000c, 0x00040015, 0x00000009, 0x00000020, 0x00000001,
      0x0004002b, 0x00000009, 0x0000000a, 0x00000000, 0x00040020, 0x0000000b,
      0x0000000c, 0x00000007, 0x0004002b, 0x00000009, 0x0000000c, 0x00000001,
      0x00050036, 0x00000005, 0x00000002, 0x00000000, 0x00000006, 0x000200f8,
      0x0000000d, 0x00050041, 0x0000000b, 0x0000000e, 0x00000003, 0x0000000a,
      0x0005003f, 0x0000000e, 0x0000000e, 0x00000028, 0x0000000c, 0x000100fd,
      0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);

   nir_intrinsic_instr *intrinsic = find_intrinsic(nir_intrinsic_barrier, 0);
   ASSERT_NE(intrinsic, nullptr);

   EXPECT_EQ(nir_intrinsic_memory_semantics(intrinsic), NIR_MEMORY_MAKE_AVAILABLE);
   EXPECT_NE(nir_intrinsic_memory_modes(intrinsic) & nir_var_mem_ssbo, 0);
   EXPECT_EQ(nir_intrinsic_memory_scope(intrinsic), SCOPE_DEVICE);
   EXPECT_EQ(nir_intrinsic_execution_scope(intrinsic), SCOPE_NONE);
}

TEST_F(AvailabilityVisibility, opcopymemory_vis)
{
   /*
               OpCapability Shader
               OpCapability VulkanMemoryModel
               OpCapability VulkanMemoryModelDeviceScope
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical Vulkan
               OpEntryPoint GLCompute %4 "main" %9
               OpExecutionMode %4 LocalSize 1 1 1
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 Block
               OpDecorate %9 DescriptorSet 0
               OpDecorate %9 Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
  %_struct_7 = OpTypeStruct %uint
%_ptr_StorageBuffer__struct_7 = OpTypePointer StorageBuffer %_struct_7
          %9 = OpVariable %_ptr_StorageBuffer__struct_7 StorageBuffer
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
  %workgroup = OpConstant %int 2
          %4 = OpFunction %void None %3
          %5 = OpLabel
         %13 = OpAccessChain %_ptr_StorageBuffer_uint %9 %int_0
               OpCopyMemory %13 %13 NonPrivatePointer|MakePointerVisible %workgroup
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010500, 0x00070000, 0x0000000f, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x000014e1, 0x00020011, 0x000014e2, 0x0006000b,
      0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
      0x00000000, 0x00000003, 0x0006000f, 0x00000005, 0x00000002, 0x6e69616d,
      0x00000000, 0x00000003, 0x00060010, 0x00000002, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00050048, 0x00000004, 0x00000000, 0x00000023,
      0x00000000, 0x00030047, 0x00000004, 0x00000002, 0x00040047, 0x00000003,
      0x00000022, 0x00000000, 0x00040047, 0x00000003, 0x00000021, 0x00000000,
      0x00020013, 0x00000005, 0x00030021, 0x00000006, 0x00000005, 0x00040015,
      0x00000007, 0x00000020, 0x00000000, 0x0003001e, 0x00000004, 0x00000007,
      0x00040020, 0x00000008, 0x0000000c, 0x00000004, 0x0004003b, 0x00000008,
      0x00000003, 0x0000000c, 0x00040015, 0x00000009, 0x00000020, 0x00000001,
      0x0004002b, 0x00000009, 0x0000000a, 0x00000000, 0x00040020, 0x0000000b,
      0x0000000c, 0x00000007, 0x0004002b, 0x00000009, 0x0000000c, 0x00000002,
      0x00050036, 0x00000005, 0x00000002, 0x00000000, 0x00000006, 0x000200f8,
      0x0000000d, 0x00050041, 0x0000000b, 0x0000000e, 0x00000003, 0x0000000a,
      0x0005003f, 0x0000000e, 0x0000000e, 0x00000030, 0x0000000c, 0x000100fd,
      0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);

   nir_intrinsic_instr *intrinsic = find_intrinsic(nir_intrinsic_barrier, 0);
   ASSERT_NE(intrinsic, nullptr);

   EXPECT_EQ(nir_intrinsic_memory_semantics(intrinsic), NIR_MEMORY_MAKE_VISIBLE);
   EXPECT_NE(nir_intrinsic_memory_modes(intrinsic) & nir_var_mem_ssbo, 0);
   EXPECT_EQ(nir_intrinsic_memory_scope(intrinsic), SCOPE_WORKGROUP);
   EXPECT_EQ(nir_intrinsic_execution_scope(intrinsic), SCOPE_NONE);
}

TEST_F(AvailabilityVisibility, opimageread_vis_opimagewrite_avail_coherent)
{
   /*
    ; Extra arguments for spirv-as --target-env vulkan1.2

                  OpCapability Shader
                  OpCapability VulkanMemoryModel
                  OpCapability VulkanMemoryModelDeviceScope
                  OpMemoryModel Logical Vulkan
                  OpEntryPoint GLCompute %main "main" %img
                  OpExecutionMode %main LocalSize 1 1 1
                  OpDecorate %img DescriptorSet 0
                  OpDecorate %img Binding 0
          %void = OpTypeVoid
            %fn = OpTypeFunction %void
          %uint = OpTypeInt 32 0
           %int = OpTypeInt 32 1
         %v2int = OpTypeVector %int 2
        %v4uint = OpTypeVector %uint 4
         %image = OpTypeImage %uint 2D 0 0 0 2 R32ui
       %ptr_img = OpTypePointer UniformConstant %image
           %img = OpVariable %ptr_img UniformConstant
        %device = OpConstant %int 1
         %int_0 = OpConstant %int 0
         %coord = OpConstantComposite %v2int %int_0 %int_0
          %main = OpFunction %void None %fn
         %entry = OpLabel
             %i = OpLoad %image %img
             %t = OpImageRead %v4uint %i %coord NonPrivateTexel|MakeTexelVisible %device
            %i2 = OpLoad %image %img
                  OpImageWrite %i2 %coord %t NonPrivateTexel|MakeTexelAvailable %device
                  OpReturn
                  OpFunctionEnd
    */
   static const uint32_t words[] = {
      0x07230203, 0x00010500, 0x00070000, 0x00000012, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x000014e1, 0x00020011, 0x000014e2, 0x0003000e,
      0x00000000, 0x00000003, 0x0006000f, 0x00000005, 0x00000001, 0x6e69616d,
      0x00000000, 0x00000002, 0x00060010, 0x00000001, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00040047, 0x00000002, 0x00000022, 0x00000000,
      0x00040047, 0x00000002, 0x00000021, 0x00000000, 0x00020013, 0x00000003,
      0x00030021, 0x00000004, 0x00000003, 0x00040015, 0x00000005, 0x00000020,
      0x00000000, 0x00040015, 0x00000006, 0x00000020, 0x00000001, 0x00040017,
      0x00000007, 0x00000006, 0x00000002, 0x00040017, 0x00000008, 0x00000005,
      0x00000004, 0x00090019, 0x00000009, 0x00000005, 0x00000001, 0x00000000,
      0x00000000, 0x00000000, 0x00000002, 0x00000021, 0x00040020, 0x0000000a,
      0x00000000, 0x00000009, 0x0004003b, 0x0000000a, 0x00000002, 0x00000000,
      0x0004002b, 0x00000006, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006,
      0x0000000c, 0x00000000, 0x0005002c, 0x00000007, 0x0000000d, 0x0000000c,
      0x0000000c, 0x00050036, 0x00000003, 0x00000001, 0x00000000, 0x00000004,
      0x000200f8, 0x0000000e, 0x0004003d, 0x00000009, 0x0000000f, 0x00000002,
      0x00070062, 0x00000008, 0x00000010, 0x0000000f, 0x0000000d, 0x00000600,
      0x0000000b, 0x0004003d, 0x00000009, 0x00000011, 0x00000002, 0x00060063,
      0x00000011, 0x0000000d, 0x00000010, 0x00000500, 0x0000000b, 0x000100fd,
      0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words);

   /* MakeTexelVisible/MakeTexelAvailable make the image access itself
    * visible/available, so the access is coherent at that scope. */
   nir_intrinsic_instr *load = find_intrinsic(nir_intrinsic_image_deref_load, 0);
   ASSERT_NE(load, nullptr);
   EXPECT_NE(nir_intrinsic_access(load) & ACCESS_COHERENT, 0);

   nir_intrinsic_instr *store = find_intrinsic(nir_intrinsic_image_deref_store, 0);
   ASSERT_NE(store, nullptr);
   EXPECT_NE(nir_intrinsic_access(store) & ACCESS_COHERENT, 0);
}

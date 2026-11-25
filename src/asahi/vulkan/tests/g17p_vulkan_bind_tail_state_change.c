/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Verify HAL200 incremental FS PPP updates and bind-tail elision across
 * push-constant-only draws and non-FS PPP state changes (scissor, viewport,
 * cull mode, depth compare op, and color blend).
 *
 * Driven under asahi_noop_drm_shim with ASAHI_SHIM_VALIDATE_G17P_STREAM=1 and
 * ASAHI_SHIM_G17P_EXPECT_BIND_TAIL_COUNTS=6,2,2:
 *   Draw 0: initial draw                    -> full bind tail (#1)
 *   Draw 1: push-constant-only change       -> elided tail (#1) + 28B FS PPP (#1)
 *   Draw 2: dynamic scissor + push const    -> full bind tail (#2)
 *   Draw 3: dynamic viewport + push const   -> full bind tail (#3)
 *   Draw 4: dynamic cull mode + push const  -> full bind tail (#4)
 *   Draw 5: dynamic depth op + push const   -> full bind tail (#5)
 *   Draw 6: blend-enabled pipe + push const -> full bind tail (#6)
 *   Draw 7: push-constant-only change       -> elided tail (#2) + 28B FS PPP (#2)
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(expr)                                                            \
   do {                                                                        \
      VkResult _r = (expr);                                                    \
      if (_r != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s failed (%d) at line %d\n", #expr, (int)_r,        \
                 __LINE__);                                                    \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

struct push_const {
   uint32_t draw_id;
   float z;
   float pad0;
   float pad1;
   float color[4];
};

/* Minimal GLSL 450 VS reading PC.{draw_id, z, color} and emitting 2 varyings. */
static const uint32_t test_vert_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x0000008b, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0009000f, 0x00000000,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x0000006e, 0x00000080,
   0x00000086, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00697274, 0x00060005,
   0x0000000b, 0x565f6c67, 0x65747265, 0x646e4978, 0x00007865, 0x00040005,
   0x00000010, 0x6e726f63, 0x00007265, 0x00040005, 0x00000016, 0x70696c66,
   0x00000000, 0x00030005, 0x0000001a, 0x00004350, 0x00050006, 0x0000001a,
   0x00000000, 0x77617264, 0x0064695f, 0x00040006, 0x0000001a, 0x00000001,
   0x0000007a, 0x00050006, 0x0000001a, 0x00000002, 0x30646170, 0x00000000,
   0x00050006, 0x0000001a, 0x00000003, 0x31646170, 0x00000000, 0x00050006,
   0x0000001a, 0x00000004, 0x6f6c6f63, 0x00000072, 0x00030005, 0x0000001c,
   0x00006370, 0x00030005, 0x00000024, 0x00000063, 0x00030005, 0x0000003a,
   0x0000786a, 0x00030005, 0x00000044, 0x0000796a, 0x00030005, 0x00000053,
   0x00000070, 0x00060005, 0x0000006c, 0x505f6c67, 0x65567265, 0x78657472,
   0x00000000, 0x00060006, 0x0000006c, 0x00000000, 0x505f6c67, 0x7469736f,
   0x006e6f69, 0x00070006, 0x0000006c, 0x00000001, 0x505f6c67, 0x746e696f,
   0x657a6953, 0x00000000, 0x00070006, 0x0000006c, 0x00000002, 0x435f6c67,
   0x4470696c, 0x61747369, 0x0065636e, 0x00070006, 0x0000006c, 0x00000003,
   0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000006e,
   0x00000000, 0x00040005, 0x00000080, 0x76755f76, 0x00000000, 0x00040005,
   0x00000086, 0x6f635f76, 0x00726f6c, 0x00040047, 0x0000000b, 0x0000000b,
   0x0000002a, 0x00030047, 0x0000001a, 0x00000002, 0x00050048, 0x0000001a,
   0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000001a, 0x00000001,
   0x00000023, 0x00000004, 0x00050048, 0x0000001a, 0x00000002, 0x00000023,
   0x00000008, 0x00050048, 0x0000001a, 0x00000003, 0x00000023, 0x0000000c,
   0x00050048, 0x0000001a, 0x00000004, 0x00000023, 0x00000010, 0x00030047,
   0x0000006c, 0x00000002, 0x00050048, 0x0000006c, 0x00000000, 0x0000000b,
   0x00000000, 0x00050048, 0x0000006c, 0x00000001, 0x0000000b, 0x00000001,
   0x00050048, 0x0000006c, 0x00000002, 0x0000000b, 0x00000003, 0x00050048,
   0x0000006c, 0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x00000080,
   0x0000001e, 0x00000000, 0x00040047, 0x00000086, 0x0000001e, 0x00000001,
   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015,
   0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007,
   0x00000006, 0x00040015, 0x00000009, 0x00000020, 0x00000001, 0x00040020,
   0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b,
   0x00000001, 0x0004002b, 0x00000006, 0x0000000e, 0x00000003, 0x00020014,
   0x00000014, 0x00040020, 0x00000015, 0x00000007, 0x00000014, 0x00030016,
   0x00000018, 0x00000020, 0x00040017, 0x00000019, 0x00000018, 0x00000004,
   0x0007001e, 0x0000001a, 0x00000006, 0x00000018, 0x00000018, 0x00000018,
   0x00000019, 0x00040020, 0x0000001b, 0x00000009, 0x0000001a, 0x0004003b,
   0x0000001b, 0x0000001c, 0x00000009, 0x0004002b, 0x00000009, 0x0000001d,
   0x00000000, 0x00040020, 0x0000001e, 0x00000009, 0x00000006, 0x0004002b,
   0x00000006, 0x0000002a, 0x00000001, 0x0004002b, 0x00000006, 0x0000002f,
   0x00000002, 0x0004002b, 0x00000006, 0x00000033, 0x00000000, 0x00040020,
   0x00000039, 0x00000007, 0x00000018, 0x0004002b, 0x00000006, 0x0000003f,
   0x00000007, 0x0004002b, 0x00000018, 0x00000042, 0x3c23d70a, 0x00040017,
   0x00000051, 0x00000018, 0x00000002, 0x00040020, 0x00000052, 0x00000007,
   0x00000051, 0x0004002b, 0x00000018, 0x00000054, 0xbf800000, 0x0004002b,
   0x00000018, 0x0000005f, 0x40400000, 0x0004001c, 0x0000006b, 0x00000018,
   0x0000002a, 0x0006001e, 0x0000006c, 0x00000019, 0x00000018, 0x0000006b,
   0x0000006b, 0x00040020, 0x0000006d, 0x00000003, 0x0000006c, 0x0004003b,
   0x0000006d, 0x0000006e, 0x00000003, 0x0004002b, 0x00000009, 0x00000070,
   0x00000001, 0x00040020, 0x00000071, 0x00000009, 0x00000018, 0x0004002b,
   0x00000018, 0x00000076, 0x358637bd, 0x0004002b, 0x00000018, 0x00000079,
   0x3f800000, 0x00040020, 0x0000007d, 0x00000003, 0x00000019, 0x00040020,
   0x0000007f, 0x00000003, 0x00000051, 0x0004003b, 0x0000007f, 0x00000080,
   0x00000003, 0x0004002b, 0x00000018, 0x00000082, 0x3f000000, 0x0004003b,
   0x0000007d, 0x00000086, 0x00000003, 0x0004002b, 0x00000009, 0x00000087,
   0x00000004, 0x00040020, 0x00000088, 0x00000009, 0x00000019, 0x00050036,
   0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
   0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0004003b, 0x00000007,
   0x00000010, 0x00000007, 0x0004003b, 0x00000015, 0x00000016, 0x00000007,
   0x0004003b, 0x00000007, 0x00000024, 0x00000007, 0x0004003b, 0x00000007,
   0x00000026, 0x00000007, 0x0004003b, 0x00000007, 0x0000002c, 0x00000007,
   0x0004003b, 0x00000039, 0x0000003a, 0x00000007, 0x0004003b, 0x00000039,
   0x00000044, 0x00000007, 0x0004003b, 0x00000052, 0x00000053, 0x00000007,
   0x0004003d, 0x00000009, 0x0000000c, 0x0000000b, 0x0004007c, 0x00000006,
   0x0000000d, 0x0000000c, 0x00050086, 0x00000006, 0x0000000f, 0x0000000d,
   0x0000000e, 0x0003003e, 0x00000008, 0x0000000f, 0x0004003d, 0x00000009,
   0x00000011, 0x0000000b, 0x0004007c, 0x00000006, 0x00000012, 0x00000011,
   0x00050089, 0x00000006, 0x00000013, 0x00000012, 0x0000000e, 0x0003003e,
   0x00000010, 0x00000013, 0x0004003d, 0x00000006, 0x00000017, 0x00000008,
   0x00050041, 0x0000001e, 0x0000001f, 0x0000001c, 0x0000001d, 0x0004003d,
   0x00000006, 0x00000020, 0x0000001f, 0x00050080, 0x00000006, 0x00000021,
   0x00000017, 0x00000020, 0x000500c7, 0x00000006, 0x00000022, 0x00000021,
   0x0000000e, 0x000500aa, 0x00000014, 0x00000023, 0x00000022, 0x0000000e,
   0x0003003e, 0x00000016, 0x00000023, 0x0004003d, 0x00000014, 0x00000025,
   0x00000016, 0x000300f7, 0x00000028, 0x00000000, 0x000400fa, 0x00000025,
   0x00000027, 0x00000036, 0x000200f8, 0x00000027, 0x0004003d, 0x00000006,
   0x00000029, 0x00000010, 0x000500aa, 0x00000014, 0x0000002b, 0x00000029,
   0x0000002a, 0x000300f7, 0x0000002e, 0x00000000, 0x000400fa, 0x0000002b,
   0x0000002d, 0x00000030, 0x000200f8, 0x0000002d, 0x0003003e, 0x0000002c,
   0x0000002f, 0x000200f9, 0x0000002e, 0x000200f8, 0x00000030, 0x0004003d,
   0x00000006, 0x00000031, 0x00000010, 0x000500aa, 0x00000014, 0x00000032,
   0x00000031, 0x0000002f, 0x000600a9, 0x00000006, 0x00000034, 0x00000032,
   0x0000002a, 0x00000033, 0x0003003e, 0x0000002c, 0x00000034, 0x000200f9,
   0x0000002e, 0x000200f8, 0x0000002e, 0x0004003d, 0x00000006, 0x00000035,
   0x0000002c, 0x0003003e, 0x00000026, 0x00000035, 0x000200f9, 0x00000028,
   0x000200f8, 0x00000036, 0x0004003d, 0x00000006, 0x00000037, 0x00000010,
   0x0003003e, 0x00000026, 0x00000037, 0x000200f9, 0x00000028, 0x000200f8,
   0x00000028, 0x0004003d, 0x00000006, 0x00000038, 0x00000026, 0x0003003e,
   0x00000024, 0x00000038, 0x0004003d, 0x00000006, 0x0000003b, 0x00000008,
   0x00050041, 0x0000001e, 0x0000003c, 0x0000001c, 0x0000001d, 0x0004003d,
   0x00000006, 0x0000003d, 0x0000003c, 0x000500c6, 0x00000006, 0x0000003e,
   0x0000003b, 0x0000003d, 0x000500c7, 0x00000006, 0x00000040, 0x0000003e,
   0x0000003f, 0x00040070, 0x00000018, 0x00000041, 0x00000040, 0x00050085,
   0x00000018, 0x00000043, 0x00000041, 0x00000042, 0x0003003e, 0x0000003a,
   0x00000043, 0x0004003d, 0x00000006, 0x00000045, 0x00000008, 0x000500c2,
   0x00000006, 0x00000046, 0x00000045, 0x0000000e, 0x00050041, 0x0000001e,
   0x00000047, 0x0000001c, 0x0000001d, 0x0004003d, 0x00000006, 0x00000048,
   0x00000047, 0x000500c6, 0x00000006, 0x00000049, 0x00000046, 0x00000048,
   0x000500c7, 0x00000006, 0x0000004a, 0x00000049, 0x0000003f, 0x00040070,
   0x00000018, 0x0000004b, 0x0000004a, 0x00050085, 0x00000018, 0x0000004c,
   0x0000004b, 0x00000042, 0x0003003e, 0x00000044, 0x0000004c, 0x0004003d,
   0x00000006, 0x0000004d, 0x00000024, 0x000500aa, 0x00000014, 0x0000004e,
   0x0000004d, 0x00000033, 0x000300f7, 0x00000050, 0x00000000, 0x000400fa,
   0x0000004e, 0x0000004f, 0x0000005a, 0x000200f8, 0x0000004f, 0x0004003d,
   0x00000018, 0x00000055, 0x0000003a, 0x00050081, 0x00000018, 0x00000056,
   0x00000054, 0x00000055, 0x0004003d, 0x00000018, 0x00000057, 0x00000044,
   0x00050081, 0x00000018, 0x00000058, 0x00000054, 0x00000057, 0x00050050,
   0x00000051, 0x00000059, 0x00000056, 0x00000058, 0x0003003e, 0x00000053,
   0x00000059, 0x000200f9, 0x00000050, 0x000200f8, 0x0000005a, 0x0004003d,
   0x00000006, 0x0000005b, 0x00000024, 0x000500aa, 0x00000014, 0x0000005c,
   0x0000005b, 0x0000002a, 0x000300f7, 0x0000005e, 0x00000000, 0x000400fa,
   0x0000005c, 0x0000005d, 0x00000065, 0x000200f8, 0x0000005d, 0x0004003d,
   0x00000018, 0x00000060, 0x0000003a, 0x00050083, 0x00000018, 0x00000061,
   0x0000005f, 0x00000060, 0x0004003d, 0x00000018, 0x00000062, 0x00000044,
   0x00050081, 0x00000018, 0x00000063, 0x00000054, 0x00000062, 0x00050050,
   0x00000051, 0x00000064, 0x00000061, 0x00000063, 0x0003003e, 0x00000053,
   0x00000064, 0x000200f9, 0x0000005e, 0x000200f8, 0x00000065, 0x0004003d,
   0x00000018, 0x00000066, 0x0000003a, 0x00050081, 0x00000018, 0x00000067,
   0x00000054, 0x00000066, 0x0004003d, 0x00000018, 0x00000068, 0x00000044,
   0x00050083, 0x00000018, 0x00000069, 0x0000005f, 0x00000068, 0x00050050,
   0x00000051, 0x0000006a, 0x00000067, 0x00000069, 0x0003003e, 0x00000053,
   0x0000006a, 0x000200f9, 0x0000005e, 0x000200f8, 0x0000005e, 0x000200f9,
   0x00000050, 0x000200f8, 0x00000050, 0x0004003d, 0x00000051, 0x0000006f,
   0x00000053, 0x00050041, 0x00000071, 0x00000072, 0x0000001c, 0x00000070,
   0x0004003d, 0x00000018, 0x00000073, 0x00000072, 0x0004003d, 0x00000006,
   0x00000074, 0x00000008, 0x00040070, 0x00000018, 0x00000075, 0x00000074,
   0x00050085, 0x00000018, 0x00000077, 0x00000075, 0x00000076, 0x00050083,
   0x00000018, 0x00000078, 0x00000073, 0x00000077, 0x00050051, 0x00000018,
   0x0000007a, 0x0000006f, 0x00000000, 0x00050051, 0x00000018, 0x0000007b,
   0x0000006f, 0x00000001, 0x00070050, 0x00000019, 0x0000007c, 0x0000007a,
   0x0000007b, 0x00000078, 0x00000079, 0x00050041, 0x0000007d, 0x0000007e,
   0x0000006e, 0x0000001d, 0x0003003e, 0x0000007e, 0x0000007c, 0x0004003d,
   0x00000051, 0x00000081, 0x00000053, 0x0005008e, 0x00000051, 0x00000083,
   0x00000081, 0x00000082, 0x00050050, 0x00000051, 0x00000084, 0x00000082,
   0x00000082, 0x00050081, 0x00000051, 0x00000085, 0x00000083, 0x00000084,
   0x0003003e, 0x00000080, 0x00000085, 0x00050041, 0x00000088, 0x00000089,
   0x0000001c, 0x00000087, 0x0004003d, 0x00000019, 0x0000008a, 0x00000089,
   0x0003003e, 0x00000086, 0x0000008a, 0x000100fd, 0x00010038,
};

/* Minimal GLSL 450 FS reading PC.{draw_id, color} + varyings. */
static const uint32_t test_frag_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x0000003d, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0008000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x0000002a, 0x0000002c,
   0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2,
   0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008,
   0x0000006d, 0x00040005, 0x0000000b, 0x76755f76, 0x00000000, 0x00030005,
   0x0000001a, 0x00004350, 0x00050006, 0x0000001a, 0x00000000, 0x77617264,
   0x0064695f, 0x00040006, 0x0000001a, 0x00000001, 0x0000007a, 0x00050006,
   0x0000001a, 0x00000002, 0x30646170, 0x00000000, 0x00050006, 0x0000001a,
   0x00000003, 0x31646170, 0x00000000, 0x00050006, 0x0000001a, 0x00000004,
   0x6f6c6f63, 0x00000072, 0x00030005, 0x0000001c, 0x00006370, 0x00050005,
   0x0000002a, 0x5f74756f, 0x6f6c6f63, 0x00000072, 0x00040005, 0x0000002c,
   0x6f635f76, 0x00726f6c, 0x00040047, 0x0000000b, 0x0000001e, 0x00000000,
   0x00030047, 0x0000001a, 0x00000002, 0x00050048, 0x0000001a, 0x00000000,
   0x00000023, 0x00000000, 0x00050048, 0x0000001a, 0x00000001, 0x00000023,
   0x00000004, 0x00050048, 0x0000001a, 0x00000002, 0x00000023, 0x00000008,
   0x00050048, 0x0000001a, 0x00000003, 0x00000023, 0x0000000c, 0x00050048,
   0x0000001a, 0x00000004, 0x00000023, 0x00000010, 0x00040047, 0x0000002a,
   0x0000001e, 0x00000000, 0x00040047, 0x0000002c, 0x0000001e, 0x00000001,
   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
   0x00000006, 0x00000020, 0x00040020, 0x00000007, 0x00000007, 0x00000006,
   0x00040017, 0x00000009, 0x00000006, 0x00000002, 0x00040020, 0x0000000a,
   0x00000001, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001,
   0x00040015, 0x0000000c, 0x00000020, 0x00000000, 0x0004002b, 0x0000000c,
   0x0000000d, 0x00000000, 0x00040020, 0x0000000e, 0x00000001, 0x00000006,
   0x0004002b, 0x00000006, 0x00000011, 0x40e00000, 0x0004002b, 0x0000000c,
   0x00000013, 0x00000001, 0x0004002b, 0x00000006, 0x00000016, 0x41300000,
   0x00040017, 0x00000019, 0x00000006, 0x00000004, 0x0007001e, 0x0000001a,
   0x0000000c, 0x00000006, 0x00000006, 0x00000006, 0x00000019, 0x00040020,
   0x0000001b, 0x00000009, 0x0000001a, 0x0004003b, 0x0000001b, 0x0000001c,
   0x00000009, 0x00040015, 0x0000001d, 0x00000020, 0x00000001, 0x0004002b,
   0x0000001d, 0x0000001e, 0x00000000, 0x00040020, 0x0000001f, 0x00000009,
   0x0000000c, 0x0004002b, 0x0000000c, 0x00000022, 0x0000001f, 0x0004002b,
   0x00000006, 0x00000025, 0x3d000000, 0x00040020, 0x00000029, 0x00000003,
   0x00000019, 0x0004003b, 0x00000029, 0x0000002a, 0x00000003, 0x00040020,
   0x0000002b, 0x00000001, 0x00000019, 0x0004003b, 0x0000002b, 0x0000002c,
   0x00000001, 0x00040017, 0x0000002d, 0x00000006, 0x00000003, 0x0004002b,
   0x0000001d, 0x00000030, 0x00000004, 0x00040020, 0x00000031, 0x00000009,
   0x00000019, 0x0004002b, 0x00000006, 0x00000038, 0x3e000000, 0x00050036,
   0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
   0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x00050041, 0x0000000e,
   0x0000000f, 0x0000000b, 0x0000000d, 0x0004003d, 0x00000006, 0x00000010,
   0x0000000f, 0x00050085, 0x00000006, 0x00000012, 0x00000010, 0x00000011,
   0x00050041, 0x0000000e, 0x00000014, 0x0000000b, 0x00000013, 0x0004003d,
   0x00000006, 0x00000015, 0x00000014, 0x00050085, 0x00000006, 0x00000017,
   0x00000015, 0x00000016, 0x00050081, 0x00000006, 0x00000018, 0x00000012,
   0x00000017, 0x00050041, 0x0000001f, 0x00000020, 0x0000001c, 0x0000001e,
   0x0004003d, 0x0000000c, 0x00000021, 0x00000020, 0x000500c7, 0x0000000c,
   0x00000023, 0x00000021, 0x00000022, 0x00040070, 0x00000006, 0x00000024,
   0x00000023, 0x00050085, 0x00000006, 0x00000026, 0x00000024, 0x00000025,
   0x00050081, 0x00000006, 0x00000027, 0x00000018, 0x00000026, 0x0006000c,
   0x00000006, 0x00000028, 0x00000001, 0x0000000a, 0x00000027, 0x0003003e,
   0x00000008, 0x00000028, 0x0004003d, 0x00000019, 0x0000002e, 0x0000002c,
   0x0008004f, 0x0000002d, 0x0000002f, 0x0000002e, 0x0000002e, 0x00000000,
   0x00000001, 0x00000002, 0x00050041, 0x00000031, 0x00000032, 0x0000001c,
   0x00000030, 0x0004003d, 0x00000019, 0x00000033, 0x00000032, 0x0008004f,
   0x0000002d, 0x00000034, 0x00000033, 0x00000033, 0x00000001, 0x00000002,
   0x00000000, 0x0004003d, 0x00000006, 0x00000035, 0x00000008, 0x00060050,
   0x0000002d, 0x00000036, 0x00000035, 0x00000035, 0x00000035, 0x0008000c,
   0x0000002d, 0x00000037, 0x00000001, 0x0000002e, 0x0000002f, 0x00000034,
   0x00000036, 0x00050051, 0x00000006, 0x00000039, 0x00000037, 0x00000000,
   0x00050051, 0x00000006, 0x0000003a, 0x00000037, 0x00000001, 0x00050051,
   0x00000006, 0x0000003b, 0x00000037, 0x00000002, 0x00070050, 0x00000019,
   0x0000003c, 0x00000039, 0x0000003a, 0x0000003b, 0x00000038, 0x0003003e,
   0x0000002a, 0x0000003c, 0x000100fd, 0x00010038,
};

/* Minimal GLSL 450 FS reading a dynamic-indexed set-0 UBO + varyings, without
 * reading push constants (agx3_uses_root = true, agx3_root_reads = 0). */
static const uint32_t test_frag_no_push_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x00000028, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0008000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000c, 0x00000018, 0x0000001a,
   0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2,
   0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008,
   0x00786469, 0x00040005, 0x0000000c, 0x76755f76, 0x00000000, 0x00050005,
   0x00000018, 0x5f74756f, 0x6f6c6f63, 0x00000072, 0x00040005, 0x0000001a,
   0x6f635f76, 0x00726f6c, 0x00030005, 0x0000001e, 0x00000055, 0x00050006,
   0x0000001e, 0x00000000, 0x6f6c6f63, 0x00007372, 0x00030005, 0x00000020,
   0x00000075, 0x00040047, 0x0000000c, 0x0000001e, 0x00000000, 0x00040047,
   0x00000018, 0x0000001e, 0x00000000, 0x00040047, 0x0000001a, 0x0000001e,
   0x00000001, 0x00040047, 0x0000001d, 0x00000006, 0x00000010, 0x00030047,
   0x0000001e, 0x00000002, 0x00050048, 0x0000001e, 0x00000000, 0x00000023,
   0x00000000, 0x00040047, 0x00000020, 0x00000021, 0x00000000, 0x00040047,
   0x00000020, 0x00000022, 0x00000000, 0x00020013, 0x00000002, 0x00030021,
   0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000,
   0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00030016, 0x00000009,
   0x00000020, 0x00040017, 0x0000000a, 0x00000009, 0x00000002, 0x00040020,
   0x0000000b, 0x00000001, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c,
   0x00000001, 0x0004002b, 0x00000006, 0x0000000d, 0x00000000, 0x00040020,
   0x0000000e, 0x00000001, 0x00000009, 0x0004002b, 0x00000009, 0x00000011,
   0x40400000, 0x0004002b, 0x00000009, 0x00000013, 0x00000000, 0x00040017,
   0x00000016, 0x00000009, 0x00000004, 0x00040020, 0x00000017, 0x00000003,
   0x00000016, 0x0004003b, 0x00000017, 0x00000018, 0x00000003, 0x00040020,
   0x00000019, 0x00000001, 0x00000016, 0x0004003b, 0x00000019, 0x0000001a,
   0x00000001, 0x0004002b, 0x00000006, 0x0000001c, 0x00000004, 0x0004001c,
   0x0000001d, 0x00000016, 0x0000001c, 0x0003001e, 0x0000001e, 0x0000001d,
   0x00040020, 0x0000001f, 0x00000002, 0x0000001e, 0x0004003b, 0x0000001f,
   0x00000020, 0x00000002, 0x00040015, 0x00000021, 0x00000020, 0x00000001,
   0x0004002b, 0x00000021, 0x00000022, 0x00000000, 0x00040020, 0x00000024,
   0x00000002, 0x00000016, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
   0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008,
   0x00000007, 0x00050041, 0x0000000e, 0x0000000f, 0x0000000c, 0x0000000d,
   0x0004003d, 0x00000009, 0x00000010, 0x0000000f, 0x00050085, 0x00000009,
   0x00000012, 0x00000010, 0x00000011, 0x0008000c, 0x00000009, 0x00000014,
   0x00000001, 0x0000002b, 0x00000012, 0x00000013, 0x00000011, 0x0004006d,
   0x00000006, 0x00000015, 0x00000014, 0x0003003e, 0x00000008, 0x00000015,
   0x0004003d, 0x00000016, 0x0000001b, 0x0000001a, 0x0004003d, 0x00000006,
   0x00000023, 0x00000008, 0x00060041, 0x00000024, 0x00000025, 0x00000020,
   0x00000022, 0x00000023, 0x0004003d, 0x00000016, 0x00000026, 0x00000025,
   0x00050085, 0x00000016, 0x00000027, 0x0000001b, 0x00000026, 0x0003003e,
   0x00000018, 0x00000027, 0x000100fd, 0x00010038,
};

static uint32_t
find_mem_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
              VkMemoryPropertyFlags req)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (mp->memoryTypes[i].propertyFlags & req) == req)
         return i;
   }
   fprintf(stderr, "No matching memory type (bits=0x%x req=0x%x)\n", bits, req);
   exit(1);
}

static VkPipeline
make_pipeline(VkDevice dev, VkRenderPass rp, VkPipelineLayout pl,
              VkShaderModule vsm, VkShaderModule fsm, VkBool32 blend_enable)
{
   VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vsm,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fsm,
         .pName = "main",
      },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = blend_enable,
      .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &cba,
   };
   VkDynamicState dyn_states[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
      VK_DYNAMIC_STATE_CULL_MODE,
      VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
   };
   VkPipelineDynamicStateCreateInfo dyn = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = sizeof(dyn_states) / sizeof(dyn_states[0]),
      .pDynamicStates = dyn_states,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vps,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pDepthStencilState = &ds,
      .pColorBlendState = &cb,
      .pDynamicState = &dyn,
      .layout = pl,
      .renderPass = rp,
      .subpass = 0,
   };
   VkPipeline pipe = VK_NULL_HANDLE;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));
   return pipe;
}

int
main(int argc, char **argv)
{
   bool vs_only_push = false;
   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--vs-only-push"))
         vs_only_push = true;
   }
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "g17p_vulkan_bind_tail_state_change",
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst = VK_NULL_HANDLE;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t pdev_count = 0;
   CHECK(vkEnumeratePhysicalDevices(inst, &pdev_count, NULL));
   if (pdev_count == 0) {
      fprintf(stderr, "No Vulkan physical devices\n");
      return 1;
   }
   VkPhysicalDevice *pdevs = calloc(pdev_count, sizeof(*pdevs));
   CHECK(vkEnumeratePhysicalDevices(inst, &pdev_count, pdevs));
   VkPhysicalDevice pdev = pdevs[0];
   free(pdevs);

   uint32_t qf_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, NULL);
   VkQueueFamilyProperties *qfs = calloc(qf_count, sizeof(*qfs));
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, qfs);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < qf_count; ++i) {
      if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         qf = i;
         break;
      }
   }
   free(qfs);
   if (qf == UINT32_MAX) {
      fprintf(stderr, "No graphics queue family\n");
      return 1;
   }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf,
      .queueCount = 1,
      .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
   };
   VkDevice dev = VK_NULL_HANDLE;
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   VkQueue queue = VK_NULL_HANDLE;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);

   const uint32_t size = 128;
   VkImageCreateInfo color_ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {size, size, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage color_img = VK_NULL_HANDLE;
   CHECK(vkCreateImage(dev, &color_ici, NULL, &color_img));
   VkMemoryRequirements cmr;
   vkGetImageMemoryRequirements(dev, color_img, &cmr);
   VkMemoryAllocateInfo cmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = cmr.size,
      .memoryTypeIndex =
         find_mem_type(&mp, cmr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory color_mem = VK_NULL_HANDLE;
   CHECK(vkAllocateMemory(dev, &cmai, NULL, &color_mem));
   CHECK(vkBindImageMemory(dev, color_img, color_mem, 0));

   VkImageCreateInfo depth_ici = color_ici;
   depth_ici.format = VK_FORMAT_D32_SFLOAT;
   depth_ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   VkImage depth_img = VK_NULL_HANDLE;
   CHECK(vkCreateImage(dev, &depth_ici, NULL, &depth_img));
   VkMemoryRequirements dmr;
   vkGetImageMemoryRequirements(dev, depth_img, &dmr);
   VkMemoryAllocateInfo dmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = dmr.size,
      .memoryTypeIndex =
         find_mem_type(&mp, dmr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory depth_mem = VK_NULL_HANDLE;
   CHECK(vkAllocateMemory(dev, &dmai, NULL, &depth_mem));
   CHECK(vkBindImageMemory(dev, depth_img, depth_mem, 0));

   VkImageViewCreateInfo cvci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = color_img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView color_view = VK_NULL_HANDLE;
   CHECK(vkCreateImageView(dev, &cvci, NULL, &color_view));

   VkImageViewCreateInfo dvci = cvci;
   dvci.image = depth_img;
   dvci.format = VK_FORMAT_D32_SFLOAT;
   dvci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
   VkImageView depth_view = VK_NULL_HANDLE;
   CHECK(vkCreateImageView(dev, &dvci, NULL, &depth_view));

   VkAttachmentDescription atts[2] = {
      {
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      },
      {
         .format = VK_FORMAT_D32_SFLOAT,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      },
   };
   VkAttachmentReference color_ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkAttachmentReference depth_ref = {
      1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
   VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
      .pDepthStencilAttachment = &depth_ref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2,
      .pAttachments = atts,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   VkRenderPass rp = VK_NULL_HANDLE;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkImageView fb_views[2] = {color_view, depth_view};
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp,
      .attachmentCount = 2,
      .pAttachments = fb_views,
      .width = size,
      .height = size,
      .layers = 1,
   };
   VkFramebuffer fb = VK_NULL_HANDLE;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkShaderModule vsm = VK_NULL_HANDLE, fsm = VK_NULL_HANDLE;
   CHECK(vkCreateShaderModule(
      dev,
      &(VkShaderModuleCreateInfo){
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = sizeof(test_vert_spv),
         .pCode = test_vert_spv,
      },
      NULL, &vsm));
   CHECK(vkCreateShaderModule(
      dev,
      &(VkShaderModuleCreateInfo){
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = vs_only_push ? sizeof(test_frag_no_push_spv)
                                  : sizeof(test_frag_spv),
         .pCode = vs_only_push ? test_frag_no_push_spv : test_frag_spv,
      },
      NULL, &fsm));

   VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
   VkDescriptorPool dpool = VK_NULL_HANDLE;
   VkDescriptorSet dset = VK_NULL_HANDLE;
   VkBuffer ubo_buf = VK_NULL_HANDLE;
   VkDeviceMemory ubo_mem = VK_NULL_HANDLE;
   if (vs_only_push) {
      VkDescriptorSetLayoutBinding ubo_binding = {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      };
      CHECK(vkCreateDescriptorSetLayout(
         dev,
         &(VkDescriptorSetLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &ubo_binding,
         },
         NULL, &dsl));
      CHECK(vkCreateBuffer(
         dev,
         &(VkBufferCreateInfo){
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = 64,
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
         },
         NULL, &ubo_buf));
      VkMemoryRequirements umr;
      vkGetBufferMemoryRequirements(dev, ubo_buf, &umr);
      CHECK(vkAllocateMemory(
         dev,
         &(VkMemoryAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = umr.size,
            .memoryTypeIndex = find_mem_type(
               &mp, umr.memoryTypeBits,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
         },
         NULL, &ubo_mem));
      CHECK(vkBindBufferMemory(dev, ubo_buf, ubo_mem, 0));
      VkDescriptorPoolSize pool_size = {
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
      };
      CHECK(vkCreateDescriptorPool(
         dev,
         &(VkDescriptorPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
         },
         NULL, &dpool));
      CHECK(vkAllocateDescriptorSets(
         dev,
         &(VkDescriptorSetAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = dpool,
            .descriptorSetCount = 1,
            .pSetLayouts = &dsl,
         },
         &dset));
      VkDescriptorBufferInfo bi = {.buffer = ubo_buf, .offset = 0, .range = 64};
      vkUpdateDescriptorSets(
         dev, 1,
         &(VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = dset,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &bi,
         },
         0, NULL);
   }

   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size = sizeof(struct push_const),
   };
   VkPipelineLayout pl = VK_NULL_HANDLE;
   CHECK(vkCreatePipelineLayout(
      dev,
      &(VkPipelineLayoutCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount = vs_only_push ? 1 : 0,
         .pSetLayouts = vs_only_push ? &dsl : NULL,
         .pushConstantRangeCount = 1,
         .pPushConstantRanges = &pcr,
      },
      NULL, &pl));

   VkPipeline pipe_opaque = make_pipeline(dev, rp, pl, vsm, fsm, VK_FALSE);
   VkPipeline pipe_blend = make_pipeline(dev, rp, pl, vsm, fsm, VK_TRUE);

   VkCommandPool cp = VK_NULL_HANDLE;
   CHECK(vkCreateCommandPool(
      dev,
      &(VkCommandPoolCreateInfo){
         .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         .queueFamilyIndex = qf,
      },
      NULL, &cp));

   VkCommandBuffer cmd = VK_NULL_HANDLE;
   CHECK(vkAllocateCommandBuffers(
      dev,
      &(VkCommandBufferAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = cp,
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      },
      &cmd));

   CHECK(vkBeginCommandBuffer(
      cmd,
      &(VkCommandBufferBeginInfo){
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      }));

   VkClearValue clears[2];
   clears[0].color.float32[0] = 0.0f;
   clears[0].color.float32[1] = 0.0f;
   clears[0].color.float32[2] = 0.0f;
   clears[0].color.float32[3] = 1.0f;
   clears[1].depthStencil.depth = 1.0f;
   clears[1].depthStencil.stencil = 0;

   vkCmdBeginRenderPass(
      cmd,
      &(VkRenderPassBeginInfo){
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = rp,
         .framebuffer = fb,
         .renderArea = {{0, 0}, {size, size}},
         .clearValueCount = 2,
         .pClearValues = clears,
      },
      VK_SUBPASS_CONTENTS_INLINE);

   VkViewport vp0 = {0.0f, 0.0f, (float)size, (float)size, 0.0f, 1.0f};
   VkViewport vp1 = {8.0f, 8.0f, (float)(size - 16), (float)(size - 16), 0.1f, 0.9f};
   VkRect2D sc0 = {{0, 0}, {size, size}};
   VkRect2D sc1 = {{16, 16}, {size - 32, size - 32}};

   struct push_const pc = {
      .draw_id = 0,
      .z = 0.5f,
      .color = {0.2f, 0.4f, 0.6f, 0.5f},
   };

   /* Draw 0: initial full bind tail (#1). */
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_opaque);
   if (vs_only_push) {
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1,
                              &dset, 0, NULL);
   }
   vkCmdSetViewport(cmd, 0, 1, &vp0);
   vkCmdSetScissor(cmd, 0, 1, &sc0);
   vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
   vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_LESS_OR_EQUAL);
   vkCmdPushConstants(cmd, pl,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(pc), &pc);
   vkCmdDraw(cmd, 3, 1, 0, 0);

   /* Draw 1: push-constant-only change -> elided bind tail (#1) + 28B FS PPP (#1). */
   pc.draw_id = 1;
   pc.color[0] = 0.8f;
   vkCmdPushConstants(cmd, pl,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(pc), &pc);
   vkCmdDraw(cmd, 3, 1, 0, 0);

   /* Draw 2: scissor change -> must re-emit full bind tail (#2). */
   vkCmdSetScissor(cmd, 0, 1, &sc1);
   pc.draw_id = 2;
   pc.color[1] = 0.9f;
   vkCmdPushConstants(cmd, pl,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(pc), &pc);
   vkCmdDraw(cmd, 3, 1, 0, 0);

   /* Draw 3: viewport change -> must re-emit full bind tail (#3). */
   vkCmdSetViewport(cmd, 0, 1, &vp1);
   pc.draw_id = 3;
   pc.color[2] = 0.9f;
   vkCmdPushConstants(cmd, pl,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(pc), &pc);
   vkCmdDraw(cmd, 3, 1, 0, 0);

   /* Draw 4: rasterizer cull-mode change -> must re-emit full bind tail (#4). */
   vkCmdSetCullMode(cmd, VK_CULL_MODE_FRONT_BIT);
   pc.draw_id = 4;
   pc.color[0] = 0.3f;
   vkCmdPushConstants(cmd, pl,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(pc), &pc);
   vkCmdDraw(cmd, 3, 1, 0, 0);

   /* Draw 5: depth-stencil compare-op change -> must re-emit full bind tail (#5). */
   vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL);
   pc.draw_id = 5;
   pc.color[1] = 0.2f;
   vkCmdPushConstants(cmd, pl,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(pc), &pc);
   vkCmdDraw(cmd, 3, 1, 0, 0);

   /* Draw 6: blend state change (pipeline bind) -> must re-emit full bind tail (#6). */
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_blend);
   pc.draw_id = 6;
   pc.color[2] = 0.1f;
   vkCmdPushConstants(cmd, pl,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(pc), &pc);
   vkCmdDraw(cmd, 3, 1, 0, 0);

   /* Draw 7: push-constant-only change after blend bind -> elided (#2) + 28B FS PPP (#2). */
   pc.draw_id = 7;
   pc.color[0] = 0.95f;
   vkCmdPushConstants(cmd, pl,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(pc), &pc);
   vkCmdDraw(cmd, 3, 1, 0, 0);

   vkCmdEndRenderPass(cmd);
   CHECK(vkEndCommandBuffer(cmd));

   CHECK(vkQueueSubmit(
      queue, 1,
      &(VkSubmitInfo){
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &cmd,
      },
      VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyPipeline(dev, pipe_blend, NULL);
   vkDestroyPipeline(dev, pipe_opaque, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   if (vs_only_push) {
      vkDestroyDescriptorPool(dev, dpool, NULL);
      vkDestroyBuffer(dev, ubo_buf, NULL);
      vkFreeMemory(dev, ubo_mem, NULL);
      vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   }
   vkDestroyShaderModule(dev, fsm, NULL);
   vkDestroyShaderModule(dev, vsm, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, depth_view, NULL);
   vkDestroyImageView(dev, color_view, NULL);
   vkDestroyImage(dev, depth_img, NULL);
   vkFreeMemory(dev, depth_mem, NULL);
   vkDestroyImage(dev, color_img, NULL);
   vkFreeMemory(dev, color_mem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return 0;
}

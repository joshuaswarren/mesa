/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_SHADER_CODE_TYPE_BINARY_EXT import of compute shader objects. An
 * exported binary must import again and export the same bytes. A binary
 * whose root prefix (hk_shader::root_descriptor_size) or root metadata --
 * descriptor set count, root-relative uniform DMA spans -- no longer matches
 * what the compiler can produce must be refused with
 * VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT, even with its digest recomputed:
 * the digest is not keyed, so an application can always reseal its edit.
 *
 * Requires the no-op DRM shim. Creates, exports and imports shader objects;
 * never records or submits work.
 *
 * usage: hk-vulkan-shader-binary-root [--expect-root-size]
 *   The binary layout is detected; --expect-root-size fails the run unless
 *   the device's binaries carry the trimmed root prefix.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "hk_cmd_buffer.h"
#include "hk_shader.h"
#include "util/mesa-blake3.h"
#include "hk_vulkan_shader_binary_root_shaders.h"

#define REQUIRE(x)                                                             \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)
#define VK_CHECK(x)                                                            \
   do {                                                                        \
      VkResult r_ = (x);                                                       \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #x, r_);                                  \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

/* vk_shader.c's binary header: magic, driver ID, UUID, version, size, then
 * the BLAKE3 of the header (with a zero digest) and the payload. */
#define BIN_HEADER_B 80
#define BIN_SIZE_B   40
#define BIN_BLAKE3_B 48

/* hk_api_shader_serialize(): stage and four flag bytes, then one marker
 * byte and hk_shader_serialize() per variant. A compute shader has one. */
#define BIN_STAGE_B   (BIN_HEADER_B + 0)
#define BIN_FLAGS2_B  (BIN_HEADER_B + 2)
#define BIN_FLAGS3_B  (BIN_HEADER_B + 3)
#define BIN_FLAGS4_B  (BIN_HEADER_B + 4)
#define BIN_MARKER_B  (BIN_HEADER_B + 5)
#define BIN_INFO_B    (BIN_HEADER_B + 6)
#define BIN_INFO_END_B (BIN_INFO_B + sizeof(struct hk_shader_info))
/* blob_write_uint32() aligns the blob first. */
#define BIN_ROOT_B    ALIGN_POT(BIN_INFO_END_B, 4)

static const uint32_t full_root = sizeof(struct hk_root_descriptor_table);

static VkDevice dev;
static VkDescriptorSetLayout set_layout;
static PFN_vkCreateShadersEXT create_shaders;
static PFN_vkDestroyShaderEXT destroy_shader;
static PFN_vkGetShaderBinaryDataEXT get_binary;
static bool root_size_field;
static bool expect_root_size;
static unsigned failures;

struct binary {
   uint8_t *data;
   size_t size;
};

static VkShaderCreateInfoEXT
create_info(VkShaderCodeTypeEXT type, const void *code, size_t size)
{
   static const VkPushConstantRange push = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 8,
   };
   return (VkShaderCreateInfoEXT){
      .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .codeType = type,
      .codeSize = size,
      .pCode = code,
      .pName = "main",
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push,
   };
}

static struct binary
export_shader(VkShaderEXT shader)
{
   struct binary b = {0};
   VK_CHECK(get_binary(dev, shader, &b.size, NULL));
   b.data = malloc(b.size);
   REQUIRE(b.data);
   VK_CHECK(get_binary(dev, shader, &b.size, b.data));
   return b;
}

/* Import a binary; on success optionally return its re-export. */
static VkResult
import(const struct binary *b, struct binary *reexport)
{
   const VkShaderCreateInfoEXT info =
      create_info(VK_SHADER_CODE_TYPE_BINARY_EXT, b->data, b->size);
   VkShaderEXT shader = VK_NULL_HANDLE;
   VkResult result = create_shaders(dev, 1, &info, NULL, &shader);
   if (result == VK_SUCCESS) {
      REQUIRE(shader != VK_NULL_HANDLE);
      if (reexport)
         *reexport = export_shader(shader);
      destroy_shader(dev, shader, NULL);
   } else {
      REQUIRE(shader == VK_NULL_HANDLE);
   }
   return result;
}

static struct binary
copy(const struct binary *b)
{
   struct binary c = {malloc(b->size), b->size};
   REQUIRE(c.data);
   memcpy(c.data, b->data, b->size);
   return c;
}

/* Recompute the binary's digest over its edited payload. */
static void
reseal(struct binary *b)
{
   blake3_hash digest;
   struct mesa_blake3 ctx;
   memset(b->data + BIN_BLAKE3_B, 0, sizeof(digest));
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, b->data, BIN_HEADER_B);
   _mesa_blake3_update(&ctx, b->data + BIN_HEADER_B, b->size - BIN_HEADER_B);
   _mesa_blake3_final(&ctx, digest);
   memcpy(b->data + BIN_BLAKE3_B, digest, sizeof(digest));
}

static uint32_t
read32(const struct binary *b, size_t offset)
{
   uint32_t v;
   REQUIRE(offset + sizeof(v) <= b->size);
   memcpy(&v, b->data + offset, sizeof(v));
   return v;
}

static void
write_bytes(struct binary *b, size_t offset, const void *v, size_t size)
{
   REQUIRE(offset + size <= b->size);
   memcpy(b->data + offset, v, size);
}

static void
expect(const char *shader, const char *what, VkResult got, VkResult want)
{
   const bool ok = got == want;
   printf("%s %s: %s (got %d, want %d)\n", ok ? "PASS" : "FAIL", shader,
          what, got, want);
   if (!ok)
      failures++;
}

/* Edit one field of a copy of b, reseal it and import it. */
static void
expect_edit(const char *shader, const char *what, const struct binary *b,
            size_t offset, const void *v, size_t size, VkResult want)
{
   struct binary t = copy(b);
   write_bytes(&t, offset, v, size);
   reseal(&t);
   expect(shader, what, import(&t, NULL), want);
   free(t.data);
}

/* Cut a copy of b short at `size` bytes, fix up the header's size field (a
 * u64 after the magic, driver ID, UUID and version) and reseal it. */
static void
expect_truncated(const char *shader, const char *what, const struct binary *b,
                 size_t size)
{
   REQUIRE(size > BIN_HEADER_B && size < b->size);
   struct binary t = {malloc(size), size};
   REQUIRE(t.data);
   memcpy(t.data, b->data, size);
   const uint64_t size64 = size;
   write_bytes(&t, BIN_SIZE_B, &size64, sizeof(size64));
   reseal(&t);
   expect(shader, what, import(&t, NULL),
          VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   free(t.data);
}

/* With the uniform DMA spans dropped, only the set-pointer floor bounds the
 * root prefix: set the set count and prefix and import. */
static void
expect_floor(const char *shader, const char *what, const struct binary *b,
             size_t dma_count_B, size_t set_count_B, uint8_t sets,
             uint32_t root, VkResult want)
{
   const uint8_t no_spans = 0;
   struct binary t = copy(b);
   write_bytes(&t, dma_count_B, &no_spans, 1);
   write_bytes(&t, set_count_B, &sets, 1);
   write_bytes(&t, BIN_ROOT_B, &root, 4);
   reseal(&t);
   expect(shader, what, import(&t, NULL), want);
   free(t.data);
}

/* Whether the compiler's shader info, which starts with its stage, starts at
 * b_info_B. Neither a root prefix nor a binary size can read as a stage. */
static bool
compiler_info_at(const struct binary *b, size_t b_info_B)
{
   mesa_shader_stage stage;
   const size_t at = b_info_B + offsetof(struct agx_shader_info, stage);
   if (at + sizeof(stage) > b->size)
      return false;
   memcpy(&stage, b->data + at, sizeof(stage));
   return stage == MESA_SHADER_COMPUTE;
}

static void
check_shader(const char *name, const uint32_t *spirv, size_t spirv_size)
{
   const VkShaderCreateInfoEXT info =
      create_info(VK_SHADER_CODE_TYPE_SPIRV_EXT, spirv, spirv_size);
   VkShaderEXT shader;
   VK_CHECK(create_shaders(dev, 1, &info, NULL, &shader));
   struct binary b = export_shader(shader);
   destroy_shader(dev, shader, NULL);

   /* Locate the single compute variant: no compute-pair, identity-copy or
    * tensor-op prefix may precede it. */
   REQUIRE(b.size > BIN_ROOT_B + 4);
   REQUIRE(b.data[BIN_STAGE_B] == MESA_SHADER_COMPUTE);
   REQUIRE((b.data[BIN_FLAGS2_B] & 32) == 0);
   REQUIRE((b.data[BIN_FLAGS3_B] & (1 | 64 | 128)) == 0);
   REQUIRE(b.data[BIN_MARKER_B] == 0);
   const size_t set_count_B =
      BIN_INFO_B + offsetof(struct hk_shader_info, set_count);
   REQUIRE(b.data[set_count_B] == 1);
   /* Chips with trimmed roots write the root prefix after hk_shader_info. */
   const bool with_root = compiler_info_at(&b, BIN_ROOT_B + 4);
   REQUIRE(with_root != compiler_info_at(&b, BIN_INFO_END_B));
   root_size_field = with_root;
   REQUIRE(root_size_field || !expect_root_size);
   const size_t b_info_B = root_size_field ? BIN_ROOT_B + 4 : BIN_INFO_END_B;
   const size_t dma_count_B =
      b_info_B + offsetof(struct agx_shader_info, agx3_uniform_dma_count);
   const uint32_t root_uniform =
      read32(&b, BIN_INFO_B + offsetof(struct hk_shader_info,
                                       agx3_root_uniform));

   /* Unmodified: imports, and exports the same bytes again. */
   struct binary again = {0};
   expect(name, "unmodified binary", import(&b, &again), VK_SUCCESS);
   expect(name, "re-export matches",
          again.size == b.size && !memcmp(again.data, b.data, b.size)
             ? VK_SUCCESS : VK_ERROR_UNKNOWN,
          VK_SUCCESS);
   free(again.data);

   /* Every device: a binary that ends inside the shader info, or right
    * after it, is refused before any of the info is used. */
   const size_t cuts[] = {BIN_INFO_B + 1,
                          BIN_INFO_B + offsetof(struct hk_shader_info, set_count),
                          BIN_INFO_END_B - 1, BIN_INFO_END_B};
   for (unsigned i = 0; i < ARRAY_SIZE(cuts); ++i) {
      char what[64];
      snprintf(what, sizeof(what), "binary cut at byte %zu", cuts[i]);
      expect_truncated(name, what, &b, cuts[i]);
   }

   /* Every device: the root layout cannot name more sets than exist, nor
    * more DMA spans than the shader info holds. */
   const uint8_t too_many_sets = HK_MAX_SETS + 1;
   expect_edit(name, "set_count past HK_MAX_SETS", &b, set_count_B,
               &too_many_sets, 1, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   const uint8_t too_many_spans = AGX3_UNIFORM_DMA_MAX_SPANS + 1;
   expect_edit(name, "DMA span count past the maximum", &b, dma_count_B,
               &too_many_spans, 1, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   uint32_t root = full_root;
   if (root_size_field) {
      root = read32(&b, BIN_ROOT_B);
      printf("INFO %s: root prefix %u of %u bytes\n", name, root, full_root);
      /* What the compiler produces: the root pointer and the set pointer
       * at least, in 16-byte steps, at most the whole table. */
      REQUIRE(root >= 16 && root <= full_root &&
              (root % 16 == 0 || root == full_root));

      static const struct {
         const char *what;
         uint32_t size;
         bool valid;
      } sizes[] = {
         {"root prefix 8 (below the set pointers)", 8, false},
         {"root prefix 15", 15, false},
         {"root prefix 24 (not a 16-byte step)", 24, false},
         {"root prefix past the table", 0, false},
         {"root prefix 0 (untrimmed)", 0, true},
         {"root prefix of the whole table", 0, true},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(sizes); ++i) {
         uint32_t size = sizes[i].size;
         if (i == 3)
            size = ALIGN_POT(full_root + 1, 16);
         else if (i == 5)
            size = full_root;
         expect_edit(name, sizes[i].what, &b, BIN_ROOT_B, &size, 4,
                     sizes[i].valid ? VK_SUCCESS
                                    : VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }

      /* A set count whose pointers end past the binary's own prefix. */
      const uint8_t sets = MIN2(root / 8, HK_MAX_SETS + 1);
      expect_edit(name, "set_count past the root prefix", &b, set_count_B,
                  &sets, 1, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

      /* The root pointer and one set pointer take 16 bytes, two take 32. */
      const VkResult refused = VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
      expect_floor(name, "no spans, 1 set, root prefix 8", &b, dma_count_B,
                   set_count_B, 1, 8, refused);
      expect_floor(name, "no spans, 1 set, root prefix 16", &b, dma_count_B,
                   set_count_B, 1, 16, VK_SUCCESS);
      expect_floor(name, "no spans, 2 sets, root prefix 16", &b, dma_count_B,
                   set_count_B, 2, 16, refused);
      expect_floor(name, "no spans, 2 sets, root prefix 32", &b, dma_count_B,
                   set_count_B, 2, 32, VK_SUCCESS);
   }

   /* Root-relative uniform DMA spans must lie inside the root the launch
    * uploads. Only compilers that publish spans emit them. */
   const unsigned spans = b.data[dma_count_B];
   REQUIRE(spans <= AGX3_UNIFORM_DMA_MAX_SPANS);
   bool root_span = false;
   for (unsigned i = 0; i < spans; ++i) {
      const size_t span_B = b_info_B +
         offsetof(struct agx_shader_info, agx3_uniform_dma) +
         i * sizeof(struct agx3_uniform_dma_span);
      struct agx3_uniform_dma_span span;
      REQUIRE(span_B + sizeof(span) <= b.size);
      memcpy(&span, b.data + span_B, sizeof(span));
      if (span.source != AGX3_UNIFORM_DMA_POINTER ||
          span.pointer != root_uniform)
         continue;

      root_span = true;
      REQUIRE(span.offset_B + 2u * span.halfs <= root);
      printf("INFO %s: root span %u reads [%u, %u)\n", name, i, span.offset_B,
             span.offset_B + 2u * span.halfs);
      const uint32_t past = ALIGN_POT(root - 2u * span.halfs + 4, 4);
      expect_edit(name, "root span past the root prefix", &b,
                  span_B + offsetof(struct agx3_uniform_dma_span, offset_B),
                  &past, sizeof(past),
                  VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      if (root_size_field && span.offset_B + 2u * span.halfs > 16) {
         const uint32_t floor = 16;
         expect_edit(name, "root prefix below a root span", &b, BIN_ROOT_B,
                     &floor, 4, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }
   }
   if (!root_span)
      printf("INFO %s: no root-relative uniform DMA span\n", name);

   free(b.data);
}

int
main(int argc, char **argv)
{
   if (argc > 2 || (argc == 2 && strcmp(argv[1], "--expect-root-size")))
      return 2;
   expect_root_size = argc == 2;

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   VK_CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice physical;
   VkResult result = vkEnumeratePhysicalDevices(instance, &count, &physical);
   REQUIRE((result == VK_SUCCESS || result == VK_INCOMPLETE) && count == 1);
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(physical, &props);
   fprintf(stderr, "Device: %s\n", props.deviceName);

   VkPhysicalDeviceShaderObjectFeaturesEXT object_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &object_features,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   REQUIRE(object_features.shaderObject);

   const float priority = 1;
   const VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *ext = VK_EXT_SHADER_OBJECT_EXTENSION_NAME;
   object_features.pNext = NULL;
   const VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &object_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = &ext,
   };
   VK_CHECK(vkCreateDevice(physical, &dci, NULL, &dev));
   create_shaders =
      (PFN_vkCreateShadersEXT)vkGetDeviceProcAddr(dev, "vkCreateShadersEXT");
   destroy_shader =
      (PFN_vkDestroyShaderEXT)vkGetDeviceProcAddr(dev, "vkDestroyShaderEXT");
   get_binary = (PFN_vkGetShaderBinaryDataEXT)vkGetDeviceProcAddr(
      dev, "vkGetShaderBinaryDataEXT");
   REQUIRE(create_shaders && destroy_shader && get_binary);

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   VK_CHECK(vkCreateDescriptorSetLayout(dev, &lci, NULL, &set_layout));

   check_shader("set", root_set_comp, sizeof(root_set_comp));
   check_shader("push", root_push_comp, sizeof(root_push_comp));

   vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   printf("%s: %u failure(s)\n", failures ? "FAIL" : "PASS", failures);
   return failures ? 1 : 0;
}

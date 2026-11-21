// SPDX-License-Identifier: MIT
/* aurora-packtest: correctness of multi-command-buffer submissions with
 * submit packing (hk_queue.c, hk_pack.c) on and off.
 *
 * Each case is a list of command buffers holding compute dispatches over a
 * storage buffer of slots, barriers, transfers and work that keeps a command
 * buffer out of submit packing (timestamps, renders, events, indirect
 * dispatches). The GPU result is compared word for word, guards included,
 * with a CPU model of the same operations. Run it once with
 * HK_PACK_CDM_SUBMIT=0 and once with =1 (HK_PACK_CDM_TRACE=1 prints what was
 * packed) to cover both paths. Plain dispatches run a program without loops,
 * heavy ones a looping program, which a packet holds at most one of.
 *
 * The driver's packing trace (HK_PACK_CDM_TRACE, turned on here) passes
 * through to stderr as it is written, and each case reports how many command
 * buffers it says were packed. With HK_PACK_CDM_SUBMIT=1 a case that must
 * pack fails if nothing was packed, and with HK_PACK_CDM_SUBMIT=0 any case
 * fails if something was.
 *
 * usage: aurora-packtest ITERATIONS [CASE...]
 */
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "aurora_packtest_shaders.h"

#define LANES 1024u
#define GUARD 64u
#define MAX_SLOTS 1024u
#define MAX_OPS 160u
#define MAX_CBS 512u
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

static void
die(const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   fputs("FATAL ", stdout);
   vprintf(fmt, ap);
   fputc('\n', stdout);
   va_end(ap);
   exit(2);
}

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS)                                                    \
         die("%s:%d %s = %d", __FILE__, __LINE__, #call, (int)r_);            \
   } while (0)

static uint64_t
now_ns(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}

enum op_kind {
   OP_MIX,      /* slot b = mix(slot a ^ salt) */
   OP_BARC,     /* compute -> compute shader write/read barrier */
   OP_BAR,      /* all commands -> all commands memory barrier */
   OP_COPY,     /* copy slot a -> slot b */
   OP_FILL,     /* fill slot b with salt */
   OP_TS,       /* timestamp write */
   OP_RENDER,   /* clear an image in a dynamic render pass */
   OP_EVENT,    /* set and wait an event, compute -> compute */
   OP_INDIRECT, /* OP_MIX through vkCmdDispatchIndirect */
   OP_HOSTBAR,  /* all writes -> host read */
   OP_HEAVY,    /* OP_MIX iterated HEAVY_ROUNDS times */
   OP_ATOM,     /* slot b[l & 63] += (slot a[l] ^ salt) & 0xff, atomically */
};

#define HEAVY_ROUNDS 20000u

struct op {
   enum op_kind kind;
   uint32_t a, b, salt;
};

struct cbdesc {
   unsigned nops;
   struct op ops[MAX_OPS];
   int same_as; /* >= 0: submit this earlier command buffer again */
};

struct testcase {
   const char *name;
   unsigned ncb, slots;
   struct cbdesc cb[MAX_CBS];
   /* Submission shape. */
   unsigned split;      /* > 0: second batch starts at this command buffer */
   bool semaphore;      /* split into two vkQueueSubmit with a timeline semaphore */
   unsigned depth;      /* > 1: submit the whole case this many times before waiting */
};

static struct {
   VkInstance inst;
   VkPhysicalDevice pdev;
   VkDevice dev;
   VkQueue queue;
   uint32_t qfam;
   VkCommandPool pool;
   VkBuffer buf, ind;
   VkDeviceMemory mem, ind_mem;
   uint32_t *words;
   VkDeviceSize size;
   VkPipeline pipe, heavy, atom;
   VkPipelineLayout layout;
   VkDescriptorSetLayout dsl;
   VkDescriptorPool dpool;
   VkDescriptorSet set;
   VkQueryPool qp;
   VkEvent event;
   VkImage img;
   VkDeviceMemory img_mem;
   VkImageView view;
   bool timeline, sync2, dynrender, hostreset;
} g;

static uint32_t
mix(uint32_t x, uint32_t salt)
{
   x ^= salt;
   x = (x ^ (x >> 13)) * 0x85ebca6bu;
   x = (x ^ (x >> 16)) * 0xc2b2ae35u;
   return x ^ (x >> 15);
}

static uint32_t
memtype(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(g.pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   die("no memory type");
   return 0;
}

static void
make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *b, VkDeviceMemory *m,
            void **map)
{
   const VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .size = size, .usage = usage};
   CHECK(vkCreateBuffer(g.dev, &bi, NULL, b));
   VkMemoryRequirements r;
   vkGetBufferMemoryRequirements(g.dev, *b, &r);
   const VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = r.size,
      .memoryTypeIndex = memtype(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   CHECK(vkAllocateMemory(g.dev, &ai, NULL, m));
   CHECK(vkBindBufferMemory(g.dev, *b, *m, 0));
   CHECK(vkMapMemory(g.dev, *m, 0, VK_WHOLE_SIZE, 0, map));
}

static VkPipeline
make_pipe(const uint32_t *code, size_t bytes)
{
   VkShaderModule sm;
   const VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                         .codeSize = bytes, .pCode = code};
   CHECK(vkCreateShaderModule(g.dev, &smi, NULL, &sm));
   const VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"},
      .layout = g.layout};
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   vkDestroyShaderModule(g.dev, sm, NULL);
   return pipe;
}

static void
setup(void)
{
   const VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                  .apiVersion = VK_API_VERSION_1_3};
   const VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                     .pApplicationInfo = &app};
   CHECK(vkCreateInstance(&ici, NULL, &g.inst));
   uint32_t n = 8;
   VkPhysicalDevice pd[8];
   CHECK(vkEnumeratePhysicalDevices(g.inst, &n, pd));
   if (!n)
      die("no device");
   g.pdev = pd[0];
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(g.pdev, &props);
   printf("DEVICE %s api %u.%u\n", props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
          VK_API_VERSION_MINOR(props.apiVersion));

   uint32_t nq = 16;
   VkQueueFamilyProperties qf[16];
   vkGetPhysicalDeviceQueueFamilyProperties(g.pdev, &nq, qf);
   g.qfam = UINT32_MAX;
   for (uint32_t i = 0; i < nq && g.qfam == UINT32_MAX; ++i)
      if ((qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
         g.qfam = i;
   if (g.qfam == UINT32_MAX)
      die("no graphics+compute queue");

   VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
   VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                           .pNext = &f13};
   VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
   vkGetPhysicalDeviceFeatures2(g.pdev, &f2);
   g.timeline = f12.timelineSemaphore;
   g.hostreset = f12.hostQueryReset;
   g.sync2 = f13.synchronization2;
   g.dynrender = f13.dynamicRendering;
   VkPhysicalDeviceVulkan13Features e13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                           .synchronization2 = g.sync2,
                                           .dynamicRendering = g.dynrender};
   VkPhysicalDeviceVulkan12Features e12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                           .pNext = &e13, .timelineSemaphore = g.timeline,
                                           .hostQueryReset = g.hostreset};
   VkPhysicalDeviceFeatures2 e2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &e12};
   const float prio = 1.0f;
   const VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                        .queueFamilyIndex = g.qfam, .queueCount = 1,
                                        .pQueuePriorities = &prio};
   const VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &e2,
                                   .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
   CHECK(vkCreateDevice(g.pdev, &dci, NULL, &g.dev));
   vkGetDeviceQueue(g.dev, g.qfam, 0, &g.queue);
   printf("FEATURES timeline=%d sync2=%d dynrender=%d hostreset=%d\n", g.timeline, g.sync2,
          g.dynrender, g.hostreset);

   const VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                        .queueFamilyIndex = g.qfam};
   CHECK(vkCreateCommandPool(g.dev, &cpi, NULL, &g.pool));

   g.size = ((VkDeviceSize)MAX_SLOTS * LANES + GUARD) * 4;
   make_buffer(g.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               &g.buf, &g.mem, (void **)&g.words);
   uint32_t *ind;
   make_buffer(16, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, &g.ind, &g.ind_mem, (void **)&ind);
   ind[0] = LANES / 64, ind[1] = 1, ind[2] = 1;

   const VkDescriptorSetLayoutBinding b0 = {.binding = 0,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   const VkDescriptorSetLayoutCreateInfo dli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
      .pBindings = &b0};
   CHECK(vkCreateDescriptorSetLayout(g.dev, &dli, NULL, &g.dsl));
   const VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 16};
   const VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .setLayoutCount = 1, .pSetLayouts = &g.dsl,
                                           .pushConstantRangeCount = 1,
                                           .pPushConstantRanges = &pcr};
   CHECK(vkCreatePipelineLayout(g.dev, &pli, NULL, &g.layout));
   g.pipe = make_pipe(packtest_mix, sizeof(packtest_mix));
   g.heavy = make_pipe(packtest_heavy, sizeof(packtest_heavy));
   g.atom = make_pipe(packtest_atom, sizeof(packtest_atom));
   const VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
   const VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                           .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
   CHECK(vkCreateDescriptorPool(g.dev, &dpi, NULL, &g.dpool));
   const VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                            .descriptorPool = g.dpool, .descriptorSetCount = 1,
                                            .pSetLayouts = &g.dsl};
   CHECK(vkAllocateDescriptorSets(g.dev, &dai, &g.set));
   const VkDescriptorBufferInfo dbi = {g.buf, 0, VK_WHOLE_SIZE};
   const VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                   .dstSet = g.set, .descriptorCount = 1,
                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                   .pBufferInfo = &dbi};
   vkUpdateDescriptorSets(g.dev, 1, &w, 0, NULL);

   const VkQueryPoolCreateInfo qpi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                      .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 4};
   CHECK(vkCreateQueryPool(g.dev, &qpi, NULL, &g.qp));
   const VkEventCreateInfo eci = {.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
   CHECK(vkCreateEvent(g.dev, &eci, NULL, &g.event));

   if (g.dynrender) {
      const VkImageCreateInfo ii = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
         .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {64, 64, 1}, .mipLevels = 1,
         .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
      CHECK(vkCreateImage(g.dev, &ii, NULL, &g.img));
      VkMemoryRequirements r;
      vkGetImageMemoryRequirements(g.dev, g.img, &r);
      const VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                       .allocationSize = r.size,
                                       .memoryTypeIndex = memtype(r.memoryTypeBits, 0)};
      CHECK(vkAllocateMemory(g.dev, &ai, NULL, &g.img_mem));
      CHECK(vkBindImageMemory(g.dev, g.img, g.img_mem, 0));
      const VkImageViewCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g.img,
         .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      CHECK(vkCreateImageView(g.dev, &vi, NULL, &g.view));
   }
}

static void
barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags sa, VkPipelineStageFlags dst,
        VkAccessFlags da)
{
   const VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = sa,
                               .dstAccessMask = da};
   vkCmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void
record_op(VkCommandBuffer cb, const struct op *o)
{
   const uint32_t push[4] = {o->a * LANES, o->b * LANES, o->salt,
                             o->kind == OP_HEAVY ? HEAVY_ROUNDS : 1};
   switch (o->kind) {
   case OP_MIX:
   case OP_HEAVY:
   case OP_ATOM:
   case OP_INDIRECT:
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                        o->kind == OP_ATOM    ? g.atom
                        : o->kind == OP_HEAVY ? g.heavy
                                              : g.pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.layout, 0, 1, &g.set, 0, NULL);
      vkCmdPushConstants(cb, g.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
      if (o->kind != OP_INDIRECT)
         vkCmdDispatch(cb, LANES / 64, 1, 1);
      else
         vkCmdDispatchIndirect(cb, g.ind, 0);
      break;
   case OP_BARC:
      barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      break;
   case OP_BAR:
      barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
              VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
      break;
   case OP_HOSTBAR:
      barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
              VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
      break;
   case OP_COPY: {
      const VkBufferCopy c = {(VkDeviceSize)o->a * LANES * 4, (VkDeviceSize)o->b * LANES * 4,
                              LANES * 4};
      vkCmdCopyBuffer(cb, g.buf, g.buf, 1, &c);
      break;
   }
   case OP_FILL:
      vkCmdFillBuffer(cb, g.buf, (VkDeviceSize)o->b * LANES * 4, LANES * 4, o->salt);
      break;
   case OP_TS:
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.qp, 0);
      break;
   case OP_RENDER: {
      const VkImageMemoryBarrier ib = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g.img,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL,
                           1, &ib);
      const VkRenderingAttachmentInfo att = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = g.view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue = {.color = {.float32 = {0.25f, 0.5f, 0.75f, 1.0f}}}};
      const VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                  .renderArea = {{0, 0}, {64, 64}}, .layerCount = 1,
                                  .colorAttachmentCount = 1, .pColorAttachments = &att};
      vkCmdBeginRendering(cb, &ri);
      vkCmdEndRendering(cb);
      break;
   }
   case OP_EVENT: {
      vkCmdSetEvent(cb, g.event, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      const VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                  .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
      vkCmdWaitEvents(cb, 1, &g.event, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1, &mb, 0, NULL, 0, NULL);
      break;
   }
   }
}

static void
model(const struct testcase *t, uint32_t *ref)
{
   for (unsigned i = 0; i < t->ncb; ++i) {
      const struct cbdesc *d = t->cb[i].same_as >= 0 ? &t->cb[t->cb[i].same_as] : &t->cb[i];
      for (unsigned k = 0; k < d->nops; ++k) {
         const struct op *o = &d->ops[k];
         uint32_t *src = ref + o->a * LANES, *dst = ref + o->b * LANES;
         switch (o->kind) {
         case OP_MIX:
         case OP_INDIRECT:
            for (unsigned l = 0; l < LANES; ++l)
               dst[l] = mix(src[l], o->salt);
            break;
         case OP_ATOM:
            for (unsigned l = 0; l < LANES; ++l)
               dst[l & 63] += (src[l] ^ o->salt) & 0xff;
            break;
         case OP_HEAVY:
            for (unsigned l = 0; l < LANES; ++l) {
               uint32_t v = src[l];
               for (unsigned r = 0; r < HEAVY_ROUNDS; ++r)
                  v = mix(v, o->salt);
               dst[l] = v;
            }
            break;
         case OP_COPY:
            memmove(dst, src, LANES * 4);
            break;
         case OP_FILL:
            for (unsigned l = 0; l < LANES; ++l)
               dst[l] = o->salt;
            break;
         default:
            break;
         }
      }
   }
}

static void
init_words(uint32_t *w, unsigned slots)
{
   for (unsigned l = 0; l < LANES; ++l)
      w[l] = l * 97u + 1234u;
   memset(w + LANES, 0xa5, ((size_t)(slots - 1) * LANES + GUARD) * 4);
}

static struct testcase T;

static void
add(unsigned cb, enum op_kind k, uint32_t a, uint32_t b, uint32_t salt)
{
   struct cbdesc *d = &T.cb[cb];
   if (d->nops == MAX_OPS)
      die("too many ops");
   d->ops[d->nops++] = (struct op){k, a, b, salt};
   if (cb + 1 > T.ncb)
      T.ncb = cb + 1;
}

/* A dependent chain slot i -> i + 1 over n command buffers, one dispatch and
 * one barrier each, starting at slot base. */
static void
chain(unsigned first_cb, unsigned n, unsigned base, enum op_kind bar)
{
   for (unsigned i = 0; i < n; ++i) {
      add(first_cb + i, OP_MIX, base + i, base + i + 1, i * 73u + 1u);
      add(first_cb + i, bar, 0, 0, 0);
   }
}

static bool
build(const char *name)
{
   memset(&T, 0, sizeof(T));
   for (unsigned i = 0; i < MAX_CBS; ++i)
      T.cb[i].same_as = -1;
   T.name = name;
   if (!strcmp(name, "chain242")) {
      chain(0, 242, 0, OP_BARC);
   } else if (!strcmp(name, "chain300")) {
      chain(0, 300, 0, OP_BARC);
   } else if (!strcmp(name, "loops")) {
      /* A dependent chain of looping dispatches between plain ones: a
       * packet takes at most one looping stream. */
      unsigned s = 0;
      for (unsigned c = 0; c < 24; ++c) {
         add(c, c % 3 ? OP_MIX : OP_HEAVY, s, s + 1, c * 13u + 5u);
         add(c, OP_BARC, 0, 0, 0);
         ++s;
      }
   } else if (!strcmp(name, "chain-all")) {
      chain(0, 64, 0, OP_BAR);
   } else if (!strcmp(name, "multi")) {
      /* 40 command buffers of 7 dependent dispatches: 280 launches. */
      unsigned s = 0;
      for (unsigned c = 0; c < 40; ++c)
         for (unsigned k = 0; k < 7; ++k, ++s) {
            add(c, OP_MIX, s, s + 1, s * 31u + 7u);
            add(c, OP_BARC, 0, 0, 0);
         }
   } else if (!strcmp(name, "multi-tail")) {
      /* Dependencies only through barriers at the start of the next buffer. */
      unsigned s = 0;
      for (unsigned c = 0; c < 48; ++c) {
         if (c)
            add(c, OP_BARC, 0, 0, 0);
         for (unsigned k = 0; k < 3; ++k, ++s) {
            add(c, OP_MIX, s, s + 1, s * 29u + 3u);
            if (k < 2)
               add(c, OP_BARC, 0, 0, 0);
         }
      }
   } else if (!strcmp(name, "disjoint")) {
      for (unsigned c = 0; c < 64; ++c) {
         const unsigned b = 1 + 3 * c;
         add(c, OP_MIX, 0, b, c * 3u + 1u);
         add(c, OP_BARC, 0, 0, 0);
         add(c, OP_MIX, b, b + 1, c * 3u + 2u);
         add(c, OP_BARC, 0, 0, 0);
         add(c, OP_MIX, b + 1, b + 2, c * 3u + 3u);
      }
   } else if (!strcmp(name, "xfer")) {
      unsigned c = 0, s = 0;
      for (unsigned r = 0; r < 12; ++r) {
         add(c, OP_MIX, s, s + 1, r * 5u + 1u), add(c++, OP_BAR, 0, 0, 0);
         add(c, OP_COPY, s + 1, s + 2, 0), add(c++, OP_BAR, 0, 0, 0);
         add(c, OP_MIX, s + 2, s + 3, r * 5u + 2u), add(c, OP_BAR, 0, 0, 0);
         add(c, OP_FILL, 0, s + 4, 0x9e3779b9u ^ r), add(c++, OP_BAR, 0, 0, 0);
         add(c, OP_MIX, s + 4, s + 5, r * 5u + 3u), add(c, OP_BAR, 0, 0, 0);
         add(c, OP_COPY, s + 5, s + 6, 0), add(c, OP_BAR, 0, 0, 0);
         add(c, OP_MIX, s + 6, s + 7, r * 5u + 4u), add(c++, OP_BAR, 0, 0, 0);
         s += 7;
      }
   } else if (!strcmp(name, "split-ts") || !strcmp(name, "split-render") ||
              !strcmp(name, "split-event") || !strcmp(name, "split-indirect")) {
      chain(0, 20, 0, OP_BARC);
      const unsigned c = 20;
      if (!strcmp(name, "split-ts"))
         add(c, OP_TS, 0, 0, 0), add(c, OP_MIX, 20, 21, 777), add(c, OP_BARC, 0, 0, 0);
      else if (!strcmp(name, "split-render"))
         add(c, OP_RENDER, 0, 0, 0), add(c, OP_MIX, 20, 21, 777), add(c, OP_BAR, 0, 0, 0);
      else if (!strcmp(name, "split-event"))
         add(c, OP_MIX, 20, 21, 777), add(c, OP_EVENT, 0, 0, 0),
            add(c, OP_MIX, 21, 21, 778), add(c, OP_BARC, 0, 0, 0);
      else
         add(c, OP_INDIRECT, 20, 21, 777), add(c, OP_BARC, 0, 0, 0);
      chain(21, 20, 21, OP_BARC);
   } else if (!strcmp(name, "dup")) {
      add(0, OP_MIX, 0, 5, 11), add(0, OP_BARC, 0, 0, 0);
      add(1, OP_MIX, 5, 5, 12), add(1, OP_BARC, 0, 0, 0);
      T.cb[2].same_as = 1, T.cb[3].same_as = 1;
      T.ncb = 4;
      add(4, OP_MIX, 5, 6, 13), add(4, OP_BARC, 0, 0, 0);
   } else if (!strcmp(name, "indep")) {
      /* One buffer of 128 independent dispatches without barriers. */
      for (unsigned i = 0; i < 128; ++i)
         add(0, OP_MIX, 0, 1 + i, i * 5u + 9u);
   } else if (!strcmp(name, "indep-order")) {
      /* Independent producers (the first one long), one barrier, then
       * consumers of every producer: the barrier must wait for all of
       * them, not only the last launch. Repeated over 6 buffers. */
      unsigned s = 1;
      for (unsigned c = 0; c < 6; ++c) {
         const unsigned p0 = s;
         add(c, OP_HEAVY, 0, s++, c * 7u + 1u);
         for (unsigned i = 1; i < 12; ++i)
            add(c, OP_MIX, 0, s++, c * 7u + 1u + i);
         add(c, OP_BARC, 0, 0, 0);
         /* MAX_OPS: 12 producers + barrier + 3 consumers. */
         add(c, OP_MIX, p0, s++, 100 + c);
         add(c, OP_MIX, p0 + 5, s++, 200 + c);
         add(c, OP_MIX, p0 + 11, s++, 300 + c);
      }
   } else if (!strcmp(name, "indep-cbs")) {
      /* 128 one-dispatch buffers without any barrier: independent. */
      for (unsigned i = 0; i < 128; ++i)
         add(i, OP_MIX, 0, 1 + i, i * 7u + 3u);
   } else if (!strcmp(name, "indep-cbs-order")) {
      /* Barrier-free producer buffers (the first one long), then a buffer
       * whose leading barrier must wait for every producer, then more
       * barrier-free buffers and a trailing-barrier buffer, three times. */
      unsigned c = 0, s = 1;
      for (unsigned r = 0; r < 3; ++r) {
         const unsigned p0 = s;
         add(c++, OP_HEAVY, 0, s++, r * 11u + 1u);
         for (unsigned i = 1; i < 8; ++i)
            add(c++, OP_MIX, 0, s++, r * 11u + 1u + i);
         add(c, OP_BARC, 0, 0, 0);
         add(c, OP_MIX, p0, s++, 500 + r);
         add(c++, OP_MIX, p0 + 7, s++, 600 + r);
         add(c, OP_MIX, s - 1, s, 700 + r), s++;
         add(c++, OP_BARC, 0, 0, 0);
         add(c++, OP_MIX, s - 1, s, 800 + r), s++;
      }
   } else if (!strcmp(name, "atom-chain")) {
      /* Counters cleared, 96 buffers of atomic adds into them, then a
       * reader, twice; the reads must see every add. */
      unsigned c = 0;
      for (unsigned r = 0; r < 2; ++r) {
         const unsigned ctr = 1 + 2 * r;
         add(c, OP_FILL, 0, ctr, 0), add(c++, OP_BAR, 0, 0, 0);
         for (unsigned i = 0; i < 96; ++i)
            add(c, OP_ATOM, 0, ctr, i * 3u + r), add(c++, OP_BARC, 0, 0, 0);
         add(c, OP_MIX, ctr, ctr + 1, 77 + r), add(c++, OP_BARC, 0, 0, 0);
      }
   } else if (!strcmp(name, "atom-multi")) {
      /* Buffers mixing plain and atomic dispatches with in-buffer barriers. */
      for (unsigned c = 0; c < 40; ++c) {
         const unsigned a = 2 + 2 * c;
         add(c, OP_MIX, 0, a, c * 5u + 1u), add(c, OP_BARC, 0, 0, 0);
         add(c, OP_ATOM, a, 1, c), add(c, OP_BARC, 0, 0, 0);
         add(c, OP_ATOM, 0, 1, c + 1000), add(c, OP_BARC, 0, 0, 0);
         add(c, OP_MIX, a, a + 1, c * 5u + 2u), add(c, OP_BARC, 0, 0, 0);
      }
      add(40, OP_MIX, 1, 90, 5), add(40, OP_BARC, 0, 0, 0);
   } else if (!strcmp(name, "sem")) {
      chain(0, 120, 0, OP_BARC);
      T.split = 60, T.semaphore = true;
   } else if (!strcmp(name, "twobatch")) {
      chain(0, 120, 0, OP_BARC);
      T.split = 60;
   } else if (!strcmp(name, "depth")) {
      unsigned s = 0;
      for (unsigned c = 0; c < 40; ++c)
         for (unsigned k = 0; k < 4; ++k, ++s)
            add(c, OP_MIX, s, s + 1, s * 13u + 5u), add(c, OP_BARC, 0, 0, 0);
      T.depth = 24;
   } else {
      return false;
   }
   /* Every case ends by publishing to the host. */
   add(T.ncb - 1 < MAX_CBS ? T.ncb - 1 : 0, OP_HOSTBAR, 0, 0, 0);
   unsigned top = 0;
   for (unsigned i = 0; i < T.ncb; ++i)
      for (unsigned k = 0; k < T.cb[i].nops; ++k)
         top = T.cb[i].ops[k].a > top ? T.cb[i].ops[k].a : top,
         top = T.cb[i].ops[k].b > top ? T.cb[i].ops[k].b : top;
   T.slots = top + 2;
   if (T.slots > MAX_SLOTS)
      die("too many slots");
   return true;
}

static bool
needs_feature(const struct testcase *t)
{
   for (unsigned i = 0; i < t->ncb; ++i)
      for (unsigned k = 0; k < t->cb[i].nops; ++k) {
         const enum op_kind kd = t->cb[i].ops[k].kind;
         if ((kd == OP_RENDER && !g.dynrender) || (kd == OP_TS && !g.hostreset) ||
             (kd == OP_ATOM && !g.atom))
            return true;
      }
   return t->semaphore && !g.timeline;
}

static int
run_case(const char *name, unsigned iterations)
{
   if (!build(name))
      die("unknown case %s", name);
   if (needs_feature(&T)) {
      printf("SKIP %s (feature)\n", name);
      return 0;
   }
   bool dup = false;
   for (unsigned i = 0; i < T.ncb; ++i)
      dup |= T.cb[i].same_as >= 0;
   const bool simultaneous = dup || T.depth > 1;

   VkCommandBuffer cbs[MAX_CBS];
   const VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                            .commandPool = g.pool,
                                            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                            .commandBufferCount = T.ncb};
   CHECK(vkAllocateCommandBuffers(g.dev, &cai, cbs));
   VkCommandBuffer list[MAX_CBS];
   for (unsigned i = 0; i < T.ncb; ++i) {
      if (T.cb[i].same_as >= 0) {
         list[i] = cbs[T.cb[i].same_as];
         continue;
      }
      const VkCommandBufferBeginInfo bi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = simultaneous ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : 0};
      CHECK(vkBeginCommandBuffer(cbs[i], &bi));
      for (unsigned k = 0; k < T.cb[i].nops; ++k)
         record_op(cbs[i], &T.cb[i].ops[k]);
      CHECK(vkEndCommandBuffer(cbs[i]));
      list[i] = cbs[i];
   }

   uint32_t *ref = malloc(((size_t)T.slots * LANES + GUARD) * 4);
   init_words(ref, T.slots);
   model(&T, ref);

   VkFence fence;
   const VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   CHECK(vkCreateFence(g.dev, &fci, NULL, &fence));
   VkSemaphore sem = VK_NULL_HANDLE;
   if (T.semaphore) {
      const VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
      const VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                         .pNext = &st};
      CHECK(vkCreateSemaphore(g.dev, &sci, NULL, &sem));
   }

   uint64_t point = 0, best = UINT64_MAX, sum = 0;
   unsigned bad = 0, timed = 0;
   for (unsigned it = 0; it < iterations; ++it) {
      init_words(g.words, T.slots);
      if (g.hostreset)
         vkResetQueryPool(g.dev, g.qp, 0, 4);
      CHECK(vkResetEvent(g.dev, g.event));
      CHECK(vkResetFences(g.dev, 1, &fence));
      const uint64_t t0 = now_ns();
      const unsigned reps = T.depth > 1 ? T.depth : 1;
      for (unsigned rep = 0; rep < reps; ++rep) {
         const bool last = rep + 1 == reps;
         if (T.split) {
            const uint64_t wait_value = point + 1;
            const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            const VkTimelineSemaphoreSubmitInfo ts1 = {
               .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
               .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &wait_value};
            const VkTimelineSemaphoreSubmitInfo ts2 = {
               .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
               .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &wait_value};
            VkSubmitInfo si[2] = {
               {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = T.split,
                .pCommandBuffers = list},
               {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = T.ncb - T.split,
                .pCommandBuffers = list + T.split},
            };
            if (T.semaphore) {
               si[0].pNext = &ts1, si[0].signalSemaphoreCount = 1, si[0].pSignalSemaphores = &sem;
               si[1].pNext = &ts2, si[1].waitSemaphoreCount = 1, si[1].pWaitSemaphores = &sem;
               si[1].pWaitDstStageMask = &wait_stage;
               point = wait_value;
               CHECK(vkQueueSubmit(g.queue, 1, &si[0], VK_NULL_HANDLE));
               CHECK(vkQueueSubmit(g.queue, 1, &si[1], last ? fence : VK_NULL_HANDLE));
            } else {
               /* The barrier ending the first batch orders the second. */
               CHECK(vkQueueSubmit(g.queue, 2, si, last ? fence : VK_NULL_HANDLE));
            }
         } else {
            const VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                     .commandBufferCount = T.ncb, .pCommandBuffers = list};
            CHECK(vkQueueSubmit(g.queue, 1, &si, last ? fence : VK_NULL_HANDLE));
         }
      }
      CHECK(vkWaitForFences(g.dev, 1, &fence, VK_TRUE, 10000000000ull));
      const uint64_t dt = now_ns() - t0;
      if (it >= 2) {
         best = dt < best ? dt : best;
         sum += dt;
         timed++;
      }
      const size_t n = (size_t)T.slots * LANES + GUARD;
      for (size_t w = 0; w < n; ++w) {
         if (g.words[w] != ref[w]) {
            if (bad < 4)
               printf("MISMATCH %s iter %u word %zu (slot %zu lane %zu): got %08x want %08x\n",
                      name, it, w, w / LANES, w % LANES, g.words[w], ref[w]);
            bad++;
            break;
         }
      }
   }
   printf("%s %s cbs=%u iterations=%u submit_to_fence_us best=%.1f mean=%.1f\n",
          bad ? "FAIL" : "PASS", name, T.ncb, iterations, best / 1e3,
          timed ? sum / 1e3 / timed : 0.0);
   vkDestroyFence(g.dev, fence, NULL);
   if (sem)
      vkDestroySemaphore(g.dev, sem, NULL);
   vkFreeCommandBuffers(g.dev, g.pool, T.ncb, cbs);
   free(ref);
   return bad ? 1 : 0;
}

/* The driver's packing trace comes through a pipe standing in for stderr. A
 * thread passes everything on to the real stderr as it arrives, so the
 * driver's messages for a case that dies are not lost, and counts the
 * command buffers the trace says were packed. trace_sync() returns that
 * count once everything written before it went through. */
#define TRACE_SYNC "\001aurora-packtest sync\n"

static struct {
   pthread_mutex_t lock;
   pthread_cond_t cond;
   pthread_t thread;
   int real_stderr, read_end;
   unsigned packed, synced, requested;
   /* What the driver's trace says of strong-flush tails: -1 not seen yet,
    * also with a driver older than the strong_tails= field (#558), where the
    * barrier-ended cases stay must-pack. A line without the field keeps the
    * value of the last one that had it. */
   int strong_tails;
   bool on;
} tr = {
   .strong_tails = -1,
   .lock = PTHREAD_MUTEX_INITIALIZER,
   .cond = PTHREAD_COND_INITIALIZER,
   .real_stderr = -1,
   .read_end = -1,
};

static void
trace_line(const char *line, size_t n)
{
   if (n == strlen(TRACE_SYNC) && !memcmp(line, TRACE_SYNC, n)) {
      pthread_mutex_lock(&tr.lock);
      tr.synced++;
      pthread_cond_broadcast(&tr.cond);
      pthread_mutex_unlock(&tr.lock);
      return;
   }
   if (write(tr.real_stderr, line, n) < 0) {
      /* The pass-through is best effort. */
   }
   unsigned runs, members;
   if (n > 13 && !memcmp(line, "HK_PACK runs=", 13)) {
      const bool counted =
         sscanf(line, "HK_PACK runs=%u members=%u", &runs, &members) == 2;
      const char *st = memmem(line, n, "strong_tails=", 13);
      pthread_mutex_lock(&tr.lock);
      if (counted)
         tr.packed += members;
      /* Only a driver with the field (#558 and later) relaxes must-pack. */
      if (st && st + 13 < line + n)
         tr.strong_tails = st[13] == '1';
      pthread_mutex_unlock(&tr.lock);
   }
}

static void *
trace_thread(void *arg)
{
   (void)arg;
   static char buf[16384];
   size_t len = 0;
   for (;;) {
      const ssize_t got = read(tr.read_end, buf + len, sizeof(buf) - len);
      if (got < 0 && errno == EINTR)
         continue;
      if (got <= 0)
         break;
      len += (size_t)got;
      size_t start = 0;
      for (size_t i = 0; i < len; ++i) {
         if (buf[i] == '\n') {
            trace_line(buf + start, i + 1 - start);
            start = i + 1;
         }
      }
      if (start == 0 && len == sizeof(buf)) {
         /* A line longer than the buffer: pass it on as it is. */
         trace_line(buf, len);
         start = len;
      }
      memmove(buf, buf + start, len - start);
      len -= start;
   }
   if (len)
      trace_line(buf, len);
   return NULL;
}

static unsigned
trace_sync(void)
{
   if (!tr.on)
      return 0;
   fflush(stderr);
   pthread_mutex_lock(&tr.lock);
   const unsigned want = ++tr.requested;
   pthread_mutex_unlock(&tr.lock);
   if (write(2, TRACE_SYNC, strlen(TRACE_SYNC)) < 0)
      return tr.packed;

   struct timespec until;
   clock_gettime(CLOCK_REALTIME, &until);
   until.tv_sec += 10;
   pthread_mutex_lock(&tr.lock);
   while (tr.synced < want &&
          pthread_cond_timedwait(&tr.cond, &tr.lock, &until) == 0)
      ;
   const unsigned packed = tr.packed;
   pthread_mutex_unlock(&tr.lock);
   return packed;
}

/* At exit, also through die(): what the driver wrote reaches stderr. */
static void
trace_end(void)
{
   trace_sync();
}

static void
trace_begin(void)
{
   setenv("HK_PACK_CDM_TRACE", "1", 1);
   int fds[2];
   if (pipe(fds) < 0)
      return;
   fflush(stderr);
   tr.real_stderr = dup(2);
   if (tr.real_stderr < 0 || dup2(fds[1], 2) < 0) {
      close(fds[0]);
      close(fds[1]);
      return;
   }
   close(fds[1]);
   tr.read_end = fds[0];
   if (pthread_create(&tr.thread, NULL, trace_thread, NULL)) {
      dup2(tr.real_stderr, 2);
      return;
   }
   tr.on = true;
   atexit(trace_end);
}

/* Whether a case submitted a render on the queue. */
static bool rendered;

static bool
listed(const char *name, const char *const *list, unsigned count)
{
   for (unsigned i = 0; i < count; ++i) {
      if (!strcmp(name, list[i]))
         return true;
   }
   return false;
}

/* Cases whose command buffers are all plain compute work that packs. Those
 * ending with a dispatch pack wherever packing is on. Those ending with a
 * barrier end with a strong flush without memory scope, which packs only
 * where the driver admits such tails (the trace's strong_tails=, see
 * hk_pack_admits_strong_tail()). A driver older than that field (#558) never
 * relaxes them: on M5 without mesa#506 those cases then FAIL as must-pack.
 * Every command buffer of chain-all ends with
 * an ALL -> ALL barrier, whose tail is a render wait once the queue ran a
 * render (split-render): it must pack only before that. */
static bool
must_pack(const char *name)
{
   static const char *const dispatch_ended[] = {"multi-tail", "disjoint"};
   static const char *const barrier_ended[] = {"chain242", "chain300", "loops",
                                               "chain-all", "multi"};
   if (listed(name, dispatch_ended, ARRAY_LEN(dispatch_ended)))
      return true;
   if (!listed(name, barrier_ended, ARRAY_LEN(barrier_ended)))
      return false;
   if (!strcmp(name, "chain-all") && rendered)
      return false;
   pthread_mutex_lock(&tr.lock);
   const int strong_tails = tr.strong_tails;
   pthread_mutex_unlock(&tr.lock);
   return strong_tails != 0;
}

static bool
has_render(const struct testcase *t)
{
   for (unsigned i = 0; i < t->ncb; ++i)
      for (unsigned k = 0; k < t->cb[i].nops; ++k)
         if (t->cb[i].ops[k].kind == OP_RENDER)
            return true;
   return false;
}

static int
check_case(const char *name, unsigned iterations)
{
   const unsigned before = trace_sync();
   const int bad = run_case(name, iterations);
   const unsigned packed = trace_sync() - before;
   const bool must = must_pack(name);
   rendered |= has_render(&T) && !needs_feature(&T);
   const char *mode = getenv("HK_PACK_CDM_SUBMIT");
   const bool on = mode && mode[0] == '1', off = mode && mode[0] == '0';
   if (!tr.on) {
      /* Nothing to count: a must-pack check cannot pass. */
      const bool fail = on && must;
      printf("%s %s trace unavailable\n", fail ? "FAIL" : "WARN", name);
      return bad + fail;
   }
   const bool wrong = (on && must && !packed) || (off && packed);
   printf("%s %s packed_cbs=%u%s\n", wrong ? "FAIL" : "PACKED", name, packed,
          wrong ? (on ? " (must pack)" : " (packing is off)") : "");
   return bad + wrong;
}

int
main(int argc, char **argv)
{
   static const char *all[] = {
      "chain242", "chain300", "loops", "chain-all", "multi", "multi-tail", "disjoint", "xfer",
      "split-ts", "split-render", "split-event", "split-indirect", "sem",
      "twobatch", "depth", "indep", "indep-order", "atom-chain", "atom-multi",
      "indep-cbs", "indep-cbs-order",
      /* Last: a recording submitted several times in one batch is refused
       * by the kernel on the current driver (pre-existing), losing the
       * device for the cases after it. */
      "dup",
   };
   setvbuf(stdout, NULL, _IOLBF, 0);
   trace_begin();
   if (argc < 2)
      die("usage: aurora-packtest ITERATIONS [CASE...]");
   setup();
   const unsigned iterations = (unsigned)strtoul(argv[1], NULL, 10);
   int fails = 0;
   if (argc > 2) {
      for (int i = 2; i < argc; ++i)
         fails += check_case(argv[i], iterations);
   } else {
      for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); ++i)
         fails += check_case(all[i], iterations);
   }
   printf("SUMMARY %s failures=%d\n", fails ? "FAIL" : "PASS", fails);
   CHECK(vkDeviceWaitIdle(g.dev));
   return fails ? 1 : 0;
}

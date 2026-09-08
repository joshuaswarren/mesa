// Minimal Vulkan reproducer: is a vkCmdSetEvent recorded at the head of a
// long-running compute command buffer visible to vkGetEventStatus while the
// job executes? mlx-omarchy's hang watchdog relies on exactly this to tell
// slow work from a wedged queue.
//
// Usage: evprobe <iters> <dispatches> [groups]
#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define CK(x)                                                              \
  do {                                                                     \
    VkResult r_ = (x);                                                     \
    if (r_ != VK_SUCCESS) {                                                \
      std::fprintf(stderr, "%s failed: %d (line %d)\n", #x, r_, __LINE__); \
      std::exit(3);                                                        \
    }                                                                      \
  } while (0)

static std::vector<uint32_t> read_spv(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(3); }
  size_t n = f.tellg();
  std::vector<uint32_t> v(n / 4);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

int main(int argc, char** argv) {
  uint32_t iters = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 50000000u;
  uint32_t dispatches = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 4u;
  uint32_t groups = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 64u;
  const char* spv = argc > 4 ? argv[4] : "spin.spv";

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  CK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 1;
  VkPhysicalDevice pd;
  CK(vkEnumeratePhysicalDevices(inst, &n, &pd));
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pd, &props);
  std::printf("device: %s\n", props.deviceName);

  uint32_t qf = 0, qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qfp(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qfp.data());
  for (uint32_t i = 0; i < qn; ++i)
    if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo dqci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  dqci.queueFamilyIndex = qf;
  dqci.queueCount = 1;
  dqci.pQueuePriorities = &prio;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.timelineSemaphore = VK_TRUE;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &f12;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &dqci;
  VkDevice dev;
  CK(vkCreateDevice(pd, &dci, nullptr, &dev));
  VkQueue queue;
  vkGetDeviceQueue(dev, qf, 0, &queue);

  // Buffer: groups*32 floats, host visible.
  const VkDeviceSize bytes = VkDeviceSize(groups) * 32 * sizeof(float);
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = bytes;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  VkBuffer buf;
  CK(vkCreateBuffer(dev, &bci, nullptr, &buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buf, &mr);
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  uint32_t mt = UINT32_MAX;
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    const VkMemoryPropertyFlags want =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if ((mr.memoryTypeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
  }
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  mai.memoryTypeIndex = mt;
  VkDeviceMemory mem;
  CK(vkAllocateMemory(dev, &mai, nullptr, &mem));
  CK(vkBindBufferMemory(dev, buf, mem, 0));
  float* host = nullptr;
  CK(vkMapMemory(dev, mem, 0, bytes, 0, reinterpret_cast<void**>(&host)));
  std::memset(host, 0, bytes);

  auto code = read_spv(spv);
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = code.size() * 4;
  smci.pCode = code.data();
  VkShaderModule sm;
  CK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

  VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 1;
  dslci.pBindings = &b;
  VkDescriptorSetLayout dsl;
  CK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl;
  CK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = sm;
  cpci.stage.pName = "main";
  cpci.layout = pl;
  VkPipeline pipe;
  CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool dp;
  CK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dp;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet ds;
  CK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  VkDescriptorBufferInfo dbi{buf, 0, bytes};
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet = ds;
  w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w.pBufferInfo = &dbi;
  vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);

  VkEventCreateInfo eci{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
  VkEvent ev;
  CK(vkCreateEvent(dev, &eci, nullptr, &ev));
  CK(vkResetEvent(dev, ev));

  VkSemaphoreTypeCreateInfo stci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &stci;
  VkSemaphore sem;
  CK(vkCreateSemaphore(dev, &sci, nullptr, &sem));

  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.queueFamilyIndex = qf;
  VkCommandPool pool;
  CK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cb;
  CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  CK(vkBeginCommandBuffer(cb, &bi));
  // Exactly what mlx-omarchy's encoder does at the head of every batch.
  vkCmdSetEvent(cb, ev, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
  vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &iters);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  for (uint32_t d = 0; d < dispatches; ++d) {
    vkCmdDispatch(cb, groups, 1, 1);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
  }
  CK(vkEndCommandBuffer(cb));

  uint64_t signal_value = 1;
  VkTimelineSemaphoreSubmitInfo tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  tsi.signalSemaphoreValueCount = 1;
  tsi.pSignalSemaphoreValues = &signal_value;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &tsi;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &sem;
  auto t0 = std::chrono::steady_clock::now();
  CK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
  auto ms = [&] {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  std::printf("submit returned at %.1f ms\n", ms());

  double first_set = -1, done = -1;
  VkResult last_ev = VK_EVENT_RESET;
  for (;;) {
    VkResult es = vkGetEventStatus(dev, ev);
    uint64_t cur = 0;
    CK(vkGetSemaphoreCounterValue(dev, sem, &cur));
    if (es == VK_EVENT_SET && first_set < 0) {
      first_set = ms();
      std::printf("event SET observed at %.1f ms (counter=%llu)\n", first_set,
                  (unsigned long long)cur);
    }
    if (es != last_ev) {
      std::printf("  event status %d at %.1f ms\n", es, ms());
      last_ev = es;
    }
    if (cur >= 1) {
      done = ms();
      std::printf("counter reached 1 at %.1f ms (event=%s)\n", done,
                  es == VK_EVENT_SET ? "SET" : "RESET");
      break;
    }
    if (ms() > 300000) {
      std::printf("TIMEOUT after 300 s: event=%d counter=%llu\n", es,
                  (unsigned long long)cur);
      return 4;
    }
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &sem;
    wi.pValues = &signal_value;
    vkWaitSemaphores(dev, &wi, 100ull * 1000 * 1000);
  }
  std::printf("result[0]=%g result[last]=%g\n", host[0], host[groups * 32 - 1]);
  std::printf("VERDICT: %s (event first seen at %.1f ms, job done at %.1f ms)\n",
              first_set >= 0 && first_set < done - 1000 ? "PROGRESS_VISIBLE"
              : first_set < 0                            ? "EVENT_NEVER_SET"
                                                        : "EVENT_ONLY_AT_END",
              first_set, done);
  vkDeviceWaitIdle(dev);
  return first_set >= 0 && first_set < done - 1000 ? 0 : 1;
}

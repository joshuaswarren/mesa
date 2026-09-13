// Build: c++ -O2 compute_runner.cpp -lvulkan -o compute_runner
// Usage: compute_runner shader.spv in.bin out.bin out_bytes groups
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CK(x)                                                              \
  do {                                                                     \
    VkResult r_ = (x);                                                     \
    if (r_ != VK_SUCCESS) {                                                \
      std::fprintf(stderr, "%s failed: %d\n", #x, (int)r_);                \
      std::exit(2);                                                        \
    }                                                                      \
  } while (0)

static std::string slurp(const char* p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { std::fprintf(stderr, "open %s\n", p); std::exit(2); }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct Buf { VkBuffer b; VkDeviceMemory m; void* map; };

static VkPhysicalDeviceMemoryProperties g_mp;
static Buf mkbuf(VkDevice dev, size_t size) {
  Buf r{};
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  CK(vkCreateBuffer(dev, &bi, nullptr, &r.b));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, r.b, &mr);
  uint32_t ti = UINT32_MAX;
  for (uint32_t i = 0; i < g_mp.memoryTypeCount; ++i) {
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if ((mr.memoryTypeBits & (1u << i)) &&
        (g_mp.memoryTypes[i].propertyFlags & want) == want) { ti = i; break; }
  }
  if (ti == UINT32_MAX) { std::fprintf(stderr, "no host memtype\n"); std::exit(2); }
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = ti;
  CK(vkAllocateMemory(dev, &ai, nullptr, &r.m));
  CK(vkBindBufferMemory(dev, r.b, r.m, 0));
  CK(vkMapMemory(dev, r.m, 0, size, 0, &r.map));
  return r;
}

int main(int argc, char** argv) {
  if (argc < 6) { std::fprintf(stderr, "usage\n"); return 2; }
  std::string spv = slurp(argv[1]);
  std::string in = slurp(argv[2]);
  size_t out_bytes = std::strtoull(argv[4], nullptr, 10);
  uint32_t groups = std::strtoul(argv[5], nullptr, 10);

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  CK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 1;
  VkPhysicalDevice pd;
  CK(vkEnumeratePhysicalDevices(inst, &n, &pd));
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pd, &props);
  vkGetPhysicalDeviceMemoryProperties(pd, &g_mp);
  std::fprintf(stderr, "device: %s driver %u.%u.%u\n", props.deviceName,
               VK_VERSION_MAJOR(props.driverVersion),
               VK_VERSION_MINOR(props.driverVersion),
               VK_VERSION_PATCH(props.driverVersion));

  uint32_t qc = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qc, nullptr);
  std::vector<VkQueueFamilyProperties> qf(qc);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qc, qf.data());
  uint32_t qi = 0;
  for (; qi < qc; ++qi) if (qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT) break;
  float prio = 1.0f;
  VkDeviceQueueCreateInfo dqi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  dqi.queueFamilyIndex = qi;
  dqi.queueCount = 1;
  dqi.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &dqi;
  VkDevice dev;
  CK(vkCreateDevice(pd, &dci, nullptr, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, qi, 0, &q);

  Buf bin = mkbuf(dev, in.size());
  Buf bout = mkbuf(dev, out_bytes);
  std::memcpy(bin.map, in.data(), in.size());
  std::memset(bout.map, 0, out_bytes);

  VkDescriptorSetLayoutBinding binds[2]{};
  for (int i = 0; i < 2; ++i) {
    binds[i].binding = i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dsl.bindingCount = 2;
  dsl.pBindings = binds;
  VkDescriptorSetLayout layout;
  CK(vkCreateDescriptorSetLayout(dev, &dsl, nullptr, &layout));
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &layout;
  VkPipelineLayout pl;
  CK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spv.size();
  smi.pCode = (const uint32_t*)spv.data();
  VkShaderModule sm;
  CK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = sm;
  cpi.stage.pName = "main";
  cpi.layout = pl;
  VkPipeline pipe;
  CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 1;
  dpi.poolSizeCount = 1;
  dpi.pPoolSizes = &ps;
  VkDescriptorPool pool;
  CK(vkCreateDescriptorPool(dev, &dpi, nullptr, &pool));
  VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsa.descriptorPool = pool;
  dsa.descriptorSetCount = 1;
  dsa.pSetLayouts = &layout;
  VkDescriptorSet ds;
  CK(vkAllocateDescriptorSets(dev, &dsa, &ds));
  VkDescriptorBufferInfo dbi[2] = {{bin.b, 0, VK_WHOLE_SIZE}, {bout.b, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet w[2]{};
  for (int i = 0; i < 2; ++i) {
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = ds;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);

  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.queueFamilyIndex = qi;
  VkCommandPool cp;
  CK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
  VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cba.commandPool = cp;
  cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cba.commandBufferCount = 1;
  VkCommandBuffer cb;
  CK(vkAllocateCommandBuffers(dev, &cba, &cb));
  VkCommandBufferBeginInfo cbb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  CK(vkBeginCommandBuffer(cb, &cbb));
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cb, groups, 1, 1);
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  CK(vkEndCommandBuffer(cb));

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  CK(vkCreateFence(dev, &fci, nullptr, &fence));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  CK(vkQueueSubmit(q, 1, &si, fence));
  CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 60000000000ull));
  std::ofstream of(argv[3], std::ios::binary);
  of.write((const char*)bout.map, out_bytes);
  vkDeviceWaitIdle(dev);
  return 0;
}

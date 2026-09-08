// Minimal Vulkan compute reproducer for Honeykrisp byte-extraction miscompiles.
// usage: bx <variant.spv> <variant-name> [N]
// Bindings: 0 = in words (uint[N]), 1 = idx (uint[N]), 2 = out (uint[N]).
// Push constant: uint count. Workgroup size 256.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(2);} } while (0)

struct Buf { VkBuffer b; VkDeviceMemory m; void* p; };

static VkDevice dev; static VkPhysicalDevice pdev;
static Buf mk(size_t bytes) {
  Buf o{};
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = bytes; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  CK(vkCreateBuffer(dev, &bi, nullptr, &o.b));
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, o.b, &mr);
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
  uint32_t ti = UINT32_MAX;
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { ti = i; break; }
  }
  if (ti == UINT32_MAX) { fprintf(stderr, "no host visible memory\n"); exit(2); }
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = ti;
  CK(vkAllocateMemory(dev, &ai, nullptr, &o.m));
  CK(vkBindBufferMemory(dev, o.b, o.m, 0));
  CK(vkMapMemory(dev, o.m, 0, bytes, 0, &o.p));
  return o;
}

static uint32_t byte_at(uint32_t w, uint32_t lane) { return (w >> (8 * (lane & 3))) & 0xffu; }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: bx <spv> <variant> [N]\n"); return 2; }
  std::string variant = argv[2];
  uint32_t N = argc > 3 ? atoi(argv[3]) : 1024;
  std::ifstream f(argv[1], std::ios::binary); std::vector<char> spv((std::istreambuf_iterator<char>(f)), {});
  if (spv.empty()) { fprintf(stderr, "empty spv\n"); return 2; }

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
  VkInstance inst; CK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 0; CK(vkEnumeratePhysicalDevices(inst, &n, nullptr));
  std::vector<VkPhysicalDevice> pds(n); CK(vkEnumeratePhysicalDevices(inst, &n, pds.data()));
  VkPhysicalDeviceProperties pp; pdev = VK_NULL_HANDLE;
  for (auto d : pds) { vkGetPhysicalDeviceProperties(d, &pp); if (pp.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) { pdev = d; break; } }
  if (pdev == VK_NULL_HANDLE) { pdev = pds[0]; vkGetPhysicalDeviceProperties(pdev, &pp); }
  uint32_t qf = 0, qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qps(qn); vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qn, qps.data());
  for (uint32_t i = 0; i < qn; ++i) if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
  float prio = 1.f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
  CK(vkCreateDevice(pdev, &dci, nullptr, &dev));
  VkQueue q; vkGetDeviceQueue(dev, qf, 0, &q);

  Buf in = mk(N * 4), idx = mk(N * 4), out = mk(N * 4);
  std::vector<uint32_t> hin(N), hidx(N), hout(N), expect(N);
  uint32_t s = 12345;
  for (uint32_t i = 0; i < N; ++i) { s = s * 1664525u + 1013904223u; hin[i] = s; s = s * 1664525u + 1013904223u; hidx[i] = s >> 5; }
  if (variant == "scan_dyn" || variant == "scan_dyn_const" || variant == "scan_dyn_bit") for (uint32_t i = 0; i < N; ++i) hin[i] = 0x00010001u;
  if (variant == "bound32") for (uint32_t i = 0; i < N; ++i) hidx[i] = 32;
  if (variant == "bound63") for (uint32_t i = 0; i < N; ++i) hidx[i] = 63;
  if (variant == "uniform40") for (uint32_t i = 0; i < N; ++i) hidx[i] = 40;
  if (variant == "bound32" || variant == "bound63" || variant == "uniform40") for (uint32_t i = 0; i < N; ++i) expect[i] = (hin[i] >> (hidx[i] & 31)) & 0xffu;
  else if (variant == "divergent8_40") for (uint32_t i = 0; i < N; ++i) expect[i] = (hin[i] >> 8) & 0xffu;
  else if (variant == "byte_dyn") for (uint32_t i = 0; i < N; ++i) expect[i] = byte_at(hin[i >> 2], hidx[i]);
  else if (variant == "byte_dyn_gid") for (uint32_t i = 0; i < N; ++i) expect[i] = byte_at(hin[i >> 2], i);
  else if (variant == "byte_dyn_const") for (uint32_t i = 0; i < N; ++i) expect[i] = byte_at(0x00010001u, i);
  else if (variant == "ushr_raw") for (uint32_t i = 0; i < N; ++i) expect[i] = hin[i] >> (hidx[i] & 31);
  else if (variant == "ishl_raw") for (uint32_t i = 0; i < N; ++i) expect[i] = hin[i] << (hidx[i] & 31);
  else if (variant == "ishr_raw") for (uint32_t i = 0; i < N; ++i) expect[i] = (uint32_t)((int32_t)hin[i] >> (hidx[i] & 31));
  else if (variant == "shift_dyn") for (uint32_t i = 0; i < N; ++i) expect[i] = hin[i] >> (hidx[i] & 31);
  else if (variant == "shl_dyn") for (uint32_t i = 0; i < N; ++i) expect[i] = hin[i] << (hidx[i] & 31);
  else if (variant == "shift_dyn_mask") for (uint32_t i = 0; i < N; ++i) expect[i] = (hin[i] >> (hidx[i] & 31)) & 0xffu;
  else if (variant == "select_bcast") for (uint32_t i = 0; i < N; ++i) expect[i] = byte_at(hin[0], i) ? hidx[i] : ~hidx[i];
  else if (variant == "select_dyn") for (uint32_t i = 0; i < N; ++i) expect[i] = byte_at(hin[i >> 2], i) ? hidx[i] : ~hidx[i];
  else if (variant == "loop_wordbase") {
    // per element i: group = i / 32 (32 elems = 8 words); walk words w=0..7 with word_base induction, sum bytes with weight.
    for (uint32_t i = 0; i < N; ++i) {
      uint32_t group = i / 32, lane = i % 32, acc = 0, word_base = group * 8;
      for (uint32_t w = 0; w < 8; ++w) { uint32_t word = hin[word_base + w]; acc += byte_at(word, lane) * (w + 1); }
      expect[i] = acc;
    }
  } else if (variant == "scan_dyn" || variant == "scan_dyn_const" || variant == "scan_dyn_bit" || variant == "scan_dyn_rand") {
    for (uint32_t wg = 0; wg < N / 256; ++wg) { uint32_t run = 0; for (uint32_t l = 0; l < 256; ++l) { uint32_t i = wg * 256 + l; run += byte_at(hin[i >> 2], i) != 0; expect[i] = run; } }
  } else { fprintf(stderr, "unknown variant %s\n", variant.c_str()); return 2; }
  memcpy(in.p, hin.data(), N * 4); memcpy(idx.p, hidx.data(), N * 4); memset(out.p, 0xEE, N * 4);

  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = spv.size(); smi.pCode = (const uint32_t*)spv.data();
  VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
  VkDescriptorSetLayoutBinding binds[3];
  for (int i = 0; i < 3; ++i) binds[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dsli.bindingCount = 3; dsli.pBindings = binds;
  VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &dsli, nullptr, &dsl));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
  VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
  cpi.layout = pl;
  VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
  VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
  VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  VkDescriptorBufferInfo dbi[3] = {{in.b, 0, VK_WHOLE_SIZE}, {idx.b, 0, VK_WHOLE_SIZE}, {out.b, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet wds[3];
  for (int i = 0; i < 3; ++i) { wds[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wds[i].dstSet = ds; wds[i].dstBinding = i; wds[i].descriptorCount = 1; wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[i].pBufferInfo = &dbi[i]; }
  vkUpdateDescriptorSets(dev, 3, wds, 0, nullptr);
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qf;
  VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  CK(vkBeginCommandBuffer(cb, &cbbi));
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
  vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &N);
  vkCmdDispatch(cb, (N + 255) / 256, 1, 1);
  CK(vkEndCommandBuffer(cb));
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; CK(vkCreateFence(dev, &fci, nullptr, &fence));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
  CK(vkQueueSubmit(q, 1, &si, fence));
  CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull));
  memcpy(hout.data(), out.p, N * 4);
  uint32_t bad = 0, first = UINT32_MAX;
  for (uint32_t i = 0; i < N; ++i) if (hout[i] != expect[i]) { if (first == UINT32_MAX) first = i; ++bad; }
  printf("%s device=%s variant=%s N=%u mismatches=%u", bad ? "FAIL" : "PASS", pp.deviceName, variant.c_str(), N, bad);
  if (bad) {
    printf(" first=%u", first);
    int shown = 0;
    for (uint32_t i = 0; i < N && shown < 8; ++i) if (hout[i] != expect[i]) { printf(" [%u: got 0x%x want 0x%x]", i, hout[i], expect[i]); ++shown; }
  }
  printf("\n");
  if (getenv("BX_DUMP")) for (uint32_t i = 0; i < N; ++i) printf("%u in=0x%08x idx=0x%08x got=0x%08x want=0x%08x %s\n", i, hin[i], hidx[i], hout[i], expect[i], hout[i]==expect[i]?"ok":"BAD");
  vkDeviceWaitIdle(dev);
  return bad ? 1 : 0;
}

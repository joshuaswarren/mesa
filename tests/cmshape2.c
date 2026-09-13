/*
 * cmshape2: extended cmshape for the honeykrisp-omarchy coopmat fixes.
 * Adds, without touching the original cmshape:
 *   arg 11: lsize  local_size_x (default 32; 16/8/1/48 exercise partial or
 *                  multi-subgroup workgroups -> software cmat path)
 *   arg 12: elem   memory VIEW element type for all four buffers:
 *                  f16 | f32 (default: same as matrix type) | u16 | u8
 *                  (narrow/wide pointer forms: fp32 matrix behind uint16_t*)
 *   mode "precise": D = precise(C * s + C) element-wise with NoContraction;
 *                  PASS requires the unfused fp32 reference bit-exactly.
 * Buffer BYTES stay component-contiguous in the matrix's own format; only the
 * shader's view type and the coopMatLoad stride units change (stride is in
 * view-element units, so strides scale by matrix_bytes / view_bytes).
 *
 * usage: cmshape2 M N K TA TB TC TD row|col identity|ones|randint|randf|precise [seed] [lsize] [elem]
 * exit: 0 pass, 1 mismatch, 2 extension missing, 3 shape not advertised,
 *       4 setup error.
 */
#include <vulkan/vulkan.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
typedef _Float16 f16;
static void die(const char *msg, int code) {
   printf("{\"result\":\"FAIL\",\"reason\":\"%s\"}\n", msg);
   exit(code);
}
static int is16(const char *t) { return strcmp(t, "f16") == 0; }
static VkComponentTypeKHR ctype(const char *t) {
   return is16(t) ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR;
}
static const char *gltype(const char *t) { return is16(t) ? "float16_t" : "float32_t"; }
static double getv(const void *p, int f16_, size_t i) {
   return f16_ ? (double)((const f16 *)p)[i] : (double)((const float *)p)[i];
}
static void setv(void *p, int f16_, size_t i, double v) {
   if (f16_) ((f16 *)p)[i] = (f16)v; else ((float *)p)[i] = (float)v;
}
static uint32_t rng_state;
static uint32_t rnd(void) {
   rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
   return rng_state;
}
int main(int argc, char **argv) {
   if (argc < 10) { fprintf(stderr, "usage: see header\n"); return 4; }
   int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
   const char *TA = argv[4], *TB = argv[5], *TC = argv[6], *TD = argv[7];
   int colmaj = strcmp(argv[8], "col") == 0;
   const char *mode = argv[9];
   rng_state = argc > 10 ? (uint32_t)atoi(argv[10]) : 0x9E3779B9u;
   if (!rng_state) rng_state = 1;
   int lsize = argc > 11 ? atoi(argv[11]) : 32;
   const char *elem = argc > 12 ? argv[12] : NULL;
   if (!elem || !strcmp(elem, "")) elem = TA;
   if (lsize <= 0 || lsize > 1024) die("bad lsize", 4);
   int precise = !strcmp(mode, "precise");
   int divergent = !strcmp(mode, "divergent");
   int subgroups = (lsize + 31) / 32;
   if (precise && !(strcmp(TA, "f32") == 0 && M == N && M == K))
      die("precise needs f32 and M==K==N", 4);
   if (divergent && strcmp(TA, "f16")) die("divergent needs f16 A", 4);

   /* view element: bytes per view element, GLSL type, required extension */
   unsigned vbytes; const char *vgltype; const char *vexts;
   if (!strcmp(elem, "f16"))       { vbytes = 2; vgltype = "float16_t"; vexts = ""; }
   else if (!strcmp(elem, "f32"))  { vbytes = 4; vgltype = "float32_t"; vexts = ""; }
   else if (!strcmp(elem, "u16"))  { vbytes = 2; vgltype = "uint16_t";
      vexts = "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable\n"; }
   else if (!strcmp(elem, "u8"))   { vbytes = 1; vgltype = "uint8_t";
      vexts = "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable\n"; }
   else die("bad elem", 4);
   int mbytes = is16(TA) ? 2 : 4; /* matrix component bytes */
   if (mbytes % vbytes) die("elem must divide matrix component size", 4);
   int ascale = mbytes / vbytes;
   int bscale = (is16(TB) ? 2 : 4) / vbytes;
   int cscale = (is16(TC) ? 2 : 4) / vbytes;
   int dscale = (is16(TD) ? 2 : 4) / vbytes;

   /* ---- instance / device ---- */
   VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
   ai.pApplicationName = "cmshape2"; ai.apiVersion = VK_API_VERSION_1_3;
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   ici.pApplicationInfo = &ai;
   VkInstance inst;
   if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) die("vkCreateInstance", 4);
   uint32_t npd = 0;
   vkEnumeratePhysicalDevices(inst, &npd, NULL);
   if (!npd) die("no physical device", 4);
   VkPhysicalDevice pds[8];
   if (npd > 8) npd = 8;
   vkEnumeratePhysicalDevices(inst, &npd, pds);
   VkPhysicalDevice pd = pds[0];
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   uint32_t nxt = 0;
   vkEnumerateDeviceExtensionProperties(pd, NULL, &nxt, NULL);
   VkExtensionProperties *exts = calloc(nxt, sizeof *exts);
   vkEnumerateDeviceExtensionProperties(pd, NULL, &nxt, exts);
   int have_cm = 0;
   for (uint32_t i = 0; i < nxt; i++)
      if (!strcmp(exts[i].extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) have_cm = 1;
   free(exts);
   printf("{\"tool\":\"cmshape2\",\"device\":\"%s\",\"driver\":\"", props.deviceName);
   {
      VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
      VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
      vkGetPhysicalDeviceProperties2(pd, &p2);
      printf("%s\",\"shape\":\"%dx%dx%d %s %s %s %s %s %s l%d %s\"", drv.driverInfo, M, N, K, TA, TB, TC, TD,
             colmaj ? "col" : "row", mode, lsize, elem);
   }
   if (!have_cm) { printf(","); die("VK_KHR_cooperative_matrix not exposed", 2); }
   PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getcm =
      (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
         inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
   if (!getcm) { printf(","); die("no vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR", 2); }
   uint32_t ncm = 0;
   getcm(pd, &ncm, NULL);
   VkCooperativeMatrixPropertiesKHR *cm = calloc(ncm ? ncm : 1, sizeof *cm);
   for (uint32_t i = 0; i < ncm; i++) cm[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
   getcm(pd, &ncm, cm);
   int advertised = 0;
   for (uint32_t i = 0; i < ncm; i++) {
      if (cm[i].MSize == (uint32_t)M && cm[i].NSize == (uint32_t)N && cm[i].KSize == (uint32_t)K &&
          cm[i].AType == ctype(TA) && cm[i].BType == ctype(TB) && cm[i].CType == ctype(TC) &&
          cm[i].ResultType == ctype(TD) && cm[i].scope == VK_SCOPE_SUBGROUP_KHR)
         advertised = 1;
   }
   free(cm);
   if (!advertised) { printf(","); die("shape not advertised", 3); }
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties qf[8];
   if (nqf > 8) nqf = 8;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qf);
   uint32_t qfi = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
   if (qfi == UINT32_MAX) die("no compute queue", 4);
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
   qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
   VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
   cmf.cooperativeMatrix = VK_TRUE;
   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &cmf};
   f12.shaderFloat16 = VK_TRUE;
   f12.vulkanMemoryModel = VK_TRUE;
   f12.vulkanMemoryModelDeviceScope = VK_TRUE;
   VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
   f11.storageBuffer16BitAccess = VK_TRUE;
   const char *devext[] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f11};
   dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
   dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = devext;
   VkDevice dev;
   if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) die("vkCreateDevice", 4);
   VkQueue queue;
   vkGetDeviceQueue(dev, qfi, 0, &queue);

   /* ---- shader ---- */
   int sa = colmaj ? M : K, sb = colmaj ? K : N, sc = colmaj ? M : N;
   const char *lay = colmaj ? "gl_CooperativeMatrixLayoutColumnMajor" : "gl_CooperativeMatrixLayoutRowMajor";
   char body[1024];
   if (precise) {
      /* C is loaded, then D = C * s + C with NoContraction (from `precise`). */
      snprintf(body, sizeof body,
         "  coopMatLoad(C, c, 0, %d, %s);\n"
         "  precise coopmat<%s, gl_ScopeSubgroup, %d, %d, gl_MatrixUseAccumulator> D =\n"
         "     C * 1.0078125 + C;\n"
         "  coopMatStore(D, d, doff, %d, %s);\n",
         sc * cscale, lay, gltype(TC), M, N, sc * dscale, lay);
   } else {
      snprintf(body, sizeof body,
         "  coopMatLoad(A, a, 0, %d, %s);\n"
         "  coopMatLoad(B, b, 0, %d, %s);\n"
         "  coopMatLoad(C, c, 0, %d, %s);\n"
         "%s"
         "  coopmat<%s, gl_ScopeSubgroup, %d, %d, gl_MatrixUseAccumulator> D = coopMatMulAdd(A, B, C);\n"
         "  coopMatStore(D, d, doff, %d, %s);\n",
         sa * ascale, lay, sb * bscale, lay, sc * cscale, lay,
         divergent ? "  if (gl_SubgroupInvocationID == 0u) A[0] = float16_t(0.0);\n" : "",
         gltype(TD), M, N, sc * dscale, lay);
   }
   char src[4096];
   snprintf(src, sizeof src,
      "#version 450\n"
      "#extension GL_KHR_cooperative_matrix : enable\n"
      "#extension GL_KHR_memory_scope_semantics : enable\n"
      "#extension GL_KHR_shader_subgroup_basic : enable\n"
      "%s"
      "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\n"
      "#extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable\n"
      "layout(local_size_x=%d) in;\n"
      "layout(binding=0) readonly buffer InA { %s a[]; };\n"
      "layout(binding=1) readonly buffer InB { %s b[]; };\n"
      "layout(binding=2) readonly buffer InC { %s c[]; };\n"
      "layout(binding=3) writeonly buffer OutD { %s d[]; };\n"
      "void main() {\n"
      "  uint doff = gl_SubgroupID * %du;\n"
      "  coopmat<%s, gl_ScopeSubgroup, %d, %d, gl_MatrixUseA> A;\n"
      "  coopmat<%s, gl_ScopeSubgroup, %d, %d, gl_MatrixUseB> B;\n"
      "  coopmat<%s, gl_ScopeSubgroup, %d, %d, gl_MatrixUseAccumulator> C;\n"
      "%s"
      "}\n",
      vexts, lsize,
      vgltype, vgltype, vgltype, vgltype, M * N * dscale,
      gltype(TA), M, K, gltype(TB), K, N, gltype(TC), M, N,
      body);
   char path[256], spvpath[256], cmd[1024];
   snprintf(path, sizeof path, "/tmp/cmshape2_%d.comp", (int)getpid());
   snprintf(spvpath, sizeof spvpath, "/tmp/cmshape2_%d.spv", (int)getpid());
   FILE *f = fopen(path, "w");
   if (!f) die("write comp", 4);
   fputs(src, f);
   fclose(f);
   snprintf(cmd, sizeof cmd, "glslangValidator -V --target-env vulkan1.3 %s -o %s > %s.log 2>&1", path,
            spvpath, path);
   if (system(cmd) != 0) die("glslangValidator failed", 4);
   f = fopen(spvpath, "rb");
   if (!f) die("open spv", 4);
   fseek(f, 0, SEEK_END);
   long spvlen = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *spv = malloc(spvlen);
   if (fread(spv, 1, spvlen, f) != (size_t)spvlen) die("read spv", 4);
   fclose(f);
   VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
   smci.codeSize = spvlen; smci.pCode = spv;
   VkShaderModule mod;
   if (vkCreateShaderModule(dev, &smci, NULL, &mod) != VK_SUCCESS)
      die("vkCreateShaderModule", 4);

   /* ---- descriptors ---- */
   VkDescriptorSetLayoutBinding bnd[4];
   for (int i = 0; i < 4; i++) {
      bnd[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
   }
   VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
   dslci.bindingCount = 4; dslci.pBindings = bnd;
   VkDescriptorSetLayout dsl;
   if (vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl) != VK_SUCCESS) die("dsl", 4);
   VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
   VkPipelineLayout pl;
   if (vkCreatePipelineLayout(dev, &plci, NULL, &pl) != VK_SUCCESS) die("pl", 4);
   VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
   cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
   cpci.stage.module = mod; cpci.stage.pName = "main";
   cpci.layout = pl;
   VkPipeline pipe;
   if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe) != VK_SUCCESS)
      die("vkCreateComputePipelines", 4);
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
   VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
   dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
   VkDescriptorPool pool;
   if (vkCreateDescriptorPool(dev, &dpci, NULL, &pool) != VK_SUCCESS) die("pool", 4);
   VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
   dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
   VkDescriptorSet set;
   if (vkAllocateDescriptorSets(dev, &dsai, &set) != VK_SUCCESS) die("set", 4);

   /* ---- buffers ---- */
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   size_t nelem[4] = {(size_t)M * K, (size_t)K * N, (size_t)M * N, (size_t)M * N * subgroups};
   int e16[4] = {is16(TA), is16(TB), is16(TC), is16(TD)};
   VkBuffer buf[4];
   VkDeviceMemory mem[4];
   void *map[4];
   for (int i = 0; i < 4; i++) {
      VkDeviceSize sz = nelem[i] * (e16[i] ? 2 : 4) + 64;
      VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bci.size = sz; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      if (vkCreateBuffer(dev, &bci, NULL, &buf[i]) != VK_SUCCESS) die("vkCreateBuffer", 4);
      VkMemoryRequirements req;
      vkGetBufferMemoryRequirements(dev, buf[i], &req);
      uint32_t mt = UINT32_MAX;
      for (uint32_t t = 0; t < mp.memoryTypeCount; t++) {
         VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
         if ((req.memoryTypeBits & (1u << t)) && (mp.memoryTypes[t].propertyFlags & want) == want) { mt = t; break; }
      }
      if (mt == UINT32_MAX) die("no host-visible memory", 4);
      VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      mai.allocationSize = req.size; mai.memoryTypeIndex = mt;
      if (vkAllocateMemory(dev, &mai, NULL, &mem[i]) != VK_SUCCESS) die("vkAllocateMemory", 4);
      vkBindBufferMemory(dev, buf[i], mem[i], 0);
      vkMapMemory(dev, mem[i], 0, VK_WHOLE_SIZE, 0, &map[i]);
      memset(map[i], 0, sz);
   }
   double *A = calloc(M * K, sizeof(double)), *B = calloc(K * N, sizeof(double)), *C = calloc(M * N, sizeof(double));
   int identity = !strcmp(mode, "identity"), ones = !strcmp(mode, "ones") || divergent, randf = !strcmp(mode, "randf") || precise;
   if (identity && M != K) die("identity needs M==K", 4);
   for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++)
         A[r * K + k] = identity ? (r == k) : ones ? 1.0 : randf ? ((int)(rnd() % 2001) - 1000) / 1000.0 : (double)((int)(rnd() % 7) - 3);
   for (int k = 0; k < K; k++)
      for (int c = 0; c < N; c++)
         B[k * N + c] = ones ? 1.0 : randf ? ((int)(rnd() % 2001) - 1000) / 1000.0 : (double)((int)(rnd() % 7) - 3);
   for (int r = 0; r < M; r++)
      for (int c = 0; c < N; c++)
         C[r * N + c] = (identity || ones) ? 0.0 : randf ? ((int)(rnd() % 2001) - 1000) / 1000.0 : (double)((int)(rnd() % 7) - 3);
   for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++) {
         size_t idx = colmaj ? (size_t)k * M + r : (size_t)r * K + k;
         setv(map[0], e16[0], idx, A[r * K + k]);
         A[r * K + k] = getv(map[0], e16[0], idx);
      }
   for (int k = 0; k < K; k++)
      for (int c = 0; c < N; c++) {
         size_t idx = colmaj ? (size_t)c * K + k : (size_t)k * N + c;
         setv(map[1], e16[1], idx, B[k * N + c]);
         B[k * N + c] = getv(map[1], e16[1], idx);
      }
   for (int r = 0; r < M; r++)
      for (int c = 0; c < N; c++) {
         size_t idx = colmaj ? (size_t)c * M + r : (size_t)r * N + c;
         setv(map[2], e16[2], idx, C[r * N + c]);
         C[r * N + c] = getv(map[2], e16[2], idx);
      }
   for (size_t i = 0; i < nelem[3]; i++) setv(map[3], e16[3], i, -12345.0);
   (void)A; (void)B;

   /* ---- dispatch ---- */
   VkDescriptorBufferInfo dbi[4];
   VkWriteDescriptorSet w[4];
   for (int i = 0; i < 4; i++) {
      dbi[i] = (VkDescriptorBufferInfo){buf[i], 0, VK_WHOLE_SIZE};
      w[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, (uint32_t)i, 0, 1,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi[i], NULL};
   }
   vkUpdateDescriptorSets(dev, 4, w, 0, NULL);
   VkCommandPoolCreateInfo cpc = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   cpc.queueFamilyIndex = qfi;
   VkCommandPool cpool;
   if (vkCreateCommandPool(dev, &cpc, NULL, &cpool) != VK_SUCCESS) die("cpool", 4);
   VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
   cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
   VkCommandBuffer cb;
   if (vkAllocateCommandBuffers(dev, &cbai, &cb) != VK_SUCCESS) die("cb", 4);
   VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   vkBeginCommandBuffer(cb, &cbbi);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
   vkCmdDispatch(cb, 1, 1, 1);
   vkEndCommandBuffer(cb);
   VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   vkCreateFence(dev, &fci, NULL, &fence);
   VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
   si.commandBufferCount = 1; si.pCommandBuffers = &cb;
   if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) die("submit", 4);
   if (vkWaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull) != VK_SUCCESS) die("fence timeout", 4);

   /* ---- check ---- */
   int bad = 0, shown = 0;
   for (int sg = 0; sg < subgroups; sg++) {
   int reduced = 0;
   for (int r = 0; r < M; r++)
      for (int c = 0; c < N; c++) {
         double ref;
         if (precise) {
            /* unfused fp32: round(c * s), then round(sum) */
            float prod = (float)C[r * N + c] * 1.0078125f;
            ref = (double)((float)(prod + (float)C[r * N + c]));
         } else {
            ref = C[r * N + c];
            for (int k = 0; k < K; k++) ref += A[r * K + k] * B[k * N + c];
            ref = (float)ref;
         }
         size_t idx = colmaj ? (size_t)c * M + r : (size_t)r * N + c;
         idx += (size_t)sg * M * N;
         double got = getv(map[3], e16[3], idx);
         if (divergent) {
            reduced += got == K - 1;
            ref = got == K - 1 ? K - 1 : K;
         }
         double err = fabs(got - ref);
         double tol = randf && !precise ? (e16[3] ? 2e-2 : 1e-4) : 0.0;
         if (!(err <= tol)) {
            bad++;
            if (shown < 300) { fprintf(stderr, "mismatch D[%d][%d]: got %g want %g\n", r, c, got, ref); shown++; }
         }
      }
   if (divergent && reduced != N) bad++;
   }
   printf(",\"elements\":%d,\"mismatches\":%d,\"result\":\"%s\"}\n", M * N * subgroups, bad,
          bad ? "FAIL" : "PASS");
   remove(path);
   remove(spvpath);
   return bad ? 1 : 0;
}

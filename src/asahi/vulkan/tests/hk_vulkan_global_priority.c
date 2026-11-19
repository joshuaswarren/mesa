/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Global queue priorities: every queue family advertises LOW, MEDIUM and
 * HIGH on every chip, and the kernel decides HIGH. Under the no-op DRM shim,
 * ASAHI_SHIM_HIGH_PRIORITY selects its answer: unset accepts, "deny" fails
 * with EACCES (no CAP_SYS_NICE or DRM master) and "unsupported" with EINVAL
 * (a GPU backend that does not schedule HIGH). Both refusals must reach the
 * application as VK_ERROR_NOT_PERMITTED from vkCreateDevice. REALTIME, which
 * the kernel reserves, is always refused; LOW and MEDIUM always succeed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

static unsigned failures, checks;

#define EXPECT(cond, ...)                                                      \
   do {                                                                        \
      checks++;                                                                \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL: " __VA_ARGS__);                                \
         fprintf(stderr, "\n");                                                \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static VkResult
create_device(VkPhysicalDevice pdev, uint32_t family,
              VkQueueGlobalPriorityKHR priority)
{
   const VkDeviceQueueGlobalPriorityCreateInfoKHR priority_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
      .globalPriority = priority,
   };
   const float queue_priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = &priority_info,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
   };
   const char *extensions[] = {VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   VkDevice dev;
   VkResult result = vkCreateDevice(pdev, &device_info, NULL, &dev);
   if (result == VK_SUCCESS)
      vkDestroyDevice(dev, NULL);
   return result;
}

int
main(void)
{
   const char *policy = getenv("ASAHI_SHIM_HIGH_PRIORITY");
   const VkResult high_expected =
      policy ? VK_ERROR_NOT_PERMITTED_KHR : VK_SUCCESS;

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice pdev;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &pdev));

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties2(pdev, &family_count, NULL);
   VkQueueFamilyGlobalPriorityPropertiesKHR prio[8];
   VkQueueFamilyProperties2 props[8];
   if (family_count > 8)
      family_count = 8;
   for (uint32_t i = 0; i < family_count; ++i) {
      prio[i] = (VkQueueFamilyGlobalPriorityPropertiesKHR){
         .sType =
            VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR,
      };
      props[i] = (VkQueueFamilyProperties2){
         .sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
         .pNext = &prio[i],
      };
   }
   vkGetPhysicalDeviceQueueFamilyProperties2(pdev, &family_count, props);
   EXPECT(family_count > 0, "no queue families");

   static const VkQueueGlobalPriorityKHR advertised[] = {
      VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR,
      VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
      VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
   };
   for (uint32_t f = 0; f < family_count; ++f) {
      EXPECT(prio[f].priorityCount == 3, "family %u: %u priorities", f,
             prio[f].priorityCount);
      for (uint32_t i = 0; i < 3 && i < prio[f].priorityCount; ++i)
         EXPECT(prio[f].priorities[i] == advertised[i],
                "family %u: priority %u is %d", f, i, prio[f].priorities[i]);

      VkResult r = create_device(pdev, f, VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR);
      EXPECT(r == VK_SUCCESS, "family %u: LOW: %d", f, r);
      r = create_device(pdev, f, VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR);
      EXPECT(r == VK_SUCCESS, "family %u: MEDIUM: %d", f, r);
      r = create_device(pdev, f, VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR);
      EXPECT(r == high_expected, "family %u: HIGH: %d, expected %d", f, r,
             high_expected);
      r = create_device(pdev, f, VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR);
      EXPECT(r == VK_ERROR_NOT_PERMITTED_KHR, "family %u: REALTIME: %d", f,
             r);
   }

   vkDestroyInstance(instance, NULL);
   printf("%u checks: %s\n", checks, failures ? "FAIL" : "PASS");
   return failures ? 1 : 0;
}

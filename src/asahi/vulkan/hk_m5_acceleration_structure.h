#pragma once
#include "hk_acceleration_structure.h"

VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_BuildAccelerationStructuresKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos);
VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos);
VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer, const VkCopyAccelerationStructureInfoKHR *info);
VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_CmdCopyAccelerationStructureToMemoryKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureToMemoryInfoKHR *info);
VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_CmdCopyMemoryToAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyMemoryToAccelerationStructureInfoKHR *info);
VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_CopyAccelerationStructureKHR(VkDevice _device,
                                VkDeferredOperationKHR deferredOperation,
                                const VkCopyAccelerationStructureInfoKHR *pInfo);
VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_CopyAccelerationStructureToMemoryKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo);
VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_CopyMemoryToAccelerationStructureKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo);
VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_CreateAccelerationStructureKHR(VkDevice device,
   const VkAccelerationStructureCreateInfoKHR *info,
   const VkAllocationCallbacks *alloc, VkAccelerationStructureKHR *out);
VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_DestroyAccelerationStructureKHR(VkDevice device, VkAccelerationStructureKHR handle,
                                   const VkAllocationCallbacks *alloc);
VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_GetAccelerationStructureBuildSizesKHR(
   VkDevice _device, VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo);
VKAPI_ATTR VkDeviceAddress VKAPI_CALL
hk_m5_backend_GetAccelerationStructureDeviceAddressKHR(
   VkDevice _device, const VkAccelerationStructureDeviceAddressInfoKHR *info);
VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice _device, const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility);
VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_WriteAccelerationStructuresPropertiesKHR(
   VkDevice _device, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, size_t dataSize, void *pData, size_t stride);
VkResult
hk_m5_backend_cmd_build_agx3_from_ir(VkCommandBuffer command_buffer,
                        const struct hk_agx3_plan_args *args,
                        VkDeviceAddress node_status,
                        VkDeviceAddress envelope_status);
VkResult
hk_m5_backend_device_init_accel_struct_state(struct hk_device *dev);

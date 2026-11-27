/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "vk_object.h"

struct hk_device;
struct hk_event_slab;

struct hk_event {
   struct vk_object_base base;
   struct agx_bo *bo;
   struct hk_event_slab *slab;
   uint32_t slot;
   uint32_t ts_handle;
   uint32_t ts_offset_B;

   uint64_t addr;
   uint64_t *status;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_event, base, VkEvent, VK_OBJECT_TYPE_EVENT)

void hk_events_init(struct hk_device *dev);
void hk_events_finish(struct hk_device *dev);

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* DRM name of the AGX kernel driver. Kernels before the Aurora rename
 * report "asahi", so both are accepted and Mesa and the kernel can update in
 * either order. Every place that recognises the AGX GPU by its kernel name
 * (the driver, the loader, the pipe-loader, kmsro) takes it from here, so a
 * rename reaches all of them at once. Needs only the C library, so code that
 * does not link the driver can include it.
 */
#define AGX_DRM_DRIVER_NAME        "aurora"
#define AGX_DRM_DRIVER_NAME_LEGACY "asahi"

/* Both names, newest first, for lists of kernel driver names. */
#define AGX_DRM_DRIVER_NAMES AGX_DRM_DRIVER_NAME, AGX_DRM_DRIVER_NAME_LEGACY

static inline bool
agx_is_native_drm_driver(const char *name)
{
   return name && (!strcmp(name, AGX_DRM_DRIVER_NAME) ||
                   !strcmp(name, AGX_DRM_DRIVER_NAME_LEGACY));
}

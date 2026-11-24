/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

/*
 * GEM_USERPTR on kernels without DRM_ASAHI_FEATURE_USERPTR_NO_MEMLOCK also
 * charges the pinned pages to the caller's RLIMIT_MEMLOCK: an import fails
 * with ENOMEM when the process's long-term pins (its VmPin, which counts
 * every pinning user, not only the GPU) plus the new range exceed the soft
 * limit, unless the process has CAP_IPC_LOCK in the initial user namespace.
 * Under systemd the limit is 8 MiB by default, less than many applications
 * import. Kernels with the feature bound the pins by the GPU memory limits
 * only.
 *
 * Limits are in bytes; UINT64_MAX stands for an unlimited RLIMIT_MEMLOCK.
 */

/*
 * Whether to offer host-memory imports without a size bound of their own
 * (VK_EXT_external_memory_host has none). Always where the kernel does not
 * charge memlock. Otherwise only when the memlock limit is no lower than the
 * GPU memory limit that applies to the client (the kernel's per-client
 * limit, else its device limit, else the system's memory): the GPU limit
 * already bounds the client's GEM_USERPTR pins, so memlock refuses an import
 * the GPU limits would take only when other pins of the process use up the
 * difference. Such a refusal is reported as an invalid handle
 * (agx_userptr_memlock_exceeded()).
 *
 * A capability check is not used: CAP_IPC_LOCK in a user namespace shows in
 * capget() but does not lift the kernel's charge.
 */
static inline bool
agx_userptr_imports_any_size(bool kernel_no_memlock, uint64_t memlock_B,
                             uint64_t gpu_limit_B)
{
   return kernel_no_memlock || memlock_B >= gpu_limit_B;
}

/*
 * Whether a kernel that charges memlock refuses pinning size_B more bytes
 * for that limit, with pinned_B already pinned by the process. An ENOMEM it
 * returns otherwise is the GPU memory limits'.
 */
static inline bool
agx_userptr_memlock_exceeded(uint64_t pinned_B, uint64_t size_B,
                             uint64_t memlock_B)
{
   return memlock_B != UINT64_MAX &&
          (pinned_B > memlock_B || size_B > memlock_B - pinned_B);
}

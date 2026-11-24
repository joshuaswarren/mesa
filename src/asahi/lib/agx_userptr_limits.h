/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

/*
 * Whether GEM_USERPTR can import a range of any size the GPU memory limits
 * admit, so that an API may offer host-memory imports without a size bound
 * of its own (VK_EXT_external_memory_host has none).
 *
 * Kernels that report DRM_ASAHI_FEATURE_USERPTR_NO_MEMLOCK bound the pinned
 * pages by the GPU memory limits only. Older kernels with GEM_USERPTR also
 * charge them to the caller's RLIMIT_MEMLOCK unless it has CAP_IPC_LOCK, and
 * refuse with ENOMEM past it: 8 MiB by default under systemd, less than many
 * applications import. There the imports are offered only when that limit
 * cannot be the one that refuses them: no lower than the GPU memory limit
 * that applies to this client.
 *
 * memlock_B is the RLIMIT_MEMLOCK soft limit in bytes (UINT64_MAX when
 * unlimited). gpu_limit_B is the client's GPU memory limit in bytes: the
 * kernel's per-client limit, else its device limit, else the system's
 * memory.
 */
static inline bool
agx_userptr_imports_any_size(bool kernel_no_memlock, bool cap_ipc_lock,
                             uint64_t memlock_B, uint64_t gpu_limit_B)
{
   return kernel_no_memlock || cap_ipc_lock || memlock_B >= gpu_limit_B;
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include <stdio.h>
#include "agx_userptr_limits.h"

#define MiB (1024ull * 1024ull)
#define GiB (1024ull * MiB)

int
main(void)
{
   /* The systemd default memlock of 8 MiB, a 3.91 GB client limit (Neo). */
   const uint64_t memlock = 8 * MiB, client = 3910000000ull;

   /* Kernels that do not charge memlock: always. */
   assert(agx_userptr_imports_any_size(true, memlock, client));
   assert(agx_userptr_imports_any_size(true, 0, client));

   /* Older kernels: the default limit would refuse a 90.7 MiB import. */
   assert(!agx_userptr_imports_any_size(false, memlock, client));
   assert(!agx_userptr_imports_any_size(false, 0, client));
   assert(!agx_userptr_imports_any_size(false, client - 1, client));

   /* ... unless the limit is lifted to the GPU limit or beyond. */
   assert(agx_userptr_imports_any_size(false, client, client));
   assert(agx_userptr_imports_any_size(false, 64 * GiB, client));
   assert(agx_userptr_imports_any_size(false, UINT64_MAX, client));

   /* Without any known limit the caller passes UINT64_MAX: only an
    * unlimited memlock clears it. */
   assert(!agx_userptr_imports_any_size(false, 64 * GiB, UINT64_MAX));
   assert(agx_userptr_imports_any_size(false, UINT64_MAX, UINT64_MAX));

   /* Refusals by memlock: existing pins plus the range past the limit. */
   const uint64_t gb6 = 95104204ull; /* 90.7 MiB */
   assert(agx_userptr_memlock_exceeded(0, gb6, memlock));
   assert(!agx_userptr_memlock_exceeded(0, 8 * MiB, memlock));
   assert(agx_userptr_memlock_exceeded(16 * 1024, 8 * MiB, memlock));
   assert(agx_userptr_memlock_exceeded(9 * MiB, 0, memlock));
   assert(!agx_userptr_memlock_exceeded(1 * GiB, gb6, client));
   assert(agx_userptr_memlock_exceeded(client - gb6 + 1, gb6, client));
   /* Unlimited: never memlock, an ENOMEM is the GPU limits'. */
   assert(!agx_userptr_memlock_exceeded(UINT64_MAX - 1, 64 * GiB, UINT64_MAX));

   printf("ok\n");
   return 0;
}

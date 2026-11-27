/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "hk_physical_device.h"

#include <stdio.h>
#include <string.h>

/* pipelineCacheUUID and the disk cache are keyed on everything that changes
 * generated code, including the kernel's feature bits; the opaque image
 * layout UUID is keyed on the generation contract only. */
#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

struct id {
   uint8_t uuid[VK_UUID_SIZE];
   uint64_t disk;
};

static uint8_t sha[32];
static uint64_t flags = 0x1234, env = 0xfeedface;

static struct id
identity(const struct agx_device *dev)
{
   struct id id;
   memset(&id, 0xa5, sizeof(id));
   hk_cache_identity(sha, dev, flags, env, id.uuid, &id.disk);
   return id;
}

static bool
same(struct id a, struct id b)
{
   return !memcmp(a.uuid, b.uuid, sizeof(a.uuid)) && a.disk == b.disk;
}

/* Both halves move: a stale disk cache is as wrong as a stale UUID. */
static bool
both_differ(struct id a, struct id b)
{
   return memcmp(a.uuid, b.uuid, sizeof(a.uuid)) && a.disk != b.disk;
}

static bool
layout_same(const struct agx_device *a, const struct agx_device *b)
{
   uint8_t ua[VK_UUID_SIZE], ub[VK_UUID_SIZE];
   hk_layout_identity(a, ua);
   hk_layout_identity(b, ub);
   return !memcmp(ua, ub, sizeof(ua));
}

int
main(void)
{
   for (unsigned i = 0; i < sizeof(sha); ++i)
      sha[i] = i * 7 + 1;

   struct agx_device dev = {.chip = AGX_CHIP_G13G};
   dev.params.gpu_generation = 13;
   dev.params.gpu_variant = 'G';
   dev.params.features = DRM_ASAHI_FEATURE_SOFT_FAULTS;

   const struct id base = identity(&dev);
   CHECK(same(base, identity(&dev)));

   /* The disk cache uses the same identity as pipelineCacheUUID. */
   CHECK(!memcmp(&base.disk, base.uuid, sizeof(base.disk)));

   /* Every kernel feature bit, all 64, is part of the identity, and none of
    * them moves the image layout (on G13G no feature changes it). */
   for (unsigned bit = 0; bit < 64; ++bit) {
      struct agx_device other = dev;
      other.params.features ^= UINT64_C(1) << bit;
      CHECK(both_differ(base, identity(&other)));
      CHECK(layout_same(&dev, &other));
   }

   /* The rest of the identity: build, compiler flags, environment, and the
    * generation contract (which also moves the layout UUID). */
   sha[31] ^= 1;
   CHECK(both_differ(base, identity(&dev)));
   sha[31] ^= 1;
   flags ^= 1;
   CHECK(both_differ(base, identity(&dev)));
   flags ^= 1;
   env ^= UINT64_C(1) << 63;
   CHECK(both_differ(base, identity(&dev)));
   env ^= UINT64_C(1) << 63;
   CHECK(same(base, identity(&dev)));

   struct agx_device rev = dev;
   rev.params.gpu_revision = 1;
   CHECK(both_differ(base, identity(&rev)));
   CHECK(!layout_same(&dev, &rev));

   struct agx_device chip = dev;
   chip.chip = AGX_CHIP_G14G;
   chip.params.gpu_generation = 14;
   CHECK(both_differ(base, identity(&chip)));
   CHECK(!layout_same(&dev, &chip));

   printf("hk_cache_identity: OK\n");
   return 0;
}

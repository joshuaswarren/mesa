/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "hk_host_alias.h"
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static enum hk_host_map_class
classify(const char *text, uintptr_t start, uintptr_t end)
{
   return hk_host_maps_classify_text(text, strlen(text), start, end);
}

static const char maps[] =
   "aaaa0000-aaab0000 r-xp 00000000 103:05 1234                       /usr/bin/app\n"
   "aaab0000-aaac0000 rw-p 00010000 103:05 1234                       /usr/bin/app\n"
   "aaac0000-aaae0000 rw-p 00000000 00:00 0                          [heap]\n"
   "b0000000-b0100000 rw-p 00000000 00:00 0 \n"
   "b0100000-b0200000 rw-p 00000000 00:00 0                  [anon:glibc: malloc]\n"
   "b0300000-b0400000 rw-s 00000000 00:01 77                 /memfd:pool (deleted)\n"
   "b0400000-b0500000 rw-s 100000000 00:0e 12                /dev/dri/renderD128\n"
   "b0500000-b0600000 r--p 00000000 103:05 99                /data/weights with space.bin\n"
   "b0600000-b0700000 rw-p 00000000 00:05 4                  /dev/zero\n"
   "b0700000-b0800000 rw-s 00000000 00:01 5                  /SYSV00000000 (deleted)\n"
   "b0800000-b0900000 rw-p 00000000 00:00 0\n"
   "b0900000-b0a00000 rw-s 00000000 00:0f 3                  anon_inode:dmabuf\n"
   "b0a00000-b0b00000 rw-p 00000000 00:00 0                  [stack]\n";

static void
test_maps(void)
{
   /* Private anonymous: plain, [heap], named, and across adjacent ones. */
   assert(classify(maps, 0xaaac0000, 0xaaae0000) == HK_HOST_MAP_PRIVATE_ANON);
   assert(classify(maps, 0xaaad0000, 0xaaad4000) == HK_HOST_MAP_PRIVATE_ANON);
   assert(classify(maps, 0xb0000000, 0xb0100000) == HK_HOST_MAP_PRIVATE_ANON);
   assert(classify(maps, 0xb00f0000, 0xb0110000) == HK_HOST_MAP_PRIVATE_ANON);

   /* Private mappings of a file: page cache until written. */
   assert(classify(maps, 0xaaab0000, 0xaaac0000) == HK_HOST_MAP_PRIVATE);
   assert(classify(maps, 0xb0500000, 0xb0504000) == HK_HOST_MAP_PRIVATE);
   assert(classify(maps, 0xaaab8000, 0xaaac8000) == HK_HOST_MAP_PRIVATE);

   /* Shared, device, special and absent memory. */
   assert(classify(maps, 0xb0300000, 0xb0304000) == HK_HOST_MAP_OTHER);
   assert(classify(maps, 0xb0400000, 0xb0404000) == HK_HOST_MAP_OTHER);
   assert(classify(maps, 0xb0600000, 0xb0604000) == HK_HOST_MAP_OTHER);
   assert(classify(maps, 0xb0700000, 0xb0704000) == HK_HOST_MAP_OTHER);
   assert(classify(maps, 0xb0900000, 0xb0904000) == HK_HOST_MAP_OTHER);
   assert(classify(maps, 0xb0a00000, 0xb0a04000) == HK_HOST_MAP_OTHER);
   /* Private anonymous, then the gap before a shared mapping. */
   assert(classify(maps, 0xb01f0000, 0xb0310000) == HK_HOST_MAP_OTHER);
   /* A gap: [b0200000, b0300000) is unmapped. */
   assert(classify(maps, 0xb0100000, 0xb0300000) == HK_HOST_MAP_OTHER);
   assert(classify(maps, 0xb0200000, 0xb0204000) == HK_HOST_MAP_OTHER);
   /* Past the last line, and empty. */
   assert(classify(maps, 0xb0b00000, 0xb0c00000) == HK_HOST_MAP_OTHER);
   assert(classify(maps, 0x1000, 0x2000) == HK_HOST_MAP_OTHER);
   assert(classify(maps, 0xaaac0000, 0xaaac0000) == HK_HOST_MAP_OTHER);
   /* An anonymous mapping with no trailing pathname padding. */
   assert(classify(maps, 0xb0800000, 0xb0900000) == HK_HOST_MAP_PRIVATE_ANON);

   /* Unparseable lines decide nothing in favour of the range. */
   assert(classify("garbage\n", 0x1000, 0x2000) == HK_HOST_MAP_OTHER);
   assert(classify("1000-2000 rw-x 0 00:00 0\n", 0x1000, 0x2000) ==
          HK_HOST_MAP_OTHER);
   assert(classify("1000-2000 rw-p 0 00:00 0\n", 0x1000, 0x2000) ==
          HK_HOST_MAP_PRIVATE_ANON);
   assert(classify("1000-2000 rw-p 0 00:00 0", 0x1000, 0x2000) ==
          HK_HOST_MAP_PRIVATE_ANON);
   assert(classify("2000-1000 rw-p 0 00:00 0\n", 0x1000, 0x2000) ==
          HK_HOST_MAP_OTHER);
   /* An anonymous name with an inode is not anonymous. */
   assert(classify("1000-2000 rw-p 0 00:05 7\n", 0x1000, 0x2000) ==
          HK_HOST_MAP_OTHER);

   /* The walk stops at the line that decides: later garbage is unread. */
   assert(classify("1000-2000 rw-p 0 00:00 0\ngarbage\n", 0x1000, 0x2000) ==
          HK_HOST_MAP_PRIVATE_ANON);
}

static void
test_self(void)
{
   const size_t page = (size_t)sysconf(_SC_PAGESIZE);
   const size_t size = 4 * page;
   int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
   if (fd < 0)
      return;
   close(fd);

   uint8_t *anon = mmap(NULL, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   assert(anon != MAP_FAILED);
   fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
   assert(hk_host_maps_classify_fd(fd, (uintptr_t)anon,
                                   (uintptr_t)anon + size) ==
          HK_HOST_MAP_PRIVATE_ANON);
   close(fd);

   uint8_t *shared = mmap(NULL, size, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_ANONYMOUS, -1, 0);
   assert(shared != MAP_FAILED);
   fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
   assert(hk_host_maps_classify_fd(fd, (uintptr_t)shared,
                                   (uintptr_t)shared + size) ==
          HK_HOST_MAP_OTHER);
   close(fd);

   /* A hole punched into the private mapping. */
   assert(!munmap(anon + page, page));
   fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
   assert(hk_host_maps_classify_fd(fd, (uintptr_t)anon,
                                   (uintptr_t)anon + size) ==
          HK_HOST_MAP_OTHER);
   close(fd);

   munmap(anon, size);
   munmap(shared, size);
}

static void
test_imports(void)
{
   struct hk_host_imports set = {0};
   int a, b, c, d;

   /* A private import aliases nothing. */
   const struct hk_host_import i0 = {0x10000, 0x20000, &a, true, false, 0};
   assert(hk_host_imports_add(&set, &i0) == 0);

   /* A second import of the same pages: both alias. */
   const struct hk_host_import i1 = {0x18000, 0x1c000, &b, true, false, 0};
   assert(hk_host_imports_add(&set, &i1) == 2);

   /* Adjacent, not overlapping. */
   const struct hk_host_import i2 = {0x20000, 0x30000, &c, true, false, 0};
   assert(hk_host_imports_add(&set, &i2) == 0);

   /* Removing the second leaves the first unaliased. */
   assert(hk_host_imports_remove(&set, &b) == 2);
   assert(hk_host_imports_remove(&set, &b) == 0);

   /* Read-only imports overlapping each other do not alias; one overlapping
    * a writable import does, with it. */
   const struct hk_host_import r0 = {0x40000, 0x50000, &b, false, false, 0};
   const struct hk_host_import r1 = {0x40000, 0x48000, &d, false, false, 0};
   assert(hk_host_imports_add(&set, &r0) == 0);
   assert(hk_host_imports_add(&set, &r1) == 0);
   assert(hk_host_imports_remove(&set, &d) == 0);
   const struct hk_host_import w = {0x4c000, 0x60000, &d, true, false, 0};
   assert(hk_host_imports_add(&set, &w) == 2);

   /* Removing the read-only one ends the aliasing of both. */
   assert(hk_host_imports_remove(&set, &b) == 2);
   assert(hk_host_imports_remove(&set, &d) == 0);
   assert(set.count == 2);
   hk_host_imports_fini(&set);

   /* An import whose own mappings alias counts once, and keeps aliasing
    * when its overlap goes. */
   const struct hk_host_import s0 = {0x10000, 0x20000, &a, true, true, 0};
   const struct hk_host_import s1 = {0x10000, 0x20000, &b, true, false, 0};
   assert(hk_host_imports_add(&set, &s0) == 1);
   assert(hk_host_imports_add(&set, &s1) == 1);
   assert(hk_host_imports_remove(&set, &b) == 1);
   assert(hk_host_imports_remove(&set, &a) == 1);
   assert(set.count == 0);
   hk_host_imports_fini(&set);

   /* The read-only rule of the mappings. */
   assert(!hk_host_import_maps_alias(true, HK_HOST_MAP_PRIVATE_ANON));
   assert(hk_host_import_maps_alias(true, HK_HOST_MAP_PRIVATE));
   assert(hk_host_import_maps_alias(true, HK_HOST_MAP_OTHER));
   assert(!hk_host_import_maps_alias(false, HK_HOST_MAP_PRIVATE_ANON));
   assert(!hk_host_import_maps_alias(false, HK_HOST_MAP_PRIVATE));
   assert(hk_host_import_maps_alias(false, HK_HOST_MAP_OTHER));
}

int
main(void)
{
   test_maps();
   test_self();
   test_imports();
   printf("hk_host_alias: ok\n");
   return 0;
}

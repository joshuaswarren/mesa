/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Execute the same indexed tuple-copy routine used by the G17P kernel. This
 * tests the memory ABI without a GPU; shader compilation is a separate gate.
 */
#include "asahi/libagx/query_copy.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>

/* Keep assertions active in release builds, too. */
#undef NDEBUG
#include <assert.h>

static void
test_flags(unsigned reports, bool timestamp, bool indexed)
{
   uint16_t remap[] = {6, 4, 1, 5, 2, 0, 3};
   uint32_t availability[] = {1, 1, 0, 7, 0, 1, 1};
   uint64_t values[7 * 11];
   for (unsigned q = 0; q < 7; ++q) {
      unsigned index = indexed ? remap[q] : q;
      for (unsigned r = 0; r < reports; ++r)
         values[index * reports + r] =
            (UINT64_C(0xfedcba98) << 32) | (100 * q + r);
      if (timestamp && !availability[q])
         values[index * reports] = UINT64_MAX;
   }

   for (unsigned flags = 0; flags < 16; ++flags) {
      /* WAIT on an unavailable query would deliberately never return. */
      if ((flags & VK_QUERY_RESULT_WAIT_BIT) ||
          (timestamp && (flags & VK_QUERY_RESULT_PARTIAL_BIT)))
         continue;

      uint64_t output[96], expected[96];
      memset(output, 0xa5, sizeof(output));
      memset(expected, 0xa5, sizeof(expected));
      unsigned word = flags & VK_QUERY_RESULT_64_BIT ? 8 : 4;
      unsigned stride = (reports + 3) * word;

      /* An offset destination, nonzero firstQuery, padding between tuples,
       * and guards before/after every write catch remap and width mistakes. */
      for (unsigned i = 0; i < 4; ++i) {
         unsigned q = i + 1;
         libagx_copy_query_g17p_result(
            timestamp ? NULL : availability, values, indexed ? remap : NULL,
            (uintptr_t)output + 16, stride, 1, flags, reports, 0, i);

         uint8_t *dst = (uint8_t *)expected + 16 + i * stride;
         if (availability[q] || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
            for (unsigned r = 0; r < reports; ++r) {
               uint64_t value = values[(indexed ? remap[q] : q) * reports + r];
               if (word == 4) {
                  uint32_t low = value;
                  memcpy(dst + r * word, &low, word);
               } else {
                  memcpy(dst + r * word, &value, word);
               }
            }
         }
         if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
            uint64_t available = availability[q] != 0;
            /* The CI hosts are little endian, as is the target GPU. */
            memcpy(dst + reports * word, &available, word);
         }
      }
      assert(memcmp(output, expected, sizeof(output)) == 0);
   }
}

/* WAIT polls memory another agent writes, so every load of the availability
 * word and of the reports must be volatile: a load the compiler may hoist out
 * of the polling loop would spin forever on a stale value. A runtime test
 * cannot force the producer's store to land while the loop is already
 * spinning, so check the property itself. */
static_assert(
   __builtin_types_compatible_p(
      __typeof__(&libagx_copy_query_g17p_result),
      void (*)(volatile uint32_t *, volatile uint64_t *, uint16_t *, uint64_t,
               uint64_t, uint32_t, uint32_t, uint16_t, uint32_t, uint32_t)),
   "WAIT must poll through volatile pointers");
static_assert(
   __builtin_types_compatible_p(
      __typeof__(&libagx_query_available),
      bool (*)(volatile uint32_t *, uint32_t, uint32_t)),
   "availability must be read through a volatile pointer");

static void
test_wait(bool timestamp)
{
   struct shared {
      uint64_t reports[2];
      uint32_t availability;
      uint64_t output[3];
   } *p = mmap(NULL, sizeof(*p), PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
   assert(p != MAP_FAILED);
   p->reports[0] = UINT64_MAX;
   p->reports[1] = UINT64_MAX;
   p->availability = 0;
   memset(p->output, 0xa5, sizeof(p->output));

   /* Use a process-shared mapping, rather than racing C threads on volatile
    * objects. The producer publishes only once the consumer has found the
    * query unavailable and is about to wait for it. */
   int start[2];
   assert(pipe(start) == 0);
   pid_t child = fork();
   assert(child >= 0);
   if (child == 0) {
      char go;
      close(start[1]);
      if (read(start[0], &go, 1) != 1)
         _exit(1);
      p->reports[1] = UINT64_C(0x8877665544332211);
      __atomic_store_n(&p->reports[0], UINT64_C(0x1122334455667788),
                       __ATOMIC_RELEASE);
      __atomic_store_n(&p->availability, 1, __ATOMIC_RELEASE);
      _exit(0);
   }

   close(start[0]);
   uint64_t output[3];
   libagx_copy_query_g17p_result(
      timestamp ? NULL : &p->availability, p->reports, NULL,
      (uintptr_t)output, 0, 0,
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT,
      timestamp ? 1 : 2, 0, 0);
   assert(output[timestamp ? 1 : 2] == 0);
   assert(write(start[1], "", 1) == 1);
   close(start[1]);

   /* Watchdog only: a correct WAIT returns as soon as the store lands. */
   alarm(5);
   libagx_copy_query_g17p_result(
      timestamp ? NULL : &p->availability, p->reports, NULL,
      (uintptr_t)p->output, 0, 0,
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT |
         VK_QUERY_RESULT_WAIT_BIT |
         (timestamp ? 0 : VK_QUERY_RESULT_PARTIAL_BIT),
      timestamp ? 1 : 2, 0, 0);
   alarm(0);
   assert(p->output[0] == UINT64_C(0x1122334455667788));
   if (!timestamp)
      assert(p->output[1] == UINT64_C(0x8877665544332211));
   assert(p->output[timestamp ? 1 : 2] == 1);
   int status;
   assert(waitpid(child, &status, 0) == child);
   assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
   assert(munmap(p, sizeof(*p)) == 0);
}

static void
test_stamp_slots(void)
{
   /* 64-bit availability slots: a retirement timestamp may have either half
    * zero, the queued store writes 1 to the low half, a reset writes both. */
   uint64_t availability[] = {0, 1, UINT64_C(0x100000000),
                              UINT64_C(0x1234567800000000), 0,
                              UINT64_C(0x00000000deadbeef)};
   uint64_t values[6];
   for (unsigned q = 0; q < 6; ++q)
      values[q] = 1000 + q;

   for (unsigned q = 0; q < 6; ++q) {
      uint64_t output[2];
      memset(output, 0xa5, sizeof(output));
      libagx_copy_query_g17p_result(
         (uint32_t *)availability, values, NULL, (uintptr_t)output, 0, q,
         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 1, 1,
         0);
      const bool available = availability[q] != 0;
      assert(output[1] == available);
      if (available)
         assert(output[0] == values[q]);
      else
         assert(output[0] == UINT64_C(0xa5a5a5a5a5a5a5a5));
   }
}

static void
test_large_stride(void)
{
   /* Reserve virtual space only: exercise 64-bit stride multiplication and
    * destination arithmetic without allocating gigabytes of physical RAM. */
   if (sizeof(uintptr_t) < 8)
      return;
   const uint64_t stride = UINT64_C(1) << 32;
   size_t size = 2 * stride + 4096;
   uint8_t *dst = mmap(NULL, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   assert(dst != MAP_FAILED);
   uint64_t values[] = {11, 22, 33};
   for (unsigned i = 0; i < 3; ++i) {
      libagx_copy_query_g17p_result(
         NULL, values, NULL, (uintptr_t)dst, stride, 0,
         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 1, 0,
         i);
      assert(*(uint64_t *)(dst + i * stride) == values[i]);
      assert(*(uint64_t *)(dst + i * stride + 8) == 1);
   }
   assert(munmap(dst, size) == 0);
}

int
main(void)
{
   test_flags(1, false, true);   /* Non-contiguous occlusion heap. */
   test_flags(2, false, false);  /* Transform feedback. */
   test_flags(11, false, false); /* All pipeline statistics. */
   test_flags(1, true, false);   /* Timestamp sentinel availability. */
   test_wait(false);
   test_wait(true);
   test_stamp_slots();
   test_large_stride();
   puts(
      "query range copies: flags, remaps, tuples, WAIT, stamp slots and 64-bit "
      "strides passed");
   return 0;
}

/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 *
 * Minimal Vulkan compute harness for the Honeykrisp miscompile reproductions.
 *
 * One device, one queue, one descriptor set with four storage buffers, one
 * pipeline per case, one submit per dispatch. Host-visible memory only, so
 * every result is read back directly. Nothing here is tuned; the harness
 * exists so each case file stays small enough to review by eye.
 */

#ifndef HK_REPRO_VK_H
#define HK_REPRO_VK_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define HK_MAX_SSBOS 4
#define HK_PUSH_SIZE 16 /* four uint32, the minimum push constant size */

struct hk_buffer {
   VkBuffer buf;
   VkDeviceMemory mem;
   void *map;
   uint64_t size;
};

struct hk_ctx {
   VkInstance instance;
   VkPhysicalDevice pdev;
   VkDevice dev;
   VkQueue queue;
   uint32_t queue_family;
   VkCommandPool cmdpool;
   VkDescriptorPool pool;
   VkDescriptorSetLayout set_layout;
   VkDescriptorSet set;
   VkPipelineLayout pipe_layout;
   char name[256];
   uint32_t api_version;
};

/* Load a case: numbers are case indices, or -1 for all. Returns nonzero if
 * any run mismatched its expected values. */
int hk_run(int only_case);

/* Case definition, filled in by each case file. */
struct hk_case {
   int number;
   const char *name;      /* short slug, used for the .spv path */
   const char *spv;       /* path to the SPIR-V produced by make */
   uint32_t groups;       /* dispatch group count, 1D */
   uint32_t push[4];      /* push constant words, may be unused */

   /* Input/output buffers. Inputs are filled from data before the
    * dispatch; outputs are zeroed. Binding i uses slot i. A NULL data
    * pointer means "leave zeroed". */
   uint32_t size[HK_MAX_SSBOS]; /* size in uint32 words per binding */
   const uint32_t *data[HK_MAX_SSBOS];

   /* Called after the dispatch with the mapped output words. Must print
    * its own verdict and return false on mismatch. */
   bool (*check)(const struct hk_case *c, const uint32_t *out[HK_MAX_SSBOS]);
};

extern const struct hk_case hk_cases[];
extern const int hk_case_count;

/* Shared helpers. */
bool hk_init(struct hk_ctx *k);
void hk_finish(struct hk_ctx *k);
bool hk_alloc_buffer(struct hk_ctx *k, struct hk_buffer *b, uint64_t bytes);
void hk_free_buffer(struct hk_ctx *k, struct hk_buffer *b);
bool hk_dispatch(struct hk_ctx *k, const char *spv_path, uint32_t groups,
                 const uint32_t *push, struct hk_buffer *ssbos,
                 unsigned n_ssbos);

/* Verdict helpers, shared by the case files. */
void hk_print_expect(const char *what);
void hk_print_case_header(const struct hk_case *c);
/* Return true when all n words match. Print the first few mismatches. */
bool hk_check_words(const char *label, const uint32_t *got,
                    const uint32_t *want, unsigned n);

#endif

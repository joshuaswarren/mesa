/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Whether host memory imported into an M5 device can give the GPU a second
 * virtual address of pages it writes (hk_cdm_alias_scope()). Pure functions,
 * so host tests run them on synthetic /proc/self/maps text.
 *
 * What the mappings over an imported range say, best first: a range takes
 * the worst class of the mappings over it.
 */
enum hk_host_map_class {
   /* Private anonymous memory: no pathname, "[heap]", or an anonymous
    * region named with PR_SET_VMA_ANON_NAME ("[anon:...]"). Its pages are
    * mapped at this one virtual address of this process and nowhere else
    * (mremap moves them, fork shares them only until a write, and pinning
    * for writing breaks that sharing first). */
   HK_HOST_MAP_PRIVATE_ANON,
   /* A private mapping of a file outside /dev: until written, its pages are
    * the file's page cache, which any shared mapping of the file maps too. */
   HK_HOST_MAP_PRIVATE,
   /* Anything else: a shared mapping (System V or POSIX shm, memfd, a shared
    * file, a mapped Vulkan or DRM buffer, a dma-buf), a mapping of a device
    * or anonymous inode, [stack] and other special regions, a gap in the
    * range, or a line that does not parse. */
   HK_HOST_MAP_OTHER,
};

/* Parses one line of /proc/<pid>/maps, without its newline. */
bool hk_host_maps_parse_line(const char *line, size_t len, uintptr_t *start,
                             uintptr_t *end, enum hk_host_map_class *cls);

/* Folds the lines of a maps file, in its (ascending) order, over the range
 * [start, end). Lines below the range are skipped, and the walk stops once
 * the range is covered, a gap shows, or a line is HK_HOST_MAP_OTHER. */
struct hk_host_maps_cover {
   uintptr_t next, end;
   enum hk_host_map_class cls;
   bool done;
};

void hk_host_maps_cover_init(struct hk_host_maps_cover *c, uintptr_t start,
                             uintptr_t end);

/* Feeds one line; returns true once the class of the range is decided. */
bool hk_host_maps_cover_line(struct hk_host_maps_cover *c, const char *line,
                             size_t len);

/* The class of the range: HK_HOST_MAP_OTHER unless the lines fed covered
 * all of it. */
enum hk_host_map_class
hk_host_maps_cover_result(const struct hk_host_maps_cover *c);

/* The class of [start, end) in maps text, or read from an open maps file
 * (only up to the line that decides it). */
enum hk_host_map_class hk_host_maps_classify_text(const char *text,
                                                  size_t len, uintptr_t start,
                                                  uintptr_t end);
enum hk_host_map_class hk_host_maps_classify_fd(int fd, uintptr_t start,
                                                uintptr_t end);

/* A live host import of a device. It aliases (hk_host_import_aliases()) when
 * its own mappings allow another address of its pages, or while it overlaps
 * another live import of the same device of which at least one of the two
 * is GPU-writable: the same pages then have two GPU addresses, one of which
 * the GPU writes. Two read-only imports of the same pages never miss a
 * write, since the GPU writes neither. */
struct hk_host_import {
   uintptr_t start, end;
   const void *owner;
   /* Bound read-write for the GPU. */
   bool gpu_writes;
   /* Its mappings allow another virtual address of its pages: for a
    * GPU-writable import anything but HK_HOST_MAP_PRIVATE_ANON, for a
    * read-only one HK_HOST_MAP_OTHER. */
   bool maps_alias;
   /* Live imports it overlaps, at least one of each pair GPU-writable. */
   uint32_t overlaps;
};

static inline bool
hk_host_import_aliases(const struct hk_host_import *import)
{
   return import->maps_alias || import->overlaps != 0;
}

/* Whether an import with these properties has mappings that allow another
 * virtual address of its pages. */
static inline bool
hk_host_import_maps_alias(bool gpu_writes, enum hk_host_map_class cls)
{
   return gpu_writes ? cls != HK_HOST_MAP_PRIVATE_ANON
                     : cls == HK_HOST_MAP_OTHER;
}

struct hk_host_imports {
   struct hk_host_import *items;
   unsigned count, capacity;
};

/* Adds an import (owner unique). Returns how many imports, it included,
 * began to alias, or -1 when out of memory (nothing changed). */
int hk_host_imports_add(struct hk_host_imports *set,
                        const struct hk_host_import *import);

/* Removes the import of owner. Returns how many imports, it included,
 * stopped aliasing; 0 when owner has no import in the set. */
unsigned hk_host_imports_remove(struct hk_host_imports *set,
                                const void *owner);

void hk_host_imports_fini(struct hk_host_imports *set);

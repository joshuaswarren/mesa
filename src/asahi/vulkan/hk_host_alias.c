/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_host_alias.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static bool
parse_hex64(const char **p, const char *end, uint64_t *out)
{
   const char *s = *p;
   uint64_t v = 0;
   unsigned digits = 0;
   for (; s < end; ++s, ++digits) {
      unsigned d;
      if (*s >= '0' && *s <= '9')
         d = *s - '0';
      else if (*s >= 'a' && *s <= 'f')
         d = *s - 'a' + 10;
      else if (*s >= 'A' && *s <= 'F')
         d = *s - 'A' + 10;
      else
         break;
      if (v > (UINT64_MAX >> 4))
         return false;
      v = (v << 4) | d;
   }
   if (!digits)
      return false;
   *p = s;
   *out = v;
   return true;
}

/* Addresses fit the process's pointer width; the file offset does not
 * (a 32-bit process maps DRM buffers at offsets above 4 GiB). */
static bool
parse_hex(const char **p, const char *end, uintptr_t *out)
{
   uint64_t v;
   if (!parse_hex64(p, end, &v) || v > UINTPTR_MAX)
      return false;
   *out = (uintptr_t)v;
   return true;
}

static bool
parse_dec(const char **p, const char *end, uint64_t *out)
{
   const char *s = *p;
   uint64_t v = 0;
   for (; s < end && *s >= '0' && *s <= '9'; ++s) {
      if (v > (UINT64_MAX - 9) / 10)
         return false;
      v = v * 10 + (uint64_t)(*s - '0');
   }
   if (s == *p)
      return false;
   *p = s;
   *out = v;
   return true;
}

static bool
expect(const char **p, const char *end, char c)
{
   if (*p == end || **p != c)
      return false;
   ++*p;
   return true;
}

static bool
starts_with(const char *s, size_t len, const char *prefix)
{
   const size_t n = strlen(prefix);
   return len >= n && !memcmp(s, prefix, n);
}

/* start-end perms offset major:minor inode [pathname], the pathname padded
 * with spaces and possibly holding spaces itself. */
bool
hk_host_maps_parse_line(const char *line, size_t len, uintptr_t *start,
                        uintptr_t *end, enum hk_host_map_class *cls)
{
   const char *p = line, *e = line + len;
   uintptr_t lo, hi, major, minor;
   uint64_t offset, inode;
   if (!parse_hex(&p, e, &lo) || !expect(&p, e, '-') ||
       !parse_hex(&p, e, &hi) || !expect(&p, e, ' ') || e - p < 5 ||
       p[4] != ' ' || hi <= lo)
      return false;

   const char *perms = p;
   p += 5;
   if (!parse_hex64(&p, e, &offset) || !expect(&p, e, ' ') ||
       !parse_hex(&p, e, &major) || !expect(&p, e, ':') ||
       !parse_hex(&p, e, &minor) || !expect(&p, e, ' ') ||
       !parse_dec(&p, e, &inode))
      return false;
   if (p != e && *p != ' ')
      return false;
   while (p != e && *p == ' ')
      ++p;
   const char *path = p;
   const size_t path_len = e - p;

   if (perms[3] == 's') {
      *cls = HK_HOST_MAP_OTHER;
   } else if (perms[3] != 'p') {
      return false;
   } else if (path_len == 0 || (path_len == 6 && !memcmp(path, "[heap]", 6)) ||
              starts_with(path, path_len, "[anon:")) {
      /* An anonymous mapping has no inode; anything else here is not. */
      *cls = inode == 0 && major == 0 && minor == 0 ? HK_HOST_MAP_PRIVATE_ANON
                                                    : HK_HOST_MAP_OTHER;
   } else if (path[0] == '/' && !starts_with(path, path_len, "/dev/") &&
              !starts_with(path, path_len, "/dmabuf")) {
      *cls = HK_HOST_MAP_PRIVATE;
   } else {
      *cls = HK_HOST_MAP_OTHER;
   }

   *start = lo;
   *end = hi;
   return true;
}

void
hk_host_maps_cover_init(struct hk_host_maps_cover *c, uintptr_t start,
                        uintptr_t end)
{
   c->next = start;
   c->end = end;
   c->done = start >= end;
   c->cls = c->done ? HK_HOST_MAP_OTHER : HK_HOST_MAP_PRIVATE_ANON;
}

bool
hk_host_maps_cover_line(struct hk_host_maps_cover *c, const char *line,
                        size_t len)
{
   if (c->done)
      return true;

   uintptr_t lo, hi;
   enum hk_host_map_class cls;
   if (!hk_host_maps_parse_line(line, len, &lo, &hi, &cls)) {
      c->cls = HK_HOST_MAP_OTHER;
      c->done = true;
      return true;
   }

   if (hi <= c->next)
      return false;

   if (lo > c->next) {
      /* Nothing maps [next, lo). */
      c->cls = HK_HOST_MAP_OTHER;
      c->done = true;
      return true;
   }

   if (cls > c->cls)
      c->cls = cls;
   c->next = hi;
   c->done = c->cls == HK_HOST_MAP_OTHER || c->next >= c->end;
   return c->done;
}

enum hk_host_map_class
hk_host_maps_cover_result(const struct hk_host_maps_cover *c)
{
   return c->done ? c->cls : HK_HOST_MAP_OTHER;
}

enum hk_host_map_class
hk_host_maps_classify_text(const char *text, size_t len, uintptr_t start,
                           uintptr_t end)
{
   struct hk_host_maps_cover c;
   hk_host_maps_cover_init(&c, start, end);
   const char *p = text, *e = text + len;
   while (!c.done && p != e) {
      const char *nl = memchr(p, '\n', e - p);
      const char *line_end = nl ? nl : e;
      hk_host_maps_cover_line(&c, p, line_end - p);
      p = nl ? nl + 1 : e;
   }
   return hk_host_maps_cover_result(&c);
}

enum hk_host_map_class
hk_host_maps_classify_fd(int fd, uintptr_t start, uintptr_t end)
{
   struct hk_host_maps_cover c;
   hk_host_maps_cover_init(&c, start, end);

   /* A line holds at most a PATH_MAX pathname after its fixed fields. */
   char buf[8192];
   size_t have = 0;
   while (!c.done) {
      const ssize_t n = read(fd, buf + have, sizeof(buf) - have);
      if (n < 0 && errno == EINTR)
         continue;
      if (n < 0)
         return HK_HOST_MAP_OTHER;
      if (n == 0) {
         if (have)
            hk_host_maps_cover_line(&c, buf, have);
         break;
      }
      have += (size_t)n;

      char *p = buf, *e = buf + have, *nl;
      while (!c.done && (nl = memchr(p, '\n', e - p))) {
         hk_host_maps_cover_line(&c, p, nl - p);
         p = nl + 1;
      }
      have = e - p;
      if (have == sizeof(buf))
         return HK_HOST_MAP_OTHER;
      memmove(buf, p, have);
   }
   return hk_host_maps_cover_result(&c);
}

static bool
hk_host_imports_overlap(const struct hk_host_import *a,
                        const struct hk_host_import *b)
{
   return a->start < b->end && b->start < a->end &&
          (a->gpu_writes || b->gpu_writes);
}

int
hk_host_imports_add(struct hk_host_imports *set,
                    const struct hk_host_import *import)
{
   if (set->count == set->capacity) {
      const unsigned capacity = set->capacity ? set->capacity * 2 : 16;
      struct hk_host_import *items =
         realloc(set->items, capacity * sizeof(*items));
      if (!items)
         return -1;
      set->items = items;
      set->capacity = capacity;
   }

   struct hk_host_import added = *import;
   added.overlaps = 0;
   int began = 0;
   for (unsigned i = 0; i < set->count; ++i) {
      struct hk_host_import *other = &set->items[i];
      if (!hk_host_imports_overlap(other, &added))
         continue;
      began += !hk_host_import_aliases(other);
      other->overlaps++;
      added.overlaps++;
   }
   began += hk_host_import_aliases(&added);
   set->items[set->count++] = added;
   return began;
}

unsigned
hk_host_imports_remove(struct hk_host_imports *set, const void *owner)
{
   unsigned index = 0;
   while (index < set->count && set->items[index].owner != owner)
      ++index;
   if (index == set->count)
      return 0;

   const struct hk_host_import removed = set->items[index];
   set->items[index] = set->items[--set->count];

   unsigned stopped = hk_host_import_aliases(&removed);
   for (unsigned i = 0; i < set->count; ++i) {
      struct hk_host_import *other = &set->items[i];
      if (!hk_host_imports_overlap(other, &removed))
         continue;
      other->overlaps--;
      stopped += !hk_host_import_aliases(other);
   }
   return stopped;
}

void
hk_host_imports_fini(struct hk_host_imports *set)
{
   free(set->items);
   set->items = NULL;
   set->count = set->capacity = 0;
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * The G18G graphics template policy (agx_g18g_template_policy.h) over draw
 * sequences, as agx_m5_graphics_program_cached() applies it: a REPLACE makes
 * a new template from agx_g18g_template_replacement(), a DROP gives the
 * shape's agx3d form up for good.
 */

#include <stdio.h>

#include "asahi/lib/agx_g18g_template_policy.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,    \
                 #x);                                                          \
         return 1;                                                             \
      }                                                                        \
   } while (0)

struct shape {
   struct agx_g18g_template_state state;
   unsigned made, patched, full;
   bool dropped;
};

/* One draw that the template patches or refuses */
static void
draw(struct shape *s, bool fits)
{
   if (s->dropped) {
      ++s->full;
      return;
   }
   if (fits)
      ++s->patched;
   else
      ++s->full;
   switch (agx_g18g_template_after_draw(&s->state, fits)) {
   case AGX_G18G_TEMPLATE_KEEP:
      break;
   case AGX_G18G_TEMPLATE_REPLACE:
      s->state = agx_g18g_template_replacement(&s->state);
      ++s->made;
      break;
   case AGX_G18G_TEMPLATE_DROP:
      s->dropped = true;
      break;
   }
}

/* A run of refused draws (addresses of another size class), then patched */
static void
run(struct shape *s, unsigned refused, unsigned patched)
{
   for (unsigned i = 0; i < refused; ++i)
      draw(s, false);
   for (unsigned i = 0; i < patched; ++i)
      draw(s, true);
}

int
main(void)
{
   /* Alternating size classes: never replaced */
   {
      struct shape s = {.made = 1};
      for (unsigned i = 0; i < 100000; ++i)
         draw(&s, i & 1);
      CHECK(s.made == 1 && !s.dropped && s.patched == 50000);
   }

   /* Runs shorter than AGX_G18G_TEMPLATE_MISSES: never replaced */
   {
      struct shape s = {.made = 1};
      for (unsigned i = 0; i < 1000; ++i)
         run(&s, AGX_G18G_TEMPLATE_MISSES - 1, 1);
      CHECK(s.made == 1 && !s.dropped);
   }

   /* A run longer than AGX_G18G_TEMPLATE_MISSES: one replacement, which then
    * patches the rest of the run's addresses */
   {
      struct shape s = {.made = 1};
      run(&s, AGX_G18G_TEMPLATE_MISSES, 1000);
      CHECK(s.made == 2 && !s.dropped && s.state.rebuilds == 1);
      CHECK(s.full == AGX_G18G_TEMPLATE_MISSES && s.patched == 1000);
   }

   /* A third run within AGX_G18G_TEMPLATE_WINDOW patched draws: the agx3d
    * form is given up, and every later draw is composed in full */
   {
      struct shape s = {.made = 1};
      run(&s, 12, 100);
      run(&s, 12, 100);
      CHECK(s.made == 3 && !s.dropped);
      run(&s, 12, 100);
      CHECK(s.made == 3 && s.dropped);
      run(&s, 0, 5000);
      CHECK(s.patched == 200);
   }

   /* A third run after AGX_G18G_TEMPLATE_WINDOW patched draws: a new window,
    * so the replacement is allowed; a size flip every so often keeps its
    * template for good */
   {
      struct shape s = {.made = 1};
      run(&s, 12, 100);
      run(&s, 12, AGX_G18G_TEMPLATE_WINDOW);
      CHECK(s.made == 3 && s.state.rebuilds == 0);
      for (unsigned i = 0; i < 100; ++i)
         run(&s, 12, AGX_G18G_TEMPLATE_WINDOW);
      CHECK(s.made == 103 && !s.dropped);
      /* Patched near 100% between switches */
      CHECK(s.full == 102 * 12 && s.patched == 100 + 101 * AGX_G18G_TEMPLATE_WINDOW);
   }

   /* Window counting starts with each new template: a window that ends in
    * a replacement does not carry over */
   {
      struct shape s = {.made = 1};
      run(&s, 12, AGX_G18G_TEMPLATE_WINDOW - 1);
      run(&s, 12, AGX_G18G_TEMPLATE_WINDOW - 1);
      CHECK(s.made == 3 && s.state.rebuilds == 2);
      run(&s, 12, 0);
      CHECK(s.dropped);
   }

   printf("template policy: all sequences as expected\n");
   return 0;
}

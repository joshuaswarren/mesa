/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

/*
 * When a G18G graphics program template (agx3_reencode_template) is kept,
 * replaced or given up. A draw whose addresses make an agx3d LDIMM take
 * another size than in the template is refused by the template and composed
 * in full. Only a run of AGX_G18G_TEMPLATE_MISSES such draws replaces the
 * template with one made from the current addresses, so addresses that
 * alternate between sizes do not rebuild it every draw. A shape that needs
 * more than AGX_G18G_TEMPLATE_REBUILDS replacements within a window of
 * AGX_G18G_TEMPLATE_WINDOW patched draws loses its agx3d form and is
 * composed in full from then on, as without templates.
 */
#define AGX_G18G_TEMPLATE_MISSES  8u
#define AGX_G18G_TEMPLATE_REBUILDS 2u
#define AGX_G18G_TEMPLATE_WINDOW  4096u

struct agx_g18g_template_state {
   /* Draws in a row the template refused */
   uint8_t misses;
   /* Templates of this shape replaced in this window */
   uint8_t rebuilds;
   /* Draws patched since this template was made, counted while rebuilds */
   uint16_t emits;
};

enum agx_g18g_template_action {
   /* Keep the template: the draw was patched, or is composed in full */
   AGX_G18G_TEMPLATE_KEEP,
   /* Replace it with one made from this draw's addresses */
   AGX_G18G_TEMPLATE_REPLACE,
   /* Give up the agx3d form: compose the shape in full from now on */
   AGX_G18G_TEMPLATE_DROP,
};

/* Account one draw of the template, patched or refused */
static inline enum agx_g18g_template_action
agx_g18g_template_after_draw(struct agx_g18g_template_state *s, bool patched)
{
   if (patched) {
      s->misses = 0;
      /* A long run of patched draws starts a new window */
      if (s->rebuilds && ++s->emits >= AGX_G18G_TEMPLATE_WINDOW)
         s->rebuilds = 0;
      return AGX_G18G_TEMPLATE_KEEP;
   }
   if (++s->misses < AGX_G18G_TEMPLATE_MISSES)
      return AGX_G18G_TEMPLATE_KEEP;
   return s->rebuilds < AGX_G18G_TEMPLATE_REBUILDS ? AGX_G18G_TEMPLATE_REPLACE
                                                   : AGX_G18G_TEMPLATE_DROP;
}

/* The state of the template that replaces one (AGX_G18G_TEMPLATE_REPLACE) */
static inline struct agx_g18g_template_state
agx_g18g_template_replacement(const struct agx_g18g_template_state *s)
{
   return (struct agx_g18g_template_state){.rebuilds = s->rebuilds + 1};
}

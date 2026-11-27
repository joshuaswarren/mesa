/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>

/* Whether a fragment shader's user varying binding can be fed from the bound
 * vertex program's output (UVS) words on the G17P native path.
 *
 * offset and count are the components the fragment shader reads,
 * vs_components the number the vertex program writes for that varying,
 * linked_slot the varying's first UVS word and uvs_size the program's total
 * UVS words. User words start at word 4, after the position.
 *
 * A fragment input may read more components than the vertex program writes:
 * translated shaders commonly declare every input as a vec4. Those components
 * are undefined, and agx_link_varyings_vs_fs() binds them to the following
 * UVS words, as it does on every other chip. That is safe as long as the
 * binding stays inside the program's UVS words. A binding that starts past the
 * written components, or a varying the vertex program does not write at all,
 * is still refused.
 */
static inline bool
hk_g17p_user_varying_binding_fits(unsigned offset, unsigned count,
                                  unsigned vs_components, unsigned linked_slot,
                                  unsigned uvs_size)
{
   return offset < vs_components && linked_slot >= 4 &&
          linked_slot + offset + count <= uvs_size;
}

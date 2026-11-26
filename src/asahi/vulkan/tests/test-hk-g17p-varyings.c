/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>

#include "hk_g17p_varyings.h"

static unsigned checks, failures;

#define CHECK(cond)                                                            \
   do {                                                                        \
      ++checks;                                                                \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

int
main(void)
{
   /* A fragment program of Mario Kart World: 31 UVS words, its vec4 inputs
    * linked at words 4, 6, 10, 14, 17, 20 and 25. VAR0 reads 4 components of
    * which the vertex program writes 2, VAR7 and VAR8 read 4 of 3. */
   CHECK(hk_g17p_user_varying_binding_fits(0, 4, 2, 4, 31));  /* VAR0 */
   CHECK(hk_g17p_user_varying_binding_fits(0, 4, 3, 14, 31)); /* VAR7 */
   CHECK(hk_g17p_user_varying_binding_fits(0, 4, 3, 17, 31)); /* VAR8 */
   CHECK(hk_g17p_user_varying_binding_fits(3, 1, 4, 25, 31)); /* VAR4.w */
   CHECK(hk_g17p_user_varying_binding_fits(0, 4, 4, 6, 31));  /* VAR5 */

   /* Reads exactly what is written, ending on the last UVS word. */
   CHECK(hk_g17p_user_varying_binding_fits(0, 4, 4, 27, 31));
   CHECK(hk_g17p_user_varying_binding_fits(1, 3, 4, 27, 31));

   /* The extra components would run past the program's UVS words. */
   CHECK(!hk_g17p_user_varying_binding_fits(0, 4, 2, 28, 31));
   CHECK(!hk_g17p_user_varying_binding_fits(1, 3, 2, 28, 31));

   /* The binding starts past the written components. */
   CHECK(!hk_g17p_user_varying_binding_fits(2, 2, 2, 4, 31));
   CHECK(!hk_g17p_user_varying_binding_fits(3, 1, 3, 4, 31));

   /* The vertex program does not write the varying. */
   CHECK(!hk_g17p_user_varying_binding_fits(0, 1, 0, 0, 31));
   CHECK(!hk_g17p_user_varying_binding_fits(0, 4, 0, 4, 31));

   /* A user varying never sits in the position words. */
   CHECK(!hk_g17p_user_varying_binding_fits(0, 2, 2, 0, 31));
   CHECK(!hk_g17p_user_varying_binding_fits(0, 1, 4, 3, 31));

   printf("hk_g17p_varyings: %u checks, %u failures\n", checks, failures);
   return failures ? 1 : 0;
}

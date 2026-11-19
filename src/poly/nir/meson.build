# Copyright © 2025 Collabora Ltd.
# SPDX-License-Identifier: MIT

libpoly_nir_files = files(
  'poly_nir_lower_gs.c',
  'poly_nir_lower_sysvals.c',
  'poly_nir_lower_tess.c',
  'poly_nir_lower_vs.c',
  'poly_nir_passthrough_gs.c',
)

libpoly_nir = static_library(
  'libpoly_nir',
  [libpoly_nir_files],
  include_directories : [inc_poly],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  dependencies: [idep_nir, idep_mesautil, idep_libpoly],
  build_by_default : false,
)

if with_tests
  test(
    'poly-gs-output-types',
    executable(
      'poly-gs-output-types',
      '../tests/poly-gs-output-types.c',
      include_directories : [inc_poly],
      link_with : [libpoly_nir],
      dependencies : [idep_nir, idep_mesautil, idep_libpoly],
    ),
    suite : 'poly',
  )
  test(
    'poly-gs-outputs',
    executable(
      'poly-gs-outputs',
      '../tests/poly_gs_output_tests.c',
      include_directories : [inc_poly],
      link_with : [libpoly_nir],
      dependencies : [idep_nir, idep_mesautil, idep_libpoly],
    ),
    suite : 'poly',
  )
  test(
    'poly-tcs-tess-levels',
    executable(
      'poly-tcs-tess-levels',
      '../tests/poly_tcs_tess_level_tests.c',
      include_directories : [inc_poly],
      link_with : [libpoly_nir],
      dependencies : [idep_nir, idep_mesautil, idep_libpoly],
    ),
    suite : 'poly',
  )
  test(
    'poly-tcs-barriers',
    executable(
      'poly-tcs-barriers',
      '../tests/poly_tcs_barrier_tests.c',
      include_directories : [inc_poly],
      link_with : [libpoly_nir],
      dependencies : [idep_nir, idep_mesautil, idep_libpoly],
    ),
    suite : 'poly',
  )
  test(
    'poly-heap-math',
    executable(
      'poly-heap-math',
      '../tests/poly_heap_math_tests.c',
      include_directories : [inc_poly],
    ),
    suite : 'poly',
  )
endif

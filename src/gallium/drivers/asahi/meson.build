# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

files_asahi = files(
  'driinfo_asahi.h',
  'agx_batch.c',
  'agx_blit.c',
  'agx_disk_cache.c',
  'agx_fence.c',
  'agx_pipe.c',
  'agx_nir_lower_sysvals.c',
  'agx_nir_lower_bindings.c',
  'agx_nir_lower_point_size.c',
  'agx_native_descriptor.c',
  'agx_query.c',
  'agx_state.c',
  'agx_streamout.c',
  'agx_uniforms.c',
  'agx_xfb_cpu.c',
  'agx_swtnl.c',
)

if with_tests
  test(
    'asahi-native-descriptor',
    executable(
      'asahi-native-descriptor',
      files(
        'agx_native_descriptor.c',
        'tests/test-agx-native-descriptor.c',
      ),
      # The AGX3 PBE encoder is the shared byte-grounded one in
      # src/asahi/lib, the same object Honeykrisp and the D3D12 UMD drive.
      # Compile it in rather than linking libasahi_lib, which would pull the
      # whole device/BO layer into a pure descriptor test.
      objects : [libasahi_lib.extract_objects('agx_pbe_agx3_core.c')],
      include_directories : [inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['asahi'],
  )

  # Offline reproduction of the G17P blit path under the no-op DRM shim. It
  # links the whole driver, so it lives after libasahi below.
endif

libasahi = static_library(
  'asahi',
  [files_asahi],
  include_directories : [inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
  c_args : [c_msvc_compat_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx, idep_libagx_shaders_h],
)

driver_asahi = declare_dependency(
  compile_args : '-DGALLIUM_ASAHI',
  link_with : [libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib, libasahi_layout, libasahi_decode]
)

if with_tests and with_tools.contains('drm-shim')
  asahi_g17g_raster_state_test = executable(
    'asahi-g17g-raster-state',
    files('tests/test-agx-g17g-raster-state.c'),
    include_directories : [
      inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm,
    ],
    dependencies : [
      idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx,
      idep_libagx_shaders_h, idep_xmlconfig,
    ],
    link_with : [
      libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib,
      libasahi_layout, libasahi_decode, libgallium, libgalliumvl_stub,
    ],
    build_by_default : false,
  )
  test(
    'asahi-g17g-raster-state',
    prog_python,
    args : [files('tests/run-shim.py'), libasahi_noop_drm_shim, asahi_g17g_raster_state_test],
    env : ['ASAHI_SHIM_CHIP=g18g', 'ASAHI_GL_NOOP_SHIM=1',
           'MESA_SHADER_CACHE_DISABLE=true'],
    suite : ['asahi'],
  )

  test(
    'asahi-g17p-texture-crawl',
    executable(
      'asahi-g17p-texture-crawl',
      files('tests/test-agx-g17p-texture-crawl.c'),
      include_directories : [
        inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi,
        inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx,
        idep_libagx_shaders_h, idep_xmlconfig,
      ],
      link_with : [
        libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib,
        libasahi_layout, libasahi_decode, libgallium, libgalliumvl_stub,
      ],
    ),
    suite : ['asahi'],
  )
  test(
    'asahi-g17p-shadow-border',
    executable(
      'asahi-g17p-shadow-border',
      files('tests/test-agx-g17p-shadow-border.c'),
      include_directories : [
        inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi,
        inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx,
        idep_libagx_shaders_h, idep_xmlconfig,
      ],
      link_with : [
        libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib,
        libasahi_layout, libasahi_decode, libgallium, libgalliumvl_stub,
      ],
    ),
    suite : ['asahi'],
  )

  asahi_vs_query_test = executable(
    'asahi-vs-query',
    files('tests/test-agx-vs-query.c'),
    include_directories : [
      inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm,
    ],
    dependencies : [
      idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx,
      idep_libagx_shaders_h, idep_xmlconfig,
    ],
    link_with : [
      libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib,
      libasahi_layout, libasahi_decode, libgallium, libgalliumvl_stub,
    ],
    build_by_default : false,
  )

  foreach chip : ['g13g', 'g15c', 'g15s', 'g17p', 't6041']
    foreach mode : ['vs', 'query']
      test(
        'asahi-' + chip + '-gallium-' + mode,
        prog_python,
        args : [
          files('tests/run-shim.py'),
          libasahi_noop_drm_shim,
          asahi_vs_query_test,
          mode,
        ],
        env : [
          'ASAHI_SHIM_CHIP=' + chip,
          'ASAHI_MESA_DEBUG=perf',
          'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
        ],
        suite : ['asahi'],
      )
    endforeach
  endforeach

  asahi_g17p_blit_test = executable(
    'asahi-g17p-blit',
    files('tests/test-agx-g17p-blit.c'),
    include_directories : [
      inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm,
    ],
    dependencies : [
      idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx,
      idep_libagx_shaders_h, idep_xmlconfig,
    ],
    link_with : [
      libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib,
      libasahi_layout, libasahi_decode, libagx3b_disasm, libgallium,
      libgalliumvl_stub,
    ],
    build_by_default : false,
  )

  asahi_priority_test = executable(
    'asahi-priority',
    files('tests/test-agx-priority.c'),
    include_directories : [
      inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm,
    ],
    dependencies : [
      idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx,
      idep_libagx_shaders_h, idep_xmlconfig,
    ],
    link_with : [
      libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib,
      libasahi_layout, libasahi_decode, libgallium, libgalliumvl_stub,
    ],
  )
  foreach policy : ['allow', 'deny', 'once']
    test(
      'asahi-priority-' + policy,
      prog_python,
      args : [files('tests/run-shim.py'), libasahi_noop_drm_shim, asahi_priority_test],
      env : ['ASAHI_SHIM_CHIP=g13g'] + (policy == 'allow' ? [] : ['ASAHI_SHIM_HIGH_PRIORITY=' + policy]),
      suite : ['asahi'],
    )
  endforeach

  test(
    'asahi-priority-shared',
    prog_python,
    args : [files('tests/run-shim.py'), libasahi_noop_drm_shim, asahi_priority_test],
    env : ['ASAHI_SHIM_CHIP=g13g', 'ASAHI_MESA_DEBUG=1queue'],
    suite : ['asahi'],
  )

  if with_egl and with_glvnd
    asahi_egl_priority_test = executable(
      'asahi-egl-priority',
      files('tests/test-agx-egl-priority.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
    )
    foreach policy : ['allow', 'deny', 'once']
      test(
        'asahi-egl-priority-' + policy,
        prog_python,
        args : [files('tests/run-shim.py'), libasahi_noop_drm_shim, asahi_egl_priority_test],
        env : [
          'ASAHI_SHIM_CHIP=g13g',
          '__EGL_VENDOR_LIBRARY_FILENAMES=' + join_paths(meson.project_build_root(), 'src/egl/devenv_glvnd_egl.json'),
        ] + (policy == 'allow' ? [] : ['ASAHI_SHIM_HIGH_PRIORITY=' + policy]),
        suite : ['asahi'],
      )
    endforeach
  endif

  asahi_submit_test = executable(
    'asahi-submit',
    files('tests/test-agx-submit.c'),
    include_directories : [
      inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm,
    ],
    dependencies : [
      idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx,
      idep_libagx_shaders_h, idep_xmlconfig,
    ],
    link_with : [
      libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib,
      libasahi_layout, libasahi_decode, libgallium, libgalliumvl_stub,
    ],
  )
  asahi_submit_retirement_test = executable(
    'asahi-submit-retirement',
    files('tests/test-agx-submit-retirement.c'),
    include_directories : [
      inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm,
    ],
    dependencies : [
      idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx,
      idep_libagx_shaders_h, idep_xmlconfig,
    ],
    link_with : [
      libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib,
      libasahi_layout, libasahi_decode, libgallium, libgalliumvl_stub,
    ],
  )
  foreach scenario : [
    ['split-pending', '0'],
    ['split-completed', '1'],
    ['mixed-pending', '2'],
    ['mixed-completed', '3'],
    ['first-ioctl-failure', '4'],
    ['prepublication-no-fence', '5'],
    ['stale-output-generation', '6'],
    ['completed-error-fence', '7'],
    ['first-ioctl-partial', '8'],
    ['split-wait-error', '9'],
    ['mixed-wait-error', '10'],
    ['publication-probe-error', '11'],
    ['interrupted-wait', '12'],
  ]
    test(
      'asahi-g17p-submit-retirement-' + scenario[0],
      prog_python,
      args : [files('tests/run-shim.py'), libasahi_noop_drm_shim,
              asahi_submit_retirement_test, scenario[1]],
      env : ['ASAHI_SHIM_CHIP=g17p'],
      suite : ['asahi'],
    )
  endforeach

  foreach chip : ['g13g', 'g17p', 't6041']
    test(
      'asahi-' + chip + '-gallium-submit',
      prog_python,
      args : [files('tests/run-shim.py'), libasahi_noop_drm_shim, asahi_submit_test],
      env : ['ASAHI_SHIM_CHIP=' + chip],
      suite : ['asahi'],
    )
  endforeach

  # The shim has to be preloaded before libc's open(), so the test runs the
  # binary through a wrapper that sets LD_PRELOAD and the chip selector.
  foreach chip : ['g13g', 'g17p', 't6041']
    test(
      'asahi-' + chip + '-gallium-xfb-program',
      prog_python,
      args : [
        files('tests/run-shim.py'),
        libasahi_noop_drm_shim,
        asahi_g17p_blit_test,
      ],
      env : [
        'ASAHI_SHIM_CHIP=' + chip,
        'AGX_TEST_XFB_COPY_PROGRAM=1',
      ],
      suite : ['asahi'],
    )

    # Linked-program vertex output shapes: clip distances and multiple user
    # varyings must reach the AGX3 vertex writeout with inlined UVS indices.
    test(
      'asahi-' + chip + '-gallium-vertex-output-shapes',
      prog_python,
      args : [
        files('tests/run-shim.py'),
        libasahi_noop_drm_shim,
        asahi_g17p_blit_test,
      ],
      env : [
        'ASAHI_SHIM_CHIP=' + chip,
        'AGX_TEST_VERTEX_OUTPUT_SHAPES=1',
      ],
      suite : ['asahi'],
    )
  endforeach

  foreach chip : ['g17p', 't6041']
    test(
      'asahi-' + chip + '-gallium-blit',
      prog_python,
      args : [
        files('tests/run-shim.py'),
        libasahi_noop_drm_shim,
        asahi_g17p_blit_test,
      ],
      env : [
        'ASAHI_SHIM_CHIP=' + chip,
        'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
        'ASAHI_SHIM_G17P_EXPECT_COMPUTE=1',
        'ASAHI_SHIM_G17P_EXPECT_STRONG_COMPUTE_BARRIER=1',
      ],
      suite : ['asahi'],
    )
  endforeach

  test(
    'asahi-g17p-gallium-blit-encode',
    prog_python,
    args : [
      files('tests/run-shim.py'),
      libasahi_noop_drm_shim,
      asahi_g17p_blit_test,
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'AGX_G17P_BLIT_REQUIRE_ENCODE=1',
    ],
    suite : ['asahi'],
  )
endif

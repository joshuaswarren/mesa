# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

libasahi_lib_files = files(
  'agx3_cdm_decode_core.c',
  'agx3_cdm_decode_core.h',
  'agx3_cdm_direct_launch_semantic_core.c',
  'agx3_cdm_direct_launch_semantic_core.h',
  'agx3_compute_program_core.c',
  'agx3_compute_program_core.h',
  'agx3_g17p_grid_helper_core.h',
  'agx3_code_container_core.c',
  'agx3_code_container_core.h',
  'agx_ioto_core.c',
  'agx_ioto_core.h',
  'agx_ioto_load_core.c',
  'agx_ioto_load_core.h',
  'agx_ioto_preshader_core.c',
  'agx_ioto_preshader_core.h',
  'agx_ioto_profile_core.c',
  'agx_ioto_profile_core.h',
  'agx_ioto_program_core.c',
  'agx_ioto_program_core.h',
  'agx_ioto_state_core.c',
  'agx_ioto_state_core.h',
  'agx_g17p_draw_state_upload.h',
  'agx_rt_result.h',
  'agx_scs_layout.c',
  'agx_scs_layout.h',
  'agx_bo.c',
  'agx_border.c',
  'agx_chip.c',
  'agx_device.c',
  'agx_bytes.h',
  'agx_linker.c',
  'agx_bg_eot.c',
  'agx_bg_eot_key.c',
  'agx_tilebuffer.c',
  'agx_nir_lower_msaa.c',
  'agx_nir_lower_sample_intrinsics.c',
  'agx_nir_lower_tilebuffer.c',
  'agx_nir_lower_uvs.c',
  'agx_nir_lower_vbo.c',
  'agx_nir_prolog_epilog.c',
  'agx_nir_legacy_epilog.c',
  'agx_ppp.h',
  'agx_ppp_hal200_graphics_core.c',
  'agx_ppp_hal200_graphics_core.h',
  'agx_ppp_hal200_raster_core.c',
  'agx_ppp_hal200_raster_core.h',
  'agx_ppp_hal300_depth_bias_scissor_core.c',
  'agx_ppp_hal300_depth_bias_scissor_core.h',
  'agx_ppp_hal300_graphics_core.c',
  'agx_ppp_hal300_graphics_core.h',
  'agx_ppp_hal300_depth_stencil_core.c',
  'agx_ppp_hal300_depth_stencil_core.h',
  'agx_ppp_decode_core.c',
  'agx_ppp_decode_core.h',
  'agx_ppp_viewport_payload_core.c',
  'agx_ppp_viewport_payload_core.h',
  'agx_ppp_w_clamp_cull2_core.c',
  'agx_ppp_w_clamp_cull2_core.h',
  'agx_pbe_agx3_core.c',
  'agx_pbe_agx3_core.h',
  'agx_scs_layout.c',
  'agx_scs_layout.h',
  'agx_scratch.c',
  'agx_vdm_barrier_core.c',
  'agx_vdm_barrier_core.h',
  'agx_vdm_hal200_bind_tail_core.c',
  'agx_vdm_hal200_bind_tail_core.h',
  'agx_vdm_hal200_direct_draw_core.c',
  'agx_vdm_hal200_direct_draw_core.h',
  'agx_vdm_decode_core.c',
  'agx_vdm_mesh_core.c',
  'agx_vdm_mesh_core.h',
  'agx_vdm_decode_core.h',
  'agx_vdm_index_list_core.c',
  'agx_vdm_index_list_core.h',
  'agx_vdm_ppp_state_core.c',
  'agx_vdm_ppp_state_core.h',
  'agx_vdm_stream_link_core.c',
  'agx_vdm_stream_link_core.h',
  'agx_vdm_stream_terminate_core.c',
  'agx_vdm_stream_terminate_core.h',
  'agx_vdm_tessellate_core.c',
  'agx_vdm_tessellate_core.h',
  'agx_vdm_state_core.c',
  'agx_vdm_state_core.h',
  'agx_vdm_state_update_core.c',
  'agx_vdm_state_update_core.h',
  'agx_va.c',
  'agx_va_runtime_hole.c',
  'agx_va_runtime_hole.h',
  'pool.c',
)

asahi_lib_has_virtio = with_vdrm
if asahi_lib_has_virtio
  libasahi_lib_files += files('agx_device_virtio.c')
endif

libasahi_decode_files = files(
  'agx3_cdm_decode_core.c',
  'agx3_cdm_direct_launch_semantic_core.c',
  'agx_ioto_core.c',
  'agx_ppp_decode_core.c',
  'agx_ppp_hal200_graphics_core.c',
  'agx_ppp_hal300_depth_bias_scissor_core.c',
  'agx_ppp_hal300_graphics_core.c',
  'agx_ppp_hal300_graphics_core.h',
  'agx_ppp_hal300_depth_stencil_core.c',
  'agx_ppp_viewport_payload_core.c',
  'agx_ppp_w_clamp_cull2_core.c',
  'agx_vdm_barrier_core.c',
  'agx_vdm_hal200_bind_tail_core.c',
  'agx_vdm_hal200_direct_draw_core.c',
  'agx_vdm_decode_core.c',
  'agx_vdm_mesh_core.c',
  'agx_vdm_mesh_core.h',
  'agx_vdm_index_list_core.c',
  'agx_vdm_ppp_state_core.c',
  'agx_vdm_state_core.c',
  'agx_vdm_state_update_core.c',
  'agx_vdm_stream_link_core.c',
  'agx_vdm_stream_terminate_core.c',
  'agx_vdm_tessellate_core.c',
  'decode.c',
)

libasahi_decode = static_library(
  'asahi_decode',
  [libasahi_decode_files, agx_pack],
  dependencies : [dep_valgrind, idep_mesautil],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  link_with : [libagx2_disasm, libagx3d_disasm],
  build_by_default : false,
)

libasahi_decode_shared = shared_library(
  'asahi_decode',
  [libasahi_decode_files, agx_pack],
  dependencies : [dep_valgrind, idep_mesautil],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  build_by_default : with_tools.contains('asahi'),
  link_with : [libagx2_disasm, libagx3d_disasm],
)

if asahi_lib_has_virtio
  libasahi_lib_includes = [inc_asahi, inc_virtio_gpu, inc_virtio_vdrm]
  libasahi_lib_links = [libasahi_decode, libvdrm, libpoly_nir]
  libasahi_lib_deps = [
    dep_libdrm, dep_valgrind, idep_nir, idep_mesautil, idep_libagx,
    idep_libpoly,
  ]
else
  libasahi_lib_includes = [inc_asahi]
  libasahi_lib_links = [libasahi_decode, libpoly_nir]
  libasahi_lib_deps = [
    dep_valgrind, idep_nir, idep_mesautil, idep_libagx, idep_libpoly,
  ]
endif

libagx_shaders = custom_target(
  input : libagx_spv,
  output : ['libagx_shaders.h', 'libagx_shaders.c'],
  command : [prog_asahi_clc, libagx_spv, '@OUTPUT0@', '@OUTPUT1@'],
)

idep_libagx_shaders_h = declare_dependency(
  sources : [libagx_shaders],
  include_directories : include_directories('.'),
)

libasahi_lib = static_library(
  'asahi_lib',
  [libasahi_lib_files, libagx_shaders, agx_pack, sha1_h],
  include_directories : libasahi_lib_includes,
  c_args : [
    no_override_init_args,
    '-Wno-c2x-extensions',
  ],
  gnu_symbol_visibility : 'hidden',
  link_with: libasahi_lib_links,
  dependencies: libasahi_lib_deps,
  build_by_default : false,
)

if with_tests
  foreach rt_test : ['rt-bvh', 'scs-layout']
    test('asahi-' + rt_test,
      executable('asahi-' + rt_test,
        ['tests/test-' + rt_test + '.c', 'agx_scs_layout.c'],
        include_directories : include_directories('.'),
        c_args : ['-UNDEBUG'],
      ),
      suite : ['asahi'],
    )
  endforeach
  test(
    'asahi-chip-admission',
    executable(
      'asahi-chip-admission',
      files('tests/test-chip-admission.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'asahi-i386-layout',
    executable(
      'asahi-i386-layout',
      files('tests/test-i386-layout.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil, idep_nir],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'asahi-g18g-reencode-cache',
    executable(
      'asahi-g18g-reencode-cache',
      files('tests/test-g18g-reencode-cache.c'),
      include_directories : [inc_asahi, inc_include, inc_src],
      dependencies : [idep_agx_pack, idep_mesautil, idep_nir],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'asahi-usc-base-page',
    executable(
      'asahi-usc-base-page',
      files('tests/test-usc-base-page.c'),
      include_directories : [inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil, idep_nir],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'asahi-agx-chip',
    executable(
      'asahi-agx-chip',
      files(
        'agx_chip.c',
        'tests/test-chip.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-g17g-tile-format-gate',
    executable(
      'asahi-agx-g17g-tile-format-gate',
      files('tests/test-agx-g17g-tile-format-gate.c'),
      include_directories : [
        inc_include, inc_src, inc_asahi, libasahi_lib_includes,
        include_directories('../libagx'),
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_libagx_shaders_h, idep_agx_pack,
      ],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-linker-container',
    executable(
      'asahi-agx-linker-container',
      files(
        'agx3_code_container_core.c',
        'tests/test-agx-linker-container.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-g17p-srgb-epilog',
    executable(
      'asahi-g17p-srgb-epilog',
      files('tests/test-agx-g17p-srgb-epilog.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-g17p-mrt-epilog',
    executable(
      'asahi-g17p-mrt-epilog',
      files('tests/test-agx-g17p-mrt-epilog.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-g17p-mrt-quad',
    executable(
      'asahi-g17p-mrt-quad',
      files('tests/test-agx-g17p-mrt-quad.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-g17p-mrt-mixed',
    executable(
      'asahi-g17p-mrt-mixed',
      files('tests/test-agx-g17p-mrt-mixed.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-va-runtime-hole',
    executable(
      'asahi-agx-va-runtime-hole',
      files(
        'agx_va_runtime_hole.c',
        'tests/test-agx-va-runtime-hole.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_mesautil],
      c_args : ['-UNDEBUG'],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-m5-va',
    executable(
      'asahi-agx-m5-va',
      files('tests/test-agx-m5-va.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      c_args : ['-UNDEBUG'],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-ioto-load-core',
    executable(
      'asahi-agx-ioto-load-core',
      files(
        'agx_ioto_core.c',
        'agx_ioto_load_core.c',
        'tests/test-agx-ioto-load-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-ioto-preshader-core',
    executable(
      'asahi-agx-ioto-preshader-core',
      files(
        'agx_ioto_core.c',
        'agx_ioto_preshader_core.c',
        'tests/test-agx-ioto-preshader-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-ioto-program-core',
    executable(
      'asahi-agx-ioto-program-core',
      files(
        'agx_ioto_core.c',
        'agx_ioto_preshader_core.c',
        'agx_ioto_profile_core.c',
        'agx_ioto_state_core.c',
        'agx_ioto_load_core.c',
        'agx_ioto_program_core.c',
        'tests/test-agx-ioto-program-core.c',
      ),
      dependencies : [idep_mesautil],
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-decode-core',
    executable(
      'asahi-vdm-decode-core',
      files(
        'agx_vdm_barrier_core.c',
        'agx_vdm_decode_core.c',
        'agx_vdm_index_list_core.c',
        'agx_vdm_mesh_core.c',
        'agx_vdm_ppp_state_core.c',
        'agx_vdm_state_core.c',
        'agx_vdm_state_update_core.c',
        'agx_vdm_stream_link_core.c',
        'agx_vdm_stream_terminate_core.c',
        'agx_vdm_tessellate_core.c',
        'tests/test-vdm-decode-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-g18g-capture-vectors',
    executable(
      'asahi-g18g-capture-vectors',
      files(
        'agx3_cdm_decode_core.c',
        'agx_pbe_agx3_core.c',
        'agx_ppp_decode_core.c',
        'agx_ppp_hal300_depth_bias_scissor_core.c',
        'agx_ppp_hal300_depth_stencil_core.c',
        'agx_ppp_viewport_payload_core.c',
        'agx_ppp_w_clamp_cull2_core.c',
        'agx_vdm_barrier_core.c',
        'agx_vdm_decode_core.c',
        'agx_vdm_index_list_core.c',
        'agx_vdm_mesh_core.c',
        'agx_vdm_ppp_state_core.c',
        'agx_vdm_state_core.c',
        'agx_vdm_state_update_core.c',
        'agx_vdm_stream_link_core.c',
        'agx_vdm_stream_terminate_core.c',
        'agx_vdm_tessellate_core.c',
        'tests/test-g18g-capture-vectors.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-ppp-hal300-g18g',
    executable(
      'asahi-ppp-hal300-g18g',
      files(
        'agx3_cdm_decode_core.c',
        'agx_pbe_agx3_core.c',
        'agx_ppp_decode_core.c',
        'agx_ppp_hal300_depth_bias_scissor_core.c',
        'agx_ppp_hal300_depth_stencil_core.c',
        'agx_ppp_hal300_graphics_core.c',
        'agx_ppp_viewport_payload_core.c',
        'agx_ppp_w_clamp_cull2_core.c',
        'agx_vdm_barrier_core.c',
        'agx_vdm_decode_core.c',
        'agx_vdm_index_list_core.c',
        'agx_vdm_mesh_core.c',
        'agx_vdm_ppp_state_core.c',
        'agx_vdm_state_core.c',
        'agx_vdm_state_update_core.c',
        'agx_vdm_stream_link_core.c',
        'agx_vdm_stream_terminate_core.c',
        'agx_vdm_tessellate_core.c',
        'tests/test-ppp-hal300-g18g.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx3-compute-program-core',
    executable(
      'asahi-agx3-compute-program-core',
      files(
        'agx3_cdm_direct_launch_semantic_core.c',
        'agx3_compute_program_core.c',
        'agx_ioto_core.c',
        'agx_ioto_load_core.c',
        'agx_ioto_preshader_core.c',
        'agx_ioto_profile_core.c',
        'agx_ioto_program_core.c',
        'agx_ioto_state_core.c',
        'tests/test-agx3-compute-program-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-index-list-core',
    executable(
      'asahi-vdm-index-list-core',
      files(
        'agx_vdm_index_list_core.c',
        'tests/test-vdm-index-list-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-pbe-agx3-core',
    executable(
      'asahi-pbe-agx3-core',
      files(
        'agx_pbe_agx3_core.c',
        'tests/test-pbe-agx3-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-ppp-hal300-graphics-core',
    executable(
      'asahi-ppp-hal300-graphics-core',
      files(
        'agx_ppp_hal300_graphics_core.c',
        'agx_vdm_state_core.c',
        'tests/test-ppp-hal300-graphics-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-state-update-core',
    executable(
      'asahi-vdm-state-update-core',
      files(
        'agx_vdm_state_core.c',
        'agx_vdm_state_update_core.c',
        'tests/test-vdm-state-update-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'libasahi_tests',
    executable(
      'libasahi_tests',
      files(
        'tests/test-chip.cpp',
        'tests/test-packing.cpp',
        'tests/test-tilebuffer.cpp',
        'tests/test-g17p-msaa-epilog.cpp',
        'tests/test-m5-depth-export.cpp',
        'tests/test-primitive-varyings.cpp',
        '../vulkan/hk_nir_widen_16bit_varyings.c',
        'tests/test-dual-source-epilog.cpp',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [
        idep_gtest, idep_agx_pack, idep_mesautil, idep_nir,
      ],
      include_directories: [inc_asahi],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout,
                   libagx3c_disasm],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )
endif

if with_tests
  test('vbo_robustness', executable('vbo_robustness_test',
    files('tests/test-vbo-robustness.c'),
    include_directories : [inc_asahi, inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil, idep_agx_pack, idep_libagx_shaders_h],
    link_with : [libasahi_lib, libasahi_compiler, libasahi_layout]),
    suite : ['asahi'])
endif

if dep_iokit.found()
  libasahi_wrap = shared_library(
    'wrap',
    'wrap.c',
    include_directories : [inc_src, inc_asahi],
    dependencies : [idep_mesautil, dep_iokit],
    c_args : [no_override_init_args, '-Wno-missing-prototypes'],
    gnu_symbol_visibility : 'hidden',
    build_by_default : with_tools.contains('asahi'),
    link_with: libasahi_decode,
  )
endif

if with_tests
  test('agx3_m5_programs', executable('agx3_m5_programs',
    '../compiler/test/test-agx3-m5-program.c',
    include_directories: [inc_include, inc_src, inc_asahi],
    dependencies: [idep_mesautil, idep_nir],
    link_with: [libasahi_compiler, libasahi_lib]), suite: ['asahi'])
endif

if with_tests
  test('agx_rt_envelope', executable('agx_rt_envelope',
    'tests/test-rt-envelope.c',
    c_args: ['-ffp-contract=off'],
    include_directories: include_directories('.')), suite: ['asahi'])
  test('agx_rt_blas_gen3', executable('agx_rt_blas_gen3',
    'tests/test-rt-blas-gen3.c',
    include_directories: include_directories('.')), suite: ['asahi'])
  test('agx_rt_bvh', executable('agx_rt_bvh',
    'tests/test-rt-bvh.c',
    include_directories: include_directories('.')), suite: ['asahi'])
  test('agx_m5_layout', executable('agx_m5_layout',
    ['tests/test-agx-m5-layout.c', 'agx3_code_container_core.c'],
    include_directories: include_directories('.')), suite: ['asahi'])
endif

if with_tests
  test('agx3_m5_instructions', executable('agx3_m5_instructions',
    '../compiler/test/test-agx3-m5-instructions.c',
    include_directories: [inc_include, inc_src, inc_asahi],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libasahi_compiler, libasahi_lib]), suite: ['asahi'])
endif

if with_tests
  test('agx_scs_nir', executable('agx_scs_nir',
    '../compiler/test/test-scs-nir.c',
    c_args: ['-UNDEBUG'],
    include_directories: [inc_include, inc_src, inc_asahi],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libasahi_compiler, libasahi_lib]), suite: ['asahi'])
endif

if with_tests
  test('agx3-m5-mesh-program', executable('agx3-m5-mesh-program',
    '../compiler/test/test-native-mesh-program.c',
    include_directories: [inc_include, inc_src, inc_asahi],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libasahi_compiler, libasahi_lib]), suite: ['asahi'])
endif

if with_tests
  test('agx_scs_query', executable('agx_scs_query',
    '../compiler/test/test-scs-query.c',
    c_args: ['-UNDEBUG'],
    include_directories: [inc_include, inc_src, inc_asahi],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libasahi_compiler, libasahi_lib]), suite: ['asahi'])
endif

if with_tests
  test('agx_scs_layout', executable('agx_scs_layout',
    ['tests/test-scs-layout.c', 'agx_scs_layout.c'],
    c_args: ['-UNDEBUG'],
    include_directories: include_directories('.')), suite: ['asahi'])
endif

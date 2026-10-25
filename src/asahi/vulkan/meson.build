# Copyright © 2022 Collabora Ltd. and Red Hat Inc.
# SPDX-License-Identifier: MIT
hk_files = files(
  'hk_barrier.c',
  'hk_barrier.h',
  'hk_acceleration_structure.c',
  'hk_m5_acceleration_structure.c',
  'hk_host_as.c',
  'hk_acceleration_structure.h',
  'hk_buffer.c',
  'hk_buffer.h',
  'hk_buffer_view.c',
  'hk_bvh_depth_core.c',
  'hk_bvh_depth_core.h',
  'hk_buffer_view.h',
  'hk_bvh.h',
  'hk_cmd_buffer.c',
  'hk_cmd_upload.c',
  'hk_cmd_buffer.h',
  'hk_conditional_render_core.h',
  'hk_cmd_lifetime.c',
  'hk_cs_root_size.h',
  'hk_cmd_clear.c',
  'hk_cmd_dispatch.c',
  'hk_gen1_query_state.c',
  'hk_cmd_draw.c',
  'hk_cmd_meta.c',
  'hk_cmd_trace_rays.c',
  'hk_cmd_pool.c',
  'hk_cmd_pool.h',
  'hk_descriptor_set.h',
  'hk_descriptor_set.c',
  'hk_descriptor_set_layout.c',
  'hk_descriptor_set_layout_hash.c',
  'hk_descriptor_set_layout.h',
  'hk_descriptor_table.c',
  'hk_descriptor_table.h',
  'hk_device.c',
  'hk_device.h',
  'hk_device_memory.c',
  'hk_device_memory.h',
  'hk_memory_budget.c',
  'hk_memory_budget.h',
  'hk_event.c',
  'hk_event.h',
  'hk_format.c',
  'hk_g17p_meta_copy_core.c',
  'hk_ioto_arena.c',
  'hk_ioto_arena_core.c',
  'hk_ioto_arena_core.h',
  'hk_g17p_meta_copy_core.h',
  'hk_agx3_ioto_program_core.c',
  'hk_agx3_ioto_program_core.h',
  'hk_agx3_vdm_decode.c',
  'hk_agx3_vdm_decode.h',
  'hk_image.c',
  'hk_image.h',
  'hk_image_layout_core.c',
  'hk_image_layout_core.h',
  'hk_image_view.c',
  'hk_image_view.h',
  'hk_instance.c',
  'hk_instance.h',
  'hk_nir_forward_image_stores.c',
  'hk_nir_forward_image_stores.h',
  'hk_nir_lower_descriptors.c',
  'hk_nir_lower_gs.c',
  'hk_nir_lower_tes_rq.c',
  'hk_nir_lower_gs.h',
  'hk_nir_root_size.c',
  'hk_nir_lower_ray_queries.c',
  'hk_nir_lower_robust_image.c',
  'hk_nir_lower_robust_image.h',
  'hk_nir_ray_tracing.c',
  'hk_ray_tracing_core.c',
  'hk_rt_inlining.c',
  'hk_ray_tracing_core.h',
  'hk_ray_tracing_pipeline.c',
  'hk_ray_tracing_pipeline.h',
  'hk_nir_ray_tracing.h',
  'hk_nir_widen_16bit_varyings.c',
  'hk_nir_widen_16bit_varyings.h',
  'hk_native_descriptor.h',
  'hk_native_vertex_template.h',
  'hk_native_descriptor_pack.c',
  'hk_native_descriptor_pack.h',
  'hk_native_sampler.c',
  'hk_native_sampler.h',
  'hk_physical_device.c',
  'hk_physical_device.h',
  'hk_pipeline_statistics_core.c',
  'hk_pipeline_statistics_core.h',
  'hk_private.h',
  'hk_query_oq.c',
  'hk_query_pool.c',
  'hk_query_reset.c',
  'hk_query_pool.h',
  'hk_drm_render_core.c',
  'hk_drm_render_core.h',
  'hk_execution_allowance.h',
  'hk_queue.c',
  'hk_queue.h',
  'hk_frame_anatomy.c',
  'hk_frame_anatomy.h',
  'hk_sampler.c',
  'hk_sampler.h',
  'hk_shader.c',
  'hk_nir_lower_image_view_min_lod.c',
  'hk_shader.h',
  'hk_sync.c',
  'hk_sync.h',
  'hk_wsi.c',
  'hk_wsi.h'
)

hk_drirc = custom_target(
  'hk_drirc',
  input  : [files('hk_drirc_gen.py'), drirc_gen, '00-hk-defaults.conf'],
  output : ['hk_drirc.c', 'hk_drirc.h'],
  command : [
    prog_python, '@INPUT0@',
    '--import-path', join_paths(dir_source_root, 'src/util'),
    '--drirc-src', '@OUTPUT0@',
    '--drirc-hdr', '@OUTPUT1@',
    '--validate', '@INPUT2@',
  ],
)

if with_tests
  test('hk_execution_allowance',
    executable('hk_execution_allowance_test',
      files('tests/test-hk-execution-allowance.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_libagx,
        idep_libagx_shaders_h, idep_agx_pack,
        idep_vulkan_runtime_headers, idep_vulkan_util_headers,
      ]),
    suite : ['asahi'])

  test('hk_memory_budget', executable('hk_memory_budget_test',
    files('tests/test-hk-memory-budget.c', 'hk_memory_budget.c'),
    include_directories : include_directories('.')),
    suite : ['asahi'])
  test(
    'hk_barrier',
    executable(
      'hk_barrier_test',
      files('tests/test-hk-barrier.c', 'hk_barrier.c'),
      include_directories : [include_directories('.'), inc_include],
    ),
    suite : ['asahi'],
  )

  test(
    'hk_gen1_query_state',
    executable(
      'hk_gen1_query_state_test',
      files('tests/test-hk-gen1-query-state.c', 'hk_gen1_query_state.c',
            '../lib/agx_scs_layout.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_libagx,
        idep_libagx_shaders_h, idep_agx_pack,
        idep_vulkan_runtime_headers, idep_vulkan_util_headers,
      ],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test('hk_root_size', executable('hk_root_size_test',
    files('tests/test-hk-root-size.c', 'hk_nir_root_size.c'),
    include_directories : [include_directories('.'), inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil]), suite : ['asahi'])
  # Unused recording functions require section garbage collection so the
  # test can link the production resolver without the rest of the driver.
  if ld_args_gc_sections.length() > 0
    test(
      'hk_query_reset',
      executable(
        'hk_query_reset_test',
        files('tests/test-hk-query-reset.c', 'hk_query_reset.c'),
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )

    # Exercise the real resolver; unused command-recording entry points in
    # hk_sync.c need no device implementation in this CPU-only executable.
    test(
      'hk_sync_prior',
      executable(
        'hk_sync_prior_test',
        files('tests/test-hk-sync-prior.c', 'hk_sync.c'),
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )

    # The deferred occlusion-availability resolver against mock control
    # streams, with the real hk_sync stream bookkeeping it drops streams from.
    test(
      'hk_query_oq',
      executable(
        'hk_query_oq_test',
        files('tests/test-hk-query-oq.c', 'hk_query_oq.c', 'hk_sync.c'),
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )

  endif

  test(
    'hk_root_fragment',
    executable(
      'hk_root_fragment_test',
      files('tests/test-hk-root-fragment.c'),
      include_directories : [
        include_directories('.'),
        inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
        inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_libagx,
        idep_libagx_shaders_h, idep_agx_pack,
        idep_vulkan_runtime_headers, idep_vulkan_util_headers,
      ],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_render_resource_span',
    executable(
      'hk_render_resource_span_test',
      files('tests/test-hk-render-resource-span.c', 'hk_drm_render_core.c'),
      include_directories : [include_directories('.'), inc_include],
      c_args : [c_msvc_compat_args],
    ),
    suite : ['asahi'],
  )

  test(
    'hk_query_program',
    executable(
      'hk_query_program_test',
      files('tests/test-hk-query-program.c'),
      include_directories : [inc_include, inc_src],
      dependencies : [
        idep_nir_headers, idep_mesautil, idep_libagx, idep_libagx_shaders_h,
      ],
      link_with : [libasahi_lib],
    ),
    suite : ['asahi'],
  )

  test(
    'hk_query_copy',
    executable(
      'hk_query_copy_test',
      files('tests/test-hk-query-copy.c'),
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'hk_g17p_compression',
    executable(
      'hk_g17p_compression_test',
      files('tests/test-hk-g17p-compression.c'),
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'hk_g16s_compression',
    executable(
      'hk_g16s_compression_test',
      files('tests/test-hk-g16s-compression.c'),
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'hk_g17p_zls',
    executable(
      'hk_g17p_zls_test',
      files('tests/test-hk-g17p-zls.c', 'hk_drm_render_core.c'),
      include_directories : [include_directories('.'), inc_include, inc_src,
                             inc_asahi],
      dependencies : [idep_mesautil, idep_nir, idep_libagx,
                      idep_libagx_shaders_h, idep_agx_pack],
      link_with : [libasahi_layout],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_g17p_metadata_grid',
    executable(
      'hk_g17p_metadata_grid_test',
      files('tests/test-hk-g17p-metadata-grid.c',
            'hk_g17p_meta_copy_core.c'),
      include_directories : [include_directories('.'), inc_include, inc_src,
                             inc_asahi],
      dependencies : [idep_mesautil, idep_vulkan_util,
                      idep_vulkan_runtime_headers],
      link_with : [libasahi_layout],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_image_layout_core',
    executable(
      'hk_image_layout_core_test',
      files('tests/test-hk-image-layout-core.c',
            'hk_image_layout_core.c'),
      include_directories : [include_directories('.')],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_native_descriptor',
    executable(
      'hk_native_descriptor_test',
      files('tests/test-hk-native-descriptor.c',
            'hk_native_descriptor_pack.c',
            '../lib/agx_pbe_agx3_core.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_native_sampler',
    executable(
      'hk_native_sampler_test',
      files('tests/test-hk-native-sampler.c', 'hk_native_sampler.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'hk_gs_ray_query',
    executable(
      'hk_gs_ray_query_test',
      files('tests/test-hk-gs-ray-query.c', 'hk_nir_lower_gs.c'),
      include_directories : [include_directories('.'), inc_asahi, inc_poly],
      dependencies : [idep_nir, idep_mesautil, idep_libpoly],
      link_with : [libpoly_nir, libasahi_compiler],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_sampler_bias',
    executable(
      'hk_sampler_bias_test',
      files('tests/test-hk-sampler-bias.c'),
      include_directories : [include_directories('.'), inc_include, inc_src, inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

endif

hk_entrypoints = custom_target(
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['hk_entrypoints.h', 'hk_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'hk',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

subdir('bvh')

hk_deps = [
  dep_libdrm,
  idep_nir,
  idep_vulkan_runtime,
  idep_vulkan_util,
  idep_vulkan_wsi,
  idep_vulkan_wsi_headers,
  idep_agx_pack,
  idep_libagx,
  idep_libagx_shaders_h,
]

libhk = static_library(
  'hk',
  [
    hk_entrypoints,
    hk_drirc,
    hk_files,
    hk_bvh_spv,
    sha1_h,
  ],
  include_directories : [
    inc_gallium,
    inc_gallium_aux,
    inc_include,
    inc_src,
    inc_asahi,
    inc_virtio_gpu,
    inc_virtio_vdrm,
    include_directories('.'),
  ],
  link_with : [libasahi_lib, libasahi_layout, libasahi_compiler],
  c_args : ['-Wno-c2x-extensions'],
  dependencies : [hk_deps],
  gnu_symbol_visibility : 'hidden',
)

# The tests below replace driver functions with -Wl,--wrap. Link-time
# optimisation resolves calls inside the driver libraries before the linker
# applies the wrap, so link them from the regular objects of fat LTO builds.
hk_wrap_test_link_args = cc.get_supported_link_arguments('-fno-lto')

if with_tests
  test('hk_rt_inlining', executable('hk_rt_inlining_test',
    files('tests/test-hk-rt-inlining.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_memory_oom', executable('hk_memory_oom_test',
    files('tests/test-hk-memory-oom.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args,
      '-Wl,--wrap=agx_bo_bind', '-Wl,--wrap=agx_bo_cache_evict_all',
      '-Wl,--wrap=agx_bo_export',
      '-Wl,--wrap=hk_get_cgroup_memory_available', '-Wl,--wrap=__vk_errorf',
      '-Wl,--wrap=os_get_available_system_memory',
      '-Wl,--wrap=os_get_total_physical_memory']),
    suite : ['asahi'])
  test('hk_va', executable('hk_va_test',
    files('tests/test-hk-va.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_tes_query', executable('hk_tes_query_test', files('tests/test-hk-tes-query.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test(
    'hk_rt_pipeline_executable',
    executable(
      'hk_rt_pipeline_executable_test',
      files('tests/test-hk-rt-pipeline-executable.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      link_with : [libhk],
      dependencies : [hk_deps],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
endif

libvulkan_asahi = shared_library(
  'vulkan_asahi',
  link_whole : [libhk],
  link_args: [ld_args_build_id],
  gnu_symbol_visibility : 'hidden',
  install : true,
)

icd_file_name = libname_prefix + 'vulkan_asahi.' + libname_suffix

asahi_icd = custom_target(
  input : [vk_icd_gen, vk_api_xml],
  output : 'asahi_icd.' + vulkan_manifest_suffix,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.4', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', vulkan_icd_lib_path,
    '--icd-filename', icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
  install_dir : with_vulkan_icd_dir,
  install_tag : 'runtime',
  install : true,
)

_dev_icdname = 'asahi_devenv_icd.@0@.json'.format(host_machine.cpu())
_dev_icd = custom_target(
  input : [vk_icd_gen, vk_api_xml],
  output : _dev_icdname,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.4', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', meson.current_build_dir(),
    '--icd-filename', icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)

devenv.append('VK_DRIVER_FILES', _dev_icd.full_path())

if with_tests and ld_args_gc_sections.length() > 0
  test(
    'hk_sparse_formats', executable('hk_sparse_formats_test',
      files('tests/test-hk-sparse-formats.c'),
      include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
      dependencies : [hk_deps], link_with : [libhk, libasahi_layout],
      link_args : [ld_args_gc_sections],
      gnu_symbol_visibility : 'hidden'), suite : ['asahi'])
  test(
    'hk_host_as', executable('hk_host_as_test',
      files('tests/test-hk-host-as.c', 'hk_host_as.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
      dependencies : [hk_deps],
      link_args : [ld_args_gc_sections, hk_wrap_test_link_args,
        '-Wl,--wrap=util_flush_range']), suite : ['asahi'])
  test('hk_agx3_as_format', executable('hk_agx3_as_format_test',
    files('tests/test-hk-agx3-as-format.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args]),
    suite : ['asahi'])
  test('hk_as_registry', executable('hk_as_registry_test',
    files('tests/test-hk-as-registry.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args, '-Wl,--wrap=agx_bo_create',
      '-Wl,--wrap=agx_bo_bind', '-Wl,--wrap=_mesa_hash_table_insert',
      '-Wl,--wrap=agx_va_alloc', '-Wl,--wrap=agx_va_free']),
    timeout : 60, suite : ['asahi'])
  test('hk_host_as_lifetime', executable('hk_host_as_lifetime_test',
    files('tests/test-hk-host-as-lifetime.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args, '-Wl,--wrap=hk_meta_begin',
      '-Wl,--wrap=hk_meta_end']), suite : ['asahi'])
  test('hk_rt_on_demand', executable('hk_rt_on_demand_test',
    files('tests/test-hk-rt-on-demand.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args,
      '-Wl,--wrap=agx_va_alloc', '-Wl,--wrap=agx_va_free',
      '-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_unreference',
      '-Wl,--wrap=agx_bo_bind', '-Wl,--wrap=__vk_errorf']), suite : ['asahi'])
  test(
    'hk_rt_cache_control',
    executable(
      'hk_rt_cache_control_test',
      files('tests/test-hk-rt-cache-control.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [hk_deps],
      link_with : [libhk],
      link_args : [
        ld_args_gc_sections,
        hk_wrap_test_link_args,
        '-Wl,--wrap=vk_pipeline_shader_stage_to_nir',
        '-Wl,--wrap=hk_compile_shader',
        '-Wl,--wrap=hk_api_shader_destroy',
        '-Wl,--wrap=ralloc_free',
      ],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'hk_rt_pipeline_binary',
    executable(
      'hk_rt_pipeline_binary_test',
      files('tests/test-hk-rt-pipeline-binary.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [hk_deps],
      link_with : [libhk],
      link_args : [
        ld_args_gc_sections,
        hk_wrap_test_link_args,
        '-Wl,--wrap=vk_pipeline_shader_stage_to_nir',
        '-Wl,--wrap=vk_common_CreatePipelineBinariesKHR',
        '-Wl,--wrap=vk_common_GetPipelineKeyKHR',
        '-Wl,--wrap=hk_compile_shader',
        '-Wl,--wrap=hk_api_shader_destroy',
        '-Wl,--wrap=ralloc_free',
      ],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
endif

if with_tests
  executable(
    'vulkan-xfb-resume',
    files('tests/vulkan_xfb_resume.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )
  executable(
    'vulkan-mipmap-layers',
    files('tests/vulkan_mipmap_layers.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'g17p-vulkan-multiview-clear',
    files('tests/g17p_vulkan_multiview_clear.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  # Also no-op shim tests, registered with the shim in ../meson.build.
  g17p_vulkan_compressed_mip_blit = executable(
    'g17p-vulkan-compressed-mip-blit',
    files('tests/g17p_vulkan_compressed_mip_blit.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  hk_vulkan_explicit_plane_layout = executable(
    'hk-vulkan-explicit-plane-layout',
    files('tests/hk_vulkan_explicit_plane_layout.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g16s_vulkan_buffer_upload = executable(
    'g16s-vulkan-buffer-upload',
    files('tests/g16s_vulkan_buffer_upload.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_compressed_layer_copy = executable(
    'g17p-vulkan-compressed-layer-copy',
    files('tests/g17p_vulkan_compressed_layer_copy.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

endif

if with_tests
  hk_cpu_upload_bench = executable(
    'hk-cpu-upload-bench',
    files('tests/hk_cpu_upload_bench.c'),
    include_directories : [
      inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm, include_directories('.'),
    ],
    dependencies : [hk_deps],
    link_with : [libhk, libasahi_lib, libasahi_layout, libasahi_compiler],
    gnu_symbol_visibility : 'hidden',
  )
  test('hk_cpu_upload', hk_cpu_upload_bench, args : ['1'], suite : ['asahi'])
endif

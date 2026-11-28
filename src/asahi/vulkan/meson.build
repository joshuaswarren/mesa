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
  'hk_2d_stencil.h',
  'hk_byte_stencil.h',
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
  'hk_compute_kick.c',
  'hk_compute_kick.h',
  'hk_device.c',
  'hk_device.h',
  'hk_device_memory.c',
  'hk_device_memory.h',
  'hk_memory_budget.c',
  'hk_memory_budget.h',
  'hk_host_alias.c',
  'hk_host_alias.h',
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
  'hk_ppp_state.h',
  'hk_image.c',
  'hk_image.h',
  'hk_image_layout_core.c',
  'hk_image_layout_core.h',
  'hk_identity_copy4.h',
  'hk_nir_wgid.h',
  'hk_image_view.c',
  'hk_nir_copy_identity4.c',
  'hk_nir_tensor_ops.c',
  'hk_nir_tensor_ops.h',
  'hk_nir_match_pointwise_clamp.c',
  'hk_image_view.h',
  'hk_instance.c',
  'hk_instance.h',
  'hk_nir_forward_image_stores.c',
  'hk_nir_forward_image_stores.h',
  'hk_nir_lower_descriptors.c',
  'hk_nir_lower_gs.c',
  'hk_nir_lower_tes_rq.c',
  'hk_nir_lower_gs.h',
  'hk_nir_vertex_preamble.c',
  'hk_nir_vertex_preamble.h',
  'hk_nir_root_size.c',
  'hk_nir_tile_2d_stencil.c',
  'hk_nir_tile_byte_stencil.c',
  'hk_nir_opt_compute_kernels.c',
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
  'hk_gpu_waits.c',
  'hk_gpu_waits.h',
  'hk_pack.c',
  'hk_pack.h',
  'hk_render_pack.c',
  'hk_render_pack.h',
  'hk_inline_blend_layout.h',
  'hk_g17p_varyings.h',
  'hk_partial_mrt_hsr.h',
  'hk_queue.c',
  'hk_queue.h',
  'hk_queue_flush.h',
  'hk_frame_anatomy.c',
  'hk_frame_anatomy.h',
  'hk_sampler.c',
  'hk_sampler.h',
  'hk_shader.c',
  'hk_nir_lower_image_view_min_lod.c',
  'hk_shader.h',
  'hk_shader_upload_core.h',
  'hk_sync.c',
  'hk_sync.h',
  'hk_tess_batch.c',
  'hk_tess_batch.h',
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

  test('hk_g17p_varyings',
    executable('hk_g17p_varyings_test',
      files('tests/test-hk-g17p-varyings.c'),
      include_directories : [include_directories('.')]),
    suite : ['asahi'])

  test('hk_inline_blend_layout',
    executable('hk_inline_blend_layout_test',
      files('tests/test-hk-inline-blend-layout.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi,
      ],
      dependencies : [idep_nir, idep_mesautil]),
    suite : ['asahi'])

  test('hk_partial_mrt_hsr',
    executable('hk_partial_mrt_hsr_test',
      files('tests/test-hk-partial-mrt-hsr.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_libagx,
        idep_libagx_shaders_h, idep_agx_pack,
        idep_vulkan_runtime_headers, idep_vulkan_util_headers,
      ],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout]),
    suite : ['asahi'])

  test('hk_memory_budget', executable('hk_memory_budget_test',
    files('tests/test-hk-memory-budget.c', 'hk_memory_budget.c'),
    include_directories : include_directories('.')),
    suite : ['asahi'])
  test('hk_host_alias', executable('hk_host_alias_test',
    files('tests/test-hk-host-alias.c', 'hk_host_alias.c'),
    include_directories : include_directories('.')),
    suite : ['asahi'])
  test('hk_shader_upload', executable('hk_shader_upload_test',
    files('tests/test-hk-shader-upload.c'),
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
    'hk_g17p_storage_image',
    executable(
      'hk_g17p_storage_image_test',
      files('tests/test-hk-g17p-storage-image.c'),
      include_directories : include_directories('.'),
    ),
    suite : ['asahi'],
  )
  test(
    'hk_pass_merge',
    executable(
      'hk_pass_merge_test',
      files('tests/test-hk-pass-merge.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
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

  test('hk_image_store_forward', executable('hk_image_store_forward_test',
    files('tests/test-hk-image-store-forward.c', 'hk_nir_forward_image_stores.c'),
    include_directories : [include_directories('.'), inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil]), suite : ['asahi'])

  test('hk_root_size', executable('hk_root_size_test',
    files('tests/test-hk-root-size.c', 'hk_nir_root_size.c'),
    include_directories : [include_directories('.'), inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil]), suite : ['asahi'])

  test('hk_vdm_indirect_count',
    executable('hk_vdm_indirect_count_test',
      files('tests/test-hk-vdm-indirect-count.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil, idep_agx_pack],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden'),
    suite : ['asahi'])
  # Unused recording functions require section garbage collection so the
  # test can link the production resolver without the rest of the driver.
  if ld_args_gc_sections.length() > 0
    test(
      'hk_tess_batch',
      executable(
        'hk_tess_batch_test',
        files('tests/test-hk-tess-batch.c', 'hk_tess_batch.c',
              'hk_drm_render_core.c'),
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

    # The internal-stage launch helper (inline in hk_cmd_buffer.h) with its
    # launches and barriers recorded.
    test(
      'hk_internal_stage_flush',
      executable(
        'hk_internal_stage_flush_test',
        files('tests/test-hk-internal-stage-flush.c'),
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

    # Timestamp carriers, copies and their launches through the real
    # hk_sync recorder, on CPU-allocated streams; no device or GPU.
    test(
      'hk_sync_stamps',
      executable(
        'hk_sync_stamps_test',
        files('tests/test-hk-sync-stamps.c', 'hk_sync.c'),
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

    # Host availability deferred to the end of the submission, resolved
    # through the real hk_sync resolver; no device or GPU.
    test(
      'hk_host_flush',
      executable(
        'hk_host_flush_test',
        files('tests/test-hk-host-flush.c', 'hk_sync.c'),
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

    # Submit packing: the member check, the planner and the dependency fold,
    # resolved through the real hk_sync resolver; no device or GPU.
    test(
      'hk_pack',
      executable(
        'hk_pack_test',
        files('tests/test-hk-pack.c', 'hk_pack.c', 'hk_sync.c'),
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

    # Submit-time render packing: the edge checks, the tail links and the
    # packet's dependency fold, resolved through the real hk_sync resolver;
    # no device or GPU.
    test(
      'hk_render_pack',
      executable(
        'hk_render_pack_test',
        files('tests/test-hk-render-pack.c', 'hk_render_pack.c',
              'hk_drm_render_core.c', 'hk_sync.c'),
        link_with : [libasahi_lib],
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
        files('tests/test-hk-query-oq.c', 'hk_query_oq.c', 'hk_sync.c',
              'hk_event.c'),
        hk_entrypoints[0],
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
    'hk_gen1_as_precomp',
    executable(
      'hk_gen1_as_precomp_test',
      files('tests/test-hk-gen1-as-precomp.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi,
      ],
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
    'hk_drm_render_core',
    executable(
      'hk_drm_render_core_test',
      files('tests/test-hk-drm-render-core.c',
            'hk_drm_render_core.c'),
      include_directories : [inc_include, inc_src],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_agx3_ioto_program_core',
    executable(
      'hk_agx3_ioto_program_core_test',
      files('tests/test-hk-agx3-ioto-program-core.c',
            'hk_agx3_ioto_program_core.c',
            '../compiler/agx3_compute_program.c',
            '../compiler/agx3_graphics_program.c',
            '../compiler/agx3_m5_graphics_program.c',
            '../compiler/agx3_copy_core.c',
            '../compiler/agx3_ioto_uniform_program.c',
            '../compiler/agx3_ioto_load_packer.c',
            '../compiler/agx3_cprog_input.c',
            '../compiler/agx3_static_spill_core.c',
            '../compiler/agx3_mem_core.c',
            '../compiler/agx3_binder.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil],
      link_with : [libasahi_lib, libasahi_layout, libagx3_compose],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_agx3_vdm_decode',
    executable(
      'hk_agx3_vdm_decode_test',
      files('tests/test-hk-agx3-vdm-decode.c',
            'hk_agx3_vdm_decode.c'),
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
    'hk_zls_stride',
    executable(
      'hk_zls_stride_test',
      files('tests/test-hk-zls-stride.c'),
      include_directories : [include_directories('.')],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_sampler_index',
    executable(
      'hk_sampler_index_test',
      files('tests/test-hk-sampler-index.c'),
      include_directories : [include_directories('.')],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

    executable(
      'g17p-render-only',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
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
    'hk_g17p_decompress',
    executable(
      'hk_g17p_decompress_test',
      files('tests/test-hk-g17p-decompress.c', 'hk_native_descriptor_pack.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir_headers, idep_mesautil, idep_libagx, idep_libagx_shaders_h,
        idep_agx_pack, idep_vulkan_util, idep_vulkan_runtime_headers,
      ],
      link_with : [libasahi_lib, libasahi_layout],
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
    'hk_g17p_restart_state',
    executable(
      'hk_g17p_restart_state_test',
      files('tests/test-hk-g17p-restart-state.c'),
      include_directories : [include_directories('.')],
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
    'hk_agx3_compressed_buf2img',
    executable(
      'hk_agx3_compressed_buf2img_test',
      files('tests/test-hk-agx3-compressed-buf2img.c',
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
    'hk_g17p_linear_slice_copy',
    executable(
      'hk_g17p_linear_slice_copy_test',
      files('tests/test-hk-g17p-linear-slice-copy.c',
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
    'hk_vertex_preamble_shift',
    executable(
      'hk_vertex_preamble_shift_test',
      files('tests/test-hk-vertex-preamble-shift.c',
            'hk_nir_vertex_preamble.c'),
      include_directories : [include_directories('.'), inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
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
  test('hk_sparse_alias', executable('hk_sparse_alias_test',
    files('tests/test-hk-sparse-alias.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args,
      '-Wl,--wrap=__vk_errorf']),
    suite : ['asahi'])
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
      '-Wl,--wrap=agx_bo_export', '-Wl,--wrap=agx_bo_import',
      '-Wl,--wrap=agx_bo_import_host_ptr',
      '-Wl,--wrap=hk_get_cgroup_memory_available', '-Wl,--wrap=__vk_errorf',
      '-Wl,--wrap=os_get_available_system_memory',
      '-Wl,--wrap=os_get_total_physical_memory']),
    suite : ['asahi'])
  test('hk_cache_identity', executable('hk_cache_identity_test',
    files('tests/test-hk-cache-identity.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args]),
    suite : ['asahi'])
  test('hk_push_descriptor_size', executable('hk_push_descriptor_size_test',
    files('tests/test-hk-push-descriptor-size.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_shader_loop_bound_cache', executable('hk_shader_loop_bound_cache_test',
    files('tests/test-hk-shader-loop-bound-cache.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args,
      '-Wl,--wrap=__vk_errorf']),
    suite : ['asahi'])
  test('hk_compute_pair_cache', executable('hk_compute_pair_cache_test',
    files('tests/test-hk-compute-pair-cache.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_nested_nir', executable('hk_nested_nir_test',
    files('tests/test-hk-nested-nir.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_host_image_copy', executable('hk_host_image_copy_test',
    files('tests/test-hk-host-image-copy.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps, dep_thread], link_with : [libhk],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args,
      '-Wl,--wrap=__vk_errorf']),
    suite : ['asahi'])
  test('hk_tes_query', executable('hk_tes_query_test', files('tests/test-hk-tes-query.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_g17p_task_kernel', executable('hk_g17p_task_kernel_test',
    files('tests/test-hk-g17p-task-kernel.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_indirect_vdm', executable('hk_indirect_vdm_test',
    files('tests/test-hk-indirect-vdm.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_coherent_image', executable('hk_coherent_image_test',
    files('tests/test-hk-coherent-image.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_binding_base', executable('hk_binding_base_test',
    files('tests/test-hk-binding-base.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_m5_store_window', executable('hk_m5_store_window_test',
    files('tests/test-hk-m5-store-window.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_aligned_uadd_sat', executable('hk_aligned_uadd_sat_test',
    files('tests/test-hk-aligned-uadd-sat.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_aligned_ubo', executable('hk_aligned_ubo_test',
    files('tests/test-hk-aligned-ubo.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_m5_vs_cprog', executable('hk_m5_vs_cprog_test',
    files('tests/test-hk-m5-vs-cprog.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args,
      '-Wl,--wrap=hk_code_alloc', '-Wl,--wrap=agx_bo_create',
      '-Wl,--wrap=hk_device_ensure_geometry_heap']), suite : ['asahi'])
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
    'hk_compression_contract', executable('hk_compression_contract_test',
      files('tests/test-hk-compression-contract.c'),
      include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
      dependencies : [hk_deps], link_with : [libhk, libasahi_layout],
      link_args : [ld_args_gc_sections],
      gnu_symbol_visibility : 'hidden'), suite : ['asahi'])
  test(
    'hk_sparse_formats', executable('hk_sparse_formats_test',
      files('tests/test-hk-sparse-formats.c'),
      include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
      dependencies : [hk_deps], link_with : [libhk, libasahi_layout],
      link_args : [ld_args_gc_sections],
      gnu_symbol_visibility : 'hidden'), suite : ['asahi'])
  test(
    'hk_buffer_view', executable('hk_buffer_view_test',
      files('tests/test-hk-buffer-view.c'),
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
  test('hk_native_feature_defaults', executable('hk_native_feature_defaults_test',
    files('tests/test-native-feature-defaults.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_part_cache', executable('hk_part_cache_test',
    files('tests/test-hk-part-cache.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('hk_index_robustness', executable('hk_index_robustness_test',
    files('tests/test-hk-index-robustness.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
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
  hk_dual_source_gpl_test = executable(
    'hk-dual-source-gpl',
    files('tests/hk_dual_source_gpl.c'),
    include_directories : [
      inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm, include_directories('.'),
    ],
    dependencies : [hk_deps, cc.find_library('vulkan')],
    # hk_cmd_buffer.h uses GNU void-pointer arithmetic in inline helpers.
    c_args : [c_msvc_compat_args, '-Wno-pointer-arith'],
    gnu_symbol_visibility : 'hidden',
  )

  # Shader-object binary import: round trip, and refusal of forged root
  # metadata. Reads the binary layout from the driver's headers.
  hk_vulkan_shader_binary_root = executable(
    'hk-vulkan-shader-binary-root',
    files('tests/hk_vulkan_shader_binary_root.c'), hk_entrypoints[0],
    include_directories : [
      inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm, include_directories('.'),
    ],
    dependencies : [hk_deps, cc.find_library('vulkan')],
    # hk_cmd_buffer.h uses GNU void-pointer arithmetic in inline helpers.
    c_args : [c_msvc_compat_args, '-Wno-pointer-arith'],
    gnu_symbol_visibility : 'hidden',
  )

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
    'vulkan-render-area',
    files('tests/vulkan_render_area.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-tess-batch',
    files('tests/vulkan_tess_batch.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  vulkan_tess_batch_order = executable(
    'vulkan-tess-batch-order',
    files('tests/vulkan_tess_batch_order.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-ubo-record-loop',
    files('tests/vulkan_ubo_record_loop.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan'), cc.find_library('m')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-sample-variant-cache',
    files('tests/vulkan_sample_variant_cache.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-border-emulation-cache',
    files('tests/vulkan_border_emulation_cache.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-min-lod-emulation-cache',
    files('tests/vulkan_min_lod_emulation_cache.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-msaa-xfer',
    files('tests/vulkan_msaa_xfer.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'aurora-r11g11b10f',
    files('tests/aurora_r11g11b10f.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'aurora-msaa-reload',
    files('tests/aurora_msaa_reload.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'aurora-tile-visit',
    files('tests/aurora_tile_visit.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'aurora-restart-switch',
    files('tests/aurora_restart_switch.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-msaa-mrt',
    files('tests/vulkan_msaa_mrt.c'),
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

  executable(
    'aurora-tilesubmit',
    files('tests/aurora_tilesubmit.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  # Submit packing on hardware: run with HK_PACK_CDM_SUBMIT=0 and =1.
  executable(
    'aurora-packtest',
    files('tests/aurora_packtest.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan'), dep_thread],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'aurora-uniform-select',
    files('tests/aurora_uniform_select.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  # Also no-op shim tests, registered with the shim in ../meson.build.
  g17p_vulkan_bind_tail_state_change = executable(
    'g17p-vulkan-bind-tail-state-change',
    files('tests/g17p_vulkan_bind_tail_state_change.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_gpl_fast_link = executable(
    'g17p-vulkan-gpl-fast-link',
    files('tests/g17p_vulkan_gpl_fast_link.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_indirect_multi = executable(
    'g17p-vulkan-indirect-multi',
    files('tests/g17p_vulkan_indirect_multi.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_compressed_mip_blit = executable(
    'g17p-vulkan-compressed-mip-blit',
    files('tests/g17p_vulkan_compressed_mip_blit.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_garbage_metadata = executable(
    'g17p-vulkan-garbage-metadata',
    files('tests/g17p_vulkan_garbage_metadata.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  hk_vulkan_global_priority = executable(
    'hk-vulkan-global-priority',
    files('tests/hk_vulkan_global_priority.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  # Submit-time render packing on reusable command buffers: packed, replayed,
  # alone and regrouped submissions, each pixel-exact (hk_render_pack.h).
  hk_vulkan_render_pack_replay = executable(
    'hk-vulkan-render-pack-replay',
    files('tests/hk_vulkan_render_pack_replay.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  hk_vulkan_sparse_bind_first_submit = executable(
    'hk-vulkan-sparse-bind-first-submit',
    files('tests/hk_vulkan_sparse_bind_first_submit.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  hk_vulkan_descriptor_buffer_sizes = executable(
    'hk-vulkan-descriptor-buffer-sizes',
    files('tests/hk_vulkan_descriptor_buffer_sizes.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  hk_vulkan_descriptor_buffer_texel = executable(
    'hk-vulkan-descriptor-buffer-texel',
    files('tests/hk_vulkan_descriptor_buffer_texel.c'),
    include_directories : [inc_include, inc_src],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_cross_queue = executable(
    'g17p-vulkan-cross-queue',
    files('tests/g17p_vulkan_cross_queue.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_clear_pass_load = executable(
    'g17p-vulkan-clear-pass-load',
    files('tests/g17p_vulkan_clear_pass_load.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  hk_vulkan_bg_eot_threads = executable(
    'hk-vulkan-bg-eot-threads',
    files('tests/hk_vulkan_bg_eot_threads.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan'), dep_thread],
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

  # Ray tracing pipeline binary import: round trip, and refusal of groups
  # that name shaders past the pipeline's stages or of the wrong stage.
  hk_vulkan_rt_binary_groups = executable(
    'hk-vulkan-rt-binary-groups',
    files('tests/hk_vulkan_rt_binary_groups.c'),
    include_directories : [include_directories('.'), inc_include, inc_src],
    dependencies : [idep_mesautil, cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  hal200_vulkan_msaa_depth_compression = executable(
    'hal200-vulkan-msaa-depth-compression',
    files('tests/hal200_vulkan_msaa_depth_compression.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_layered_depth = executable(
    'g17p-vulkan-layered-depth',
    files('tests/g17p_vulkan_layered_depth.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan'), dep_m],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-storage-compression',
    files('tests/vulkan_storage_compression.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan'), dep_m],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-storage-alias',
    files('tests/vulkan_storage_alias.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-dynamic-offset-draws',
    files('tests/vulkan_dynamic_offset_draws.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-compute-store-visibility',
    files('tests/vulkan_compute_store_visibility.c'),
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

if with_tests
  test('hk_submit_admission', executable('hk_submit_admission_test',
    files('tests/test-hk-submit-admission.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk]),
    timeout : 30, suite : ['asahi'])
  test('hk_fused_cprog', executable('hk_fused_cprog_test',
    files('tests/test-hk-fused-cprog.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  # The queue's own-timeline wait of a kernel submission: one entry at the
  # latest point any reason asks for. Header only; no device.
  test('hk_own_wait', executable('hk_own_wait_test',
    files('tests/test-hk-own-wait.c'),
    include_directories : [include_directories('.'), inc_include, inc_src]),
    suite : ['asahi'])
  test('hk_errored_submit', executable('hk_errored_submit_test',
    files('tests/test-hk-errored-submit.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [hk_wrap_test_link_args]),
    timeout : 30, suite : ['asahi'])
  test('hk_lost_object_submit', executable('hk_lost_object_submit_test',
    files('tests/test-hk-lost-object-submit.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [hk_wrap_test_link_args,
      '-Wl,--wrap=vk_queue_wait_lost_submissions']),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('hk_sparse_submit_wait', executable('hk_sparse_submit_wait_test',
    files('tests/test-hk-sparse-submit-wait.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('hk_queue_sharing', executable('hk_queue_sharing_test',
    files('tests/test-hk-queue-sharing.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk]),
    timeout : 30, suite : ['asahi'])
  test('hk_pack_atomics', executable('hk_pack_atomics_test',
    files('tests/test-hk-pack-atomics.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('hk_cmd_pending', executable('hk_cmd_pending_test',
    files('tests/test-hk-cmd-pending.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk, libasahi_lib],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('hk_query_stamps', executable('hk_query_stamps_test',
    files('tests/test-hk-query-stamps.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk]),
    timeout : 30, suite : ['asahi'])
  test('hk_query_no_access', executable('hk_query_no_access_test',
    files('tests/test-hk-query-no-access.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
  test('hk_byte_stencil', executable('hk_byte_stencil_test',
    files('tests/test-hk-byte-stencil.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk, libasahi_compiler]),
    timeout : 30, suite : ['asahi'])
  test('hk_2d_stencil', executable('hk_2d_stencil_test',
    files('tests/test-hk-2d-stencil.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk, libasahi_compiler]),
    timeout : 30, suite : ['asahi'])
  test('hk_compute_opt', executable('hk_compute_opt_test',
    files('tests/test-hk-compute-opt.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk, libasahi_compiler]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('hk_attachmentless', executable('hk_attachmentless_test',
    files('tests/test-hk-attachmentless.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('hk_ray_tracing_core', executable('hk_ray_tracing_core_test',
    files('tests/test-hk-ray-tracing-core.c', 'hk_ray_tracing_core.c'),
    include_directories : [include_directories('.'), inc_include, inc_src]),
    suite : ['asahi'])
  test('hk_root_only_flush', executable('hk_root_only_flush_test',
    files('tests/test-hk-root-only-flush.c'),
    c_args : ['-DNDEBUG'],
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('hk_uma_heap_sizing', executable('hk_uma_heap_sizing_test',
    files('tests/test-hk-uma-heap-sizing.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps, dep_m], link_with : [libhk, libasahi_lib],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args,
      '-Wl,--wrap=drmIoctl',
      '-Wl,--wrap=os_get_available_system_memory',
      '-Wl,--wrap=os_get_total_physical_memory']),
    timeout : 30, suite : ['asahi'])

  test('hk_cdm_barrier_stream', executable('hk_cdm_barrier_stream_test',
    files('tests/test-hk-cdm-barrier-stream.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk, libasahi_lib],
    link_args : [ld_args_gc_sections, hk_wrap_test_link_args,
      '-Wl,--wrap=hk_ensure_cs_has_space']),
    timeout : 30, suite : ['asahi'])

  test('hk_resolve_fold_resume', executable('hk_resolve_fold_resume_test',
    files('tests/test-hk-resolve-fold-resume.c'),
    hk_entrypoints[0],
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [hk_deps], link_with : [libhk, libasahi_lib],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('hk_identity_copy4', executable('hk_identity_copy4_test',
    files('tests/test-hk-identity-copy4.c', 'hk_nir_copy_identity4.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi],
    dependencies : [idep_nir, idep_mesautil, idep_libagx, dep_m],
    link_with : [libasahi_compiler],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('hk_compute_pair', executable('hk_compute_pair_test',
    files('tests/test-hk-compute-pair.c', 'hk_nir_match_pointwise_clamp.c',
      'hk_nir_tensor_ops.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi],
    dependencies : [idep_nir, idep_mesautil, idep_libagx],
    link_with : [libasahi_compiler],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])

  test('hk_tensor_ops', executable('hk_tensor_ops_test',
    files('tests/test-hk-tensor-ops.c', 'hk_nir_tensor_ops.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi],
    dependencies : [idep_nir, idep_mesautil, idep_libagx, dep_m],
    link_with : [libasahi_compiler],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif

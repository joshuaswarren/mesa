# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# SPDX-License-Identifier: MIT

# This is used only for the r/e tooling on macOS
dep_iokit = dependency('IOKit', required : false)

inc_asahi = include_directories([
   '.', 'layout', 'lib', 'genxml', 'compiler'
])

subdir('isa')

with_asahi_common = with_gallium_asahi or with_asahi_vk or \
                    with_tools.contains('asahi')

if with_asahi_common
   subdir('genxml')
   subdir('libagx')
   subdir('compiler')
   subdir('clc')
endif

if with_gallium_asahi or with_asahi_vk
   subdir('layout')
   subdir('lib')
elif dep_iokit.found()
   # Just build enough for libwrap.dylib
   if not with_asahi_common
      subdir('genxml')
   endif
   subdir('lib')
endif

if with_asahi_vk
   subdir('vulkan')
endif

if with_tools.contains('drm-shim')
  subdir('drm-shim')

  if with_asahi_vk and with_tests and meson.can_run_host_binaries()
    dual_source_glslang = find_program('glslangValidator', required : false)
    if dual_source_glslang.found()
      test(
        'hk_dual_source_gpl',
        prog_python,
        args : [
          files('vulkan/tests/run-dual-source-gpl.py'),
          hk_dual_source_gpl_test, libvulkan_asahi, libasahi_noop_drm_shim,
          dual_source_glslang.full_path(),
        ],
        suite : ['asahi'],
      )
    endif
  endif
endif

if with_tools.contains('drm-shim') and with_asahi_vk and with_tests
  # Shader-object binaries: an exported binary imports and exports again
  # unchanged; one whose root prefix or root metadata was edited and resealed
  # is refused. M5 and G17P binaries carry the trimmed root prefix.
  foreach chip : [['g17s', ['--expect-root-size']],
                  ['g17p', ['--expect-root-size']], ['g13g', []]]
    test(
      'hk-' + chip[0] + '-shader-binary-root',
      prog_python,
      args : [
        files('../gallium/drivers/asahi/tests/run-shim.py'),
        libasahi_noop_drm_shim,
        hk_vulkan_shader_binary_root,
      ] + chip[1],
      env : [
        'ASAHI_SHIM_CHIP=' + chip[0],
        'MESA_SHADER_CACHE_DISABLE=true',
        'VK_DRIVER_FILES=' + _dev_icd.full_path(),
      ],
      depends : [_dev_icd, libvulkan_asahi],
      suite : ['asahi'],
    )
  endforeach

  # A draw with a pipeline fast-linked from libraries compiles nothing.
  test(
    'hk-g17p-gpl-fast-link',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_gpl_fast_link,
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'MESA_SHADER_CACHE_DISABLE=true',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # HAL200 bind-tail elision / incremental 28B FS PPP update vs non-FS PPP
  # state changes (scissor, viewport, cull mode, depth op, color blend).
  test(
    'hk-g17p-bind-tail-state-change',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_bind_tail_state_change,
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_BIND_TAIL_COUNTS=6,3,3,4',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Vertex-only push-constant changes with a root-using FS that does not read
  # push constants must reuse the prior fragment root and elide both the full
  # bind tail and the incremental 28B FS PPP update.
  test(
    'hk-g17p-bind-tail-vs-only-push',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_bind_tail_state_change,
      '--vs-only-push',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_MESA_DEBUG=verify',
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_BIND_TAIL_COUNTS=6,3,0,4',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  test(
    'hk-g16g-bind-tail-vs-only-push',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_bind_tail_state_change,
      '--vs-only-push',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g16g',
      'ASAHI_MESA_DEBUG=verify',
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_BIND_TAIL_COUNTS=6,3,0,4',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Ray tracing pipeline binaries: an exported binary creates the pipeline
  # again; one whose shader groups were edited and resealed is refused.
  # G17P with a kernel that admits render ray queries has ray tracing
  # pipelines and loads their binaries natively.
  test(
    'hk-g17p-rt-binary-groups',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      hk_vulkan_rt_binary_groups,
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_SHIM_RENDER_RAY_QUERY=1',
      'MESA_SHADER_CACHE_DISABLE=true',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # HAL200 2x/4x depth compression with zink's feedback-loop usage and
  # sample-location flag, through vkCreateImage and the host-copy query.
  foreach chip : ['g17p', 'g16g']
    test(
      'hk-' + chip + '-msaa-depth-compression',
      prog_python,
      args : [
        files('../gallium/drivers/asahi/tests/run-shim.py'),
        libasahi_noop_drm_shim,
        hal200_vulkan_msaa_depth_compression,
        '--expect-1x-compressed',
      ],
      env : [
        'ASAHI_SHIM_CHIP=' + chip,
        'VK_DRIVER_FILES=' + _dev_icd.full_path(),
      ],
      depends : [_dev_icd, libvulkan_asahi],
      suite : ['asahi'],
    )
  endforeach

  # AGX3 multi-draw indirect recording fast path across HAL200, G15 and
  # HAL300/M5 (fixed loader reuse, dynamic DRAW_PARAMS/DRAW_ID relocs,
  # robust-index batching, empty draws, VDM chunk rollover and Ioto growth).
  foreach chip : ['g17p', 'g16g', 'g15s', 'g17s-kernel']
    test(
      'hk-' + chip + '-indirect-multi',
      prog_python,
      args : [
        files('../gallium/drivers/asahi/tests/run-shim.py'),
        libasahi_noop_drm_shim,
        g17p_vulkan_indirect_multi,
        '--shim',
      ],
      env : [
        'ASAHI_SHIM_CHIP=' + chip,
        'ASAHI_MESA_DEBUG=verify',
        'VK_DRIVER_FILES=' + _dev_icd.full_path(),
      ],
      depends : [_dev_icd, libvulkan_asahi],
      suite : ['asahi'],
    )
  endforeach

  # Static G17P VDM stream validation on the CPU-encoded non-indexed multi-draw
  # indirect cases (bind-tail elision, dynamic relocs, and VDM chunk rollover).
  # Indexed robust-batch slots are written by libagx_draw_robust_index_batch on
  # the GPU and remain zeroed under drm-shim, so they are verified on hardware
  # and via ASAHI_MESA_DEBUG=verify above.
  test(
    'hk-g17p-indirect-multi-stream',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_indirect_multi,
      '--shim',
      '--non-indexed-only',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_MESA_DEBUG=verify',
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Graphics-path copies from one layer of a block-compressed image into a
  # compressed G17P image; encoded and stream-validated, not executed.
  test(
    'hk-g17p-compressed-layer-copy',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_compressed_layer_copy,
      '--shim',
      '--expect-compressed',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'ASAHI_MESA_DEBUG=perf',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Compressed M4 Pro depth arrays through every transfer, clear and
  # attachment path; recorded and submitted, not executed.
  test(
    'hk-g16s-layered-depth',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_layered_depth,
      '--shim',
      '--expect-compressed',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g16s',
      'ASAHI_MESA_DEBUG=perf',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Mip chains on compressed images take compute dispatches, no render pass.
  test(
    'hk-g17p-compressed-mip-blit',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_compressed_mip_blit,
      '--shim',
      '--expect-compute',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'ASAHI_MESA_DEBUG=perf',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Aliased compressed attachments over host-filled memory; recorded and
  # submitted, not executed (the hardware run needs the GPU).
  test(
    'hk-g17p-garbage-metadata',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_garbage_metadata,
      '--shim',
      '--expect-compressed',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_MESA_DEBUG=perf',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Aliased compressed attachments over host-filled memory; recorded and
  # submitted, not executed (the hardware run needs the GPU).
  test(
    'hk-g16s-garbage-metadata',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_garbage_metadata,
      '--shim',
      '--expect-compressed',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g16s',
      'ASAHI_MESA_DEBUG=perf',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Batched software tessellation (hk_tess_batch.h): its compute streams put
  # the direct launches of a stage back to back, which the validator admits;
  # encoded and stream-validated, not executed.
  test(
    'hk-g17p-tess-batch-stream',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim, vulkan_tess_batch_order, '--shim',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_COMPUTE=1',
      'ASAHI_SHIM_G17P_EXPECT_BACK_TO_BACK=1',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # An indirect tessellated draw is not batched: its tessellation launches
  # are indirect precompiled dispatches that ask for AGX_BARRIER_ALL, so each
  # must end in the strong cache flush its consumer relies on.
  test(
    'hk-g17p-tess-indirect-stream',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim, vulkan_tess_batch_order, '--shim', '--indirect',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
      'ASAHI_SHIM_G17P_EXPECT_COMPUTE=1',
      'ASAHI_SHIM_G17P_EXPECT_STRONG_INDIRECT_COMPUTE_BARRIER=1',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Global queue priorities: HIGH is advertised on every chip and passed to
  # the kernel, whose refusal (no CAP_SYS_NICE: EACCES; a GPU backend without
  # HIGH: EINVAL) is VK_ERROR_NOT_PERMITTED; REALTIME is always refused.
  foreach chip : ['g13g', 'g17p']
    foreach policy : ['allow', 'deny', 'unsupported']
      test(
        'hk-' + chip + '-global-priority-' + policy,
        prog_python,
        args : [
          files('../gallium/drivers/asahi/tests/run-shim.py'),
          libasahi_noop_drm_shim,
          hk_vulkan_global_priority,
        ],
        env : [
          'ASAHI_SHIM_CHIP=' + chip,
          'VK_DRIVER_FILES=' + _dev_icd.full_path(),
        ] + (policy == 'allow' ? [] : ['ASAHI_SHIM_HIGH_PRIORITY=' + policy]),
        depends : [_dev_icd, libvulkan_asahi],
        suite : ['asahi'],
      )
    endforeach
  endforeach

  # The M5 Ultra (G17D) profile: two dies, eight clusters. Admission, device
  # creation and a queue at each priority, on the shim.
  test(
    'hk-g17d-global-priority-allow',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      hk_vulkan_global_priority,
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17d',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # A sparse image bind as a queue's first submission, with no command buffer.
  test(
    'hk-g17p-sparse-bind-first-submit',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      hk_vulkan_sparse_bind_first_submit,
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  foreach chip : ['g15g', 'g15s', 'g15c', 'g16g', 'g17p', 'g17s']
    # Descriptor buffer sizes and strides as vkd3d-proton lays out its heaps.
    test(
      'hk-' + chip + '-descriptor-buffer-sizes',
      prog_python,
      args : [
        files('../gallium/drivers/asahi/tests/run-shim.py'),
        libasahi_noop_drm_shim,
        hk_vulkan_descriptor_buffer_sizes,
      ],
      env : [
        'ASAHI_SHIM_CHIP=' + chip,
        'VK_DRIVER_FILES=' + _dev_icd.full_path(),
      ],
      depends : [_dev_icd, libvulkan_asahi],
      suite : ['asahi'],
    )
  endforeach

  foreach chip : ['g17p', 'g17s']
    # Texel and storage buffer descriptors from vkGetDescriptorEXT: their
    # software metadata.
    test(
      'hk-' + chip + '-descriptor-buffer-texel',
      prog_python,
      args : [
        files('../gallium/drivers/asahi/tests/run-shim.py'),
        libasahi_noop_drm_shim,
        hk_vulkan_descriptor_buffer_texel,
      ] + (chip == 'g17s' ? ['m5'] : []),
      env : [
        'ASAHI_SHIM_CHIP=' + chip,
        'VK_DRIVER_FILES=' + _dev_icd.full_path(),
      ],
      depends : [_dev_icd, libvulkan_asahi],
      suite : ['asahi'],
    )
  endforeach

  # A render on one queue read by compute on another, ordered by timeline
  # semaphores, including waits made by submissions without command buffers;
  # recorded and submitted, not executed (the ordering check needs the GPU).
  test(
    'hk-g17p-cross-queue',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_cross_queue,
      '--shim',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
    timeout : 120,
  )

  # Background/end-of-tile shader cache shared by command buffers recording
  # on several threads.
  test(
    'hk-g17p-bg-eot-threads',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      hk_vulkan_bg_eot_threads,
      '16', '20',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
    timeout : 120,
  )

  # Explicit DRM plane layouts: placement, requirements, subresource layouts
  # and validation, without executing on the GPU.
  foreach chip : ['g13g', 'g17p']
    test(
      'hk-' + chip + '-explicit-plane-layout',
      prog_python,
      args : [
        files('../gallium/drivers/asahi/tests/run-shim.py'),
        libasahi_noop_drm_shim,
        hk_vulkan_explicit_plane_layout,
      ],
      env : [
        'ASAHI_SHIM_CHIP=' + chip,
        'VK_DRIVER_FILES=' + _dev_icd.full_path(),
      ],
      depends : [_dev_icd, libvulkan_asahi],
      suite : ['asahi'],
    )
  endforeach
endif

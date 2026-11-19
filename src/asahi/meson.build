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

  # Descriptor buffer sizes and strides as vkd3d-proton lays out its heaps.
  test(
    'hk-g17p-descriptor-buffer-sizes',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      hk_vulkan_descriptor_buffer_sizes,
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Texel buffer descriptors from vkGetDescriptorEXT: their software metadata.
  test(
    'hk-g17p-descriptor-buffer-texel',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      hk_vulkan_descriptor_buffer_texel,
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

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

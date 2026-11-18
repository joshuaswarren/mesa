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

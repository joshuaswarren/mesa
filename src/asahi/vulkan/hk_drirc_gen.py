#!/usr/bin/env python3
# Copyright 2024 Valve Corporation
# Copyright 2024 Alyssa Rosenzweig
# SPDX-License-Identifier: MIT

import argparse
import sys

VALID_COMMON_VK_OPTIONS = {
    "force_vk_devicename",
    "force_vk_vendor",
    "heap_memory_percent",
}

def declare_options():
    import drirc_gen

    B = drirc_gen.DrircBool

    misc_options = [
        B("hk_disable_border_emulation", False,
          "Disable custom border colour emulation",
          c_name="disable_border_emulation"),
        B("hk_fake_minmax", False,
          "Fake support for min/max filtering",
          c_name="fake_minmax"),
        B("hk_sparse_strict_residency", False,
          "Report sparseResidencyNonResidentStrict on G13, G14 and G17P, "
          "where a write to an unbound page can still be read back",
          c_name="sparse_strict_residency"),
        B("hk_host_memory_read_only", False,
          "Import host memory the kernel can only pin for reading (a "
          "read-only file mapping) and bind it read-only for the GPU, "
          "instead of failing the allocation. A GPU write to such memory is "
          "not performed: it is discarded, or faults the job on a kernel "
          "without soft faults",
          c_name="host_memory_read_only"),
        B("hk_host_memory_import_read_only", False,
          "Accept a host import only for a range the kernel cannot pin for "
          "writing (a read-only file mapping, say), bound read-only for the "
          "GPU, and refuse a range it pins for writing, also when "
          "hk_host_memory_read_only is set. The application may still write "
          "the same pages through another mapping, but the GPU writes "
          "through no host import, so on the M5 the extension does not keep "
          "compute barriers from skipping the write-back to memory",
          c_name="host_memory_import_read_only"),
        B("hk_d3d_nearest_mip", False,
          "Round a nearest mip selection halfway between levels up, as "
          "Direct3D does, rather than down, as Vulkan does",
          c_name="d3d_nearest_mip"),
        B("hk_image_view_min_lod", False,
          "Emulate VK_EXT_image_view_min_lod (conformant but slower)",
          c_name="image_view_min_lod"),
        B("hk_enable_vertex_pipeline_stores_atomics", False,
          "Enable vertexPipelineStoresAndAtomics",
          c_name="enable_vertex_pipeline_stores_atomics"),
        B("hk_mutable_format_srgb_pair", False,
          "Assume a mutable-format colour image created without a format "
          "list is only viewed as itself or its sRGB/UNORM twin, so that it "
          "can be compressed",
          c_name="mutable_format_srgb_pair"),
        drirc_gen.DrircInt("hk_compute_kick_ms", 10, 0, 1000,
          "M5 Max: once a queue's compute work has completed nothing for "
          "50 ms, submit a small compute job on a second queue every this "
          "many ms until it does, which lets other applications' compute "
          "work and rendering run meanwhile (0: off)",
          c_name="compute_kick_ms"),
        drirc_gen.DrircInt("hk_compute_timeout_ms", 0, 0, 3600000,
          "Watchdog allowance of each compute command on the device's "
          "queues, in ms (0: the kernel's default). The kernel clamps the "
          "request. It is an allowance, not a deadline: the kernel's "
          "watchdog may let a command run longer before it fails it",
          c_name="compute_timeout_ms"),
        B("hk_report_honeykrisp_driver_id", False,
          "Report VK_DRIVER_ID_MESA_HONEYKRISP instead of the Borealis ID, "
          "for translation layers that do not know the Borealis ID yet",
          c_name="report_honeykrisp_driver_id"),
    ]

    debug_options = []
    features_options = []
    performance_options = []

    drirc_gen.add_common_vk_options(debug_options, features_options, misc_options,
                                    valid_options=VALID_COMMON_VK_OPTIONS,
                                    defaults={"heap_memory_percent": 0.0})
    drirc_gen.add_common_vk_wsi_options(debug_options, performance_options)

    return [
        drirc_gen.DrircSection("Debugging", debug_options, c_name="debug"),
        drirc_gen.DrircSection("Performance", performance_options, c_name="performance"),
        drirc_gen.DrircSection("Miscellaneous", misc_options, c_name="misc"),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--import-path', required=True)
    parser.add_argument('--drirc-src', required=True)
    parser.add_argument('--drirc-hdr', required=True)
    parser.add_argument('--validate', required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.import_path)
    import drirc_gen

    options = declare_options()

    drirc_gen.drirc_validate([args.validate], options)

    drirc_gen.drirc_generate(args.drirc_src, args.drirc_hdr, "hk", options)


if __name__ == '__main__':
    main()

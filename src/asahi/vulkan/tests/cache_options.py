#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run with VK_DRIVER_FILES pointing to the Honeykrisp ICD under test."""
import os
import re
import subprocess

OPTIONS = ("AGX_SIMDMAT", "AGX_DECODE_Q4", "AGX_HWMAT_NO_FLATADDR", "AGX_HWMAT_VEC2")


def inspect(options):
    env = {k: v for k, v in os.environ.items() if k not in OPTIONS}
    env.update(options)
    result = subprocess.run(["vulkaninfo"], env=env, capture_output=True,
                            text=True, check=True, timeout=30)
    names = re.findall(r"^\s*deviceName\s*=\s*(.+)$", result.stdout, re.MULTILINE)
    assert len(names) == 1 and "Apple" in names[0], names
    ids = tuple(re.findall(r"^\s*(?:pipelineCacheUUID|shaderBinaryUUID)\s*=\s*(\S+)",
                           result.stdout, re.MULTILINE))
    assert len(ids) == 2, ids
    advertised = bool(re.search(r"^\s*VK_KHR_cooperative_matrix\s+:",
                                result.stdout, re.MULTILINE))
    print(options, ids, "cooperative_matrix=" + str(advertised), flush=True)
    return ids, advertised


def main():
    base, advertised = inspect({})
    assert not advertised, "cooperative matrices must be opt-in"
    assert inspect({"AGX_SIMDMAT": "0"}) == (base, False)
    enabled, advertised = inspect({"AGX_SIMDMAT": "1"})
    assert advertised and all(a != b for a, b in zip(base, enabled))
    seen = {enabled}
    for option in OPTIONS[1:]:
        changed, advertised = inspect({"AGX_SIMDMAT": "1", option: "1"})
        assert advertised and all(a != b for a, b in zip(enabled, changed)), option
        assert changed not in seen, option
        seen.add(changed)
        assert inspect({"AGX_SIMDMAT": "1", option: ""}) == (changed, True)
    assert inspect({}) == (base, False), "unsetting options must restore identity"
    print("PASS: opt-in advertisement and cache identities track compiler options")


if __name__ == "__main__":
    main()

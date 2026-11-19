# G18 background reload and multisample resolve check

This is a manual end-to-end GL test. It uploads a red RGB8 or RGBA8 texture,
draws green over its right half, and checks every readback pixel. At 2x/4x it
first clears and retires the multisample target, then draws and resolves it.
Each execution writes a 64x64 RGBA8 image even when pixels are wrong.

Build against EGL and desktop GL (a compatibility context is required):

```sh
cc -O2 -Wall -Wextra g18-reload.c $(pkg-config --cflags --libs egl gl) -o g18-reload
```

Select the intended Mesa install through the usual loader environment. Run
each case separately. A compiler refusal, submit failure (the driver reports
it on stderr), GPU fault or retirement failure fails the case even if the
process exits zero:

```sh
MESA_SHADER_CACHE_DISABLE=true ./g18-reload 0 rgb-1x.rgba
MESA_SHADER_CACHE_DISABLE=true RELOAD_RGBA=1 ./g18-reload 0 rgba-1x.rgba
MESA_SHADER_CACHE_DISABLE=true ./g18-reload 2 rgb-2x.rgba
MESA_SHADER_CACHE_DISABLE=true ./g18-reload 4 rgb-4x.rgba
```

`RELOAD_PACKED_FLOAT=1` selects `GL_R11F_G11F_B10F` instead, so the tile holds
one packed-float word and the end-of-tile program stores it as a raw 32-bit
word on G18G. Pure red and green are exact in that format, so the same
per-pixel check applies. Run it at 1x, 2x and 4x:

```sh
MESA_SHADER_CACHE_DISABLE=true RELOAD_PACKED_FLOAT=1 ./g18-reload 0 packed-1x.rgba
MESA_SHADER_CACHE_DISABLE=true RELOAD_PACKED_FLOAT=1 ./g18-reload 2 packed-2x.rgba
MESA_SHADER_CACHE_DISABLE=true RELOAD_PACKED_FLOAT=1 ./g18-reload 4 packed-4x.rgba
```

Success requires exit zero and `wrong=0` for all 4096 pixels. Retain the raw
process logs, image hashes and matching kernel submission/retirement evidence.
The expected left half is opaque red and right half opaque green, with no
allowance for unrendered pixels. A software run validates the probe; a drm-shim
run exercises compilation only. Neither qualifies M6 rendering.

The baseline M6 port rejects background fetches after split-coordinate
lowering, fails validation on full-width texture levels, and advertises zero
multisample capacity. This probe exercises those paths through the real GL,
Gallium and compiler code rather than constructing driver-internal objects.

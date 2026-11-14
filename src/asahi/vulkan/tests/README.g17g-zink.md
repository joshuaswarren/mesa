# M5 Zink rendering probe

`g17g_zink_draw.c` exercises a 256x256 surfaceless OpenGL triangle through
Zink and Honeykrisp. It rejects other renderer names and uses no depth,
stencil, textures, or fragment varyings. Build it against EGL and OpenGL:

```sh
cc -Wall -Wextra -O2 g17g_zink_draw.c -lEGL -lGL -o g17g_zink_draw
```

Select the experimental M5 ICD and matching Mesa libraries when running it.
Set `MESA_LOADER_DRIVER_OVERRIDE=zink` and `GALLIUM_DRIVER=zink` to
select Zink for this specific probe. The default path checks all 65,536 RGBA pixels returned
by `glReadPixels` against `ff470dff`. `--submit-only` deliberately skips that
check and must not be interpreted as rendering proof.

`--blit-only` copies the rendered image to a second framebuffer and skips
CPU readback. Its completion messages still require independent pixel and
queue validation. The deliberate `HK_M5_ABORT_BEFORE_IOCTL` error makes Zink
device-lost after the first submit, so that mode cannot preflight later GL
blits reliably. For independent draw-plus-blit recording, use the existing
Vulkan fixture instead:

```sh
cc -O2 -Wall -Wextra -DAGX_TEST_DEVICE_ID=0x8142 \
   -DIMAGE_WIDTH=256 -DIMAGE_HEIGHT=256 g17p_render_only.c -ldl -o m5-render-blit
AGX_RENDER_BLIT=1 AGX_RENDER_RECORD_ONLY=1 ./m5-render-blit
```

With the M5 ICD selected, the record-only mode compiles and records both
commands without submitting either. Removing `AGX_RENDER_RECORD_ONLY`
submits the draw, checks its pixels, then submits the blit and checks the
separate destination against the expected color and its prefilled sentinel.
`AGX_RENDER_UNCACHED=1 AGX_RENDER_HOLD=1` selects the diagnostic memory type
and retains both images for external inspection. This test is limited to
one frame and does not use an image-to-buffer compute copy.

`AGX_RENDER_READBACK=1` selects a separate image-to-buffer command instead
of the fragment blit. It prefills the destination and 16-byte guards at
both ends. After a successful hardware submission it checks every pixel
and both guards. This mode is mutually exclusive with `AGX_RENDER_BLIT`.
Use `AGX_RENDER_RECORD_ONLY=1` to validate command construction without
submitting work. On M5 this now records both command buffers successfully,
including a 342-byte compute shader and its native 148-byte Ioto program.
Compute execution and returned readback pixels remain unqualified.

This Vulkan draw-plus-blit test passed on M5 hardware on 2026-09-12. Both
submissions retired both graphics queues. All 65,536 source and destination
pixels matched `ff470dff`, and no destination pixels retained the `a5` fill.
The downloaded 262,144-byte destination matched the expected bytes exactly
(SHA-256 `8ebe40523a836923ad5f331124b2caf011a971c9ce6232acfa6a188331125e36`).
This qualifies the bounded fragment-copy path, including native indexed
array sampling and its conditional shader. It does not qualify compute
readback, other sizes/formats, or general texture sampling.

`AGX_GL_DEFER_READBACK=1` retains the live context after `glFinish`; send SIGUSR1
to the reported PID to request readback. `AGX_GL_HOLD=1` also retains resources
after that check. These modes let an external diagnostic verify queue
completion and read the actual render target while its allocations are live.
Do not access GPU MMIO as part of that RAM inspection.

On M5 hardware on 2026-09-12, both graphics queues retired and all 65,536
render-target pixels independently matched `0d47ffff` in the selected BGRA
surface. This exercised Mesa-generated shaders, the AGX3c robust-fetch integer
select, position plus point-size UVS layout, and the inverted full-frame Zink
viewport. No native reference shader was substituted.

OpenGL readback did **not** pass. The fragment-copy draw is now hardware
validated, including M5 array sampling and its conditional shader. The
complete image-to-buffer compute shader now compiles after adding native
indexed texture reads and extended-wait buffer stores. M5-native compute
Ioto and direct CDM construction now pass recording tests. The kernel
application adapter still accepts only Render commands; compute needs
SKSM queue integration, shared VM/resource ownership, and verified fence
completion before that shader can run. Zink can report these Vulkan failures
while `glFinish` and `glGetError` appear successful, so always inspect the
compiler/submission log and verify the pixels independently. The rendering
result does not establish general OpenGL support, glxgears, or display-server
presentation. Depth, fragment varyings, transfers, sustained queue reuse, and
the combined GPU/DCP desktop remain separate validation work.

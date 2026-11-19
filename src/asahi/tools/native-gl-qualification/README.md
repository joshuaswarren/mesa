# Native GL qualification

This kit selects a relocatable Mesa build for one process. It does not install
Mesa, select a new system default, or establish hardware conformance by itself.
Run GPU commands only inside a reservation approved for the named test device.
Keep raw machine logs and captures private; publish only sanitized results.

## Inputs and identity

Use an aarch64 overlay built from the exact candidate commit. Record the Mesa commit,
archive SHA256, ELF architecture and runtime dependencies, kernel revision,
CTS revision, binary SHA256 and deqp-runner version. Do not mix builds between
the two driver paths. Do not use an x86 DRM-shim build as a hardware result.

An overlay has `usr/lib64/` or `usr/lib/` with Mesa EGL, GLX, GBM, Gallium,
Honeykrisp, the `gbm/dri_gbm.so` backend, and relative `dri/asahi_dri.so` and
`dri/zink_dri.so` links.
`with-mesa.py` checks that its library links remain inside the overlay, creates
absolute loader JSON files, and overrides library search paths only in the
child. Native runs unset both driver selectors so the loader selects Asahi.
Zink runs select Zink and the overlay's Honeykrisp ICD explicitly. Mesa's shader
cache is disabled by default; `--cache on` explicitly enables it for warm runs.

Package a native aarch64 release build with `DESTDIR` (never a system install):

```sh
STAGE=$(mktemp -d)
DESTDIR="$STAGE" meson install -C build --no-rebuild
python3 src/asahi/tools/native-gl-qualification/make-overlay.py \
  --stage "$STAGE" --commit "$(git rev-parse HEAD)" --architecture AArch64 \
  --output /var/tmp/mesa-native-qualification/overlay
```

Alternatively extract the matching native RPM artifact into a fresh staging
directory with `rpm2cpio`/`cpio`, then run the same packager with the recorded
source commit. Select the native artifact, not FEX or source RPMs. The packager
checks source identity in Gallium/Honeykrisp, ELF architecture and absence of
LLVM/SPIRV-Tools runtime dependencies, and records SHA256 hashes and dynamic
dependencies in `MANIFEST.json`. Preserve the original RPM/archive checksums.
It includes Mesa's license notices from the source tree containing the kit;
use `--source` if the matching public source is in a different directory.
Resolve any missing runtime dependencies in the test environment before a run;
do not mix a different Mesa driver into the overlay to satisfy them.

```sh
export KIT=$PWD/src/asahi/tools/native-gl-qualification
export OVERLAY=/var/tmp/mesa-native-qualification/overlay
export OUT=/var/tmp/mesa-native-qualification/results
mkdir -p "$OUT"
cc -std=c11 -O2 -Wall -Wextra -Werror "$KIT/index-coverage.c" \
  $(pkg-config --cflags --libs egl glesv2) -o "$OUT/index-coverage"
for driver in native zink; do
  python3 "$KIT/with-mesa.py" --overlay "$OVERLAY" --driver "$driver" -- \
    "$OUT/index-coverage" > "$OUT/index-$driver.txt" 2>&1
done
```

Require `tests=14 failures=0` on both paths and **4096/4096** green pixels
for every draw. This covers U8 offsets 0/1/2/3, U16 offsets 0/2 and U32 offset
0, each after a CPU upload and after a GPU buffer copy without an intervening
finish. Record renderer strings: native must report the Apple/Asahi renderer,
Zink must say Zink, and neither may use llvmpipe/softpipe. Retain failure logs.
The historical half-strip defect produces roughly half coverage, not a subtle
image difference. A no-op shim cannot validate these pixels.

## GLES and focused GL cases

Use one frozen public VK-GL-CTS source and build, with its licensed test data.
Set `CTS` to its build root. Keep each binary's working directory and data
beside it. Generate complete supported-suite lists with the candidate driver:

```sh
export CTS=/var/tmp/VK-GL-CTS-build
export LISTS="$OUT/lists"
mkdir -p "$LISTS"
for suite in gles2 gles3 gles31; do
  (
    cd "$CTS/modules/$suite"
    EGL_PLATFORM=surfaceless python3 "$KIT/with-mesa.py" --overlay "$OVERLAY" --driver native -- \
      "./deqp-$suite" --deqp-runmode=txt-caselist
    sed -n 's/^TEST: //p' "dEQP-$(echo "$suite" | tr '[:lower:]' '[:upper:]')-cases.txt" \
      > "$LISTS/$suite.txt"
    test -s "$LISTS/$suite.txt"
  )
done
for driver in native zink; do
  JOBS=1 TIMEOUT=120 bash "$KIT/run-cts.sh" \
    "$OVERLAY" "$driver" "$CTS" "$LISTS" "$OUT/full-$driver"
done
for suite in gles2 gles3 gles31; do
  python3 "$KIT/compare-results.py" "$LISTS/$suite.txt" \
    "$OUT/full-zink/$suite/results/results.csv" \
    "$OUT/full-native/$suite/results/results.csv"
done
```

Freeze and checksum the generated lists before either run. Require exact
coverage of every listed case, no unexpected failure/crash/timeout/internal
error, no new warnings, and no Pass-to-NotSupported changes. Unsupported
features need a capability-based explanation, not a new skip list. Preserve
baseline failures too. The comparator rejects missing, extra and duplicate
rows; review warnings separately. Repeat failures in isolated processes with
shader sources enabled and preserve QPA files. Check kernel logs over the run
interval for GPU faults, resets, OOMs and new warnings.

The checked-in `gles*-aosp.txt` files are the focused native/Zink comparison
cohort, not full mustpass lists. Use them as `gles2.txt`, `gles3.txt`, and
`gles31.txt` in a separate list directory for a quick initial run.
`gl46a.txt`, `gl46b.txt`, and `gles-khr.txt` run through
`$CTS/external/openglcts/modules/glcts` from that binary's working directory:

```sh
for driver in native zink; do
  for suite in gl46a gl46b gles-khr; do
    (
      cd "$CTS/external/openglcts/modules"
      EGL_PLATFORM=surfaceless python3 "$KIT/with-mesa.py" --overlay "$OVERLAY" --driver "$driver" -- \
        deqp-runner run --deqp "$PWD/glcts" --caselist "$KIT/$suite.txt" \
        --output "$OUT/$driver-$suite" --jobs 1 --timeout 180 \
        --tests-per-group 1 -- --deqp-surface-type=pbuffer \
        --deqp-gl-config-name=rgba8888d24s8ms0 --deqp-visibility=hidden \
        --deqp-shadercache=disable --deqp-log-shader-sources=enable
    )
  done
done
```

`remaining13.txt` identifies eight sample-mask and five sparse-buffer cases;
`fp64.txt` identifies the three dmat2/3/4 subroutine-array compile-cost cases.
Sparse-buffer NotSupported on native GL reflects absent GL_ARB_sparse_buffer;
it is not a pass. Software FP64 is advertised on native Asahi, so these three
cases must be timed on **both** paths. They cannot be waived as unreachable.

## Piglit subset

Use the same public Piglit revision and build on both paths. Run from its root
with `PIGLIT_PLATFORM=surfaceless_egl`. Verify the profile selected nonzero tests
and record test names/counts from `results.json.bz2`; an empty run fails.

```sh
export PIGLIT=/var/tmp/piglit
for driver in native zink; do
  (
    cd "$PIGLIT"
    EGL_PLATFORM=surfaceless PIGLIT_PLATFORM=surfaceless_egl python3 "$KIT/with-mesa.py" \
      --overlay "$OVERLAY" --driver "$driver" -- \
      ./piglit run gpu "$OUT/piglit-$driver" \
      -t 'draw-elements|primitive-restart|arb_gpu_shader_fp64|sample.mask|multisample|fbo-blit'
  )
done
"$PIGLIT/piglit" summary console "$OUT/piglit-zink" "$OUT/piglit-native"
```

Require no new failures/crashes/timeouts or loss of previously passing tests.
Run the same subsets on G13/G14 reference builds before claiming no regression
there; compiler unit tests and shim controls alone do not establish parity.

## Compositor and greeter

First run a nested compositor in the existing graphical test environment:

```sh
python3 "$KIT/with-mesa.py" --overlay "$OVERLAY" --driver native -- \
  dbus-run-session -- gnome-shell --nested --wayland
```

Exercise overview, application windows, translucent menus, avatar clipping,
animations and logout. Capture the loaded overlay Gallium path in that
gnome-shell process's `/proc/PID/maps`, renderer information and its journal.
Inspect the actual circular avatar, including its lower-right quadrant.

The GDM greeter is a separate test. A user-session environment or GDM service
drop-in does not prove the greeter used the overlay: PAM may reset it. During
an approved exclusive configuration-test reservation, temporarily put the
wrapper's absolute loader paths in `/etc/environment`, unset the Zink
selectors there, and restart the display manager. Pre-create loader JSON
files with the index test. All parent directories and libraries must be
readable/traversable by the greeter. Keep the original file and an automatic
rollback scheduled within the reservation before restarting GDM. Restore it
on every exit, including a failed greeter start. This changes configuration
temporarily; Mesa libraries stay entirely in the overlay.

Confirm the **greeter** gnome-shell PID (not the user's compositor) maps the
candidate `libgallium` path. Zink symlinks also resolve to libgallium, so absence
of `zink_dri.so` in maps is not proof of native GL; inspect the filtered driver
selectors in its environment and absence of `libvulkan_asahi.so` too. Repeat
the avatar test on the real greeter, with a fresh launch. Preserve capture,
journal interval, loaded library identity and result.

For direct scanout captures record DRM format, width, height and pitch. XR30
is little-endian XRGB2101010: B bits 0..9, G 10..19, R 20..29, X 30..31. Do
not interpret it as 8-bit BGRX or use byte-3 histograms as an alpha oracle.
If the capture tool cannot report the format/pitch, use the numerical index
test and mark greeter visual qualification pending.

## Performance

Use identical dimensions, build, compositor/display path and power/thermal
conditions. Alternate native/Zink order for seven paired repetitions; retain
raw logs, scores, elapsed/user/system CPU times and renderer strings. Warm
each path once, then measure with separate persistent shader-cache
directories. Report medians and spread, including any loss beyond run noise.

```sh
export GLMARK2=glmark2-es2-wayland # use the installed matching display backend
for rep in $(seq 0 7); do
  drivers='native zink'
  test $((rep % 2)) = 0 || drivers='zink native'
  for driver in $drivers; do
    MESA_SHADER_CACHE_DIR="$OUT/cache-$driver" /usr/bin/time \
      -f 'elapsed=%e user=%U system=%S maxrss_kb=%M' \
      -o "$OUT/glmark-$driver-$rep.time" \
      python3 "$KIT/with-mesa.py" --overlay "$OVERLAY" --driver "$driver" --cache on -- \
      "$GLMARK2" --off-screen --size 1280x720 \
      > "$OUT/glmark-$driver-$rep.log" 2>&1
  done
done
```

Repetition 0 warms the cache and is excluded. For a gpubench comparison use
the public aurora-silicon/gpubench GL executor, `gpubench-gl` (built with
`-DGB_BUILD_GL=ON`), on both paths and check the renderer it reports. Vulkan
timings do not compare native GL with Zink; do not invent numbers or relabel a
Vulkan run. This kit has no measured Neo scores.

## Default selection and rollback

Only after all qualification gates pass, use the tested Mesa build in each
distro and remove the global `MESA_LOADER_DRIVER_OVERRIDE=zink` assignment.
Remove an accompanying exact `GALLIUM_DRIVER=zink` assignment if present.
Keep every unrelated environment setting byte-for-byte. Inspect user and
greeter environment.d files for remaining selectors; their configuration
owners must remove those separately. Restart the display manager or reboot
inside the approved reservation so both session types get fresh environments.
Verify native renderer identity again with installed Mesa after the overlay
variables have been removed. Re-run aurora-diag GL/GLES checks.

Rollback restores the saved `/etc/environment` with its original metadata and
restarts the display manager, or reboots using the existing boot fallback.
Restoring Zink is configuration-only and must retain the same Mesa version
while diagnosing a rendering regression. Do not leave overlay paths globally
configured. Distribution packaging already builds both drivers; keep Zink
available as a per-application fallback:

```sh
MESA_LOADER_DRIVER_OVERRIDE=zink GALLIUM_DRIVER=zink application
```

Do not promote a selector removal while any rendering classification, complete
suite, native FP64 cost, index coverage, greeter, or performance result is still
pending. A test bug requires spec evidence; a slower driver workaround is not
a substitute for correcting that test.

# lapstack — Laplacian-pyramid focus stacking, native and in the browser

`lapstack` fuses a focus-bracketed series of photographs into one all-in-focus
image and a dense depth map. It is a from-scratch implementation of focus
stacking on the Laplacian pyramid, written from two papers kept in `docs/`:

- Adelson, Anderson, Bergen, Burt, Ogden, *Pyramid methods in image
  processing* (RCA Engineer, 1984) — REDUCE/EXPAND, the band-pass
  decomposition, and the "multifocus composite": pick, node by node, the
  pyramid coefficient with the larger magnitude, then expand-and-add; the
  blending between frames happens in the reconstruction itself.
- Wang & Chang, *A Multi-focus Image Fusion Method Based on Laplacian
  Pyramid* (J. Computers 6(12), 2011) — the 5×5 binomial generating kernel,
  **maximum region energy** selection for the band-pass levels, and a local
  deviation + entropy rule for the residual.

## Layout

```
crates/lapstack-core   library: pyramid, fusion, depth from focus, aligner, I/O, CUDA path
crates/lapstack-cli    `lapstack` command-line tool
crates/lapstack-web    wasm32 + WebGPU engine for the browser app
web/                   the browser app (static files) and its headless test
docs/                  the two papers the algorithm is written from
```

## Build & run (native)

`just` lists the developer commands (build, test, serve, chrome, dev, …);
`just serve` also kills a stale server on the port first.

```
cargo build --release
target/release/lapstack --align-coarsen 2 --save-depth -o out.png frames/*.tif
target/release/lapstack --no-align -o out.png aligned/*.png    # streams from disk
cargo build --release --features gpu -p lapstack-cli               # CUDA fusion + aligner
target/release/lapstack --gpu --gpu-align --align-coarsen 2 -o out.png frames/*.tif
```

Input is PNG, JPEG or TIFF, 8- or 16-bit; the output keeps the input bit
depth (16-bit needs PNG or TIFF). `lapstack --help` lists every option.

**Transform** (`pyramid.rs`): separable kernel `[1 4 6 4 1]/16` (Burt's
a = 0.375), reflect-101 borders, so REDUCE/EXPAND are plain linear operators
at any image size (odd or even at every level) and `collapse(build(x)) == x`
to float precision. The number of band-pass levels defaults to as many as
keep the residual's short side ≥ 32 px (7 levels on 8280×5520, residual
65×44); `--levels N` overrides.

**Fusion** (`fuse.rs`), generalised from the paper's two frames to N:

- Band-pass levels: region energy `RE = Σ ω·L²` over a binomial window
  (`--energy-radius`, default 1 = 3×3) of the *luminance* coefficient
  (Y = .299R+.587G+.114B of the RGB coefficients — the pyramid is linear, so
  that is the luma pyramid), winner-take-all per coefficient, ties to the
  earlier frame. The selection is applied to all three channels so colour
  never splits at a selection edge. Radius 0 is Adelson's per-node |L| max.
- Residual: local deviation D and local entropy E (`--top-radius`, default
  2 = 5×5; `--entropy-bins`, default 256). The paper's rule takes A when A
  wins both measures, B when B wins both, and averages otherwise; for N
  frames that is kept as a Pareto rule — a frame is dominated if another is
  at least as good on both and strictly better on one, and the fused value is
  the mean of the non-dominated frames (`--top de`; `dev` = deviation only,
  `avg` = plain mean).
- The accumulator folds frames in one at a time: only the running fused
  pyramid, one best-energy plane per level and the tiny residuals are held,
  so memory does not grow with the stack. Without alignment (`--no-align`)
  frames are decoded on demand with a bounded read-ahead.

**Alignment** (`align.rs`): 4-DOF similarity registration (shift, scale,
rotation), direct intensity-based, coarse-to-fine on a Gaussian pyramid of
the luma with a DC-removed RMS objective, Spline4x4 resampling and a bounded
Nelder-Mead search, chained sequentially to frame 0. `--align-coarsen N`
stops N levels short of full resolution (the transform is resolution
independent, so this is a large speed-up at sub-pixel accuracy);
`--no-shift/--no-scale/--no-rotation` restrict the model; `--save-aligned DIR`
writes the registered frames.

`--save-depth` writes the depth map produced by the depth-from-focus pass
below (`--depth winner` instead reports the raw winning frame index read
from pyramid level `--depth-level`, default 2 — the finest level's winner
map is noise wherever the scene is flat).

### Depth from focus (`depth.rs`)

After fusion, lapstack streams the aligned frames a second time and builds a
dense, sub-frame depth map with the modern non-learned depth-from-focus
recipe, written from the papers:

1. **Focus measure** — the *ring difference filter* of Jeon, Surh, Im &
   Kweon (IEEE TIP 2019), `|mean(disk r≤1) − mean(ring 1<r≤3)|` on luma at
   full resolution (`--depth-focus rdf:RIN:ROUT`; `sml` = Nayar &
   Nakagawa's sum-modified Laplacian).
2. **Cost aggregation** — block-mean to a working grid (`--depth-scale`,
   default 1 = half resolution) and a *guided filter* (He, Sun & Tang 2013)
   on every slice of the focus volume with the fused all-in-focus luma as
   guide (`--depth-agg R:EPS`, default 3:1e-4), the edge-aware aggregation
   of fast cost-volume filtering.
3. **Peak search** — streamed over the frame axis with O(1) memory in the
   stack size: the global peak with its two neighbours (Gaussian
   interpolation of Nayar & Nakagawa 1994 → fractional frame index), the
   second-best local maximum (peak-ratio confidence), the profile mean
   (prominence) and a noise gate against the median profile minimum
   (`--depth-gate`). Confidence is normalised so its 90th percentile is 1.
4. **Regularisation** — edge-aware *weighted least squares* (Farbman et al.
   2008) with the confidence as data weight: flat, noisy or ambiguous
   pixels take their depth from confident neighbours without crossing image
   edges. The separable fast global smoother (Min et al. 2014) gives the
   initial guess, a Jacobi-preconditioned conjugate gradient solves the 2-D
   system (`--depth-lambda`, `--depth-sigma`, `--depth-cg`), and one Huber
   reweighting pass (`--depth-robust`, default 1 frame) removes outliers
   that a least-squares fit would otherwise average in.
5. **Upsampling** — guided-filter upsampling on the full-resolution luma
   (`--depth-upsample guided:R:EPS` | `bilinear`), so depth edges land on
   image edges.

`--save-depth` writes the 8-bit visualisation, `--depth-raw PATH` a 16-bit
PNG with a fixed scale (65535 = last frame) for numeric use, `--save-conf`
the confidence map.

### Performance

On a 25-frame stack of 8280×5520 16-bit TIFFs (RTX 3060, 128-thread host),
same frames and alignment settings for every row:

| run | wall | of which align / fuse | peak RSS |
|---|--:|--:|--:|
| `lapstack --align-coarsen 2` (CPU) | 83 s | 56 s / 21 s (0.8 s per frame) | 31 GB |
| `lapstack --gpu --gpu-align --align-coarsen 2` | 40 s | 32 s / 2.0 s | 31 GB |
| `lapstack --no-align` on the aligned 16-bit PNGs, CPU or GPU | 26 s | – / 24 s | 4.4 GB |

With `--gpu` (build with `--features gpu`; CUDA is loaded at run time, no
toolkit needed at build time) the fusion runs in `gpu.rs`: the same kernels
transcribed to CUDA, the accumulator pyramid, best-energy planes and winner
map stay on the device, and only the three RGB planes go up per frame and
the tiny residual comes back. `--gpu-align` runs the aligner's Nelder-Mead
cost search on the GPU. Both GPU pipelines are then alignment-bound (32 s of
the 40 s). The streaming row is 16-bit PNG decode-bound (~1 s per frame with
four decoder threads); uncompressed TIFF input decodes an order of magnitude
faster.

GPU fusion output is bit-exact with the CPU output (one run out of four
differed in 637 of 45.7 M pixels on energy ties and could not be
reproduced). CPU reruns are byte-identical. GPU alignment evaluates the cost
in FP32, so `--gpu-align` results differ from CPU-aligned ones by ~0.5 % of
pixels.

Parameter sweep on the same aligned frames (default = 3×3 window, 7 levels,
`--top de`): the energy window is the only knob that matters.
`--energy-radius 0` (per-node max) is grainier, `--energy-radius 2` (5×5)
slightly smoother. `--top avg|dev`, `--levels 5|9` and `--use-chroma` change
< 1 % of tiles: as Burt & Adelson note, the low-pass content is shared
between frames, so the residual rule is nearly moot on a deep stack.

## lapstack in the browser (`web/`, WebGPU + WASM)

`crates/lapstack-web` + `web/` run the whole lapstack pipeline in a browser
tab: frames are decoded in WASM (PNG/JPEG/TIFF, 8- and 16-bit, via the `image`
crate), aligned and fused on **WebGPU** (`shaders.wgsl`, the same kernels as
the CUDA path transcribed to WGSL), one frame at a time in a Web Worker, so
browser memory stays at a couple of frames regardless of stack size. Only the
residual rule and PNG encoding run on the CPU side (lapstack-core in WASM).

```
./web/build.sh          # wasm32 build + wasm-bindgen glue into web/pkg
./web/serve.sh          # http://localhost:8765/  (WebGPU needs a secure context; localhost is one)
```

Needs the `wasm32-unknown-unknown` std, `wasm-bindgen-cli` matching the crate
(`cargo install wasm-bindgen-cli --version 0.2.128`) and a wasm linker
(rustup's `rust-lld`, or `wasm-ld` from LLVM/lld — on NixOS
`nix-shell -p lld --run ./web/build.sh`).

**Chrome on Linux ships WebGPU behind command-line switches.** Stock Chrome
finds no adapter; the `chrome://flags` pair (`#enable-unsafe-webgpu` +
`#enable-vulkan`) only yields the SwiftShader *software* adapter (1 GB buffer
cap, slow). Leave those flags at *Default* and use the launcher instead:

```
./web/serve.sh &
./web/chrome.sh     # google-chrome --user-data-dir=~/.config/lapstack-chrome \
                    #   --enable-unsafe-webgpu --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan
```

It runs a **separate profile**, so your normal Chrome is untouched and the
switches take effect even while it is open (Chrome ignores switches when a
window with the same profile already exists), and it forces the **X11
backend** (`--ozone-platform=x11`, i.e. XWayland): Chrome's native Wayland
backend refuses to present with Vulkan enabled and the window stays black.
`VulkanFromANGLE,DefaultANGLEVulkan` is what selects the hardware adapter.
The flags page's "ANGLE graphics backend = Vulkan" (`--use-angle=vulkan`)
exposes it too, but on NVIDIA + Wayland it makes Chrome's accelerated 2D
canvas paint black, so the app shows nothing — leave it at Default.
`probe.html` shows what the browser offers in the main thread and in a
worker. The app warns when it lands on a software adapter.
Chrome/Edge on Windows/macOS have WebGPU on by default; Firefox needs
`dom.webgpu.enabled` (and `dom.webgpu.workers.enabled`); Safari 26+.

The UI follows the workflow as three steps in the top bar:
**1 Stack** (add frames, set parameters, run, inspect the result),
**2 Retouch** (paint from source frames into the result) and **3 Save**
(format, file name, depth map). Keys 1/2/3 switch steps.

*Stack* is the workbench: a filmstrip (thumbnails arrive as frames are
added, with each frame's registration once aligned), a parameter panel (all
`lapstack` knobs, persisted in localStorage), Run/Cancel with progress and a
log, and a viewer whose header is a segmented **Source / Stack / Depth**
control with a second-level control for the group's layers — **LAP / DFR**
under Stack (DFR only when the depth-map render ran), **Focus depth / Winner** under
Depth — followed by controls that only show for what is on screen: on
Source a **peaking** toggle with a threshold stepper; on Depth a Gray/Turbo
LUT and a **slice** toggle; a frame slider whenever the shown layers depend
on a frame; and on the right a **compare** toggle whose "vs" dropdown lists
the other layers (the divider is draggable, `flip` or space swaps sides).
Scroll-zoom at the cursor, drag pan, double-click fit/100 %, wheel / ←/→
scrub (shift = 10; ctrl+wheel zooms on scrubbable views). Drag-and-drop
works. Added frames are decoded and downscaled in the worker straight away
(the browser cannot decode TIFF itself), so the filmstrip and Source view
are populated before a run; after the run the Source view shows the aligned
screen-resolution proxies produced during it — full frames are not retained.

*Save* writes the (retouched) image as PNG at the input bit depth, 8-bit
PNG or JPEG with a quality slider, under a chosen file name, and the depth
map as an 8-bit gray PNG (min–max scaled) or a 16-bit PNG with a fixed
scale (65535 = last frame, the same encoding as the CLI's `--depth-raw`).

**Depth map**: every run ends with the **depth from focus** pass — the
pipeline of `crates/lapstack-core/src/depth.rs` as WGSL kernels
(`crates/lapstack-web/src/depth.rs`): ring difference filter per frame
during the run, block-averaged to a working grid of 1/2^N the frame size
(*grid*, default N = 2) and kept as a quantised u16 slice; after the
collapse the slices are aggregated with the guided filter (fused luma as
guide), the peaks are tracked with sub-frame interpolation and confidence,
the confidence-weighted WLS with its robust reweight runs as fast-global-
smoother sweeps plus conjugate gradient on the device, and the map is
guided-upsampled to full resolution for saving. The viewer shows the
working-grid map under **Depth → Focus depth**; the 16-bit save writes the full-resolution
one. The browser map matches the native one on the same frames to
4 × 10⁻⁴ frames (the u16 quantisation), see `web/test.html`. The pass costs
no measurable wall time on a 25 × 45 MP run and keeps one u16 slice per
frame (5.7 MB at 45 MP and N = 2). The raw pyramid **winner map** (which
frame won at pyramid level *winner map level*, a free by-product of fusion)
is the second depth layer, under **Depth → Winner**; both take the Gray/Turbo LUT
and the slice overlay, can be compared against each other, and each has a
save button.

**Depth-map rendering (DFR)**: with *also render from the depth map*
checked in the Run button's ▾ menu (off by default; the button then reads
*Run LAP + DFR*) the run makes a second stacked image from the depth
map: every frame is decoded again, warped with the registration found during
the run, and blended in with weight `1 − |index − depth|` at each pixel, so
a pixel is the average of the one or two frames nearest its depth index. The
run then opens the **side by side** compare (two panes, one zoom/pan;
*swipe* is the divider mode) with the LAP result on the left and DFR on
the right; DFR has its own layer under Stack, can be the retouch target
(*paint into*) and the saved image (*result*). On the 25 × 45 MP stack the second pass
adds 6.7 s to a 27.7 s run.

**Retouch** (step 2, after a run): two panes with one zoom/pan —
the selected source frame on the left, the fused result on the right — and
a soft brush that copies the *aligned* source into the result. Drag on
either pane to paint (a circle shows the brush on both), shift+drag pans,
the wheel zooms, the panel's sliders (or `[` / `]`) set the brush size and
hardness, and ctrl+z / ctrl+shift+z undo and redo whole strokes. The frame
slider, filmstrip or ←/→ choose the source; since full frames are not kept
after the run, the chosen one is decoded again and re-warped with the
registration found during the run (about a second at 45 MP; the pane shows
its proxy until "loaded" appears). While dragging, the stroke is previewed
on the display copy; on release the worker applies it to the 16-bit master,
sends back the exact patch, and Save writes the retouched image. Undo
history is capped at ~600 MB of patches.

**Depth slice**: with `slice` on, the Depth view (gray or Turbo) paints a
60 % magenta band over the pixels the depth map assigns to the scrubbed
frame (rounded), so scrolling through the stack sweeps the band through the
depth map.

**Focus peaking** (Source view, same magenta band): during the run every
frame's level-1 region energy is area-averaged to proxy resolution and
kept. A pixel is painted for frame *i* when that frame's contrast is at
least *threshold* × the largest contrast any frame of the stack has at that
pixel (and that maximum is above a small noise floor), so scrubbing shows
the in-focus band sweep through the scene and each filmstrip entry shows its
% in focus — computed from the pyramid the fusion already built rather than
a separate contrast pass.

**Alignment on the GPU**: the streaming aligner chains each frame to the
previous warped one like the native one, runs Nelder-Mead on the CPU side
(an async transcription of lapstack-core's optimiser) and evaluates every
cost on WebGPU (Spline4x4 warp + DC-removed RMS partial sums, one small
readback per evaluation). It stops `align coarsen` levels short of full
resolution (default 2). The final warp of the 16-bit frame also runs on the
GPU.

Measured in headless Chrome on the RTX 3060 (`web/test/headless.mjs`):

| stack | browser | native (`lapstack --gpu --gpu-align`) |
|---|--:|--:|
| 25 × 8280×5520 16-bit TIFF, align coarsen 2 | **30 s** (≈1 s/frame: decode 0.4 s, align 0.4 s, fuse 0.1 s) | 40 s |
| 8 × 1024×768 crops, aligned | 2.0 s | – |

The browser is faster end-to-end because it streams: alignment never waits
for the whole stack to load, and the per-frame GPU work is the same. Fusion
output matches the native CPU path to one 8-bit step on 0.01 % of pixels
(`node web/test/headless.mjs`); the aligner produces the same kind of
transforms as the native one (scale 1.000 → 1.064 over the 25-frame sweep,
sub-pixel shifts), not bit-identical since the objective is evaluated in
FP32 on a slightly different pyramid.

Limits: every full-resolution plane is one storage buffer, so a frame needs
`3 × w × h × 4` bytes per buffer (550 MB at 45 MP) within the adapter's
`maxBufferSize` / `maxStorageBufferBindingSize`, and about 3.5 GB of GPU
memory in total at 45 MP; the app reports the adapter limits and refuses
frames that do not fit. Discrete desktop GPUs are fine; integrated GPUs will
want smaller frames.

Testing: `node web/test/headless.mjs [--align]` runs `web/test.html` in
headless Chrome over CDP (the test frames come from
`web/test/make-frames.sh`, the fused reference and the depth reference
`expected_dff.u16` from the native CLI at the same working grid; the page
reports the max / mean depth difference in frames); `PAGE=... SHOT=out.png`
screenshots any page after an autorun (`index.html?autorun=test/frames`,
`&render=1` also runs the depth-map rendering; `test.html?render=1` times
it and reports its mean difference to the pyramid image). `FILES="dir/*.tif"
PRE_EXPR="..." node web/test/headless.mjs` feeds real files to the page's
file input over CDP (no in-memory copies, so 100 × 45 MP stacks work) and
runs an expression, e.g. ticking *also render from the depth map* and
clicking Run. `web/serve.sh` sends `Cache-Control: no-store`; after a
rebuild a plain reload is enough (a hard reload alone can keep a cached
worker / WASM and the page then waits for messages the old worker never
sends).

## GPU acceleration (CUDA, optional)

```
cargo build --release --features gpu -p lapstack-cli
# Linux: needs libcuda (driver) + libnvrtc (toolkit) on the path, and nvidia_uvm loaded.
# Windows: nvcuda.dll ships with the display driver; drop the two DLLs from
# NVIDIA's cuda_nvrtc redist zip (bin/nvrtc64_120_0.dll + nvrtc-builtins) next
# to lapstack.exe. Match the nvrtc major.minor to the driver's CUDA version
# (nvidia-smi) or the PTX JIT will reject the kernels.
```

`cudarc` is built with `dynamic-loading`, so a binary built with the feature
still runs on a machine without CUDA; `--gpu` / `--gpu-align` fail at run
time with a message, everything else works.

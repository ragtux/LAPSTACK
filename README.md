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
depth (16-bit needs PNG or TIFF) and carries the **first frame's metadata**:
its EXIF (camera, lens, exposure, date, orientation, resolution — rebuilt
without the MakerNote and thumbnail, Software set to lapstack, the pixel
dimensions set to the result's), ICC profile and XMP packet, and, for a TIFF
without a profile, its white point and primaries (`meta.rs`; `--no-metadata`
turns it off). PNG carries them as eXIf, iCCP / cHRM and iTXt chunks, JPEG as
APP1 / APP2 segments, TIFF in IFD0 with the Exif and GPS sub-IFDs — TIFF
output is written by lapstack's own uncompressed writer for that.
`lapstack --help` lists every option.

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

**Slabs** (`--slabs SIZE[:OVERLAP]`, `--slab-dir DIR`): Zerene's slabbing
for the native path — after the result, every run of SIZE consecutive
frames overlapping by OVERLAP (default 2) is fused on its own with the same
settings over the same aligned, equalised frames, cropped like the result,
and written as it is made to DIR (default `<output stem>_slabs`) in the
output's format with the same metadata, as `slab_01_000-009.tif` and so on
(0-based frame indices). Slabs are thick planes of focus to retouch from in
another editor; the browser app makes them on demand instead (below). In the
streaming path each slab streams its frames from disk again. Where a warped frame does not reach, the warp
repeats its edge, so the output (image, depth and confidence maps) is
**cropped to the largest rectangle every frame covers** with real pixels
(`align::common_area`: each frame's sound area is a convex quad, cut per pixel
row into an interval, intersected over frames, and the best rectangle over
consecutive rows is taken; the warp's 2 px interpolation support is kept
out); `--no-crop` keeps the full frame.

**Brightness** (`brightness.rs`): flash recycling, mains-powered lights and
a shutter that is not quite repeatable make frames differ in exposure by a
percent or two, and the region-energy rule sees it (energy grows with the
square of the gain, so a brighter frame wins ties it should not and the
seams between winners show as patches). Every frame is brought to frame 0's
brightness by one gain per channel — the ratio of the two frames' channel
*means* over the pixels the frame's warp covers, since a blur leaves a
mean alone while a pixel-wise fit would slope towards zero with the defocus;
per channel, so a light that flickers in colour is corrected too. Gains are
clamped to [1/4, 4] and logged; `--no-brightness` turns it off. The browser
app does the same on the GPU (block means of frame 0 kept, one small readback
per frame), *equalise brightness* in the parameter panel, and each filmstrip
entry shows its gain.

**Synthetic stereo and rocking** (`view.rs`; `--stereo PCT[:LAYOUT]`,
`--rocking PCT[:N]`, `--far-first`): the depth map makes the result a relief,
and a view from the side is that relief sheared — every pixel slides
sideways in proportion to its depth. Zerene gets the same picture by
shifting each frame by its index before stacking; Helicon projects the
textured 3D model it builds from the depth map; lapstack shears the stacked
image and its depth map in one pass, with Zerene's parameter: the far end of
the stack moves PCT % of the width relative to the near end (their "maximum
X shift"; ±3 % suits most subjects — for a scene d deep and w wide a viewing
angle a is tan(a)·d/w), the middle of the stack staying put. The shear is a
forward warp per row: consecutive samples less than 2 px apart in the view
form a patch of surface, rasterised with a nearness test so a near edge
slides over the background; a larger gap is a depth discontinuity, and the
hole it opens is filled from the farther side (the background shows
through, the foreground is not stretched); pixels are sampled linearly.
`--stereo` writes the views from the left and the right (−PCT / +PCT) as
`<stem>_stereo.<ext>` — side by side for parallel viewing (`sbs`),
cross-eyed (`cross`) or as a red–cyan anaglyph; `--rocking` writes N views
(default 24) whose shift sweeps ±PCT in one sine cycle to
`<stem>_rocking/view_NN.<ext>` (join them with ffmpeg or ImageMagick). Which
end is near decides who wins where surfaces overlap: frame 0 is taken as the
near end (the focus went front to back); `--far-first` says otherwise — the
symptom of the wrong choice is a relief that looks inside out.

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
./web/build.sh          # wasm32 build + wasm-bindgen glue into web/pkg (+ web/pkg-cc)
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

The UI follows the workflow as two steps in the top bar:
**1 Stack** (add frames, set parameters, run, inspect and retouch the
result) and **2 Save** (a file list of everything the run can produce, file
names, animations, content credentials). Keys 1/2 switch steps.

*Stack* is the workbench: a filmstrip (thumbnails arrive as frames are
added, with each frame's registration once aligned), a parameter panel (all
`lapstack` knobs, persisted in localStorage), Run/Cancel with progress and a
log, and a viewer whose header is a segmented **Source / Stack / Depth**
control with a second-level control for the group's layers — **LAP / DFR**
under Stack (DFR only when the depth-map render ran), **Focus depth / In focus**
under Depth — followed by controls that only show for what is on screen: on
Source a **peaking** toggle with a threshold stepper; on Focus depth a
Gray/Turbo LUT and a **slice** toggle; a frame slider whenever the shown
layers depend on a frame; and on the right a **compare** toggle whose "vs" dropdown lists
the other layers (the divider is draggable, `flip` or space swaps sides);
a **Retouch** button in the top-right corner whenever LAP or DFR is on
screen (see below).
Scroll-zoom at the cursor, drag pan, double-click fit/100 %, wheel / ←/→
scrub (shift = 10; ctrl+wheel zooms on scrubbable views). Drag-and-drop
works. Added frames are decoded and downscaled in the worker straight away
(the browser cannot decode TIFF itself), so the filmstrip and Source view
are populated before a run; after the run the Source view shows the aligned
screen-resolution proxies produced during it — full frames are not retained.

*Save* takes over the whole window: one card with a file list (a checkbox,
a thumbnail and the exact file name per output) and the settings. Stills:
the (retouched) LAP and DFR images as PNG at the input bit depth, 8-bit PNG
or JPEG with a quality slider; the depth map as an 8-bit gray PNG (min–max
scaled) or a 16-bit PNG with a fixed scale (65535 = last frame, the same
encoding as the CLI's `--depth-raw`); the winner map. The stacked images
carry the **first frame's EXIF, ICC profile and XMP** like the CLI's output
(the run reads them from the first frame's bytes; the card's *Metadata*
section says what was found and has the switch). Every file, animations
included, is **cropped to the area all aligned frames cover** like the CLI's
output (the run reports the window; the card's *Crop* section shows its size
and has the switch, and the viewer shows the window as the bright part of
the image while it is on). Animations, as GIF:
**Focus depth** in Turbo with the slice sweeping through the frames, the
**In focus** sweep, the aligned **Source** frames under their peaking
band, each frame rendered like the viewer shows it (the Source and In focus
frames are decoded and re-aligned at full resolution by the engine, then
scaled to the chosen *long edge*), and **Rocking**, the stacked image
rocking from side to side; *every Nth frame*, speed and loop
(back and forth or forward) are settings, and the card estimates the size —
a full-resolution GIF of a long 45 MP stack runs to gigabytes, so pick a
long edge or a frame step for those. Frames are quantised (median cut,
Floyd–Steinberg) and LZW-encoded in the worker (`crates/lapstack-web/src/gif.rs`),
the bytes streaming back so the file never sits in wasm memory whole.
**Stereo and rocking** are the CLI's synthetic stereo (above) in the
browser: the card's section has the stereo shift, the pair's layout
(parallel, cross-eyed, anaglyph), the rocking shift and frames per cycle,
the image to shear (LAP, or DFR when it was rendered) and the near-end
switch. The engine cuts the master and the full-resolution depth map to the
crop and shrinks them to the view size once (`view_prepare`), then shears a
view per call (`view_rgba` for the GIF's frames at the animation size,
`view_stereo` for the pair, saved at the crop's size in the chosen format and
bit depth with the metadata). The **Stereo pair** still and the **Rocking**
GIF are rows of the file list like the others. The section's *method*
picks the shear or the **refold**, Zerene's own way: the frames are read
again and the stack is fused once per view with every frame shifted
sideways in proportion to its index (`Refold` in `lapstack-web/src/lib.rs`:
`refold_begin`, one `refold_pass_begin` / `refold_push` per frame /
`refold_pass_finish` per batch of views, `refold_view` / `refold_stereo`,
`refold_end`). Each view is then a real LAP stack — no depth map is
involved, so hair and bristles come out as they would from a camera moved
to the side — at the cost of one pass over the frames per batch. The stereo
pair is folded at full resolution (its two accumulators, 1 GB at 45 MP, are
one batch: the run's own accumulator serves as the first); a rocking
sequence is folded at the animation's size, each warped frame
block-averaged by an integer factor and shifted by a whole number of source
pixels, so a 24-view cycle at 1600 px is one batch and one pass. Views are
batched to a 1 GB budget of accumulator memory. The refold has the method's
halo: beside a near object the far frames win (the background is sharp in
them) and carry the object's defocused copy displaced by the shift, so a
soft ghost stands next to it, wider with a larger shift and a deeper scene;
the shear has no halo but only one surface.
File names are built from tokens joined with `_`, lower case: `lapstack`,
the EXIF date/time of the first frame (read from the JPEG APP1 / TIFF /
PNG eXIf structure), a date/time found in the first frame's name, the
current date/time, a custom text, and the layer name (`lap`, `dfr`, `stereo`,
`depth`, `depth16`, `winner`, `depth-slice`, `infocus`, `peaking`, `rocking`). The EXIF date
falls back to the XMP packet's CreateDate, which is all a raw converter's
TIFF may have.
**Content credentials** ([C2PA](https://contentcredentials.org/)) can be
attached to every saved file: a self-signed ES256 certificate is generated
in the browser for the *signed as* name (kept in localStorage, the key never
leaves the browser) and a manifest with a `c2pa.created` action
(`compositeCapture`) and an `org.lapstack.stack` assertion (frames, fusion
parameters, retouch strokes) is embedded and signed. Verifiers show the
manifest, assertions and content hash as valid and the signer as unknown
(a self-signed certificate is on no trust list). The c2pa crate is large, so
it is a second wasm module (`crates/lapstack-cc` → `web/pkg-cc`) that the
worker loads on the first signed save.

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
is not shown as a layer; it drives the ctrl+G pixel lookup and has its own
save button.

**In focus** (the second Depth layer) is the scrubbed frame's own pixels,
in colour, showing only the parts of it the result uses — Helicon Focus's
"source map": every pixel is darkened by how far, in frames, the depth map
puts it from that frame (full brightness within ±0.5 frames, 8 % beyond
±0.75, linear between), so the plane of focus stands out with a crisp edge
and scrubbing sweeps it through the scene. The out-of-focus part keeps its
outlines: the local contrast of its luminance (|luma − box blur|, radius
≈ width/1000) is added back in grey, so blurred edges and fibres read as
light lines against the dimmed colour. The page builds a preview from the
proxy and the depth map's working grid straight away; the engine then
returns the full-resolution aligned frame rendered the same way from the
guided-upsampled depth map (`source_focus`, ~1 s at 45 MP, cached with the
full-res sources), which replaces the preview. It can be compared against any layer, LAP in
particular.

**Depth-map rendering (DFR)**: with *also render from the depth map*
checked in the Run button's ▾ menu (off by default; the button then reads
*Run LAP + DFR*) the run makes a second stacked image from the depth
map: every frame is decoded again, warped with the registration found during
the run, and blended in with weight `1 − |index − depth|` at each pixel, so
a pixel is the average of the one or two frames nearest its depth index. The
run then opens the **side by side** compare (two panes, one zoom/pan;
*swipe* is the divider mode) with the LAP result on the left and DFR on
the right; DFR has its own layer under Stack, is the retouch target while
it is the shown layer, and can be the saved image (*result*). On the 25 × 45 MP stack the second pass
adds 6.7 s to a 27.7 s run.

*From slabs* in the same menu is Zerene's slabbing as a rendering mode:
the stack is cut into slabs of *slab size* frames overlapping by
*overlap* (defaults 10 and 2), each slab is fused on its own with the
run's settings (LAP within the slab, the same fold as the retouch slab)
and DFR blends the slab images instead of the frames, with weight
`1 − dist(depth, [lo, hi])` per pixel — full weight for every slab whose
frames hold the pixel's depth, a one-frame fall-off outside, so a frame is
the special case `lo = hi`. Fine detail and crossing structures come from
LAP within a slab, while the far-out-of-focus frames that build up noise
and halos over a whole stack never blend in, since a slab's frames all lie
near its depth. Frames in the overlaps are decoded once per slab, and the
blend keeps its own accumulator on the GPU (4 floats per pixel, freed
afterwards) while the run's fuses the slabs. A single slab of the whole
stack reproduces the LAP result exactly, which the test page checks.

**Retouch** is a mode of the Stack step, not a step of its own: the
**Retouch** button in the viewer's top-right corner (or `R`) is there
whenever LAP or DFR is on screen. It turns the view into a side-by-side
compare with one zoom/pan — the stacked image on the left (LAP or DFR,
whichever is selected: that is the paint target), the brush source on the
right — and shows the brush controls above the run parameters in the
panel. The source is the scrubbed source frame; a **slab**; or, after a
*Run LAP + DFR*, the **other stacked result** (the panel's *Source* section,
or `S` to cycle): DFR while LAP is painted, LAP while DFR is, so the
pyramid's fine detail can be brushed into the depth-map rendering and its
smooth areas back into the pyramid image, the way Zerene retouches PMax into
DMap. A slab is Zerene's slab made on demand: the scrubbed frame and its
neighbours (*slab ± frames*, default 5) fused on their own, with the run's
registration, brightness gains and fusion settings, so the brush copies a
thick plane of focus instead of one frame's sliver of it — on a deep stack
that is what most retouching paints from. The engine fuses it when the scrub
settles (each frame is decoded again, ~0.6 s per 45 MP frame), reusing the
run's accumulator on the GPU, keeps one slab at a time, and drops a build the
scrub has moved away from between frames; the filmstrip tints the slab's
frames, the right pane's label says how far the build is, and once a slab
exists it is also offered as a compare layer. A soft brush copies
the *aligned* source into the stacked image, and it shows its work before
the button goes down: wherever the cursor is, the
dab a click would lay down is composited into the paint pane, under the
circle that marks the brush edge and its hard core (the circle shows on both
panes, the preview on the paint pane, built at the pane's own scale and cut
to it so a brush wider than the window costs no more than the window). The
preview lives on the display canvas alone — no pixel is committed until a
stroke is painted — and it stands aside for one pointer move after an undo or
redo, so a stroke reverted under the cursor is seen going. Drag on either
pane to paint, shift+drag pans, ctrl+wheel zooms, the panel's sliders (or `[`
/ `]`) set the brush size and hardness, and ctrl+z / ctrl+shift+z undo and
redo whole strokes.
With a frame as the source, the wheel, frame slider, filmstrip or ←/→ choose
it (the wheel goes on zooming while a stroke is being dragged, so the frame
cannot change under the brush mid-stroke); since full frames are not kept
after the run, the chosen one is decoded again and re-warped with the
registration found during the run (about a second at 45 MP; the pane label
says *loading full res…* until then). The other result is on hand at once,
and the wheel zooms. The button again, `Esc`,
unticking *vs*, or leaving the Stack layers ends the mode and brings back
the compare that was open before it. While dragging, the stroke is previewed
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

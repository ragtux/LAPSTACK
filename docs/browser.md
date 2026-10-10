# lapstack in the browser (`web/`, WebGPU + WASM)

`crates/lapstack-web` + `web/` run the whole lapstack pipeline in a browser
tab: frames are decoded in WASM (PNG/JPEG/TIFF, 8- and 16-bit, via the `image`
crate; camera raws developed via rawler, `engine.md` *Camera raw input*),
aligned and fused on **WebGPU** (`shaders.wgsl`, the same kernels as
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
`lapstack` knobs, persisted in localStorage — *rotate frames* and *draft*
among them: the CLI's `--rotate` and `--draft`, the thumbnails turned with
the frames, the Run button reading *draft ÷4*; every section's heading
collapses the rows under it and carries a ↺ that puts that section's
settings back to the defaults, lit only while the section holds something
other than them, so the panel also says which groups have been changed),
Run/Cancel with progress and a log, and a viewer whose header is a segmented **Source / Stack / Depth**
control with a second-level control for the group's layers — **LAP / DFR**
under Stack (DFR only when the depth-map render ran), **Focus depth /
Confidence / In focus** under Depth — followed by controls that only show
for what is on screen: on Source a **peaking** toggle with a threshold
stepper; on Focus depth and Confidence a
Gray/Turbo LUT and a **slice** toggle; a frame slider whenever the shown
layers depend on a frame; and on the right a **compare** toggle whose "vs" dropdown lists
the other layers (the divider is draggable, `flip` or space swaps sides);
a **Retouch** button in the top-right corner whenever LAP or DFR is on
screen (see below), and beside it **Crop** (`C`), which arms a drag on the
stacked image: the rectangle drawn becomes the run's crop window — the
CLI's `--crop`, cut to the area every frame covers by the engine
(`crop_set`), shown as the bright window, applied to every file the Save
step writes, kept by a project file and cleared with the ✕ beside the button
or in the Save step's *Crop* section (which also has *set a window…*).
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
encoding as the CLI's `--depth-raw`); the confidence map as a 16-bit PNG
(65535 = 1, as `--save-conf`); the winner map. The stacked images
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
long edge or a frame step for those. Frames are quantized (median cut,
Floyd–Steinberg) and LZW-encoded in the worker (`crates/lapstack-web/src/gif.rs`),
the bytes streaming back so the file never sits in wasm memory whole.
The section's *format* makes the same animations **videos** instead —
**MP4 (H.264)** or **WebM (VP9)**: each frame is
drawn as for the GIF, encoded by the browser's own encoder (WebCodecs
`VideoEncoder`: H.264 High, Main or Constrained Baseline at the level the
size needs, or VP9 profile 0, VP8 failing that — whichever the browser
supports at that size, so a 4K rocking that no encoder takes asks for a
smaller long edge) and written into the container by the page
(`web/mux.js`, a plain MP4 with its sample table and a WebM of
SimpleBlocks, both from scratch). *Video quality* sets the bit rate in
bits per pixel per second (0.2 / 0.1 / 0.05); the size is made even for the
4:2:0 chroma; a keyframe goes in every two seconds; and the rocking is one
sine cycle, so a player's loop is seamless. Videos carry no content
credentials. A browser without WebCodecs (Firefox before 130, older Safari)
gets the GIF alone.
**Stereo and rocking** are the CLI's synthetic stereo (`engine.md`, *Synthetic stereo and rocking*) in the
browser: the card's section has the stereo shift, the pair's layout
(parallel, cross-eyed, anaglyph), the rocking shift and frames per cycle,
the image to shear (LAP, or DFR when it was rendered) and the *near end*:
frame 0 the nearest, the farthest, or *what the frames say* — the EXIF
SubjectDistance of the first and last frames, read by the page's own EXIF
reader (`subjectDistance`; a MakerNote's distance is beyond it, so a stack
whose camera keeps it there falls back to frame 0 near, and the section's
line says which cue, if any, decided). The engine cuts the master and the full-resolution depth map to the
crop and shrinks them to the view size once (`view_prepare`), then shears a
view per call (`view_rgba` for the GIF's frames at the animation size,
`view_stereo` for the pair, saved at the crop's size in the chosen format and
bit depth with the metadata). The **Stereo pair** still and the **Rocking**
GIF are rows of the file list like the others. The section's *method*
picks the shear or the **refold**: the frames are read
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
**3D model**: the CLI's `--mesh` in the browser, a row of the file list
(`3d`): the card's section has the format (GLB, one file with the texture
inside; OBJ + MTL + texture image, three files, which the browser may ask to
allow; STL, the relief alone), the relief as a percentage of the width, the
mesh's vertices along the long edge and the texture's long edge, with the
vertex and triangle counts and a size estimate; the image (LAP or DFR) and
the near end are the stereo section's, the texture is 8-bit in the card's
image format (JPEG at its quality, or PNG), and the model is cropped like
the other files. The row's thumbnail is the stacked image lit as the relief
would be. The engine builds it from the full-resolution depth map and the
master (`mesh` in `lapstack-web/src/lib.rs`, core `mesh.rs`); no content
credentials are attached to 3D files.
File names are built from tokens joined with `_`, lower case: `lapstack`,
the EXIF date/time of the first frame (read from the JPEG APP1 / TIFF /
PNG eXIf structure), a date/time found in the first frame's name, the
current date/time, a custom text, and the layer name (`lap`, `dfr`, `stereo`,
`3d`, `depth`, `depth16`, `winner`, `depth-slice`, `infocus`, `peaking`, `rocking`). The EXIF date
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
(*grid*, default N = 2) and kept as a quantized u16 slice; after the
collapse the slices are aggregated with the guided filter (fused luma as
guide), the peaks are tracked with sub-frame interpolation and confidence,
the confidence-weighted WLS with its robust reweight runs as fast-global-
smoother sweeps plus the multigrid-preconditioned conjugate gradient on
the device (the residual read back every 8 iterations, so the solve stops
when it has converged instead of at the iteration cap), and the map is
guided-upsampled to full resolution for saving. The viewer shows the
working-grid map under **Depth → Focus depth** and its confidence under
**Depth → Confidence** — the peak-ratio confidence of each pixel's focus
profile, normalized so the 90th percentile is 1: the weight the WLS gave the
pixel's own depth, so the dark parts are where the map was filled in from
the neighbors (flat, noisy or ambiguous areas); the ctrl+G readout logs
both values for the clicked pixel. The 16-bit saves write the full-resolution
depth and the confidence bilinearly upsampled (65535 = 1, as `--save-conf`).
The browser map matches the native one on the same frames to
4 × 10⁻⁴ frames (the u16 quantization) and the confidence to 3 × 10⁻³ on
average (0.1 % of the pixels, near-ties between two peaks, differ by
more), see `web/test.html`. The pass costs
no measurable wall time on a 25 × 45 MP run and keeps one u16 slice per
frame (5.7 MB at 45 MP and N = 2). The raw pyramid **winner map** (which
frame won at pyramid level *winner map level*, a free by-product of fusion)
is not shown as a layer; it drives the ctrl+G pixel lookup and has its own
save button.

**In focus** (the second Depth layer) is the scrubbed frame's own pixels,
in color, showing only the parts of it the result uses — a source map:
every pixel is darkened by how far, in frames, the depth map
puts it from that frame (full brightness within ±0.5 frames, 8 % beyond
±0.75, linear between), so the plane of focus stands out with a crisp edge
and scrubbing sweeps it through the scene. The out-of-focus part keeps its
outlines: the local contrast of its luminance (|luma − box blur|, radius
≈ width/1000) is added back in gray, so blurred edges and fibers read as
light lines against the dimmed color. The page builds a preview from the
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

*From slabs* in the same menu is slabbing as a rendering mode:
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
afterward) while the run's fuses the slabs. A single slab of the whole
stack reproduces the LAP result exactly, which the test page checks.

**Frame list**: the filmstrip is editable. Its head counts the frames in the
run and the excluded ones, with *reverse* (the stack was shot back to front)
and, once any frame is excluded, *include all*. Hovering a thumb shows its
tools — exclude ⊘ / include ↩, move ▲ ▼, remove ✕ — and a thumb can be
dragged to a new place; `X` excludes the selected frame, `Delete` removes
it, alt+↑/↓ moves it, and shift on a thumb's exclude or remove takes every
frame from the last one marked through that one. An excluded frame stays in
the list at its place, dashed and dimmed, and is left out of the run, the
split into stacks, the name tokens and the save; internally it leaves
`st.files` (which the run, the split and everything after the run index) for
a side list anchored to its position (`st.off`), and the filmstrip merges the
two back in order (`frameList` / `setFrameList` in `app.js`, every edit going
through `editFrames`). An edit after a run drops the result — the result's
frame indices are the list's — and the status bar and log say so; the
aligned proxies stay as the thumbs. A batch's stack in hand cannot be edited
(*all frames* first), nor can the list while a run or save is going. After a
run each thumb says what share of the detail the frame won (the CLI's line,
from the same winner map, read back with its best-energy plane at the end of
the fold), in red under the threshold beside the head's **cull** button,
which excludes every frame under it — the CLI's `--cull`, done by hand: the
result is dropped as after any edit, and the next run goes without them. The
threshold is kept in the browser.

**WAV** (the same ▾ menu, *WAV (weighted average)*, with *contrast power*,
*weight smoothing*, *noise gate %* and *edge-aware weights*, the box mean
in place of the guided filter when it is off): the CLI's `--wav` in the browser — a third stacked
image, made in one more pass over the frames after LAP (and DFR): each frame
is decoded again, warped with the run's registration, its contrast taken on
the GPU with the depth pass's focus measure on the working grid
(box-smoothed, less the gate's multiple of the cell's noise floor the depth
pass kept, raised to the power, smoothed again: `record_weight` in
`lapstack-web/src/depth.rs`) and blended in with that weight (`wav_acc`;
`render_push` in its wav mode, the same accumulator as DFR's). WAV is a
layer of the Stack group, the retouch's third target — the brush's *Result*
source lists the run's other images and the kept results together — a row of
the Save step, an image for the stereo, rocking and 3D model, and a kept
result like the others.

**Linear DNG** (the same ▾ menu): the CLI's `-o stacked.dng` in the browser.
With the box ticked the run develops each raw to its camera space and fuses
in the look of frame 0's white balance and matrix (`Engine::decode`, the
same `dng.rs` as the CLI; a frame that is not a raw is the look already), the
Run button reads *→ DNG*, the log says whose space the run is in, and the
Save step's *format* offers **DNG, linear (camera space)** — disabled, with
a line saying why, after a run made without the box. The stacked images, the
kept results of such a run and the stereo pair (sheared or refolded) are
then written as linear DNGs by `encode` / `view_stereo` / `refold_stereo`
through the core writer, with the first frame's EXIF and XMP; the maps and
animations are what they always are. The browser's frames are 16-bit, so a
highlight past the balanced white is clipped here where the CLI keeps it;
the color matrices, neutral and metadata are the same. Checked headless on
five NEFs: the DNG darktable develops from the browser matches the CLI's.

**Batch** (the same ▾ menu, *split into stacks*): the CLI's `--split` for
the browser. *Add folder…* takes a whole folder (its subfolders too; a
dropped folder works the same), files sort by their path so a folder's
frames stay together, and the rule cuts the list into stacks — every N
frames, at every pause in the capture times longer than S seconds, or one
per folder. The filmstrip shows the split at once as a header over each
stack (its frames, capture times and the pause before it in the tooltip),
the menu says what it makes (*3 stacks of 25–31 frames · frames 1.0 s
apart, the longest pause 48 s*, so the threshold can be tuned by eye), and
the Run button reads *Run LAP ×3*. The capture time comes from EXIF
DateTimeOriginal, else DateTimeDigitized, DateTime or the XMP CreateDate,
read through a chunked reader so a 274 MB TIFF whose IFD trails the pixels
costs a few 64 KB reads; a frame without one uses its file date and the
menu says so. The batch runs the stacks in turn through the ordinary run
— each becomes the frame list in hand — and as each finishes saves the
files ticked in the Save step, with that step's names and settings (the
*stack number* token, `s01`, goes into the names unless another per-stack
token — the EXIF date, the date in the name, the first frame's name — is
on), to a folder chosen with *save to folder…* (the File System Access API,
Chrome and Edge; the Save step's *folder…* is the same choice) or as
downloads. A stack that fails is logged and the batch goes on; the banner
over the filmstrip counts saved and failed, the last stack stays on screen
with its result, and *all frames* brings every frame back with each
stack's status in its header. The headless harness exercises it with
`PAGE=index.html` and a `PRE_EXPR` that sets the rule and a
`window.__saveHook`, waiting on `window.__batch_done`.

**Retouch** is a mode of the Stack step, not a step of its own: the
**Retouch** button in the viewer's top-right corner (or `R`) is there
whenever LAP or DFR is on screen. It turns the view into a side-by-side
compare with one zoom/pan — the stacked image on the left (LAP or DFR,
whichever is selected: that is the paint target), the brush source on the
right — and shows the brush controls above the run parameters in the
panel. The source is the scrubbed source frame; a **slab**; a **kept result** of
the run's size (below); or, after a
*Run LAP + DFR*, the **other stacked result** (the panel's *Source* section,
or `S` to cycle): DFR while LAP is painted, LAP while DFR is, so the
pyramid's fine detail can be brushed into the depth-map rendering and its
smooth areas back into the pyramid image. A slab is fused on demand: the
scrubbed frame and its
neighbors (*slab ± frames*, default 5) fused on their own, with the run's
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
/ `]` and alt+wheel for the size, alt+shift+wheel for the hardness) set the brush size and hardness, and ctrl+z / ctrl+shift+z undo and
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

<p align="center">
  <img src="media/retouch.gif" width="960" alt="The retouch mode: the stacked image on the left with a dark registration ghost on a chip's edge, a slab of the frames around the sharpest one on the right, the brush circle on both panes, and the ghost painted over from the slab">
</p>

A retouch in the browser app, reduced for this page: ctrl+G and a click
on a spot beside the flaw jump to the frame that won it, `R` opens
retouch, `S` makes the source a slab around that frame, alt+wheel sets
the brush size and alt+shift+wheel its hardness, the brush previews under
the cursor, and a click or drag paints the slab over the flaw.

**Results** (the panel's *Results* section): a result stays on past its run,
kept in a list with the other runs' output. When the next run
starts — or the frame list changes under the result — its LAP and DFR images
become *kept results*: the engine takes the 16-bit masters out of the run
(`keep` in `lapstack-web/src/lib.rs`), the page keeps the display copies, and
each is named after its run (*run 3 · LAP*, its frames, settings and time in
the tooltip). A kept result is a layer of the **Stack** group, so two runs
with different settings can be compared with *vs*, swipe or side by side; a
kept result of the run's size is a brush source for the retouch (*Result* in
the brush's *Source* section, `S` cycles to it), so the retouch can paint
from any saved output; and each is a row of the Save step, cut to its own
crop window and carrying its own metadata (`encode` takes `kept:ID`). *Load
result…* takes a saved result — an earlier session's, the CLI's, another
program's — as one more (`keep_file`: decoded in the engine, its EXIF / ICC /
XMP kept, no crop); a file of another size can be viewed and compared but
not brushed from. The masters are held in wasm memory, so *results kept MB*
in the section bounds them (default 1536 MB; a 45 MP image is 270 MB; 0
keeps none): over it, the oldest go first, the newest always stays, and
each row's ✕ lets one go by hand. Frames apart, kept results are all a
session holds — *Clear* drops them — and a batch keeps nothing, since its
stacks are saved as they finish (the result on screen when a batch starts
is kept). With no run's result on hand the Stack group and the Save step
still open on the kept ones.

**Project files** (*Open project…* / *Save project…* in the toolbar): a
project is one JSON file, `<name>.lapstack.json`, holding what it takes to
come back to a session — the frames (names, paths, sizes, which are
excluded), every setting of the panel and the Save step, the last run's
registration (each frame's shift, scale and rotation, as the filmstrip
shows them) and its retouch strokes (target, source and dabs, kept in step
with undo and redo). The images are not in it: a result is saved as a file,
or loaded again as a kept result. Opening a project clears the session and
puts placeholders in the filmstrip, in the project's order; the frames come
from the folders the browser remembers — *Add folder…* and a dropped folder
go through the File System Access API in Chrome and Edge, and the folder's
handle is stored in the browser (IndexedDB) under an id the project names,
so the banner over the filmstrip offers *open folder "shoot"* and, once
reading is allowed, the frames are read from it — or they are added by
hand and matched by name and size (files not in the project are left out).
*Run* then makes the stack again with the registration as it was: the
frames' transforms go to the engine (`push` takes them) and the alignment
search, most of a frame's time, is skipped (*reuse the project's
registration* in the Run menu turns that off; a changed frame list turns it
off by itself), and the retouch strokes are painted again in order, each
from its source brought back — a frame decoded and warped, a slab fused,
the other stacked image, a kept result of the same label if one is loaded —
through the worker's `replay`, so the retouched image comes back pixel for
pixel (`web/test/headless.mjs` with a project script checks it). A stroke
whose source is not on hand is left out and the log says why. Saving again
after the run writes this session's run and strokes; saving before it keeps
the project's own. **Copy command** beside it puts on the clipboard (and in
the log) the `lapstack` command that stacks the frames in hand with the
panel's and the Save step's settings — the same table of keys to flags that
`lapstack --config` applies to a project file (`cliCommand` in `app.js` and
`project.rs` are kept in step) — with the frames by their names, to run in
their folder.

**Scale bar and text** (the panel's *Scale bar and text* section; the CLI's
`--scale-bar` and `--text`, above): the *scale bar* tick, the calibration in
µm per pixel (empty: the bar is labeled in pixels of the frames), the bar's
length (empty = the 1-2-5 value nearest a fifth of the width), a caption with
the same tokens (`{date}`, `{time}`, `{frames}`, `{first}`, `{n}`; `\n` breaks
a line), the corners, the size, the color and the style. The engine renders
the overlay for the saved file's size (`overlay_patches`, core `overlay.rs`)
and returns RGBA patches over the corners it occupies, which the viewer draws
over the Stack layers — LAP, DFR, WAV and the kept results — inside the crop
window, so what is on screen is what the file will carry (not while
retouching: the brush preview lives there); the section's line says what the
bar came to (*scale bar 100 µm = 308 px at 0.325 µm/px, bottom right*), and
after a run the calibration the first frame carries (ImageJ, OME-TIFF) is
offered with a *use* button. Saving burns the same overlay into the 16-bit
master in the engine (`encode`, `view_stereo` and `refold_stereo` take it as
JSON) — the stacked images, the kept results, the stereo pair (once per
view) — and the page draws it, rendered at the animation's size, over every
frame of the GIFs and videos; the Save step's *Scale bar and text* switch
leaves it out of the files (and off the screen) without losing the
settings. The settings are part of the panel's parameters, so they persist
and travel with a project file.

**Depth slice**: with `slice` on, the Depth view (gray or Turbo) paints a
60 % magenta band over the pixels the depth map assigns to the scrubbed
frame (rounded), so scrolling through the stack sweeps the band through the
depth map.

**Halo control** (the panel's *halo control* stepper, the CLI's
`--halo-control`; off by default): the fold selects only up to the depth
level, turns that level's region energy into the weights on the GPU
(`wgt`), REDUCEs them down the pyramid and folds every coarser level and
the residual as Σ w·L and Σ w (`wacc`), normalized at the collapse
(`wnorm`); no residual crosses to the host. Slabs and refolds use the same
fold, so a slab or a stereo view is halo-controlled like the run. The
browser result matches the native one as the plain fold does — a handful
of pixels one 8-bit count off
(`test.html?halo=2&expected=test/expected_halo2.png&expected_dff=&expected_conf=`,
against `make-frames.sh`'s `expected_halo2.png`).

**Focus peaking** (Source view, same magenta band): during the run every
frame's level-1 region energy is area-averaged to proxy resolution and
kept. A pixel is painted for frame *i* when that frame's contrast is at
least *threshold* × the largest contrast any frame of the stack has at that
pixel (and that maximum is above a small noise floor), so scrubbing shows
the in-focus band sweep through the scene and each filmstrip entry shows its
% in focus — computed from the pyramid the fusion already built rather than
a separate contrast pass. On a thumbnail the mask is box-averaged down, and
the in-focus fraction of each thumbnail pixel is scaled by that frame's
densest ones, so the band keeps its strength however long the stack (and
however thin each frame's band) instead of averaging away to nothing.

**Alignment on the GPU**: the streaming aligner chains each frame to the
previous warped one like the native one, runs Nelder-Mead on the CPU side
(an async transcription of lapstack-core's optimiser, on its per-level
schedule) and evaluates every cost on WebGPU (Spline4x4 warp + DC-removed
RMS partial sums), over the panel's *model* (the CLI's `--align-model`:
similarity, affine, projective). A readback is the expensive part in the
browser — a round trip through Chrome's GPU process — so the points an
iteration may need (the reflection, and the expansion or the contraction
that follows it) are dispatched together and read back at once, the
decisions made from the values as the sequential search makes them; at a
level whose cost dispatch outweighs a round trip (above 8 MP) the reflection
goes alone. It stops `align coarsen` levels short of full resolution
(default 2). A project saved with a run
records each frame's transform with all eight terms; older projects with
four are read as similarities. The final warp of the 16-bit frame also runs on the
GPU, with the kernel of the panel's *interpolation* (the CLI's
`--interpolation`: nearest, bilinear, bicubic, spline 4×4, spline 6×6,
Lanczos 3); a frame brought back after the run (the Source view, a slab, the
depth-map render, a refold) is warped with the same kernel.

Measured in headless Chrome on the RTX 3060 (`web/test/headless.mjs`):

| stack | browser | native (`lapstack --gpu --gpu-align`) |
|---|--:|--:|
| 25 × 8280×5520 16-bit TIFF, align coarsen 2 | **16 s** (≈0.65 s/frame: decode 0.4 s, align 0.2 s, fuse 0.1 s; 30 s before the search's per-level schedule and batched readbacks) | 6.5 s (align + fuse + focus measure, the frames on the device; the depth pass is another 29 s) |
| 8 × 1024×768 crops, aligned | 1.2 s (2.0 s before) | – |

Both stream now, and the per-frame GPU work is the same. Fusion
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
clicking Run. `index.html?debug=1` sends the worker's console errors (a wasm panic, a
WebGPU validation error) to the page's log, where the harness sees them.
`web/serve.sh` sends `Cache-Control: no-store`; after a
rebuild a plain reload is enough (a hard reload alone can keep a cached
worker / WASM and the page then waits for messages the old worker never
sends).

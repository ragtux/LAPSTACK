# lapstack — Laplacian-pyramid focus stacking, native and in the browser

`lapstack` fuses a focus-bracketed series of photographs into one all-in-focus
image and a dense depth map. It is a from-scratch implementation of focus
stacking on the Laplacian pyramid, written from the papers cited in
`docs/README.md`:

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
crates/lapstack-core   library: pyramid, fusion, depth from focus, aligner, dust map, stereo views, 3D model, batch splitting, I/O, linear DNG, frame preparation, CUDA path, pooling allocator
crates/lapstack-cli    `lapstack` command-line tool
crates/lapstack-web    wasm32 + WebGPU engine for the browser app
crates/lapstack-raw    the camera raw decoder (rawler, LGPL) as a module of its own: a shared library the CLI loads at run time, a wasm module the browser app loads beside its engine
web/                   the browser app (static files) and its headless test
docs/                  the papers the algorithm is written from, cited
lightroom/             the Lightroom Classic plugin (the CLI as an export target and a Library menu item)
desktop/               the browser app as an Electron desktop application
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

Input is PNG, JPEG or TIFF, 8- or 16-bit, or a camera raw; the output keeps the input bit
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

**Fusion** (`fuse.rs`), generalized from the paper's two frames to N:

- Band-pass levels: region energy `RE = Σ ω·L²` over a binomial window
  (`--energy-radius`, default 1 = 3×3) of the *luminance* coefficient
  (Y = .299R+.587G+.114B of the RGB coefficients — the pyramid is linear, so
  that is the luma pyramid), winner-take-all per coefficient, ties to the
  earlier frame. The selection is applied to all three channels so color
  never splits at a selection edge. Radius 0 is Adelson's per-node |L| max.
- Residual: local deviation D and local entropy E (`--top-radius`, default
  2 = 5×5; `--entropy-bins`, default 256). The paper's rule takes A when A
  wins both measures, B when B wins both, and averages otherwise; for N
  frames that is kept as a Pareto rule — a frame is dominated if another is
  at least as good on both and strictly better on one, and the fused value is
  the mean of the non-dominated frames (`--top de`; `dev` = deviation only,
  `avg` = plain mean).
- **Halo control** (`--halo-control P`, off by default): the pyramid's
  halo comes from each level picking its own winner. Beside a bright (or
  dark) object, the frames focused behind it carry the object's defocused
  copy spread over the background — strong *coarse* energy where the frame
  that has the object sharp has none — so the coarse levels collect that
  glow from one frame after another while the fine levels take the sharp
  background, and the result has a soft rim around the object. With halo
  control the levels coarser than `--depth-level` (the *guide*, default 2)
  do not select: each, the residual included, is the mean of the frames
  weighed by `((RE_guide + ε) / ρ)^P`, the guide's region energy raised to
  the hardness P and REDUCEd to the level's size (the Gaussian pyramid of a
  weight mask, as a multiresolution spline blends with it). The coarse
  structure then follows the frames the guide found sharp, and where none
  is (a flat area) the frames average. P = 1 weighs by the energy itself,
  higher values approach a hard pick; 8 is the cap. Checked on the halo
  in isolation (`fuse.rs` tests): a sharp fine texture against the same
  texture blurred away with a wide soft bump added — coarse structure
  only, as a defocused copy of a bright object has. Every level picking
  its own winner lets the bump in (RMS error above 0.05 of full scale);
  with the guide finding the sharp frame everywhere the coarse levels
  follow it and the error falls by more than 85 % at every hardness from
  1 to 8. On the fruit stack (dark grapes on a bright table) the change is
  about 1 % of full scale along the silhouettes and invisible elsewhere:
  the halo is a coarse-level effect, and a deep stack of small focus steps
  leaves little of it to remove.
- The accumulator folds frames in one at a time: only the running fused
  pyramid, one best-energy plane per level (a weight sum at the levels halo
  control guides) and the tiny residuals are held, so memory does not grow
  with the stack. Frames are decoded on demand with
  a bounded read-ahead, aligned or not.

**Alignment** (`align.rs`): similarity registration (shift, scale,
rotation), direct intensity-based, coarse-to-fine on a Gaussian pyramid of
the luma with a DC-removed RMS objective, Spline4x4 resampling and a bounded
Nelder-Mead search, chained sequentially to frame 0. `--align-coarsen N`
stops N levels short of full resolution (the transform is resolution
independent, so this is a large speed-up at sub-pixel accuracy);
`--no-shift/--no-scale/--no-rotation` restrict the model; `--save-aligned DIR`
writes the registered frames. The search's schedule follows the pyramid:
at each level the simplex starts one pixel of that level wide (in every
parameter, measured by how far it moves the frame's edge) and stops at a
tenth of one, so a coarse level only hands the next one a start within its
pixel and the finest level searched settles sub-pixel — about 300 cost
evaluations for a 45 MP pair at coarsen 2 (a fixed simplex and tolerance at
every level took 450 to 700). Each evaluation is one row-parallel pass that
warps and reduces at once, nothing allocated: 1.6 ms at level 2 of 45 MP on
128 threads, 20 ms at full resolution. The same schedule drives the CUDA and
browser searches.

**Alignment model** (`--align-model M`; `align::AlignModel`): how much a
frame may be deformed to land on the previous one. `similarity` (the
default) is shift, scale and rotation — the usual model, and
what a focus rail or a focus ring produces: the image breathes, shifts and
turns a little. `affine` adds an aspect ratio and a shear; `projective`
adds the two perspective terms, for a camera that tilted against the
subject as it stepped, so the frames keystone. The transform (`align::Sim`)
is one 3×3 homography whichever the model, the perspective row being
[0, 0, 1] below projective, so every warp, the crop and the brightness
sampling read pixels through the same rational map — the division is by
exactly 1 for the affine models, and the similarity results are unchanged to
the bit. The search box around the previous frame's transform is 10 % of
the frame in shift, 10 % in scale, 5° in rotation, 5 % in aspect, shear and
each perspective term. More parameters take longer to search (the simplex
grows with them), and on a stack that needs none the extra ones only fit
noise, so the default stays the similarity. Checked on a frame keystoned by
a known perspective (ImageMagick's `-distort Perspective`, the top corners
moved 12 px inwards): against the original, the frame the projective fit
brings back differs by an RMSE of 0.0029 of full scale over the center,
the floor of resampling (a pure scale and rotation, which the similarity
model recovers exactly, leaves 0.0028), where the similarity fit leaves
0.0076 and the affine 0.0071; the browser's aligner finds the same eight
terms to three decimals. The stack is streamed through the alignment
like the browser streams it (`stack::AlignedFrames`): each frame is decoded
with the read-ahead, registered against the previous aligned frame's luma,
warped, brought to frame 0's brightness and folded into the fusion, then
dropped; only the transforms, the gains and frame 0 on the brightness
sampling grid are kept, so memory is that of a few frames whatever the
stack's length (a 100-frame 45 MP stack needed over 100 GB resident). The
depth pass takes no second look at the frames: each frame's focus slice
(`depth::focus_slice`, its focus measure on the depth pass's working grid,
45 MB per 45 MP frame at the default half grid) is taken as the frame is
folded and kept, the one thing that grows with the stack's length. The
weighted average and the slabs decode and warp the frames again with the
transforms found, as they already did without alignment.

**Interpolation** (`--interpolation K`; `align::Interp`): the kernel each
aligned frame is resampled with once its transform is found — the choice a
focus stacker usually offers. `nearest` (the nearest source pixel:
nothing blurred, nothing rung, and sub-pixel shifts land jagged — for stacks
already aligned to the pixel, or to see the pixels as shot), `bilinear`
(soft), `bicubic` (Keys' cubic convolution, a = −0.5), `spline4x4`
(Panorama Tools' spline16, our default), `spline6x6` (spline36,
sharper, a little ringing at hard edges) and `lanczos3` (three lobes, the
sharpest and the most ringing). All are separable and interpolating (a
pixel-centered sample comes back as it is; the weights sum to 1, Lanczos's
normalized to make it so). The registration search itself always resamples
with spline4x4 — the fit does not depend on the kernel, and the CUDA and
WebGPU cost kernels stay one thing — so the choice changes only how the
frames are read, not where they land. The crop below keeps each kernel's own
support out (nearest 0 px, bilinear 1, the 4-taps 2, the 6-taps 3), so a
wider kernel loses a pixel or two more at the border.

**Slabs** (`--slabs SIZE[:OVERLAP]`, `--slab-dir DIR`): slabbing
for the native path — after the result, every run of SIZE consecutive
frames overlapping by OVERLAP (default 2) is fused on its own with the same
settings over the same aligned, equalized frames, cropped like the result,
and written as it is made to DIR (default `<output stem>_slabs`) in the
output's format with the same metadata, as `slab_01_000-009.tif` and so on
(0-based frame indices). Slabs are thick planes of focus to retouch from in
another editor; the browser app makes them on demand instead (below). Each
slab streams its frames from disk again. Where a warped frame does not reach, the warp
repeats its edge, so the output (image, depth and confidence maps) is
**cropped to the largest rectangle every frame covers** with real pixels
(`align::common_area`: each frame's sound area is a convex quad, cut per pixel
row into an interval, intersected over frames, and the best rectangle over
consecutive rows is taken; the interpolation kernel's support — 2 px for
spline4x4 — is kept out); `--no-crop` keeps the full frame.

**Batch runs and stack splitting** (`batch.rs`; `--split RULE`, `--dry-run`):
a directory among the inputs stands for the image files in it (PNG, JPEG,
TIFF, in natural order, `f2` before `f10`), and `--split` cuts the frame list
into stacks that are then run one after the other with the same settings.
`count:N` makes a stack of every N frames,
`gap:SECONDS` starts a new one wherever the capture time jumps by more than
that (a rail shoots every second or two and a pause between subjects is tens
of seconds; the time is EXIF DateTimeOriginal with its sub-seconds, else
DateTimeDigitized, DateTime, or the XMP packet's CreateDate — raw converters
write TIFFs with XMP and no EXIF — and a frame with none takes its file's
modification time and says so), `dir` makes one stack per folder, so
`--split dir shoot/*/` stacks a folder of folders. With more than one stack
the output paths (`-o`, `--slab-dir`, `--save-aligned`, `--depth-raw`) are
templates: `{n}` is the stack's number, `{first}` its first frame's stem,
`{dir}` its folder's name, and a path with no field gets `_NN` before its
extension (`-o out.tif` writes `out_01.tif`, `out_02.tif`, …); the folder a
template names is created. Every other output (depth, slabs, stereo, rocking,
model) follows the stem as always. Stacks run in turn, so memory stays that
of one stack; a stack that fails (frames of two sizes, say) is reported and
the batch goes on, with a summary and exit status 1 at the end. `--dry-run`
prints the stacks — frames, output names, and for `gap` the capture times and
the pause before each — and stops: worth a look before hours of stacking,
and the way to see whether the split is what you meant. Reading the times
costs little: only the head of each file is read, or, for a TIFF whose IFD
follows the pixels, a window around that IFD (100 frames of 274 MB in 0.2 s).

**Camera raw input** (`raw.rs`): NEF, CR2 and CR3, ARW, DNG, RAF, ORF, RW2,
PEF, IIQ, 3FR and the rest of what [rawler](https://github.com/dnglab/dnglab)
(dnglab's library) decodes are taken as frames, natively and in the browser,
by their extension. The decoding is not in lapstack itself but in
`lapstack-raw` (`crates/lapstack-raw`, LGPL-2.1), a component loaded at run
time: natively `liblapstack_raw.so` / `.dylib` / `lapstack_raw.dll`, which
`cargo build --release` puts next to the binary and the binary looks for
beside itself (then `../lib`, then the system path; `LAPSTACK_RAW_LIB` names
one outright) — without it a raw is an error that says so and every other
input works, as with CUDA; in the browser `web/pkg-raw`, a wasm module the
worker imports when the first raw comes in and the engine reaches through
three globals (`lapstack-web`'s `raw_bridge.rs`). `raw.rs` is the client of
that interface: the five functions the rest of lapstack calls, a `RawBackend`
either side installs, and the JSON shapes of the camera's color and
metadata that cross it. Each is developed as shot — black and white levels,
demosaic, the white balance the camera recorded, the camera's color matrix
to sRGB, the sRGB curve — into the 16-bit RGB every other input becomes,
turned the way the EXIF orientation says; a monochrome sensor gives gray.
There is no exposure or tone adjustment: the frames of a stack are shot
alike, and what the stack needs is that they are developed alike. When the
look matters, develop the stack in a raw converter first and give lapstack
its TIFFs. Developing costs more than decoding — about a second per 24 MP
frame natively on all cores, several in the browser on one thread — and a
raw's capture time for the batch split is read from its TIFF structure
where it has one (NEF, CR2, ARW, DNG, …) and else by rawler's own reader
(CR3, RAF, …), the whole file in either case. In the browser the filmstrip
shows the JPEG preview the camera wrote into the file, and the run develops
the frame. The linear DNG output (below) develops the raws another way. The core crate's `raw` feature (on by default) carries rawler,
which is vendored in `vendor/rawler` (the workspace's `[patch.crates-io]`)
with one change, so that it runs in the browser: `std::time::Instant`, which
the demosaic and the CR3 decoder use for a timing log line, has no
implementation on wasm32 and panics there, so a shim reads zero on wasm
(`vendor/rawler/LAPSTACK-PATCH.md` says how to move to a newer rawler);
without the feature a raw file is refused.

**Linear DNG output** (`dng.rs`; `-o stacked.dng`): raw in, DNG out. With a
`.dng` output the raws are not developed to sRGB: each is
decoded to the camera's own linear space — black and white levels, demosaic,
the sensor's crop, turned by its orientation, and nothing else
(`raw::develop_linear`) — and the stack is fused in a *look* space made from
frame 0's as-shot white balance and the camera's D65 matrix followed by the
sRGB curve, the ordinary development except that nothing is clipped: the
curve is extended above 1 and mirrored below 0, so a highlight past white or
a color outside sRGB keeps its value and the transform stays invertible
(`DngInfo::to_look`). The fusion therefore makes the same decisions it makes
on a normally developed stack (the luma it selects by is the ordinary one),
every frame is developed alike (frame 0's white balance is applied to all,
as the raw section says a stack needs), and the result is taken back through
the inverse curve, the inverse matrix and the inverse white balance to camera
space (`DngInfo::from_look`) and written as a DNG 1.4 with
`PhotometricInterpretation` LinearRaw, 16 bits per sample, `WhiteLevel`
65535, the camera's `ColorMatrix1` / `ColorMatrix2` with their illuminants
(the cooler light first, as Adobe writes them), `AsShotNeutral` (the
reciprocal of the white balance, green = 1), `UniqueCameraModel`, Make and
Model, the first frame's Exif IFD and XMP (no ICC profile: a DNG's color is
its matrices) and Orientation 1 (`meta::write_dng`). A raw converter then
develops the stacked image like one of the raws — exposure, white balance,
profile and highlight recovery still open, since the camera data above the
balanced white is kept where an sRGB development would have clipped it.
Checked on five NEFs of the Z 8: darktable develops the DNG and the first NEF
to the same colors and brightness. Frames that are not raws are taken as
sRGB — their linear values are the "camera space", `ColorMatrix1` is XYZ →
sRGB at D65 and the neutral is 1, 1, 1 — so any stack can come out as a DNG,
but only a raw's carries more than its file did; a stack must be all raws or
none. The slabs, the weighted average and the stereo pair are DNGs as well;
the rocking views are TIFFs (a video is made of them), the depth maps PNGs.
A raw's EXIF Orientation is reset to 1 in every output now (the frame was
turned as decoded; a viewer must not turn the result again).

**Frame preparation** (`prep.rs`): what a frame goes through as it is decoded,
after the dust map (which is in the sensor's orientation and size).
`--rotate 90 | 180 | 270` turns every frame clockwise, for a camera held
sideways (a raw is already turned by its EXIF orientation; a TIFF or JPEG is
not). `--draft N` block-averages every frame by 2^N, so the whole run —
alignment, fusion, depth pass, every view and file — is a quick check of the
settings at 1/2^N the size: five 45 MP raws at `--draft 2` align, fuse and
run the depth pass in 4 s, the raw development included; `--crop` and the
scale bar's calibration follow the frame's own
pixels, so a draft's output is the full run's shrunk. A frame of another size
than frame 0 (a JPEG among the raws, another camera mode) is no longer
refused: it is resampled to frame 0's size (each axis on its own, with
spline 4×4) and the run says so; the alignment takes care of what is left,
and a stretched frame of the wrong aspect is visible in the output. `--crop
X,Y,W,H` cuts every output to a window of the (turned) frame, in
full-resolution pixels, inside the automatic crop to the area every frame
covers; a window outside that area stops the run.

**Weighted average** (`wav.rs`; `--wav`, `--wav-power P`, `--wav-smooth R`,
`--wav-gate G`):
a second image, `<stem>_wav.<ext>` — every
frame's pixels averaged with weights that follow their local contrast, so
the frame in focus at a pixel counts most and the rest fade in with their
sharpness. No pixel is ever picked outright: the seams and halos a
winner-take-all rule can leave cannot arise, and where nothing is sharp (a
flat area) the frames simply average and the noise drops by the square root
of their number — at the cost of some softness where a hard pick would have
kept one frame's detail. It suits short, smooth, low-contrast or noisy
stacks; the pyramid is the sharper tool, and the browser's retouch brushes
one into the other. The contrast is the depth pass's focus measure (the
ring difference filter on the luma, so `--wav` needs the depth-from-focus
pass, not `--depth winner`), block-averaged to the depth pass's working
grid and box-smoothed there by `--wav-smooth` grid pixels (default 3; the
depth pass aggregates its slices the same way before it takes their
statistics): a cell's pick is its region's — one
cell's measure strays over the gate below by chance and picks a noisy frame
where a flat area should average them all, and along a silhouette, where
one frame holds the edge and another the blurred halo over it, neighboring
cells picked different frames and the 2 px ramp between them showed as
jagged speckle along every depth edge; over the window the frames
cross-fade instead. What weighs is the contrast *above the cell's noise
floor*: sensor
noise alone gives the measure a floor that is the same in every frame, and
at full resolution it is of the order of a sharp frame's own texture (a
45 MP lemon peel reads under 3× its out-of-focus frames), so the contrast
itself, shared with dozens of frames of noise, made an average as soft as
no stacking at all. The depth pass tracks the least contrast any frame
shows at every cell — the frames farthest out of focus leave only the
noise — and `--wav-gate G` (default 0.5) says how far above it a contrast
counts: the weight is the contrast less (1+G) × the floor, clamped at zero,
raised to `--wav-power` (default 2; 1 = plain, higher = a keener pick of
the sharpest frame), so what the noise could explain weighs nothing (a fade
in place of the cut let dozens of frames of noise back in); where no frame
rises above the floor a tiny even weight makes the frames a plain average.
The weights are smoothed by the same window after the cut, so a region's
border is a cross-fade and not a step wherever a cell sits at the gate, and
taken back to full resolution bilinearly. One more pass over the frames.

**Frame list** (`--skip LIST`, `--reverse`): `--skip` leaves frames out of the
list — 1-based positions and ranges, comma-separated (`--skip 3,7-9,12`),
counted after directories are expanded and before any split, each skipped
frame reported; a position past the end or a backward range is an error,
since a typo should not stack the wrong frames. `--reverse` turns the order
round for a stack shot back to front, so frame 0 is the near end again for
`--stereo` and `--mesh` (the alternative is `--far-first`, which leaves the
fusion alone and tells only those two); with `--split` each stack is reversed
on its own, which is what a rail run backward means for every stack of the
batch. Reordering beyond that is the shell's: the frames are stacked in the
order they are given.

**Brightness** (`brightness.rs`): flash recycling, mains-powered lights and
a shutter that is not quite repeatable make frames differ in exposure by a
percent or two, and the region-energy rule sees it (energy grows with the
square of the gain, so a brighter frame wins ties it should not and the
seams between winners show as patches). Every frame is brought to frame 0's
brightness by one gain per channel — the ratio of the two frames' channel
*means* over the pixels the frame's warp covers, since a blur leaves a
mean alone while a pixel-wise fit would slope toward zero with the defocus;
per channel, so a light that flickers in color is corrected too. Gains are
clamped to [1/4, 4] and logged; `--no-brightness` turns it off. The browser
app does the same on the GPU (block means of frame 0 kept, one small readback
per frame), *equalize brightness* in the parameter panel, and each filmstrip
entry shows its gain.

**Dust map** (`dust.rs`; `--dust-map FILE`, `--dust-threshold PCT`,
`--dust-margin PX`, `--dust-mode fill|flat`, `--save-dust-map PATH`): the dust
map. Sensor dust shows in every frame at the same place as a
soft dark spot, and the stack keeps it — worse, the region-energy rule takes
the spot's edge for detail and picks it, so the spot comes out sharper than
in any frame. A frame of an evenly lit, featureless surface shot out of
focus (a white wall, the sky, a sheet of paper) at the stack's aperture
shows nothing but the dust, and that frame is the dust map: its luma at
half resolution (one REDUCE, which also tames the noise) is divided by its
own large-scale background — a plane fitted under each cell four pyramid
levels up (a first-order normalized convolution: the illumination's falloff
is followed out to the frame's edges, a spot is not; estimated twice, the
second time with the first pass's spots weighted out) — and a pixel darker
than that by more than the threshold (default 3 %) is dust. The connected
components, grown by the margin (default 3 px: the soft edge of the shadow,
and the little a spot moves between apertures), are the spots; a component
under 16 px is noise, and a blob wider than a quarter of the frame is not
dust and is reported. The spots are then taken out of every frame **as
decoded, before alignment** — the dust is fixed on the sensor, and a fixed
pattern in every frame is exactly what pulls a registration toward zero
shift. `fill` (the default) interpolates each spot from its
surroundings by pull-push (Gortler et al. 1996: the window's pyramid is
built with the dust weighted out, and on the way down every hole takes the
coarser level's value), a smooth patch that meets its edges; `flat` divides
the spot by the attenuation the map measured, a flat-field correction that
keeps whatever detail lies under the spot — right when the map was shot at
the stack's aperture and lighting, a ring when it was not. The run logs the
spots found (count, size, share of the frame); `--save-dust-map` writes the
mask (white = dust) to check the map before a long batch; a map of another
size than the frames stops the run. The browser app has the same under
*Dust map* in the parameter panel: load the frame, the spots are found in
the worker and outlined on a preview (click it for a larger view in a new tab), the
threshold, margin and mode take effect at once, and the map applies to
every frame the engine decodes — the run, the DFR and WAV renders, the
slabs, the Source view and the retouch brush (the filmstrip's thumbnails
are of the frames as shot). A project file records the map's name, and the
file is taken from the ones added when the project is opened again.

**Scale bar and caption** (`overlay.rs`; `--scale-bar CAL[:LENGTH]`, `--text TEXT`,
`--overlay-pos`, `--overlay-size`, `--overlay-color`, `--overlay-style`): for
the microscope, where the stacked image is a figure. `CAL` is the size of one
pixel of the frames in µm (`0.325`, or with a unit, `325nm`) — the camera's
pixel pitch over the magnification, or what a stage micrometer measures — or
`auto`, which reads it from the first frame's TIFF (`Meta::pixel_size_um`):
ImageJ's `unit=micron` in the ImageDescription with XResolution in pixels per
unit, OME-XML's `PhysicalSizeX` / `PhysicalSizeXUnit`, or a plain resolution
in cm or inch from a writer that is not a camera (no Make tag: a camera's 72
or 300 dpi says nothing about the subject). Alignment brings every frame onto
frame 0's pixel grid and the crop only cuts that grid, so one number serves
every output at full size. The bar is the 1-2-5 value nearest a fifth of the
width, or `LENGTH` (`100um`, `2mm`, `500nm`; one that does not fit is brought
down to the largest 1-2-5 value that does, and the log says so), snapped to
whole pixels, with its length written over it in the unit that keeps the
number under a thousand (500 nm, 100 µm, 2.5 mm). `--scale-bar px` asks for
a bar without a calibration: a round count of the frames' pixels, labeled so
(*500 px*; `px:LENGTH` takes the count), so a figure that is not calibrated
still carries a scale. `--text` is a caption:
`\n` breaks a line, and `{date}` / `{time}` (the first frame's capture time,
as the batch split reads it), `{frames}`, `{first}` (the first frame's stem)
and `{n}` (the stack's number in a batch) are filled in. Bar and text each
take a corner (`--overlay-pos BAR[,TEXT]`, `tl | tr | bl | br`, default
bottom right and bottom left; in one corner the text goes above the bar,
below it at the top), in white with a black halo — the ink dilated by a
disk, a thin outline that reads on any background — or black with a white
one (`--overlay-color`), or on a translucent box, or plain
(`--overlay-style halo | box | plain`). Every size follows the image: the
font's em is `--overlay-size` % of the image height (3 by default), and the
margins, the bar's thickness, the gap under the label and the halo's width
are fractions of it, so the same settings give the same figure at every
resolution, and an animation shrunk to a long edge carries the same bar,
shrunk with it. The text is set in Fira Sans, the browser app's own face — a
30 KB subset of the Regular weight embedded in the core crate
(`crates/lapstack-core/fonts`, made by `subset.py` there from Mozilla's TTF;
SIL OFL) — and rasterized by lapstack itself: a TrueType outline reader
(glyf / loca / cmap / hmtx, simple and composite glyphs, no hinting) and the
signed-area coverage accumulation of font-rs / stb_truetype v2, where each
edge deposits the area it sweeps into the pixels it crosses and a running
sum along the row gives the exact coverage of the nonzero-winding fill,
anti-aliased for free. The overlay is a few patches of coverage over the
corners it occupies, so nothing the size of the image is allocated; it is
burned into the fused image, the weighted average and every stereo and
rocking view (after the shear — a bar sheared by the depth map would bend),
not into the depth maps, the slabs or the 3D model's texture. The browser
app draws the same patches (below).

**Synthetic stereo and rocking** (`view.rs`; `--stereo PCT[:LAYOUT]`,
`--rocking PCT[:N]`, `--far-first`): the depth map makes the result a relief,
and a view from the side is that relief sheared — every pixel slides
sideways in proportion to its depth. The same picture can be had by shifting
each frame by its index before stacking, or by projecting a textured 3D
model built from the depth map; lapstack shears the stacked image and its
depth map in one pass: the far end of the stack moves PCT % of the width
relative to the near end (the maximum X shift; ±3 % suits most subjects —
for a scene d deep and w wide a viewing angle a is tan(a)·d/w), the middle
of the stack staying put. The shear is a
forward warp per row: consecutive samples less than 2 px apart in the view
form a patch of surface, rasterized with a nearness test so a near edge
slides over the background; a larger gap is a depth discontinuity, and the
hole it opens is filled from the farther side (the background shows
through, the foreground is not stretched); pixels are sampled linearly.
`--stereo` writes the views from the left and the right (−PCT / +PCT) as
`<stem>_stereo.<ext>` — side by side for parallel viewing (`sbs`),
cross-eyed (`cross`) or as a red–cyan anaglyph; `--rocking` writes N views
(default 24) whose shift sweeps ±PCT in one sine cycle to
`<stem>_rocking/view_NN.<ext>` (join them with ffmpeg or ImageMagick), and
`--video FPS` joins them itself into `<stem>_rocking.mp4` (H.264, crf 18,
the size made even) when ffmpeg is on the path — the command to run by hand
is printed when it is not. Which end is near decides who wins where surfaces
overlap, and `--near-end auto | first | last` says which end frame 0 is
(`--far-first` = `last`). `auto`, the default, reads the focus distance the
camera recorded in the first and last frames — the focus went from the
smaller distance to the larger — and takes frame 0 as the near end when
there is none. The standard EXIF SubjectDistance is read by lapstack itself
(`meta::subject_distance`, the TIFF structure or the XMP); most cameras keep
the distance in their MakerNote instead (Nikon's is even encrypted), so when
`exiftool` is on the path it is asked for SubjectDistance, FocusDistance,
Canon's FocusDistanceUpper / Lower and Sony's FocusDistance2
(`io::exiftool_distance`, two calls of 0.1 s), which covers the Z 8's NEFs
here (0.78 m in the first frame, 1.44 m in the last: frame 0 near). The
frames' scale over the stack is logged as a hint but does not decide: on a
rail, or with a lens that extends to focus closer, the near frames are the
larger ones and a stack shot near to far is enlarged toward its end to fit
frame 0; an internal-focus lens that widens as it focuses closer breathes
the other way, and the sign says nothing without knowing the lens. The
symptom of the wrong choice is a relief that looks inside out; the run logs
which cue decided.

**3D model** (`mesh.rs`; `--mesh glb,obj,stl`): the depth map makes the
result a relief, and the model is that relief as a
mesh, a grid of vertices over the image (`--mesh-grid N` along the long
edge, default 1000; each vertex takes the mean depth of the cell of pixels
around it, so the mesh is smooth at its own scale; each cell is cut along the
diagonal with the smaller depth step) raised by the depth and textured with
the stacked image. Coordinates are right-handed with the width as the unit:
x along the width, y up, z toward the viewer, the far end of the stack on
z = 0 and the near end at `--mesh-relief PCT` % of the width (default 25 —
the depth of the stack is the one thing the depth map cannot know, so
measure the subject or set it by eye; `--far-first` applies). `glb` writes
`<stem>.glb`, a self-contained glTF 2.0 binary with the texture embedded
(Windows 3D Viewer, macOS Quick Look, Blender, every web viewer); `obj`
writes `<stem>.obj` + `<stem>.mtl` + `<stem>_texture.jpg`, plain text most
tools read; `stl` writes the geometry alone as binary STL for printing. The
texture is 8-bit, JPEG (quality 92) or PNG, capped at `--mesh-texture
EDGE[:jpeg[:Q] | png]` px on the long edge (default 8192, the largest most
viewers accept). A depth discontinuity becomes a wall between the near
surface and the far one — a heightfield has no way to show what is behind
an edge.

`--save-depth` writes the depth map produced by the depth-from-focus pass
below (`--depth winner` instead reports the raw winning frame index read
from pyramid level `--depth-level`, default 2 — the finest level's winner
map is noise wherever the scene is flat).

### Depth from focus (`depth.rs`)

Each frame's focus measure is taken as it is folded, and after the fusion,
whose result is the guide, lapstack builds a dense, sub-frame depth map
from those slices with the modern non-learned depth-from-focus recipe,
written from the papers:

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
   stack size: the global peak with its two neighbors (Gaussian
   interpolation of Nayar & Nakagawa 1994 → fractional frame index), the
   second-best local maximum (peak-ratio confidence), the profile mean
   (prominence) and a noise gate against the median profile minimum
   (`--depth-gate`). Confidence is normalized so its 90th percentile is 1.
4. **Regularization** — edge-aware *weighted least squares* (Farbman et al.
   2008) with the confidence as data weight: flat, noisy or ambiguous
   pixels take their depth from confident neighbors without crossing image
   edges. The separable fast global smoother (Min et al. 2014) gives the
   initial guess, and a conjugate gradient solves the 2-D system
   (`--depth-lambda`, `--depth-sigma`, `--depth-cg`) to a relative residual
   of 1e-5, preconditioned by one multigrid V-cycle: the system is
   aggregated 2×2 down to a 16-px grid (the data weights summed over each
   block, the edges a block boundary cuts summed into the coarse edge — the
   Galerkin operator of a piecewise-constant prolongation, so the
   preconditioner stays symmetric) with one damped-Jacobi sweep before and
   after each coarse correction. A Jacobi-preconditioned CG left the low
   modes to the CG itself: on the 4140×2760 half grid of a 45 MP stack it
   needed 300 iterations for 1e-5 (the `--depth-cg 200` cap stopped it at
   7e-5, 0.4 frames off the converged map in places), and where the
   confidence is low over a large area — a flat wall, a synthetic stack
   with no peaks — it did not converge at all; the V-cycle takes the same
   solve to 1e-5 in about 20 iterations (50 on the pathological case),
   3.5 s to 1 s on the CPU. One Huber reweighting pass (`--depth-robust`,
   default 1 frame) then removes outliers that a least-squares fit would
   otherwise average in; the second solve reuses the hierarchy.
5. **Upsampling** — guided-filter upsampling on the full-resolution luma
   (`--depth-upsample guided:R:EPS` | `bilinear`), so depth edges land on
   image edges.

`--save-depth` writes the 8-bit visualization, `--depth-raw PATH` a 16-bit
PNG with a fixed scale (65535 = last frame) for numeric use, `--save-conf`
the confidence map.

### Performance

On a stack of 8280×5520 16-bit TIFFs (RTX 3060, 128-thread host), the same
frames and alignment settings for every row; the whole run, depth pass
included:

| run | wall | of which fold (align + fuse + focus measure) / depth | peak RSS |
|---|--:|--:|--:|
| 25 frames, `lapstack --align-coarsen 2` (CPU) | 26 s | 19 s (0.75 s per frame) / 6.0 s | 8.7 GB |
| 25 frames, `lapstack --gpu --gpu-align --align-coarsen 2` | 9.6 s | 5.1 s (0.20 s per frame) / 1.0 s | 6.2 GB |
| 100 frames, CPU | 85 s | 74 s (0.74 s per frame) / 8.0 s | 12.0 GB |
| 100 frames, `--gpu --gpu-align` | 26 s | 21 s (0.21 s per frame) / 1.9 s | 9.4 GB |
| 25 frames, `lapstack --no-align` on the aligned 16-bit PNGs, CPU or GPU | 26 s | 24 s / – (no depth pass then) | 4.4 GB |

Before the pooling allocator below, the same rows read 51 s (41 s fold,
1.6 s per frame, 6.8 GB), 13 s (6.5 s, 0.26 s, 3.7 GB), 183 s (169 s, 1.7 s)
and 34 s (28 s, 0.28 s, 7.0 GB): more than half of the CPU fold was page
faults on fresh planes. Before the aligner's cost was fused into one pass
and its search put on the per-level schedule, the rows read 118 s (75 s align + fuse,
3.0 s per frame), 74 s (30 s, 1.2 s), 449 s (310 s, 3.1 s) and 265 s (126 s,
1.3 s): the CPU search alone was 1.2 s of every frame, the two registration
pyramids 0.5 s; and until the frames moved onto the device with
`--gpu-align`, the CUDA rows read 50 s (18 s fold) and 161 s (74 s), the
CPU-side warps, pyramids and focus measure being 0.5 s of every frame. The
depth pass then decoded and warped every frame a second
time to take its focus measure, 1.2 s per frame on either path (42 s of the
25-frame runs); now the fold takes each frame's focus slice as it passes
(0.15 s per frame) and the depth pass is the guided-filter aggregation of
the slices, the WLS solve and the upsampling. That pass was 29 s of the
25-frame runs and 83 s of the 100-frame ones on either path, the aggregation
0.75 s per slice and the WLS 9 s: the box filters allocated four fresh
45 MB planes per slice, and faulting a fresh plane in from 128 threads at
once costs 150 ms (the sums themselves 20 ms), so the filters now keep
their planes from one slice to the next (0.045 s per slice, 16×), and the
CG is preconditioned by the multigrid V-cycle above (2.8 s for both solves
instead of 9 s, and converged). With `--gpu` the whole pass runs on the
device (`gpu::depth_from_slices`, the same stages as the browser's, the
multigrid included): the slices are uploaded one at a time, only the two
medians (noise floor, confidence scale) are taken on the host, and the
maps come back — 1.7 s for 25 frames, 2.3 s for 100, the maps within
4e-4 of full scale of the CPU's (a few pixels in a million of the
confidence differ, where a near-tie in the peak search falls the other
way in float order). The peak RSS swings between runs by how far the read-ahead
decoders (four frames) get ahead of the fold; the slices add 45 MB per
frame. Otherwise memory is flat over the stack's length since the frames
stream through the alignment: before that, when every frame was loaded and
aligned at once, the 25-frame CPU run took 56 s to align and 21 s to fuse at
31 GB peak, the CUDA run 32 s and 2.0 s at the same 31 GB, and 100 frames
did not fit.

With `--gpu` (build with `--features gpu`; CUDA is loaded at run time, no
toolkit needed at build time) the fusion runs in `gpu.rs`: the same kernels
transcribed to CUDA, the accumulator pyramid, best-energy planes and winner
map stay on the device, and only the three RGB planes go up per frame and
the tiny residual comes back. With `--gpu-align` the frames live on the
device (`gpu::GpuFrames`): a decoded frame is uploaded once, and its luma
and registration pyramid, the Nelder-Mead cost search, the warp with the
kernel of `--interpolation`, the brightness gains and the depth pass's
focus slice are all computed there; the fuser folds the warped planes in
place, and a frame comes back to the host only when something there asks
for it (`--save-aligned`, the weighted average). What is left of a GPU
frame is the decode and the upload, and the host holds no warped copies,
which is where the peak memory went. The `--no-align` row is 16-bit PNG
decode-bound (~1 s per frame with four decoder threads); uncompressed TIFF
input decodes an order of magnitude faster.

The CLI runs under a pooling allocator (`pool.rs`, installed as the
process's global allocator): a block of 4 MB or more is not returned to the
system on free but kept, up to 64 blocks and 4 GB, and the next request of
its size class gets it back with its pages still mapped. Every stage of the
fold makes fresh planes — the decoder's buffers, the warp, the luma, the
Laplacian pyramid's levels, the energies, the focus slice — and glibc
unmaps a block above its mmap threshold on free, so each frame took the
page faults again; with 128 threads first-touching a plane at once that is
~150 ms per 45 MB plane, and it was more than half of the fold: the 25-frame
CPU fold is 18.7 s with the pool and 40.8 s with it switched off
(`LAPSTACK_NO_POOL=1`, same binary), the output bit-identical, for 1.8 GB
more peak memory (the idle blocks). A 2 GB cap loses most of the gain (the
frame's working set of planes is larger than that); 6 GB gains nothing over
4. The CUDA rows gain too — the decode and the slices are host planes —
and the depth pass's host side with them (1.7 s to 1.0 s for 25 frames).
The browser build does not use it. Zeroed requests on a recycled block
are cleared with memset, ~4 ms per 45 MB.

GPU fusion output is bit-exact with the CPU output (one run out of four
differed in 637 of 45.7 M pixels on energy ties and could not be
reproduced); with halo control the weights are made, REDUCEd and folded
on the device too (`wgtk`, `wacck`, `wnormk`) and the output differs from
the CPU's by float rounding only. CPU reruns are byte-identical. GPU alignment evaluates the cost
in FP32 and warps the frames on the device (coordinates in double like the
CPU's warp, taps in FP32), so `--gpu-align` results differ from CPU-aligned
ones by ~0.5 % of pixels; the transforms it finds are the same as when the
registration pyramids were built on the CPU, to the last printed digit.

Parameter sweep on the same aligned frames (default = 3×3 window, 7 levels,
`--top de`): the energy window is the only knob that matters.
`--energy-radius 0` (per-node max) is grainier, `--energy-radius 2` (5×5)
slightly smoother. `--top avg|dev`, `--levels 5|9` and `--use-chroma` change
< 1 % of tiles: as Burt & Adelson note, the low-pass content is shared
between frames, so the residual rule is nearly moot on a deep stack.

## lapstack in the browser (`web/`, WebGPU + WASM)

`crates/lapstack-web` + `web/` run the whole lapstack pipeline in a browser
tab: frames are decoded in WASM (PNG/JPEG/TIFF, 8- and 16-bit, via the `image`
crate; camera raws developed via rawler, see above,
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
**Stereo and rocking** are the CLI's synthetic stereo (above) in the
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
(*all frames* first), nor can the list while a run or save is going.

**WAV** (the same ▾ menu, *WAV (weighted average)*, with *contrast power*,
*weight smoothing* and *noise gate %*): the CLI's `--wav` in the browser — a third stacked
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
the project's own.

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

## Lightroom Classic plugin (`lightroom/`)

`lightroom/lapstack.lrplugin` is the usual focus-stacking round trip from
Lightroom: the frames of a stack go out of the catalog
to the `lapstack` CLI and the stacked image comes back into it. Add the
folder in File › Plug-in Manager and set the path to the lapstack binary and
the stacking settings in its section (output format tif / png / dng, a name
template with `{first}` and `{n}`, alignment and its coarsening, halo
control, `--wav`, `--save-depth`, CUDA, extra options appended verbatim,
whether the rendered frames are kept). Then either select the frames of one
stack and choose Library › Plug-in Extras › **Stack with lapstack**, which
renders them as 16-bit TIFFs (or hands over the original raws, for a DNG
output) into a temporary folder, runs lapstack over them in the order of
the selection and imports the result stacked above the first frame,
selected; or File › Export with **lapstack** as the *Export To* target, which
keeps Lightroom's own file, size and metadata sections, adds the plug-in's,
and does the same with what Lightroom rendered — an export preset carries
the settings, and an export makes them the menu item's too. The result lands
next to the first frame as `<first>_stacked.<ext>` with the CLI's log
(`<stem>.lapstack.log`) beside it; a failure shows the log's tail. The CLI
cannot be interrupted once it runs (LrTasks.execute blocks its task); a
selection is one stack (`--split` in the extra options cuts it, and the
templates then number the outputs). The plugin was written against the
SDK 6.0 reference and its sources parse under Lua 5.1 (Lightroom's), but it
has not been run inside Lightroom here: the export-settings keys the menu
item uses (`LR_format`, `LR_export_bitDepth`, `LR_export_colorSpace`,
`LR_export_destinationPathPrefix`, …) are the ones published plugins use,
and whether Lightroom takes a lapstack-written linear DNG on `addPhoto` is
untested — if it refuses, the plugin says the file was written and leaves
it. `lightroom/README.md` has the details.

## Desktop application (`desktop/`)

`desktop/` wraps the browser app in **Electron**, so it runs as a desktop
application on Linux, Windows and macOS without the WebGPU launcher script:
Electron bundles Chromium, which has WebGPU on every platform (Tauri's
WebKitGTK does not, on Linux). `main.js` serves `web/` from the main process
over a static server on 127.0.0.1 with an ephemeral port (localhost is the
secure context WebGPU needs; a `file://` page cannot fetch its wasm), with
the MIME types and `Cache-Control: no-store` of `serve.sh`, and opens one
sandboxed window on it (no Node in the page, context isolation on, a CSP).
The Chromium switches are `chrome.sh`'s: `enable-unsafe-webgpu` everywhere,
and on Linux the Vulkan trio and the X11 ozone platform — which the browser
process only honors on its real command line, so on Linux the app
relaunches itself once with `--ozone-platform=x11` (a Wayland window with an
X11 Vulkan surface never maps; `LAPSTACK_SWITCHES` replaces the Linux
switches for troubleshooting). The File System Access pickers the app uses
for folders and saving are granted through Electron's permission handlers,
and a download goes to a Save As dialog. `just desktop-install` once, then
`just desktop` runs it (after `just build-web`), `just desktop-smoke` opens
a hidden window, prints the adapter, runs the 8-frame test stack and exits
0 or 1, and `just desktop-dist` packages it into `desktop/dist` (AppImage
and deb on Linux, dmg on macOS, nsis on Windows; `web/` minus its tests goes
in as a resource). On this machine (NixOS, KDE on Wayland, RTX 3060) the
smoke test passes on the nvidia adapter with nixpkgs' Electron, the official
binary, the unpacked build and the AppImage, and the window maps and
presents; NixOS needs the FHS environment `desktop/nix-fhs.nix` for the npm
Electron binary and electron-builder's tools (`desktop/README.md`). The
pickers, the Save As dialog and the Windows and macOS builds have not been
exercised here.

## GPU acceleration (CUDA, optional)

```
cargo build --release --features gpu -p lapstack-cli
# Linux: needs libcuda (driver) + libnvrtc (toolkit) on the path, and nvidia_uvm loaded.
# NixOS: LD_LIBRARY_PATH=/run/opengl-driver/lib:<cudaPackages.cuda_nvrtc's lib output>/lib
# Windows: nvcuda.dll ships with the display driver; drop the two DLLs from
# NVIDIA's cuda_nvrtc redist zip (bin/nvrtc64_120_0.dll + nvrtc-builtins) next
# to lapstack.exe. Match the nvrtc major.minor to the driver's CUDA version
# (nvidia-smi) or the PTX JIT will reject the kernels.
```

`cudarc` is built with `dynamic-loading`, so a binary built with the feature
still runs on a machine without CUDA; `--gpu` / `--gpu-align` fail at run
time with a message, everything else works.

## License

lapstack is proprietary software: Copyright (C) 2026 RAGTUX LLC, all rights
reserved. `LICENSE` (SPDX `LicenseRef-RAGTUX-Proprietary`) is the whole of
it — no license to the source, and an end-user license for the programs built
from it. The revisions from 2026-09-24 to 2026-09-30 were published under the
AGPL-3.0-only; that license stays with the copies distributed under it and
does not extend to later revisions.

Third-party components keep their own licenses and are listed with their
notices in `THIRD-PARTY.md`, which `just third-party` regenerates from the
Cargo metadata and the crates' own license files; a distributed build carries
it. Two need more than a listing:

- The camera raw decoder is LGPL-2.1: `vendor/rawler` (MIT and LGPL-2.1 per
  file, copyright Daniel Vogelbacher) inside `crates/lapstack-raw`, RAGTUX's
  own LGPL-2.1 shim around it. The LGPL lets a proprietary program use the
  library on the condition that the user can replace it, so lapstack never
  links it: `lapstack-raw` is built as a shared library the CLI loads at run
  time and as a wasm module the browser app loads beside its engine, and its
  source — the crate and the patched rawler, with a workspace file so it
  builds as it is — travels with every build (`web/dist.sh` puts
  `legal/lapstack-raw-src.tar.gz` in the app; the CLI's downloads carry the
  same tarball). `vendor/rawler/LAPSTACK-PATCH.md` has the obligations in
  full; `crates/lapstack-raw/README.md` how to rebuild and drop in a
  replacement.
- The papers the algorithm is written from are cited in `docs/README.md`
  rather than redistributed: they are their authors' and publishers' work,
  under their own copyright.

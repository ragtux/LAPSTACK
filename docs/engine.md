# The engine, natively

The design of the command-line tool and the core it is built on, section by
section, each naming the file it describes; every formula the engine
implements is stated here with the deviations from the papers called out.
The papers are cited in `README.md` beside this file, the build commands are
in the repository README, and `browser.md` is the same for the browser app.

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
another editor; the browser app makes them on demand instead (`browser.md`, *Retouch*). Each
slab streams its frames from disk again. Where a warped frame does not reach, the warp
repeats its edge, so the output (image, depth and confidence maps) is
**cropped to the largest rectangle every frame covers** with real pixels
(`align::common_area`: each frame's sound area is a convex quad, cut per pixel
row into an interval, intersected over frames, and the best rectangle over
consecutive rows is taken; the interpolation kernel's support — 2 px for
spline4x4 — is kept out); `--no-crop` keeps the full frame, and `--restretch`
resamples the cropped outputs (the image, the depth and confidence maps, the
weighted average, the slabs) back to the frame's size with the alignment's
kernel, each axis on its own — a stretch of a percent or so, since the crop's
aspect is not quite the frame's — for a session whose stacks must all come out
one size.

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

**A project's settings on the command line** (`--config FILE`; `project.rs`):
the browser app's project file (`<name>.lapstack.json`, `browser.md`, *Project files*) carries every
setting of its panel and Save step, and `--config` reads it: the settings go
in front of the command line as the flags they stand for — alignment, fusion,
depth, the weighted average, the batch split, the dust map (its file looked
for beside the project file), the scale bar and caption, the crop window,
the Save step's metadata and crop switches, the near end, and the outputs
ticked there (depth, confidence, stereo, rocking, 3D model) — so the flags
given after them override, `-o` is `<name>_stacked.<ext>` in the Save step's
format unless given, and with no frames on the command line the project's
frames are the inputs (the excluded ones left out), each looked for beside
the project file by its path and then its name. The run logs the flags it
took and notes what has no command-line form (the depth-map rendering, the
animations, the content credentials) or was not found. So a stack is dialed
in by eye in the app, saved as a project, and run — or rerun on a bigger
machine, or scripted over a session's folders with `--split dir` — from the
shell; the app's **Copy command** button (`browser.md`, *Project files*) writes the same command
without the file.

**Tethered capture** (`--watch`, `--watch-quiet S`; `batch::Watcher`): with
`--watch` the inputs are folders, and after the frames already in them are
stacked as above the program stays up and watches them (and their direct
subfolders, for `--split dir`) for new image files. A file counts as arrived
once its size and modification time have held for 2 s — a camera or a tether
writes it in pieces — and a stack closes after `--watch-quiet` seconds without
a new frame (the gap of `--split gap:S` when there is one, else 10 s): the
frames that arrived are then split by the rule, numbered on from the last
stack and run, and the watch goes on, every stack's outputs named like a
batch's (`out_03.tif`, or the template's fields), so `-o` must point outside
the watched folder — an output landing there would be taken for a frame, and
the program refuses to start. A stack that fails is reported and the watch
goes on. `--skip` has no meaning for a list that grows and is refused;
`--cull` and every other setting apply to each stack as it closes. `--dry-run`
lists the stacks already there and says what would be watched. Ctrl-C ends
it; a stack being fused at that moment is lost. On the tether's side nothing
is needed beyond a folder the frames land in — Lightroom's tethered capture
and the camera makers' own tools all write one.

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
the frame. The linear DNG output (below) develops the raws another way. The core crate's `raw` feature (on by default) is only the
loader — `libloading` and the interface in `raw.rs`; without it a raw file
is refused. rawler itself is built into `lapstack-raw` alone, from the copy
vendored in `vendor/rawler` (which `crates/lapstack-raw` names by path; the
workspace excludes it as a member) with one change, so that it runs in the
browser: `std::time::Instant`, which the demosaic and the CR3 decoder use
for a timing log line, has no implementation on wasm32 and panics there, so
a shim reads zero on wasm (`vendor/rawler/LAPSTACK-PATCH.md` says how to
move to a newer rawler).

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
border is a cross-fade and not a step wherever a cell sits at the gate — by
the *guided filter* (He, Sun & Tang 2013, the depth pass's aggregation
filter) with the fused luma on the grid as guide and `--depth-agg`'s
regularizer, so the cross-fade follows the image's edges: along a silhouette
the near object's weight stops at its outline instead of bleeding over the
background behind it, and the background's stops at the object, where a box
mean carried each a window's width across and left a halo band of the
blurred frames along every depth edge (`--wav-box` keeps the box mean; on
the fruit stack the guided weights leave the banana's edge clean where the
box's scalloped it, thin the tomato's halo, sharpen the grapes a little, and
change a flat area by nothing measurable) — and taken back to full
resolution bilinearly. One more pass over the frames.

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

**Redundant frames** (`--cull PCT`; `fuse::winner_shares`): a stack shot with
a small focus step holds frames the pyramid takes nothing from — the near
duplicate of a neighbor, or a frame focused in front of or behind the subject
where nothing is sharp. The fold knows: its winner map (the level-2 map
`--depth winner` reports) says which frame won each cell, and every run logs
each frame's *share of the detail* — the fraction of the cells with detail
(a winning region energy above 1 % of the plane's largest, so a flat area's
noise winners do not hand every frame its 1/N) it won — as a line naming the
least and the most and the frames under 1 %, redundant to the pyramid.
`--cull PCT` leaves such frames out before the run: a quick fold at `--draft
2` (the run's own draft when coarser) with the alignment as set and no other
output finds the shares, the frames under PCT % are dropped and listed, and
the run goes without them; with `--split` every stack is culled on its own.
On z-stackr's 28-frame 8 MP sample `--cull 2` keeps 17 frames — the first
four and a run of six in the middle each won under 2 % — and the fold at
quarter size costs 0.7 s with CUDA.
The browser app shows each frame's share under its thumbnail after a run and
has a *cull* button in the filmstrip's head (`browser.md`, *Frame list*). The threshold is a
judgment: 1 % names the frames that contribute nothing, 2 to 5 % trims a
deep stack to the frames that carry it, and a frame culled still sits in the
pyramid's loss column — what it won, however little, is now taken from its
neighbors, slightly less sharp there.

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

## Depth from focus (`depth.rs`)

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

## Performance

On a stack of 8280×5520 16-bit TIFFs (RTX 3060, 128-thread host), the same
frames and alignment settings for every row; the whole run, depth pass
included:

| run | wall | of which fold (align + fuse + focus measure) / depth | peak RSS |
|---|--:|--:|--:|
| 25 frames, `lapstack --align-coarsen 2` (CPU) | 26 s | 19 s (0.75 s per frame) / 6.0 s | 8.7 GB |
| 25 frames, `lapstack --gpu --gpu-align --align-coarsen 2` | 9.6 s | 5.1 s (0.20 s per frame) / 1.0 s | 6.2 GB |
| 25 frames, the same through wgpu (`--gpu-backend wgpu`, Vulkan on the same card) | 11.3 s | 8.3 s (0.33 s per frame) / 1.5 s | 6.2 GB |
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

**The wgpu engine** (`crates/lapstack-core/src/wg/`, the `wgpu` feature) is
the browser app's: the WGSL kernels (`shaders.wgsl`), the fold (`fold.rs`),
the aligner (`align.rs`, the cost search's evaluations on the device and the
simplex on the host) and the depth pass (`depth.rs`) moved from
`lapstack-web` into the core so the two have one copy, with a layer
(`gpu.rs`) that is WebGPU in the browser and Vulkan, Metal or DX12 natively
— the differences are the backend asked for, how the device is waited on
(the browser's event loop runs the map callbacks; natively `Gpu::wait`
blocks on the queue and `block_on` drives the futures) and nothing in the
kernels. `engine.rs` is the native driver: `WgFrames` keeps the frames on
the device (upload as 16-bit, registration, warp, brightness gains — the
twin of `gpu::GpuFrames`) and `WgFuser` folds them, takes each frame's focus
slice as it passes and runs the depth pass after the collapse (the twin of
`gpu::GpuFuser` and `gpu::depth_from_slices`), one device per thread shared
by the two. The fusion is the browser's to the bit: on the 8-frame test
stack the native wgpu result matches the CPU's to one 8-bit count on three
pixels in a million (70.8 dB), the depth map to 59 dB (the slices are
quantized to 16 bits on the way, as in the browser). The registration is the
browser's too: on the 25-frame fruit stack the wgpu aligner's transforms
differ from the CPU's by up to 0.8 px at the far end of the chain (frame 24:
dx −2.87 against −2.24 px, dy −0.79 against 0.00, the scale 1.06481 against
1.06452), within the search's own stopping tolerance at `--align-coarsen 2`
(a tenth of a level-2 pixel, 0.4 px of the frame), where the CUDA aligner, a
transcription of the CPU's kernel for kernel on the same pyramid, lands on
the CPU's numbers to the last digit printed (the images agree to 82 dB; the
wgpu image to 40 dB, the sub-pixel difference on 45 MP of texture). What the CUDA
engine has that the wgpu one has not: nothing the CLI exposes; what the
browser has that the native wgpu run has not: the proxies, the peaking map,
the renders and the retouch, which are the page's.

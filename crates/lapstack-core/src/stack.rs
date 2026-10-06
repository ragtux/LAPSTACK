// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

//! Orchestration: decode → (align) → fuse → (depth) → (slabs). Frames are
//! consumed one at a time, so the stack is streamed from disk with a bounded
//! read-ahead and memory stays at a few frames regardless of stack size; with
//! alignment each frame is registered to the previous one and warped as it is
//! decoded (`AlignedFrames`), and only the transforms are kept. The depth
//! pass, the weighted average and each slab stream the frames again.

use crate::depth::{self, DepthMap, DepthParams};
use crate::fuse::{FuseParams, Fuser};
use crate::align::{self, AlignParams, Backend, Interp, Rect, Sim, common_area};
use crate::brightness;
use crate::dng::DngInfo;
use crate::dust::DustMap;
use crate::prep;
use crate::io::{self, Depth};
use crate::pyramid::{crop_plane, Img3};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::VecDeque;
use std::thread::JoinHandle;
use std::time::Instant;

#[derive(Clone)]
pub struct Params {
    pub fuse: FuseParams,
    /// `None` = frames are already registered.
    pub align: Option<AlignParams>,
    pub save_aligned: Option<String>,
    /// Where the fusion and the depth pass run (`align::Backend`); a backend
    /// the build lacks falls back to the CPU with a note.
    pub backend: Backend,
    /// Depth-from-focus pass after fusion (`depth.rs`); `None` = report the
    /// pyramid winner map of `fuse.depth_level` instead (no extra pass).
    pub depth: Option<DepthParams>,
    /// Crop the result (image, depth, confidence) to the area every aligned
    /// frame covers with real pixels (`align::common_area`).
    pub crop: bool,
    /// Bring every frame to frame 0's brightness, one gain per channel over
    /// the area the frame covers (`brightness`): exposure flicker.
    pub brightness: bool,
    /// Slabs: after the result, fuse every run of `size`
    /// consecutive frames overlapping by `overlap` on its own, with the same
    /// settings, and hand each to `run_with`'s callback — thick planes of
    /// focus to retouch from elsewhere. `None` = no slabs.
    pub slabs: Option<(usize, usize)>,
    /// The weighted average (`wav.rs`) as a second image,
    /// from the depth pass's focus measure: needs `depth`.
    pub wav: Option<crate::wav::WavParams>,
    /// Dust map (`dust.rs`): the spots taken out of every frame as decoded,
    /// before alignment. Must be of the frames' size.
    pub dust: Option<DustMap>,
    /// Quarter turns clockwise every frame is given as decoded (`--rotate`),
    /// after the dust map (which is in the sensor's orientation).
    pub rotate: u8,
    /// A draft run: every frame block-averaged by `2^reduce` as decoded
    /// (`--draft`), so the whole run is a quick check of the settings.
    pub reduce: usize,
    /// A linear-DNG run (`dng.rs`): raws are developed to their camera space
    /// and fused in the look space made from frame 0's white balance and
    /// matrix, which `Output::dng` carries for the writer.
    pub dng: bool,
    /// A window of the frame (after `rotate`, in full-resolution pixels) the
    /// outputs are cut to (`--crop`), on top of the automatic crop.
    pub crop_rect: Option<Rect>,
    /// Resample the cropped outputs back to the frame's size (`--restretch`),
    /// each axis on its own, so every stack of a session comes out one size.
    pub restretch: bool,
}

/// The slab ranges of a `count`-frame stack: `size` frames each, consecutive
/// slabs overlapping by `overlap` (clamped below `size`, so every slab
/// advances), the last one cut short at the last frame.
pub fn slab_ranges(count: usize, size: usize, overlap: usize) -> Vec<(usize, usize)> {
    let mut slabs = Vec::new();
    if count == 0 {
        return slabs;
    }
    let size = size.clamp(1, count);
    let overlap = overlap.min(size - 1);
    let mut lo = 0;
    loop {
        let hi = (lo + size - 1).min(count - 1);
        slabs.push((lo, hi));
        if hi + 1 >= count {
            return slabs;
        }
        lo = hi + 1 - overlap;
    }
}

/// One fused slab, as `run_with` hands it out: cropped like the result.
pub struct Slab<'a> {
    pub index: usize,
    pub count: usize,
    pub lo: usize,
    pub hi: usize,
    pub image: &'a Img3,
    pub bit_depth: Depth,
    /// The look space of a linear-DNG run, for a slab written as a DNG.
    pub dng: Option<DngInfo>,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            fuse: FuseParams::default(),
            align: Some(AlignParams::default()),
            save_aligned: None,
            backend: Backend::Cpu,
            depth: Some(DepthParams::default()),
            crop: true,
            brightness: true,
            slabs: None, wav: None, dust: None,
            rotate: 0, reduce: 0, dng: false, crop_rect: None, restretch: false,
        }
    }
}

/// CPU or GPU accumulator behind one interface.
enum AnyFuser {
    Cpu(Fuser),
    #[cfg(feature = "gpu")]
    Gpu(crate::gpu::GpuFuser),
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    Wg(crate::wg::engine::WgFuser),
}

impl AnyFuser {
    fn levels(&self) -> usize {
        match self {
            AnyFuser::Cpu(f) => f.levels,
            #[cfg(feature = "gpu")]
            AnyFuser::Gpu(f) => f.levels,
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            AnyFuser::Wg(f) => f.levels,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            AnyFuser::Cpu(_) => "CPU",
            #[cfg(feature = "gpu")]
            AnyFuser::Gpu(_) => "GPU (CUDA)",
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            AnyFuser::Wg(_) => "GPU (wgpu)",
        }
    }
    /// The fuser takes the frames' focus slices itself (the wgpu engine).
    fn measures(&self) -> bool {
        match self {
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            AnyFuser::Wg(f) => f.measures(),
            _ => false,
        }
    }
    fn push(&mut self, frame: &Img3) -> Result<(), String> {
        match self {
            AnyFuser::Cpu(f) => {
                f.push(frame);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            AnyFuser::Gpu(f) => f.push(frame),
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            AnyFuser::Wg(f) => f.push(frame),
        }
    }
    fn shares(&self) -> Result<Vec<f32>, String> {
        match self {
            AnyFuser::Cpu(f) => Ok(f.shares()),
            #[cfg(feature = "gpu")]
            AnyFuser::Gpu(f) => f.shares(),
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            AnyFuser::Wg(f) => f.shares(),
        }
    }
    /// The fused image, the winner map, and the depth map when the fuser
    /// took the slices itself (`measures`).
    fn finish(self, log: &mut dyn FnMut(String)) -> Result<(Img3, Vec<f32>, Option<DepthMap>), String> {
        let _ = &log;
        match self {
            AnyFuser::Cpu(f) => {
                let (i, d) = f.finish();
                Ok((i, d, None))
            }
            #[cfg(feature = "gpu")]
            AnyFuser::Gpu(f) => {
                let (i, d) = f.finish()?;
                Ok((i, d, None))
            }
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            AnyFuser::Wg(mut f) => {
                let (i, d) = f.finish()?;
                let dm = if f.measures() { Some(f.depth(log)?) } else { None };
                Ok((i, d, dm))
            }
        }
    }
}

/// Sequential access to the (aligned) frames of a stack. Fusion and the depth
/// pass consume frames strictly one at a time, so the source can be the
/// in-memory stack or a decode-on-demand reader.
pub trait FrameSource {
    fn len(&self) -> usize;
    /// Level-0 frame dimensions.
    fn dims(&self) -> (usize, usize);
    fn get(&mut self, i: usize) -> Result<Cow<'_, Img3>, String>;
    /// Frame `i` on the CUDA device, with its focus slice when `measure`
    /// asks, for a source that keeps its frames there (`AlignedFrames` with
    /// `--gpu-align`); `None` from one that does not, and `get` serves it.
    #[cfg(feature = "gpu")]
    fn get_gpu(&mut self, _i: usize, _measure: Option<&DepthParams>) -> Result<Option<crate::gpu::DeviceFrame<'_>>, String> {
        Ok(None)
    }
    /// Frame `i` on the wgpu device (3 f32 planes), for a source that keeps
    /// its frames there (`AlignedFrames` with the wgpu aligner); `None` from
    /// one that does not.
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn get_wg(&mut self, _i: usize) -> Result<Option<&wgpu::Buffer>, String> {
        Ok(None)
    }
    /// Log lines the source made while serving frames (a frame's
    /// registration), taken out once; the caller logs them.
    fn take_log(&mut self) -> Vec<String> {
        Vec::new()
    }
}

impl FrameSource for &[Img3] {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn dims(&self) -> (usize, usize) {
        (self[0].w, self[0].h)
    }
    fn get(&mut self, i: usize) -> Result<Cow<'_, Img3>, String> {
        Ok(Cow::Borrowed(&self[i]))
    }
}

pub struct Output {
    pub image: Img3,
    /// Depth map: fractional frame index per pixel (DFF), or the pyramid
    /// winner map when `Params::depth` is `None`.
    pub depth: Vec<f32>,
    /// DFF confidence in [0,1] per pixel (`None` for the winner map).
    pub conf: Option<Vec<f32>>,
    pub bit_depth: Depth,
    pub align: Vec<Sim>,
    pub levels: usize,
    /// The window of the full frame the outputs were cropped to (`Params::crop`),
    /// `None` when nothing was cut.
    pub crop: Option<Rect>,
    /// The weighted average (`Params::wav`), cropped like `image`.
    pub wav: Option<Img3>,
    /// The space of a linear-DNG run (`Params::dng`): what the DNG writer
    /// needs to take the look image back to the camera's space.
    pub dng: Option<DngInfo>,
    /// The draft factor: the outputs are `1/2^reduce` of the frames' size.
    pub reduce: usize,
    /// Which end of the stack is near, as far as the run can tell
    /// (`near_end`): `Some((near_first, why))` when a cue decided it.
    pub near: Option<(bool, String)>,
    /// Each frame's share of the detail the pyramid took from it
    /// (`fuse::winner_shares`); empty when the winner map was not recorded.
    pub shares: Vec<f32>,
}

/// Decoder threads kept in flight ahead of the fuser (each holds one decoded
/// frame, ~550 MB at 45 MP). The GPU fuser needs ~80 ms per frame and a
/// 16-bit PNG decode takes ~1.2 s, so a few decoders are needed to feed it.
const READ_AHEAD: usize = 4;

/// Decode-on-demand frame source with a bounded read-ahead queue.
struct LazyFrames {
    paths: Vec<String>,
    w: usize,
    h: usize,
    depth: Depth,
    /// Last decoded frame and its index (a second pass re-decodes).
    cur: Option<(usize, Img3)>,
    pending: VecDeque<(usize, JoinHandle<Result<Decoded, String>>)>,
    notes: Vec<String>,
    /// Brightness normalization: frame 0's channel means, and each frame's
    /// gains once found (the depth pass decodes the frames a second time).
    ref_means: Option<[f64; 3]>,
    gains: Vec<Option<[f32; 3]>>,
    /// The dust map, applied to every frame as it is decoded.
    dust: Option<DustMap>,
    /// The frames' size as decoded (the dust map's), before the turn and the draft reduction.
    raw_w: usize,
    raw_h: usize,
    /// `Params::rotate` and `Params::reduce`.
    rotate: u8,
    reduce: usize,
    /// The look space of a linear-DNG run: frame 0's (`Params::dng`).
    look: Option<DngInfo>,
}

/// One frame decoded: the image, its bit depth, and its own color space in a
/// linear-DNG run (to catch a raw among TIFFs or the reverse).
type Decoded = (Img3, Depth, Option<DngInfo>);

/// A decoded frame turned (`Params::rotate`) and block-averaged for a draft
/// (`Params::reduce`), in that order.
fn prep_frame(img: Img3, rotate: u8, reduce: usize) -> Img3 {
    let img = if rotate % 4 != 0 { prep::rotate(&img, rotate, false) } else { img };
    if reduce > 0 { prep::reduce(&img, reduce) } else { img }
}

fn decode(path: &str, look: Option<&DngInfo>) -> Result<Decoded, String> {
    match look {
        Some(l) => io::load_look(path, l).map(|(i, d, own)| (i, d, Some(own))),
        None => io::load_rgb(path).map(|(i, d)| (i, d, None)),
    }
}

impl LazyFrames {
    fn open(paths: Vec<String>, brightness: bool, params: &Params) -> Result<LazyFrames, String> {
        let dust = params.dust.as_ref();
        let (mut f0, depth, look) = if params.dng {
            let (i, d, info) = io::load_look_first(&paths[0])?;
            (i, d, Some(info))
        } else {
            let (i, d) = io::load_rgb(&paths[0])?;
            (i, d, None)
        };
        let (raw_w, raw_h) = (f0.w, f0.h);
        if let Some(d) = dust {
            check_dust(d, f0.w, f0.h)?;
            d.apply(&mut f0);
        }
        let f0 = prep_frame(f0, params.rotate, params.reduce);
        let ref_means = brightness.then(|| brightness::means(&f0));
        let n = paths.len();
        let mut lf = LazyFrames { w: f0.w, h: f0.h, depth, paths, cur: Some((0, f0)), pending: VecDeque::new(), notes: Vec::new(), ref_means, gains: vec![None; n], dust: dust.cloned(), raw_w, raw_h, rotate: params.rotate, reduce: params.reduce, look };
        lf.gains[0] = Some([1.0; 3]);
        lf.prefetch(1);
        Ok(lf)
    }
    /// One line on the gains found, for the log (after the run).
    fn brightness_note(&self) -> Option<String> {
        self.ref_means?;
        let gs: Vec<f32> = self.gains.iter().flatten().flat_map(|g| g.iter().copied()).collect();
        let (lo, hi) = gs.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        Some(format!("brightness equalized to frame 0: gains {lo:.3} … {hi:.3}"))
    }
    /// Keep decoders running for frames `from..from+READ_AHEAD`.
    fn prefetch(&mut self, from: usize) {
        let mut next = self.pending.back().map_or(from, |(i, _)| i + 1).max(from);
        while self.pending.len() < READ_AHEAD && next < self.paths.len() {
            let p = self.paths[next].clone();
            let look = self.look.clone();
            self.pending.push_back((next, std::thread::spawn(move || decode(&p, look.as_ref()))));
            next += 1;
        }
    }
    fn take(&mut self, i: usize) -> Result<Img3, String> {
        if let Some((ci, _)) = &self.cur {
            if *ci == i {
                return Ok(self.cur.take().unwrap().1);
            }
        }
        // drop stale entries (random access is not expected, but stay correct)
        while self.pending.front().is_some_and(|(pi, _)| *pi < i) {
            let (_, jh) = self.pending.pop_front().unwrap();
            let _ = jh.join();
        }
        let (img, d, own) = match self.pending.front() {
            Some((pi, _)) if *pi == i => {
                let (_, jh) = self.pending.pop_front().unwrap();
                jh.join().map_err(|_| "decoder thread panicked".to_string())??
            }
            _ => decode(&self.paths[i], self.look.as_ref())?,
        };
        if d != self.depth {
            self.notes.push(format!(
                "{} is {}-bit but frame 0 is {}-bit; writing {}-bit output",
                self.paths[i], d.bits(), self.depth.bits(), self.depth.bits()
            ));
        }
        if let (Some(own), Some(look)) = (&own, &self.look) {
            if own.is_camera() != look.is_camera() {
                return Err(format!(
                    "{}: {} but frame 0 is {}; a linear DNG needs frames of one kind",
                    self.paths[i],
                    if own.is_camera() { "a camera raw" } else { "not a camera raw" },
                    if look.is_camera() { "a camera raw" } else { "not one" }
                ));
            }
        }
        let mut img = img;
        // the dust map is in the sensor's orientation and size: a frame of another size goes without it
        if let Some(d) = &self.dust {
            if img.w == self.raw_w && img.h == self.raw_h {
                d.apply(&mut img);
            } else {
                self.notes.push(format!("{}: {}x{} is not the dust map's size; no dust taken out of it", self.paths[i], img.w, img.h));
            }
        }
        let mut img = prep_frame(img, self.rotate, self.reduce);
        if img.w != self.w || img.h != self.h {
            // a frame of another size (a JPEG among the raws, another camera mode) is brought to
            // frame 0's size rather than refused: the alignment takes care of what is left
            self.notes.push(format!(
                "{}: {}x{} differs from frame 0 ({}x{}); resampled to frame 0's size",
                self.paths[i], img.w, img.h, self.w, self.h
            ));
            img = prep::resize(&img, self.w, self.h, crate::align::Interp::Spline4x4);
        }
        if let Some(r) = self.ref_means {
            let g = *self.gains[i].get_or_insert_with(|| brightness::gains_to(r, &img));
            brightness::apply(&mut img, g);
        }
        Ok(img)
    }
}

impl FrameSource for LazyFrames {
    fn len(&self) -> usize {
        self.paths.len()
    }
    fn dims(&self) -> (usize, usize) {
        (self.w, self.h)
    }
    fn get(&mut self, i: usize) -> Result<Cow<'_, Img3>, String> {
        if self.cur.as_ref().is_none_or(|(ci, _)| *ci != i) {
            let img = self.take(i)?;
            self.cur = Some((i, img));
        }
        self.prefetch(i + 1);
        Ok(Cow::Borrowed(&self.cur.as_ref().unwrap().1))
    }
}

/// The aligned stack, streamed: every frame is decoded on demand (`LazyFrames`,
/// with its read-ahead, dust map and size checks), registered to the previous
/// aligned frame the first time it is asked for — the fusion asks for the
/// frames in order, first — and warped with its transform every time, then
/// brought to frame 0's brightness. Between passes only the transforms, the
/// gains and frame 0 on the brightness grid are kept, so memory is that of a
/// few frames whatever the stack's length; the weighted average and the
/// slabs decode and warp the frames again (the depth pass runs on the focus
/// slices the fold took).
struct AlignedFrames {
    src: LazyFrames,
    a: AlignParams,
    free: [bool; Sim::N],
    /// With `--gpu-align`, the frames live on the device (`gpu::GpuFrames`):
    /// registration, warp, gains and focus slice all happen there, and a
    /// frame comes to the host only when something here asks for it.
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::GpuFrames>,
    /// The frame the device holds, once processed.
    #[cfg(feature = "gpu")]
    on_device: Option<usize>,
    /// The same on the wgpu device (`wg::engine::WgFrames`).
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    wg: Option<crate::wg::engine::WgFrames>,
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    on_wg: Option<usize>,
    /// The transforms found so far (frame 0's is the identity).
    sims: Vec<Sim>,
    /// The previous aligned frame's luma, while the transforms are being found.
    prev_ref: Option<Vec<f32>>,
    /// Frame 0 on the brightness grid (`Params::brightness`), and each frame's
    /// gains once found.
    reference: Option<brightness::Reference>,
    brightness: bool,
    gains: Vec<Option<[f32; 3]>>,
    save_dir: Option<String>,
    cur: Option<(usize, Img3)>,
    /// Each frame's registration as it is found, for `take_log`.
    lines: Vec<String>,
}

impl AlignedFrames {
    fn open(paths: Vec<String>, a: AlignParams, params: &Params) -> Result<AlignedFrames, String> {
        let src = LazyFrames::open(paths, false, params)?;
        #[cfg(feature = "gpu")]
        let gpu = if a.backend == Backend::Cuda { Some(crate::gpu::GpuFrames::new(src.w, src.h, a.interp)?) } else { None };
        #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
        let wg = if a.backend == Backend::Wgpu { Some(crate::wg::engine::WgFrames::new(src.w, src.h, a.interp)?) } else { None };
        let n = src.len();
        Ok(AlignedFrames {
            src,
            a,
            free: a.free(),
            #[cfg(feature = "gpu")]
            gpu,
            #[cfg(feature = "gpu")]
            on_device: None,
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            wg,
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            on_wg: None,
            sims: Vec::with_capacity(n),
            prev_ref: None,
            reference: None,
            brightness: params.brightness,
            gains: vec![None; n],
            save_dir: params.save_aligned.clone(),
            cur: None,
            lines: Vec::new(),
        })
    }

    /// Frame `i`'s transform: the one found before, or found now against the
    /// previous aligned frame (the frames come in order the first time).
    fn register(&mut self, i: usize, img: &Img3) -> Result<Sim, String> {
        if let Some(s) = self.sims.get(i) {
            return Ok(*s);
        }
        if i != self.sims.len() {
            return Err(format!("frame {i} asked for before frame {} was aligned", self.sims.len()));
        }
        let (w, h) = (img.w, img.h);
        let y = align::luma(img);
        let sim = match &self.prev_ref {
            None => Sim::id(),
            Some(rf) => align::multiscale_align(rf, &y, w, h, *self.sims.last().unwrap(), self.free, self.a.coarsen),
        };
        // the next frame is registered to this one as aligned; after the last, nothing is
        self.prev_ref = (i + 1 < self.src.len()).then(|| if sim == Sim::id() { y } else { align::warp_plane(&y, w, h, &sim, w, h, self.a.interp).0 });
        self.sims.push(sim);
        Ok(sim)
    }

    fn brightness_note(&self) -> Option<String> {
        if !self.brightness {
            return None;
        }
        let gs: Vec<f32> = self.gains.iter().flatten().flat_map(|g| g.iter().copied()).collect();
        let (lo, hi) = gs.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        Some(format!("brightness equalized to frame 0: gains {lo:.3} … {hi:.3}"))
    }

    /// Frame `i` through the device: decoded here, then registered (the first
    /// time), warped and equalized there; its focus slice when `measure`
    /// asks. Afterward the device holds it.
    #[cfg(feature = "gpu")]
    fn process_gpu(&mut self, i: usize, measure: Option<&DepthParams>) -> Result<Option<Vec<f32>>, String> {
        let img = self.src.take(i)?;
        self.src.prefetch(i + 1);
        let (w, h) = (img.w, img.h);
        let first = self.sims.len() <= i;
        if first && i != self.sims.len() {
            return Err(format!("frame {i} asked for before frame {} was aligned", self.sims.len()));
        }
        let known = (!first).then(|| (self.sims[i], self.gains[i].unwrap_or([1.0; 3])));
        let guess = self.sims.last().copied().unwrap_or(Sim::id());
        let gpu = self.gpu.as_mut().unwrap();
        let (sim, gain, focus) = gpu.process(&img, guess, self.free, self.a.coarsen, known, self.brightness, measure)?;
        drop(img);
        self.on_device = Some(i);
        if first {
            self.sims.push(sim);
            self.gains[i] = Some(gain);
            if i > 0 {
                self.lines.push(format!(
                    "  frame {i:>3}: {}{}",
                    align::report(&sim, w, h),
                    if self.brightness { format!("  brightness {}", brightness::describe(gain)) } else { String::new() }
                ));
            }
            if let Some(dir) = &self.save_dir {
                let img = self.gpu.as_ref().unwrap().download()?;
                io::save_rgb(&img, &format!("{dir}/aligned_{i:03}.png"), self.src.depth, None)?;
            }
        }
        Ok(focus)
    }
}

impl AlignedFrames {
    /// Frame `i` through the wgpu device (`process_gpu`'s twin): the fuser
    /// takes its focus slice itself there.
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn process_wg(&mut self, i: usize) -> Result<(), String> {
        let img = self.src.take(i)?;
        self.src.prefetch(i + 1);
        let (w, h) = (img.w, img.h);
        let first = self.sims.len() <= i;
        if first && i != self.sims.len() {
            return Err(format!("frame {i} asked for before frame {} was aligned", self.sims.len()));
        }
        let known = (!first).then(|| (self.sims[i], self.gains[i].unwrap_or([1.0; 3])));
        let guess = self.sims.last().copied().unwrap_or(Sim::id());
        let wg = self.wg.as_mut().unwrap();
        let (sim, gain) = wg.process(&img, guess, self.free, self.a.coarsen, known, self.brightness)?;
        drop(img);
        self.on_wg = Some(i);
        if first {
            self.sims.push(sim);
            self.gains[i] = Some(gain);
            if i > 0 {
                self.lines.push(format!(
                    "  frame {i:>3}: {}{}",
                    align::report(&sim, w, h),
                    if self.brightness { format!("  brightness {}", brightness::describe(gain)) } else { String::new() }
                ));
            }
            if let Some(dir) = &self.save_dir {
                let img = self.wg.as_ref().unwrap().download()?;
                io::save_rgb(&img, &format!("{dir}/aligned_{i:03}.png"), self.src.depth, None)?;
            }
        }
        Ok(())
    }
}

impl FrameSource for AlignedFrames {
    fn len(&self) -> usize {
        self.src.len()
    }
    fn dims(&self) -> (usize, usize) {
        (self.src.w, self.src.h)
    }
    #[cfg(feature = "gpu")]
    fn get_gpu(&mut self, i: usize, measure: Option<&DepthParams>) -> Result<Option<crate::gpu::DeviceFrame<'_>>, String> {
        if self.gpu.is_none() {
            return Ok(None);
        }
        let focus = self.process_gpu(i, measure)?;
        Ok(Some(crate::gpu::DeviceFrame { planes: self.gpu.as_mut().unwrap().planes(), focus }))
    }
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn get_wg(&mut self, i: usize) -> Result<Option<&wgpu::Buffer>, String> {
        if self.wg.is_none() {
            return Ok(None);
        }
        self.process_wg(i)?;
        Ok(Some(self.wg.as_ref().unwrap().frame()))
    }
    fn get(&mut self, i: usize) -> Result<Cow<'_, Img3>, String> {
        #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
        if self.wg.is_some() {
            if self.cur.as_ref().is_none_or(|(ci, _)| *ci != i) {
                if self.on_wg != Some(i) {
                    self.process_wg(i)?;
                }
                let img = self.wg.as_ref().unwrap().download()?;
                self.cur = Some((i, img));
            }
            return Ok(Cow::Borrowed(&self.cur.as_ref().unwrap().1));
        }
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            if self.cur.as_ref().is_none_or(|(ci, _)| *ci != i) {
                if self.on_device != Some(i) {
                    self.process_gpu(i, None)?;
                }
                let img = self.gpu.as_ref().unwrap().download()?;
                self.cur = Some((i, img));
            }
            return Ok(Cow::Borrowed(&self.cur.as_ref().unwrap().1));
        }
        if self.cur.as_ref().is_none_or(|(ci, _)| *ci != i) {
            let mut img = self.src.take(i)?;
            self.src.prefetch(i + 1);
            let (w, h) = (img.w, img.h);
            let first = self.sims.len() <= i;
            let sim = self.register(i, &img)?;
            if sim != Sim::id() {
                let (mut warped, valid) = align::warp_img3(&img, &sim, self.a.interp);
                // where the warp reaches outside the frame, the pixel as shot (the edge repeated would smear)
                for (o, s) in warped.p.iter_mut().zip(&img.p) {
                    o.par_chunks_mut(w).zip(s.par_chunks(w)).zip(valid.par_chunks(w)).for_each(|((o, s), v)| {
                        for x in 0..w {
                            if v[x] == 0 {
                                o[x] = s[x];
                            }
                        }
                    });
                }
                img = warped;
            }
            let mut gain = [1f32; 3];
            if self.brightness {
                if i == 0 {
                    self.reference.get_or_insert_with(|| brightness::Reference::new(&img));
                    self.gains[0] = Some(gain);
                } else if let Some(r) = &self.reference {
                    gain = *self.gains[i].get_or_insert_with(|| r.gains(&img, &sim));
                    brightness::apply(&mut img, gain);
                }
            }
            if first {
                if i > 0 {
                    self.lines.push(format!(
                        "  frame {i:>3}: {}{}",
                        align::report(&sim, w, h),
                        if self.brightness { format!("  brightness {}", brightness::describe(gain)) } else { String::new() }
                    ));
                }
                if let Some(dir) = &self.save_dir {
                    io::save_rgb(&img, &format!("{dir}/aligned_{i:03}.png"), self.src.depth, None)?;
                }
            }
            self.cur = Some((i, img));
        }
        Ok(Cow::Borrowed(&self.cur.as_ref().unwrap().1))
    }
    fn take_log(&mut self) -> Vec<String> {
        std::mem::take(&mut self.lines)
    }
}

fn fuse_all(
    src: &mut dyn FrameSource,
    params: &FuseParams,
    backend: Backend,
    measure: Option<&DepthParams>,
    log: &mut dyn FnMut(String),
) -> Result<Fused, String> {
    let last = src.len() - 1;
    fuse_range(src, 0, last, params, backend, measure, log)
}

/// What a fold leaves: the image, the winner map, the level count, the
/// frames' focus slices (when the pass is on and the fuser did not run it
/// itself), each frame's share of the detail, and the depth map when the
/// fuser ran the pass (the wgpu engine).
struct Fused {
    image: Img3,
    winner: Vec<f32>,
    levels: usize,
    slices: Vec<Vec<f32>>,
    shares: Vec<f32>,
    depth: Option<DepthMap>,
}

/// Fuse frames `lo..=hi` of the source. With `measure`, each frame's focus
/// slice (`depth::focus_slice`) is taken as it passes and the slices come
/// back with the result, so the depth pass needs no second pass over the
/// frames.
fn fuse_range(
    src: &mut dyn FrameSource,
    lo: usize,
    hi: usize,
    params: &FuseParams,
    backend: Backend,
    measure: Option<&DepthParams>,
    log: &mut dyn FnMut(String),
) -> Result<Fused, String> {
    let (w, h) = src.dims();
    let mut fuser = match backend {
        Backend::Cpu => AnyFuser::Cpu(Fuser::new(w, h, params.clone())),
        #[cfg(feature = "gpu")]
        Backend::Cuda => AnyFuser::Gpu(crate::gpu::GpuFuser::new(w, h, params.clone())?),
        #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
        Backend::Wgpu => AnyFuser::Wg(crate::wg::engine::WgFuser::new(w, h, params.clone(), measure)?),
        #[allow(unreachable_patterns)]
        other => {
            log(format!("the {} backend is not in this build; fusing on the CPU", other.name()));
            AnyFuser::Cpu(Fuser::new(w, h, params.clone()))
        }
    };
    let levels = fuser.levels();
    log(format!(
        "fusing {} frames{} on the {}: {} band-pass levels + residual ({}x{}), energy window {}x{}, top rule {:?}",
        hi + 1 - lo,
        if lo == 0 && hi + 1 == src.len() { String::new() } else { format!(" ({lo}..{hi})") },
        fuser.name(),
        levels,
        {
            let mut x = w;
            for _ in 0..levels { x = crate::pyramid::half(x); }
            x
        },
        {
            let mut y = h;
            for _ in 0..levels { y = crate::pyramid::half(y); }
            y
        },
        2 * params.energy_radius + 1,
        2 * params.energy_radius + 1,
        params.top_rule
    ));
    let t = Instant::now();
    let mut slices = Vec::with_capacity(if measure.is_some() { hi + 1 - lo } else { 0 });
    for i in lo..=hi {
        #[allow(unused_mut)]
        let mut done = false;
        #[cfg(feature = "gpu")]
        if let AnyFuser::Gpu(f) = &mut fuser {
            if let Some(df) = src.get_gpu(i, measure)? {
                f.push_device(df.planes)?;
                slices.extend(df.focus);
                done = true;
            }
        }
        #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
        if let AnyFuser::Wg(f) = &mut fuser {
            if let Some(buf) = src.get_wg(i)? {
                f.push_buf(buf)?;
                done = true;
            }
        }
        if !done {
            let f = src.get(i)?;
            let own = fuser.measures();
            fuser.push(&f)?;
            if let Some(dp) = measure.filter(|_| !own) {
                slices.push(depth::focus_slice(&f, dp));
            }
        }
        for line in src.take_log() {
            log(line);
        }
        log(format!(
            "  frame {:>3}/{} folded{}  ({:.1}s)",
            i + 1 - lo,
            hi + 1 - lo,
            if measure.is_some() { " and measured" } else { "" },
            t.elapsed().as_secs_f64()
        ));
    }
    let shares = fuser.shares()?;
    let (image, winner, depth) = fuser.finish(log)?;
    log(format!("collapsed  ({:.1}s)", t.elapsed().as_secs_f64()));
    Ok(Fused { image, winner, levels, slices, shares, depth })
}

/// Fusion, with the optional depth from focus: the frames' focus slices are
/// taken during the fold and the depth pass runs on them once the fused
/// image, its guide, exists. Returns the image, the depth map, its
/// confidence, the pyramid's level count and the depth pass's noise floor
/// (`DepthMap::floor`, for the weighted average).
fn fuse_and_depth(
    src: &mut dyn FrameSource,
    params: &Params,
    log: &mut dyn FnMut(String),
) -> Result<(Img3, Vec<f32>, Option<Vec<f32>>, usize, Option<Vec<f32>>, Vec<f32>), String> {
    let Fused { image, winner, levels, slices, shares, depth } = fuse_all(src, &params.fuse, params.backend, params.depth.as_ref(), log)?;
    share_note(&shares, log);
    match (&params.depth, depth) {
        (Some(_), Some(dm)) => Ok((image, dm.depth, Some(dm.conf), levels, Some(dm.floor), shares)),
        (Some(dp), None) => {
            let n = slices.len();
            let mut it = slices.into_iter().map(Ok);
            #[cfg(feature = "gpu")]
            let dm = if params.backend == Backend::Cuda { crate::gpu::depth_from_slices(&mut it, n, &image, dp, log)? } else { depth::depth_from_slices(&mut it, n, &image, dp, log)? };
            #[cfg(not(feature = "gpu"))]
            let dm = depth::depth_from_slices(&mut it, n, &image, dp, log)?;
            Ok((image, dm.depth, Some(dm.conf), levels, Some(dm.floor), shares))
        }
        (None, _) => Ok((image, winner, None, levels, None, shares)),
    }
}

/// The weighted average of `Params::wav`, another pass over the frames; it
/// borrows the depth pass's focus measure and noise floor, so it needs
/// `Params::depth`.
fn weighted(src: &mut dyn FrameSource, params: &Params, floor: Option<&[f32]>, guide: &Img3, log: &mut dyn FnMut(String)) -> Result<Option<Img3>, String> {
    match (&params.wav, &params.depth, floor) {
        (Some(wp), Some(dp), Some(floor)) => Ok(Some(crate::wav::weighted_average(src, dp, wp, floor, Some(guide), log)?)),
        (Some(_), _, _) => {
            log("the weighted average needs the depth-from-focus pass (not the winner map): skipped".into());
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// After the result: the slabs of `Params::slabs`, each fused on its own
/// over the same (aligned, equalized) frames, cropped like the result, and
/// handed to `on_slab` one at a time — none is kept.
fn fuse_slabs(
    src: &mut dyn FrameSource,
    params: &Params,
    bit_depth: Depth,
    crop: Option<&Rect>,
    restretch: Option<(usize, usize)>,
    dng: Option<&DngInfo>,
    log: &mut dyn FnMut(String),
    on_slab: &mut dyn FnMut(Slab<'_>) -> Result<(), String>,
) -> Result<(), String> {
    let Some((size, overlap)) = params.slabs else { return Ok(()) };
    let ranges = slab_ranges(src.len(), size, overlap);
    log(format!("{} slabs of {size} frames, overlap {overlap}", ranges.len()));
    for (k, &(lo, hi)) in ranges.iter().enumerate() {
        log(format!("slab {}/{}: frames {lo}..{hi}", k + 1, ranges.len()));
        let image = fuse_range(src, lo, hi, &params.fuse, params.backend, None, log)?.image;
        let image = match crop { Some(r) => image.crop(r), None => image };
        let image = match restretch { Some((w, h)) => prep::resize(&image, w, h, Interp::Spline4x4), None => image };
        on_slab(Slab { index: k, count: ranges.len(), lo, hi, image: &image, bit_depth, dng: dng.cloned() })?;
    }
    Ok(())
}

/// Which end of the stack is near, from the cues a run has: the focus
/// distance the camera wrote into the first and last frames (EXIF
/// SubjectDistance) decides it — the focus went from the smaller distance to
/// the larger one. The frames' scale over the stack is reported as a hint
/// but does not decide: on a rail, or with a lens that extends to focus
/// closer, the near frames are the larger ones and a stack shot near to far
/// shrinks with its index; with an internal-focus lens the field widens as
/// it focuses closer and the same stack grows — the sign says nothing
/// without knowing the lens. `None` when nothing decides it: the caller's
/// default (frame 0 near) stands.
fn near_end(inputs: &[String], sims: &[Sim], w: usize, h: usize, log: &mut dyn FnMut(String)) -> Option<(bool, String)> {
    let _ = (w, h);
    if inputs.len() < 2 {
        return None;
    }
    if let (Some(a), Some(b)) = (io::load_subject_distance(&inputs[0]), io::load_subject_distance(&inputs[inputs.len() - 1])) {
        if (a - b).abs() > 1e-6 * a.max(b) {
            let near_first = a < b;
            let why = format!("focus distance {:.3} m in the first frame, {:.3} m in the last (the focus distance the camera recorded): frame 0 is the {} end", a, b, if near_first { "near" } else { "far" });
            log(format!("near end: {why}"));
            return Some((near_first, why));
        }
    }
    if let (Some(first), Some(last)) = (sims.first(), sims.last()) {
        let k = last.scale / first.scale;
        if (k - 1.0).abs() > 0.002 {
            log(format!(
                "near end: no focus distance in the frames' metadata; the last frame is {} by {:.2} % to fit the first — on a rail, or with a lens that extends to focus closer, that means frame 0 is the {} end; an internal-focus lens breathes the other way. Frame 0 is taken as the near end unless --near-end says otherwise",
                if k > 1.0 { "enlarged" } else { "shrunk" }, (k - 1.0).abs() * 100.0, if k > 1.0 { "near" } else { "far" }
            ));
        }
    }
    None
}

/// What the winner shares say, in a line or two: the least share and its
/// frame, and the frames that won under `REDUNDANT_PCT` % of the detail —
/// what `--cull` would leave out.
fn share_note(shares: &[f32], log: &mut dyn FnMut(String)) {
    if shares.len() < 2 {
        return;
    }
    let (imin, &min) = shares.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    let (imax, &max) = shares.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    let low: Vec<String> = shares.iter().enumerate().filter(|(_, s)| **s * 100.0 < REDUNDANT_PCT).map(|(i, s)| format!("{i} ({:.1} %)", s * 100.0)).collect();
    log(format!("detail won per frame: least {:.1} % (frame {imin}), most {:.1} % (frame {imax}){}", min * 100.0, max * 100.0,
        if low.is_empty() { String::new() } else { format!("; under {REDUNDANT_PCT} %, redundant to the pyramid: {} — --cull {REDUNDANT_PCT} leaves such frames out", low.join(", ")) }));
}

/// Below this share of the detail a frame is reported as redundant.
pub const REDUNDANT_PCT: f32 = 1.0;

/// `Params::restretch`: the cropped outputs brought back to the frame's size
/// `w × h`, each axis on its own — a slight stretch, since the crop's aspect
/// differs a little from the frame's. Returns the size the slabs are to be
/// brought to as well (`None` when nothing was cropped).
#[allow(clippy::too_many_arguments)]
fn restretched(
    params: &Params,
    crop: Option<&Rect>,
    w: usize,
    h: usize,
    interp: Interp,
    image: Img3,
    depth: Vec<f32>,
    conf: Option<Vec<f32>>,
    wav: Option<Img3>,
    log: &mut dyn FnMut(String),
) -> (Img3, Vec<f32>, Option<Vec<f32>>, Option<Img3>, Option<(usize, usize)>) {
    let Some(r) = crop.filter(|_| params.restretch) else { return (image, depth, conf, wav, None) };
    if r.w == w && r.h == h {
        return (image, depth, conf, wav, None);
    }
    log(format!("restretched from {}x{} to the frame's {w}x{h} ({}; x by {:.3}, y by {:.3})", r.w, r.h, interp.name(), w as f64 / r.w as f64, h as f64 / r.h as f64));
    let image = prep::resize(&image, w, h, interp);
    let depth = prep::resize_plane(&depth, r.w, r.h, w, h, interp);
    let conf = conf.map(|c| prep::resize_plane(&c, r.w, r.h, w, h, interp).into_iter().map(|v| v.clamp(0.0, 1.0)).collect());
    let wav = wav.map(|i| prep::resize(&i, w, h, interp));
    (image, depth, conf, wav, Some((w, h)))
}

/// `Params::crop_rect` on the run's grid: given in full-resolution pixels of
/// the turned frame, brought down by the draft factor, cut to the frame.
fn user_crop(params: &Params, w: usize, h: usize, log: &mut dyn FnMut(String)) -> Result<Option<Rect>, String> {
    let Some(r) = params.crop_rect else { return Ok(None) };
    let k = 1usize << params.reduce;
    let r = Rect { x: r.x / k, y: r.y / k, w: r.w.div_ceil(k), h: r.h.div_ceil(k) };
    let r = intersect(&r, &Rect::full(w, h)).ok_or_else(|| format!("--crop: the window {}x{} at ({}, {}) lies outside the {w}x{h} frame", r.w, r.h, r.x, r.y))?;
    log(format!("cropped to the window asked for: {}x{} at ({}, {})", r.w, r.h, r.x, r.y));
    Ok(Some(r))
}

/// The overlap of two windows, `None` when they do not meet.
pub fn intersect(a: &Rect, b: &Rect) -> Option<Rect> {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    (x1 > x0 && y1 > y0).then(|| Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
}

/// A dust map of another size than the frames is a mistake (another camera, a
/// crop): the run stops rather than leave the dust in.
fn check_dust(d: &DustMap, w: usize, h: usize) -> Result<(), String> {
    if d.w != w || d.h != h {
        return Err(format!("the dust map is {}x{} but the frames are {w}x{h}; it must be shot with the same camera at the same size", d.w, d.h));
    }
    Ok(())
}

/// Run the whole pipeline. `log` receives human-readable progress lines.
pub fn run(inputs: &[String], params: &Params, log: &mut dyn FnMut(String)) -> Result<Output, String> {
    run_with(inputs, params, log, &mut |_| Ok(()))
}

/// `run`, with every slab of `Params::slabs` handed to `on_slab` as it is fused.
pub fn run_with(
    inputs: &[String],
    params: &Params,
    log: &mut dyn FnMut(String),
    on_slab: &mut dyn FnMut(Slab<'_>) -> Result<(), String>,
) -> Result<Output, String> {
    if inputs.is_empty() {
        return Err("no input images; use --help".into());
    }
    let inputs: Vec<String> = inputs.to_vec();
    match params.align {
        Some(a) if inputs.len() > 1 => {
            let a = {
                let mut a = a;
                if (a.backend == Backend::Cuda && !cfg!(feature = "gpu")) || (a.backend == Backend::Wgpu && !cfg!(all(feature = "wgpu", not(target_arch = "wasm32")))) {
                    log(format!("the {} aligner is not in this build; aligning on the CPU", a.backend.name()));
                    a.backend = Backend::Cpu;
                }
                a
            };
            if let Some(dir) = &params.save_aligned {
                std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {dir}: {e}"))?;
            }
            let mut src = AlignedFrames::open(inputs.clone(), a, params)?;
            let (w, h, bit_depth) = (src.src.w, src.src.h, src.src.depth);
            if let Some(d) = &params.dust {
                log(format!("dust map applied to each frame as decoded ({}): {}", d.params.mode.name(), d.describe()));
            }
            log(format!(
                "{} frames @ {w}x{h}, {}-bit ({} threads); streaming from disk, each frame aligned as it is folded ({} shift={} scale={} rot={} coarsen={} {}{})",
                src.len(), bit_depth.bits(), rayon::current_num_threads(), a.model.name(), a.shift, a.scale, a.rotation, a.coarsen, a.interp.name(),
                if a.backend.is_gpu() { format!(", {} GPU", a.backend.name()) } else { String::new() }
            ));
            let (image, depth, conf, levels, floor, shares) = fuse_and_depth(&mut src, params, log)?;
            let wav = weighted(&mut src, params, floor.as_deref(), &image, log)?;
            // the borders some frames only reach with smeared edge pixels go
            let area = common_area(&src.sims, w, h, a.interp);
            let mut window = if params.crop && !area.is_full(w, h) {
                log(format!("cropped to the area every frame covers: {}x{} at ({}, {})", area.w, area.h, area.x, area.y));
                Some(area)
            } else {
                None
            };
            if let Some(r) = user_crop(params, w, h, log)? {
                window = Some(match window { Some(a) => intersect(&a, &r).ok_or("the crop window lies outside the area every frame covers")?, None => r });
            }
            let (image, depth, conf, wav, crop) = match window {
                Some(r) => (image.crop(&r), crop_plane(&depth, w, &r), conf.map(|c| crop_plane(&c, w, &r)), wav.map(|i| i.crop(&r)), Some(r)),
                None => (image, depth, conf, wav, None),
            };
            let (image, depth, conf, wav, restretch) = restretched(params, crop.as_ref(), w, h, a.interp, image, depth, conf, wav, log);
            let look = src.src.look.clone();
            fuse_slabs(&mut src, params, bit_depth, crop.as_ref(), restretch, look.as_ref(), log, on_slab)?;
            for n in &src.src.notes {
                log(format!("note: {n}"));
            }
            if let Some(n) = src.brightness_note() {
                log(n);
            }
            if let Some(dir) = &params.save_aligned {
                log(format!("wrote aligned frames to {dir}/"));
            }
            let near = near_end(&inputs, &src.sims, w, h, log);
            let dng = src.src.look.take();
            Ok(Output { image, depth, conf, bit_depth, align: src.sims, levels, crop, wav, dng, reduce: params.reduce, near, shares })
        }
        _ => {
            let mut src = LazyFrames::open(inputs.clone(), params.brightness, params)?;
            if let Some(d) = &params.dust {
                log(format!("dust map applied to each frame as decoded ({}): {}", d.params.mode.name(), d.describe()));
            }
            log(format!(
                "{} frames @ {}x{}, {}-bit ({} threads); alignment skipped, streaming from disk",
                src.len(), src.w, src.h, src.depth.bits(), rayon::current_num_threads()
            ));
            let bit_depth = src.depth;
            let (w, h) = (src.w, src.h);
            let (image, depth, conf, levels, floor, shares) = fuse_and_depth(&mut src, params, log)?;
            let wav = weighted(&mut src, params, floor.as_deref(), &image, log)?;
            let crop = user_crop(params, w, h, log)?;
            let (image, depth, conf, wav) = match &crop {
                Some(r) => (image.crop(r), crop_plane(&depth, w, r), conf.map(|c| crop_plane(&c, w, r)), wav.map(|i| i.crop(r))),
                None => (image, depth, conf, wav),
            };
            let (image, depth, conf, wav, restretch) = restretched(params, crop.as_ref(), w, h, Interp::Spline4x4, image, depth, conf, wav, log);
            let look = src.look.clone();
            fuse_slabs(&mut src, params, bit_depth, crop.as_ref(), restretch, look.as_ref(), log, on_slab)?;
            for n in &src.notes {
                log(format!("note: {n}"));
            }
            if let Some(n) = src.brightness_note() {
                log(n);
            }
            let near = near_end(&inputs, &[], w, h, log);
            let dng = src.look.take();
            Ok(Output { image, depth, conf, bit_depth, align: vec![Sim::id(); inputs.len()], levels, crop, wav, dng, reduce: params.reduce, near, shares })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{intersect, slab_ranges};
    use crate::align::Rect;

    #[test]
    fn windows_meet_or_do_not() {
        let a = Rect { x: 10, y: 10, w: 100, h: 50 };
        assert_eq!(intersect(&a, &Rect { x: 50, y: 0, w: 100, h: 30 }), Some(Rect { x: 50, y: 10, w: 60, h: 20 }));
        assert_eq!(intersect(&a, &a), Some(a));
        assert_eq!(intersect(&a, &Rect { x: 110, y: 10, w: 5, h: 5 }), None);
        assert_eq!(intersect(&a, &Rect { x: 0, y: 60, w: 500, h: 5 }), None);
    }

    #[test]
    fn slabs_cover_the_stack_and_overlap_as_asked() {
        assert_eq!(slab_ranges(8, 3, 1), vec![(0, 2), (2, 4), (4, 6), (6, 7)]);
        assert_eq!(slab_ranges(20, 10, 2), vec![(0, 9), (8, 17), (16, 19)]);
        assert_eq!(slab_ranges(8, 8, 0), vec![(0, 7)]);
        assert_eq!(slab_ranges(8, 20, 5), vec![(0, 7)]);        // a slab larger than the stack is the stack
        assert_eq!(slab_ranges(5, 2, 7), vec![(0, 1), (1, 2), (2, 3), (3, 4)]);   // overlap clamped below the size
        assert_eq!(slab_ranges(0, 3, 1), vec![]);
        for (count, size, overlap) in [(100, 10, 2), (7, 3, 2), (9, 4, 0), (1, 5, 3)] {
            let r = slab_ranges(count, size, overlap);
            assert_eq!(r[0].0, 0);
            assert_eq!(r.last().unwrap().1, count - 1);
            for w in r.windows(2) {
                assert!(w[1].0 > w[0].0 && w[1].0 <= w[0].1 + 1, "{r:?}");   // advances, leaves no gap
            }
        }
    }
}

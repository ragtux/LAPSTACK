// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! Orchestration: decode → (align) → fuse → (depth) → (slabs). Frames are
//! consumed one at a time, so without alignment the stack is streamed from
//! disk with a bounded read-ahead and memory stays at a few frames regardless
//! of stack size (the depth pass, and each slab, stream the frames again).

use crate::depth::{self, DepthParams};
use crate::fuse::{FuseParams, Fuser};
use crate::align::{self, AlignParams, CancelToken, Rect, Sim, common_area};
use crate::brightness;
use crate::dust::DustMap;
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
    /// Fuse on the CUDA GPU (needs the `gpu` build feature; falls back to CPU).
    pub gpu: bool,
    /// Depth-from-focus pass after fusion (`depth.rs`); `None` = report the
    /// pyramid winner map of `fuse.depth_level` instead (no extra pass).
    pub depth: Option<DepthParams>,
    /// Crop the result (image, depth, confidence) to the area every aligned
    /// frame covers with real pixels (`align::common_area`).
    pub crop: bool,
    /// Bring every frame to frame 0's brightness, one gain per channel over
    /// the area the frame covers (`brightness`): exposure flicker.
    pub brightness: bool,
    /// Slabs (Zerene's slabbing): after the result, fuse every run of `size`
    /// consecutive frames overlapping by `overlap` on its own, with the same
    /// settings, and hand each to `run_with`'s callback — thick planes of
    /// focus to retouch from elsewhere. `None` = no slabs.
    pub slabs: Option<(usize, usize)>,
    /// The weighted average (`wav.rs`, Helicon's method A) as a second image,
    /// from the depth pass's focus measure: needs `depth`.
    pub wav: Option<crate::wav::WavParams>,
    /// Dust map (`dust.rs`): the spots taken out of every frame as decoded,
    /// before alignment. Must be of the frames' size.
    pub dust: Option<DustMap>,
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
}

impl Default for Params {
    fn default() -> Self {
        Params {
            fuse: FuseParams::default(),
            align: Some(AlignParams::default()),
            save_aligned: None,
            gpu: false,
            depth: Some(DepthParams::default()),
            crop: true,
            brightness: true,
            slabs: None, wav: None, dust: None,
        }
    }
}

/// CPU or GPU accumulator behind one interface.
enum AnyFuser {
    Cpu(Fuser),
    #[cfg(feature = "gpu")]
    Gpu(crate::gpu::GpuFuser),
}

impl AnyFuser {
    fn levels(&self) -> usize {
        match self {
            AnyFuser::Cpu(f) => f.levels,
            #[cfg(feature = "gpu")]
            AnyFuser::Gpu(f) => f.levels,
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
        }
    }
    fn finish(self) -> Result<(Img3, Vec<f32>), String> {
        match self {
            AnyFuser::Cpu(f) => Ok(f.finish()),
            #[cfg(feature = "gpu")]
            AnyFuser::Gpu(f) => f.finish(),
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
    pending: VecDeque<(usize, JoinHandle<Result<(Img3, Depth), String>>)>,
    notes: Vec<String>,
    /// Brightness normalisation: frame 0's channel means, and each frame's
    /// gains once found (the depth pass decodes the frames a second time).
    ref_means: Option<[f64; 3]>,
    gains: Vec<Option<[f32; 3]>>,
    /// The dust map, applied to every frame as it is decoded.
    dust: Option<DustMap>,
}

impl LazyFrames {
    fn open(paths: Vec<String>, brightness: bool, dust: Option<&DustMap>) -> Result<LazyFrames, String> {
        let (mut f0, depth) = io::load_rgb(&paths[0])?;
        if let Some(d) = dust {
            check_dust(d, f0.w, f0.h)?;
            d.apply(&mut f0);
        }
        let ref_means = brightness.then(|| brightness::means(&f0));
        let n = paths.len();
        let mut lf = LazyFrames { w: f0.w, h: f0.h, depth, paths, cur: Some((0, f0)), pending: VecDeque::new(), notes: Vec::new(), ref_means, gains: vec![None; n], dust: dust.cloned() };
        lf.gains[0] = Some([1.0; 3]);
        lf.prefetch(1);
        Ok(lf)
    }
    /// One line on the gains found, for the log (after the run).
    fn brightness_note(&self) -> Option<String> {
        self.ref_means?;
        let gs: Vec<f32> = self.gains.iter().flatten().flat_map(|g| g.iter().copied()).collect();
        let (lo, hi) = gs.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        Some(format!("brightness equalised to frame 0: gains {lo:.3} … {hi:.3}"))
    }
    /// Keep decoders running for frames `from..from+READ_AHEAD`.
    fn prefetch(&mut self, from: usize) {
        let mut next = self.pending.back().map_or(from, |(i, _)| i + 1).max(from);
        while self.pending.len() < READ_AHEAD && next < self.paths.len() {
            let p = self.paths[next].clone();
            self.pending.push_back((next, std::thread::spawn(move || io::load_rgb(&p))));
            next += 1;
        }
    }
    fn take(&mut self, i: usize) -> Result<Img3, String> {
        // drop stale entries (random access is not expected, but stay correct)
        while self.pending.front().is_some_and(|(pi, _)| *pi < i) {
            let (_, jh) = self.pending.pop_front().unwrap();
            let _ = jh.join();
        }
        let (img, d) = match self.pending.front() {
            Some((pi, _)) if *pi == i => {
                let (_, jh) = self.pending.pop_front().unwrap();
                jh.join().map_err(|_| "decoder thread panicked".to_string())??
            }
            _ => io::load_rgb(&self.paths[i])?,
        };
        if d != self.depth {
            self.notes.push(format!(
                "{} is {}-bit but frame 0 is {}-bit; writing {}-bit output",
                self.paths[i], d.bits(), self.depth.bits(), self.depth.bits()
            ));
        }
        if img.w != self.w || img.h != self.h {
            return Err(format!(
                "{}: {}x{} differs from frame 0 ({}x{}); frames must share one size",
                self.paths[i], img.w, img.h, self.w, self.h
            ));
        }
        let mut img = img;
        if let Some(d) = &self.dust {
            d.apply(&mut img);
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

fn fuse_all(
    src: &mut dyn FrameSource,
    params: &FuseParams,
    gpu: bool,
    log: &mut dyn FnMut(String),
) -> Result<(Img3, Vec<f32>, usize), String> {
    let last = src.len() - 1;
    fuse_range(src, 0, last, params, gpu, log)
}

/// Fuse frames `lo..=hi` of the source.
fn fuse_range(
    src: &mut dyn FrameSource,
    lo: usize,
    hi: usize,
    params: &FuseParams,
    gpu: bool,
    log: &mut dyn FnMut(String),
) -> Result<(Img3, Vec<f32>, usize), String> {
    let (w, h) = src.dims();
    #[cfg(feature = "gpu")]
    let mut fuser = if gpu {
        AnyFuser::Gpu(crate::gpu::GpuFuser::new(w, h, params.clone())?)
    } else {
        AnyFuser::Cpu(Fuser::new(w, h, params.clone()))
    };
    #[cfg(not(feature = "gpu"))]
    let mut fuser = {
        if gpu {
            log("--gpu requested but built without the 'gpu' feature; fusing on the CPU".into());
        }
        AnyFuser::Cpu(Fuser::new(w, h, params.clone()))
    };
    let levels = fuser.levels();
    log(format!(
        "fusing {} frames{} on the {}: {} band-pass levels + residual ({}x{}), energy window {}x{}, top rule {:?}",
        hi + 1 - lo,
        if lo == 0 && hi + 1 == src.len() { String::new() } else { format!(" ({lo}..{hi})") },
        match fuser { AnyFuser::Cpu(_) => "CPU", #[cfg(feature = "gpu")] AnyFuser::Gpu(_) => "GPU" },
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
    for i in lo..=hi {
        let f = src.get(i)?;
        fuser.push(&f)?;
        log(format!("  frame {:>3}/{} folded  ({:.1}s)", i + 1 - lo, hi + 1 - lo, t.elapsed().as_secs_f64()));
    }
    let (img, depth) = fuser.finish()?;
    log(format!("collapsed  ({:.1}s)", t.elapsed().as_secs_f64()));
    Ok((img, depth, levels))
}

/// Fusion followed by the optional depth-from-focus pass over the same frames.
fn fuse_and_depth(
    src: &mut dyn FrameSource,
    params: &Params,
    log: &mut dyn FnMut(String),
) -> Result<(Img3, Vec<f32>, Option<Vec<f32>>, usize), String> {
    let (image, winner, levels) = fuse_all(src, &params.fuse, params.gpu, log)?;
    match &params.depth {
        Some(dp) => {
            let dm = depth::depth_from_focus(src, &image, dp, log)?;
            Ok((image, dm.depth, Some(dm.conf), levels))
        }
        None => Ok((image, winner, None, levels)),
    }
}

/// The weighted average of `Params::wav`, another pass over the frames; it
/// borrows the depth pass's focus measure, so it needs `Params::depth`.
fn weighted(src: &mut dyn FrameSource, params: &Params, log: &mut dyn FnMut(String)) -> Result<Option<Img3>, String> {
    match (&params.wav, &params.depth) {
        (Some(wp), Some(dp)) => Ok(Some(crate::wav::weighted_average(src, dp, wp, log)?)),
        (Some(_), None) => {
            log("the weighted average needs the depth-from-focus pass (not the winner map): skipped".into());
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// After the result: the slabs of `Params::slabs`, each fused on its own
/// over the same (aligned, equalised) frames, cropped like the result, and
/// handed to `on_slab` one at a time — none is kept.
fn fuse_slabs(
    src: &mut dyn FrameSource,
    params: &Params,
    bit_depth: Depth,
    crop: Option<&Rect>,
    log: &mut dyn FnMut(String),
    on_slab: &mut dyn FnMut(Slab<'_>) -> Result<(), String>,
) -> Result<(), String> {
    let Some((size, overlap)) = params.slabs else { return Ok(()) };
    let ranges = slab_ranges(src.len(), size, overlap);
    log(format!("{} slabs of {size} frames, overlap {overlap}", ranges.len()));
    for (k, &(lo, hi)) in ranges.iter().enumerate() {
        log(format!("slab {}/{}: frames {lo}..{hi}", k + 1, ranges.len()));
        let (image, _, _) = fuse_range(src, lo, hi, &params.fuse, params.gpu, log)?;
        let image = match crop { Some(r) => image.crop(r), None => image };
        on_slab(Slab { index: k, count: ranges.len(), lo, hi, image: &image, bit_depth })?;
    }
    Ok(())
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
            // Alignment needs every frame resident.
            let t = Instant::now();
            log(format!("loading {} frames ...", inputs.len()));
            let loaded: Vec<(Img3, Depth)> =
                inputs.par_iter().map(|p| io::load_rgb(p)).collect::<Result<_, _>>()?;
            let bit_depth = loaded[0].1;
            for (i, (_, d)) in loaded.iter().enumerate() {
                if *d != bit_depth {
                    log(format!(
                        "note: {} is {}-bit but frame 0 is {}-bit; writing {}-bit output",
                        inputs[i], d.bits(), bit_depth.bits(), bit_depth.bits()
                    ));
                }
            }
            let mut frames: Vec<Img3> = loaded.into_iter().map(|(f, _)| f).collect();
            let (w, h) = (frames[0].w, frames[0].h);
            if frames.iter().any(|f| f.w != w || f.h != h) {
                return Err("frames differ in size; pre-size them to a common resolution".into());
            }
            if let Some(d) = &params.dust {
                // the dust is fixed on the sensor: out before the frames are warped
                check_dust(d, w, h)?;
                frames.par_iter_mut().for_each(|f| d.apply(f));
                log(format!("dust map applied ({}): {}", d.params.mode.name(), d.describe()));
            }
            log(format!(
                "{} frames @ {w}x{h}, {}-bit ({} threads)  ({:.1}s)",
                frames.len(), bit_depth.bits(), rayon::current_num_threads(), t.elapsed().as_secs_f64()
            ));
            let t = Instant::now();
            log(format!("aligning (shift={} scale={} rot={} coarsen={}{}) ...", a.shift, a.scale, a.rotation, a.coarsen, if a.gpu && cfg!(feature = "gpu") { ", GPU" } else { "" }));
            if a.gpu && !cfg!(feature = "gpu") {
                log("--gpu-align requested but built without the 'gpu' feature; aligning on the CPU".into());
            }
            let mut on_frame = |idx: usize, sim: Sim| log(format!("  frame {idx:>3}: {}", align::report(&sim, w, h)));
            #[cfg(feature = "gpu")]
            let res = if a.gpu {
                crate::gpu::align_gpu(&frames, a.shift, a.scale, a.rotation, a.coarsen, &CancelToken::new(), &mut on_frame)
            } else {
                align::align_stack(&frames, a.shift, a.scale, a.rotation, a.coarsen, &CancelToken::new(), &mut on_frame)
            };
            #[cfg(not(feature = "gpu"))]
            let res = align::align_stack(&frames, a.shift, a.scale, a.rotation, a.coarsen, &CancelToken::new(), &mut on_frame);
            let (mut aligned, sims) = res.map_err(|_| "cancelled".to_string())?;
            drop(frames);
            log(format!("aligned  ({:.1}s)", t.elapsed().as_secs_f64()));
            if params.brightness {
                // one gain per channel brings each frame to frame 0's brightness over the area it covers
                let (f0, rest) = aligned.split_at_mut(1);
                let gains: Vec<[f32; 3]> = rest.par_iter().zip(&sims[1..]).map(|(f, s)| brightness::gains(&f0[0], f, s)).collect();
                rest.par_iter_mut().zip(&gains).for_each(|(f, g)| brightness::apply(f, *g));
                let shown: Vec<String> = gains.iter().enumerate().map(|(i, g)| format!("{}:{}", i + 1, brightness::describe(*g))).collect();
                log(format!("brightness equalised to frame 0: {}", shown.join(" ")));
            }
            if let Some(dir) = &params.save_aligned {
                std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {dir}: {e}"))?;
                for (i, f) in aligned.iter().enumerate() {
                    io::save_rgb(f, &format!("{dir}/aligned_{i:03}.png"), bit_depth, None)?;
                }
                log(format!("wrote aligned frames to {dir}/"));
            }
            let mut src: &[Img3] = &aligned;
            let (image, depth, conf, levels) = fuse_and_depth(&mut src, params, log)?;
            let wav = weighted(&mut src, params, log)?;
            // the borders some frames only reach with smeared edge pixels go
            let area = common_area(&sims, w, h);
            let (image, depth, conf, wav, crop) = if params.crop && !area.is_full(w, h) {
                log(format!("cropped to the area every frame covers: {}x{} at ({}, {})", area.w, area.h, area.x, area.y));
                (image.crop(&area), crop_plane(&depth, w, &area), conf.map(|c| crop_plane(&c, w, &area)), wav.map(|i| i.crop(&area)), Some(area))
            } else {
                (image, depth, conf, wav, None)
            };
            fuse_slabs(&mut src, params, bit_depth, crop.as_ref(), log, on_slab)?;
            Ok(Output { image, depth, conf, bit_depth, align: sims, levels, crop, wav })
        }
        _ => {
            let mut src = LazyFrames::open(inputs.clone(), params.brightness, params.dust.as_ref())?;
            if let Some(d) = &params.dust {
                log(format!("dust map applied to each frame as decoded ({}): {}", d.params.mode.name(), d.describe()));
            }
            log(format!(
                "{} frames @ {}x{}, {}-bit ({} threads); alignment skipped, streaming from disk",
                src.len(), src.w, src.h, src.depth.bits(), rayon::current_num_threads()
            ));
            let bit_depth = src.depth;
            let (image, depth, conf, levels) = fuse_and_depth(&mut src, params, log)?;
            let wav = weighted(&mut src, params, log)?;
            fuse_slabs(&mut src, params, bit_depth, None, log, on_slab)?;
            for n in &src.notes {
                log(format!("note: {n}"));
            }
            if let Some(n) = src.brightness_note() {
                log(n);
            }
            Ok(Output { image, depth, conf, bit_depth, align: vec![Sim::id(); inputs.len()], levels, crop: None, wav })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::slab_ranges;

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

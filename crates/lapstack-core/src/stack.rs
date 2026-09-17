// Copyright (c) 2026 MATCHMUSEUM.COM
// INTERNAL USE ONLY

//! Orchestration: decode → (align) → fuse → (depth). Frames are consumed one
//! at a time, so without alignment the stack is streamed from disk with a
//! bounded read-ahead and memory stays at a few frames regardless of stack
//! size (the depth pass streams the frames a second time).

use crate::depth::{self, DepthParams};
use crate::fuse::{FuseParams, Fuser};
use crate::align::{self, AlignParams, CancelToken, Sim};
use crate::io::{self, Depth};
use crate::pyramid::Img3;
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
}

impl Default for Params {
    fn default() -> Self {
        Params {
            fuse: FuseParams::default(),
            align: Some(AlignParams::default()),
            save_aligned: None,
            gpu: false,
            depth: Some(DepthParams::default()),
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
}

impl LazyFrames {
    fn open(paths: Vec<String>) -> Result<LazyFrames, String> {
        let (f0, depth) = io::load_rgb(&paths[0])?;
        let mut lf = LazyFrames { w: f0.w, h: f0.h, depth, paths, cur: Some((0, f0)), pending: VecDeque::new(), notes: Vec::new() };
        lf.prefetch(1);
        Ok(lf)
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
        "fusing {} frames on the {}: {} band-pass levels + residual ({}x{}), energy window {}x{}, top rule {:?}",
        src.len(),
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
    for i in 0..src.len() {
        let f = src.get(i)?;
        fuser.push(&f)?;
        log(format!("  frame {:>3}/{} folded  ({:.1}s)", i + 1, src.len(), t.elapsed().as_secs_f64()));
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

/// Run the whole pipeline. `log` receives human-readable progress lines.
pub fn run(inputs: &[String], params: &Params, log: &mut dyn FnMut(String)) -> Result<Output, String> {
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
            let frames: Vec<Img3> = loaded.into_iter().map(|(f, _)| f).collect();
            let (w, h) = (frames[0].w, frames[0].h);
            if frames.iter().any(|f| f.w != w || f.h != h) {
                return Err("frames differ in size; pre-size them to a common resolution".into());
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
            let (aligned, sims) = res.map_err(|_| "cancelled".to_string())?;
            drop(frames);
            log(format!("aligned  ({:.1}s)", t.elapsed().as_secs_f64()));
            if let Some(dir) = &params.save_aligned {
                std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {dir}: {e}"))?;
                for (i, f) in aligned.iter().enumerate() {
                    io::save_rgb(f, &format!("{dir}/aligned_{i:03}.png"), bit_depth)?;
                }
                log(format!("wrote aligned frames to {dir}/"));
            }
            let mut src: &[Img3] = &aligned;
            let (image, depth, conf, levels) = fuse_and_depth(&mut src, params, log)?;
            Ok(Output { image, depth, conf, bit_depth, align: sims, levels })
        }
        _ => {
            let mut src = LazyFrames::open(inputs.clone())?;
            log(format!(
                "{} frames @ {}x{}, {}-bit ({} threads); alignment skipped, streaming from disk",
                src.len(), src.w, src.h, src.depth.bits(), rayon::current_num_threads()
            ));
            let bit_depth = src.depth;
            let (image, depth, conf, levels) = fuse_and_depth(&mut src, params, log)?;
            for n in &src.notes {
                log(format!("note: {n}"));
            }
            Ok(Output { image, depth, conf, bit_depth, align: vec![Sim::id(); inputs.len()], levels })
        }
    }
}

// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

//! The native wgpu engine: the browser app's kernels (`fold.rs`, `align.rs`,
//! `depth.rs`) driving a desktop GPU through Vulkan, Metal or DX12 for the
//! CLI's `--gpu` where there is no CUDA. `WgFrames` keeps the frames on the
//! device — registration (the cost search's evaluations), warp, brightness
//! gains — the twin of `gpu::GpuFrames`; `WgFuser` folds them and runs the
//! depth pass, the twin of `gpu::GpuFuser` and `gpu::depth_from_slices`. One
//! device per thread (`gpu()`), shared by the two, so a frame goes from the
//! one to the other without leaving the GPU. Everything here is synchronous:
//! the kernels' futures (map callbacks) are driven by `gpu::block_on`.

use super::align::{Aligner, LumaPyr};
use super::depth::DepthGpu;
use super::fold::{FoldBufs, record_collapse_in, record_fold_in, record_reset};
use super::gpu::{Gpu, P, Rec, block_on, grid1, grid2};
use crate::align::{Interp, Sim, inverse};
use crate::depth::{DepthMap, DepthParams, blocks, upsample_bilinear};
use crate::fuse::{FuseParams, binomial, halo_guide, upsample_index, winner_shares};
use crate::pyramid::{Img3, auto_levels, half};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    static GPU: RefCell<Option<Rc<Gpu>>> = const { RefCell::new(None) };
}

/// The thread's device, made on first use.
pub fn gpu() -> Result<Rc<Gpu>, String> {
    GPU.with(|g| {
        if let Some(g) = g.borrow().as_ref() {
            return Ok(g.clone());
        }
        let made = Rc::new(block_on(Gpu::new())?);
        *g.borrow_mut() = Some(made.clone());
        Ok(made)
    })
}

/// Whether a GPU is there, and which: the adapter's name and backend.
pub fn available() -> Result<String, String> {
    Ok(describe(gpu()?.as_ref()))
}

pub fn describe(g: &Gpu) -> String {
    format!("{} ({:?}, {} MB per buffer)", g.info.name, g.info.backend, g.limits.max_buffer_size >> 20)
}

/// A frame needs `3 × w × h` floats in one buffer.
fn check_size(g: &Gpu, w: usize, h: usize) -> Result<(), String> {
    let need = (3 * w * h * 4) as u64;
    if need > g.limits.max_buffer_size || need > g.limits.max_storage_buffer_binding_size {
        return Err(format!(
            "{w}x{h} needs {} MB storage buffers; this GPU allows {} MB per buffer / {} MB per binding",
            need >> 20,
            g.limits.max_buffer_size >> 20,
            g.limits.max_storage_buffer_binding_size >> 20
        ));
    }
    Ok(())
}

/// The frame as the upload buffer holds it: RGB u16 interleaved, an even
/// count of samples (two per u32).
fn to_u16(img: &Img3) -> Vec<u16> {
    let n = img.w * img.h;
    let mut v = vec![0u16; (3 * n).div_ceil(2) * 2];
    v[..3 * n].par_chunks_mut(3).enumerate().for_each(|(i, o)| {
        for c in 0..3 {
            o[c] = (img.p[c][i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    });
    v
}

/// Read a 3-plane f32 image back.
fn read_img3(g: &Gpu, buf: &wgpu::Buffer, w: usize, h: usize) -> Result<Img3, String> {
    let n = w * h;
    let v = block_on(g.read_f32(buf, 3 * n))?;
    Ok(Img3 { w, h, p: [v[..n].to_vec(), v[n..2 * n].to_vec(), v[2 * n..].to_vec()] })
}

/// Record the warp of the uploaded frame `up` by `sim` into `dst` (3 f32
/// planes), `aff` holding the transform's translation and perspective terms.
fn record_warp(g: &Gpu, rec: &mut Rec<'_>, up: &wgpu::Buffer, aff: &wgpu::Buffer, dst: &wgpu::Buffer, w: usize, h: usize, sim: Sim, interp: Interp) {
    let identity = sim == Sim::id();
    let mut p = P { w: w as u32, h: h as u32, flag: identity as u32, klen: interp.id(), ..Default::default() };
    if !identity {
        let inv = inverse(sim.matrix(w, h));
        g.queue.write_buffer(aff, 0, bytemuck::cast_slice(&[inv[0][2] as f32, inv[1][2] as f32, inv[2][0] as f32, inv[2][1] as f32]));
        p.f0 = inv[0][0] as f32;
        p.f1 = inv[0][1] as f32;
        p.f2 = inv[1][0] as f32;
        p.f3 = inv[1][1] as f32;
    }
    rec.dispatch("warp", [None, None, Some(dst), None, Some(aff), Some(up)], p, grid2(w, h));
}

/// The brightness sampling block (`blk_mean` / `bright` kernels).
const BLK: usize = 64;

/// The frames on the device: each uploaded as shot, registered against the
/// previous aligned frame (the search's evaluations on the GPU, the simplex
/// on the host — `wg::align`), warped and brought to frame 0's brightness,
/// and left in `frame()` for the fuser.
pub struct WgFrames {
    g: Rc<Gpu>,
    w: usize,
    h: usize,
    interp: Interp,
    up: wgpu::Buffer,
    cur: wgpu::Buffer,
    tmp_half: wgpu::Buffer,
    aff: wgpu::Buffer,
    ref_pyr: LumaPyr,
    tgt_pyr: LumaPyr,
    aligner: Aligner,
    /// Frame 0's channel means per block (blocks across, blocks down), and
    /// the partial sums of a frame against them.
    ref_blk: (wgpu::Buffer, usize, usize),
    bright: wgpu::Buffer,
    have_ref: bool,
}

impl WgFrames {
    pub fn new(w: usize, h: usize, interp: Interp) -> Result<WgFrames, String> {
        let g = gpu()?;
        check_size(&g, w, h)?;
        let n = w * h;
        let (bx, by) = (w.div_ceil(BLK), h.div_ceil(BLK));
        Ok(WgFrames {
            up: g.buffer("upload u16", (3 * n).div_ceil(2) as u64 * 4),
            cur: g.buffer_f32("frame", 3 * n),
            tmp_half: g.buffer_f32("tmp_half", (half(w) * h).max(half(h) * w)),
            aff: g.buffer_init("affine", bytemuck::cast_slice(&[0f32; 4])),
            ref_pyr: LumaPyr::new(&g, w, h, "ref"),
            tgt_pyr: LumaPyr::new(&g, w, h, "tgt"),
            aligner: Aligner::new(&g, w, h),
            ref_blk: (g.buffer_f32("brightness reference", bx * by * 3), bx, by),
            bright: g.buffer_f32("brightness partials", bx * by * 8),
            have_ref: false,
            g,
            w,
            h,
            interp,
        })
    }

    /// Register (unless `known`), warp and equalize one frame; the device
    /// then holds it (`frame`). `guess` seeds the search, `free` says which
    /// terms move, `coarsen` how many levels short of full resolution it
    /// stops. Returns the transform and the gains.
    pub fn process(&mut self, img: &Img3, guess: Sim, free: [bool; Sim::N], coarsen: usize, known: Option<(Sim, [f32; 3])>, brightness: bool) -> Result<(Sim, [f32; 3]), String> {
        if img.w != self.w || img.h != self.h {
            return Err("frame size mismatch".into());
        }
        let g = self.g.clone();
        let (w, h, n) = (self.w, self.h, self.w * self.h);
        block_on(g.upload(&self.up, bytemuck::cast_slice(&to_u16(img))))?;
        let first = !self.have_ref && known.is_none();
        let sim = match known {
            Some((s, _)) => s,
            None if first => Sim::id(),
            None => {
                let mut rec = g.rec();
                rec.dispatch("luma_u16", [None, None, Some(&self.tgt_pyr.lv[0].0), None, None, Some(&self.up)], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(n));
                self.tgt_pyr.reduce_chain(&mut rec, &self.tmp_half);
                rec.submit();
                block_on(self.aligner.align(&g, &self.ref_pyr, &self.tgt_pyr, guess, free, coarsen))
            }
        };
        let mut rec = g.rec();
        record_warp(&g, &mut rec, &self.up, &self.aff, &self.cur, w, h, sim, self.interp);
        // the gains against frame 0: its block means are kept; a frame's channel means
        // over the pixels its warp covers are compared with frame 0's over the same
        // pixels (see `brightness` for why means)
        let (bx, by) = (self.ref_blk.1, self.ref_blk.2);
        let gains = match known {
            Some((_, gn)) => gn,
            None if !brightness => [1.0; 3],
            None if first => {
                rec.dispatch("blk_mean", [Some(&self.cur), None, Some(&self.ref_blk.0), None, None, None], P { w: w as u32, h: h as u32, ow: bx as u32, oh: by as u32, ..Default::default() }, grid2(bx, by));
                [1.0; 3]
            }
            None => {
                let identity = sim == Sim::id();
                let mut pb = P { w: w as u32, h: h as u32, ow: bx as u32, oh: by as u32, flag: identity as u32, klen: self.interp.id(), ..Default::default() };
                if !identity {
                    let inv = inverse(sim.matrix(w, h));
                    pb.f0 = inv[0][0] as f32;
                    pb.f1 = inv[0][1] as f32;
                    pb.f2 = inv[1][0] as f32;
                    pb.f3 = inv[1][1] as f32;
                }
                rec.dispatch("bright", [Some(&self.cur), Some(&self.ref_blk.0), None, Some(&self.bright), Some(&self.aff), None], pb, (bx as u32, by as u32));
                rec.submit();
                let part = block_on(g.read_f32(&self.bright, bx * by * 8))?;
                let mut s = [0f64; 8];
                for r in part.chunks_exact(8) {
                    for k in 0..7 {
                        s[k] += r[k] as f64;
                    }
                }
                rec = g.rec();
                if s[3] >= 64.0 {
                    [0, 1, 2].map(|c| if s[c] > 1e-9 { (s[4 + c] / s[c]) as f32 } else { 1.0 }.clamp(crate::brightness::GAIN_MIN, crate::brightness::GAIN_MAX))
                } else {
                    [1.0; 3]
                }
            }
        };
        if !crate::brightness::is_unity(gains) {
            rec.dispatch("gain3", [None, None, Some(&self.cur), None, None, None], P { w: n as u32, f0: gains[0], f1: gains[1], f2: gains[2], ..Default::default() }, grid1(3 * n));
        }
        // the next search's reference: this frame as aligned and equalized
        if known.is_none() {
            rec.dispatch("luma_f32", [Some(&self.cur), None, Some(&self.ref_pyr.lv[0].0), None, None, None], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(n));
            self.ref_pyr.reduce_chain(&mut rec, &self.tmp_half);
            self.have_ref = true;
        }
        rec.submit();
        Ok((sim, gains))
    }

    /// The frame last processed, as warped and equalized: 3 f32 planes.
    pub fn frame(&self) -> &wgpu::Buffer {
        &self.cur
    }

    pub fn download(&self) -> Result<Img3, String> {
        read_img3(&self.g, &self.cur, self.w, self.h)
    }
}

/// The fold on the device (`fold.rs`), the frames' focus slices taken as
/// they pass when a depth pass is wanted, and the depth pass itself after
/// the collapse (`depth.rs`).
pub struct WgFuser {
    g: Rc<Gpu>,
    pub w: usize,
    pub h: usize,
    pub levels: usize,
    depth_level: usize,
    fp: FuseParams,
    dims: Vec<(usize, usize)>,
    cur: Vec<wgpu::Buffer>,
    acc: Vec<wgpu::Buffer>,
    best: Vec<wgpu::Buffer>,
    tmp_half: wgpu::Buffer,
    tmp_full: wgpu::Buffer,
    en: wgpu::Buffer,
    en2: wgpu::Buffer,
    wt: wgpu::Buffer,
    klen: u32,
    up: wgpu::Buffer,
    aff: wgpu::Buffer,
    tops: Vec<Img3>,
    count: usize,
    dff: Option<DepthGpu>,
    collapsed: bool,
}

impl WgFuser {
    /// A fuser for `w × h` frames; with `measure`, each frame's focus slice is
    /// taken as it is folded and `depth` runs the pass after `finish`.
    pub fn new(w: usize, h: usize, fp: FuseParams, measure: Option<&DepthParams>) -> Result<WgFuser, String> {
        let g = gpu()?;
        check_size(&g, w, h)?;
        let n = w * h;
        let levels = fp.levels.unwrap_or_else(|| auto_levels(w, h, 32)).max(1);
        let depth_level = fp.depth_level.min(levels - 1);
        let mut dims = vec![(w, h)];
        for _ in 0..levels {
            let (cw, ch) = *dims.last().unwrap();
            dims.push((half(cw), half(ch)));
        }
        let cur: Vec<_> = dims.iter().enumerate().map(|(l, &(lw, lh))| g.buffer_f32(&format!("cur{l}"), 3 * lw * lh)).collect();
        let acc: Vec<_> = dims.iter().enumerate().map(|(l, &(lw, lh))| g.buffer_f32(&format!("acc{l}"), if l == depth_level { 4 } else { 3 } * lw * lh)).collect();
        let best: Vec<_> = dims.iter().enumerate().map(|(l, &(lw, lh))| g.buffer_f32(&format!("best{l}"), lw * lh)).collect();
        let wtv = binomial(fp.energy_radius);
        let klen = wtv.len() as u32;
        let wt = g.buffer_init("window", bytemuck::cast_slice(&wtv));
        let dff = match measure {
            Some(dp) => Some(DepthGpu::new(&g, w, h, dp.clone())?),
            None => None,
        };
        let f = WgFuser {
            w,
            h,
            levels,
            depth_level,
            fp,
            dims,
            cur,
            acc,
            best,
            tmp_half: g.buffer_f32("tmp_half", (half(w) * h).max(half(h) * w)),
            tmp_full: g.buffer_f32("tmp_full", n),
            en: g.buffer_f32("en", n),
            en2: g.buffer_f32("en2", n),
            wt,
            klen,
            up: g.buffer("upload u16", (3 * n).div_ceil(2) as u64 * 4),
            aff: g.buffer_init("affine", bytemuck::cast_slice(&[0f32; 4])),
            tops: Vec::new(),
            count: 0,
            dff,
            collapsed: false,
            g,
        };
        let g = f.g.clone();
        let mut rec = g.rec();
        record_reset(&f.bufs(), &mut rec);
        rec.submit();
        Ok(f)
    }

    fn bufs(&self) -> FoldBufs<'_> {
        FoldBufs { dims: &self.dims, levels: self.levels, halo: halo_guide(&self.fp, self.levels), cur: &self.cur, acc: &self.acc, best: &self.best, tmp_half: &self.tmp_half, en: &self.en }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether the fuser takes the frames' focus slices (made with a depth pass).
    pub fn measures(&self) -> bool {
        self.dff.is_some()
    }

    /// Fold a frame from the host (planes in [0, 1]).
    pub fn push(&mut self, img: &Img3) -> Result<(), String> {
        if img.w != self.w || img.h != self.h {
            return Err("frame size mismatch".into());
        }
        let g = self.g.clone();
        block_on(g.upload(&self.up, bytemuck::cast_slice(&to_u16(img))))?;
        let mut rec = g.rec();
        record_warp(&g, &mut rec, &self.up, &self.aff, &self.cur[0], self.w, self.h, Sim::id(), Interp::Nearest);
        self.fold(rec)
    }

    /// Fold a frame the device holds (3 f32 planes, `WgFrames::frame`).
    pub fn push_buf(&mut self, frame: &wgpu::Buffer) -> Result<(), String> {
        let g = self.g.clone();
        let mut rec = g.rec();
        rec.copy(frame, 0, &self.cur[0], 0, (3 * self.w * self.h * 4) as u64);
        self.fold(rec)
    }

    fn fold(&mut self, mut rec: Rec<'_>) -> Result<(), String> {
        let g = self.g.clone();
        let (w, h, n) = (self.w, self.h, self.w * self.h);
        if let Some(dff) = &self.dff {
            rec.dispatch("luma_f32", [Some(&self.cur[0]), None, Some(&self.tmp_full), None, None, None], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(n));
            dff.record_measure(&mut rec, &self.tmp_full, &self.en, w, h);
        }
        record_fold_in(&self.bufs(), self.klen, &self.wt, self.fp.use_chroma, &mut rec, &self.en2, Some((self.count, self.depth_level)), None);
        rec.submit();
        let (tw, th) = self.dims[self.levels];
        let top = block_on(g.read_f32(&self.cur[self.levels], 3 * tw * th))?;
        self.tops.push(Img3 { w: tw, h: th, p: [top[..tw * th].to_vec(), top[tw * th..2 * tw * th].to_vec(), top[2 * tw * th..].to_vec()] });
        if let Some(dff) = &mut self.dff {
            block_on(dff.take_slice(&g))?;
        }
        self.count += 1;
        Ok(())
    }

    /// `fuse::winner_shares` of the frames folded so far.
    pub fn shares(&self) -> Result<Vec<f32>, String> {
        let nsel = halo_guide(&self.fp, self.levels).map_or(self.levels, |(gd, _)| gd + 1);
        if self.count == 0 || self.depth_level >= nsel {
            return Ok(Vec::new());
        }
        let (dw, dh) = self.dims[self.depth_level];
        let wn = dw * dh;
        let acc = block_on(self.g.read_f32(&self.acc[self.depth_level], 4 * wn))?;
        let best = block_on(self.g.read_f32(&self.best[self.depth_level], wn))?;
        Ok(winner_shares(&acc[3 * wn..], &best, self.count))
    }

    /// Fuse the residuals and collapse. Returns the fused image (clamped to
    /// [0, 1]) and the winner map of `depth_level` nearest-upsampled to full
    /// resolution. The image stays on the device for `depth`.
    pub fn finish(&mut self) -> Result<(Img3, Vec<f32>), String> {
        if self.count == 0 {
            return Err("no frames pushed".into());
        }
        let g = self.g.clone();
        let mut rec = g.rec();
        record_collapse_in(&g, &self.bufs(), &self.fp, &mut rec, &self.tops);
        rec.submit();
        self.collapsed = true;
        let img = read_img3(&g, &self.acc[0], self.w, self.h)?;
        let (dw, dh) = self.dims[self.depth_level];
        let wn = dw * dh;
        let winner = block_on(g.read_f32(&self.acc[self.depth_level], 4 * wn))?[3 * wn..].to_vec();
        Ok((img, upsample_index(&winner, dw, dh, self.w, self.h, self.depth_level)))
    }

    /// The depth pass over the slices taken during the fold, guided by the
    /// fused image left on the device by `finish` (`wg::depth::DepthGpu`).
    pub fn depth(&mut self, log: &mut dyn FnMut(String)) -> Result<DepthMap, String> {
        if !self.collapsed {
            return Err("the depth pass comes after finish".into());
        }
        let g = self.g.clone();
        let dff = self.dff.as_mut().ok_or("no focus slices were taken: the fuser was made without a depth pass")?;
        let (w, h, n) = (self.w, self.h, self.w * self.h);
        let mut rec = g.rec();
        rec.dispatch("luma_f32", [Some(&self.acc[0]), None, Some(&self.en), None, None, None], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(n));
        rec.submit();
        let lines: RefCell<Vec<String>> = RefCell::new(Vec::new());
        let (_, conf_w, full) = block_on(dff.finish(&g, w, h, &self.en, &self.tmp_full, &self.en2, &self.up, &|s| lines.borrow_mut().push(s.to_string())))?;
        for l in lines.into_inner() {
            log(l.trim_start_matches("[lapstack] ").to_string());
        }
        let k = 1usize << dff.params.scale;
        let (dw, dh) = (blocks(w, k), blocks(h, k));
        let last = (self.count.max(2) - 1) as f32;
        let depth: Vec<f32> = full.par_iter().map(|&v| v as f32 * last / 65535.0).collect();
        let conf = upsample_bilinear(&conf_w, dw, dh, w, h, k);
        let floor = block_on(dff.floor_values(&g))?;
        Ok(DepthMap { depth, conf, w, h, dw, dh, floor })
    }
}

// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! Alignment — 4-DOF similarity registration. Direct intensity-based,
//! coarse-to-fine, DC-removed-RMS on luminance, Spline4x4 resampling,
//! sequential chaining to frame 0. Uses a bounded Nelder-Mead optimiser.
//!
//! The registration search runs on its own Gaussian pyramid (Burt's
//! generating kernel with a = 0.33, border-renormalised, halved while
//! `h > 64 && w > 8`), independent of the fusion pyramid in `pyramid.rs`.
//! Also home to the small pieces the pipeline shares: `AlignParams`, the
//! cooperative `CancelToken` and the `Cancelled` marker.

use crate::pyramid::{Img3, for_rows};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy)]
pub struct AlignParams {
    pub shift: bool,
    pub scale: bool,
    pub rotation: bool,
    /// Skip the N finest pyramid levels during the fit (0 = full res).
    pub coarsen: usize,
    /// Run the cost search on the CUDA GPU (needs the `gpu` build feature).
    pub gpu: bool,
}

impl Default for AlignParams {
    fn default() -> AlignParams {
        AlignParams { shift: true, scale: true, rotation: true, coarsen: 0, gpu: false }
    }
}

/// Cooperative cancellation flag; clone it into the caller, pass it to the aligner.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> CancelToken {
        CancelToken::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
    /// Stage-internal check: `cancel.check()?` at frame boundaries.
    pub fn check(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() { Err(Cancelled) } else { Ok(()) }
    }
}

/// Marker error returned by the alignment stage when the token fired.
#[derive(Debug)]
pub struct Cancelled;

#[derive(Clone, Copy, PartialEq)]
pub struct Sim {
    pub xoff: f64, // fraction of width
    pub yoff: f64, // fraction of height
    pub scale: f64,
    pub rot: f64, // radians
}

impl Sim {
    pub fn id() -> Sim {
        Sim { xoff: 0.0, yoff: 0.0, scale: 1.0, rot: 0.0 }
    }
    pub fn from_vec(v: &[f64]) -> Sim {
        Sim { xoff: v[0], yoff: v[1], scale: v[2], rot: v[3] }
    }
    pub fn as_vec(&self) -> [f64; 4] {
        [self.xoff, self.yoff, self.scale, self.rot]
    }

    /// Forward 2x3 affine mapping SOURCE(target) -> REFERENCE coords.
    pub fn matrix(&self, w: usize, h: usize) -> [[f64; 3]; 2] {
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let (c, s) = (self.rot.cos(), self.rot.sin());
        let sc = self.scale;
        let (a, b, d, e) = (sc * c, -sc * s, sc * s, sc * c);
        let tx = cx + self.xoff * w as f64 - (a * cx + b * cy);
        let ty = cy + self.yoff * h as f64 - (d * cx + e * cy);
        [[a, b, tx], [d, e, ty]]
    }
}

/// An axis-aligned pixel rectangle: `x, y` its top-left corner, `w, h` its size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    pub fn full(w: usize, h: usize) -> Rect {
        Rect { x: 0, y: 0, w, h }
    }
    pub fn is_full(&self, w: usize, h: usize) -> bool {
        self.x == 0 && self.y == 0 && self.w == w && self.h == h
    }
    pub fn area(&self) -> usize {
        self.w * self.h
    }
}

/// Interpolation support of the warp: a destination pixel is only sound when
/// its source point lies this far inside the frame (Spline4x4 reads two
/// samples each way; outside, the warp repeats the edge).
const WARP_MARGIN: f64 = 2.0;

/// The largest axis-aligned rectangle every aligned frame covers with real
/// pixels: the area to crop the result to, so no smeared border of any warped
/// frame is left in it. For each frame, a destination pixel is covered when its
/// source point (the frame's inverse transform of it) lies `WARP_MARGIN` inside
/// the source; that is a convex quadrilateral, and its cut with a pixel row is
/// one interval, so each row's common interval is an intersection over frames
/// and the best rectangle is the largest one spanning consecutive rows. Frames
/// at the identity are sampled straight and cover everything. Rows are pixel
/// centres, rectangles pixel-aligned; the full frame comes back when nothing
/// is cut. The whole frame when the frames leave no common area.
pub fn common_area(sims: &[Sim], w: usize, h: usize) -> Rect {
    let full = Rect::full(w, h);
    if w == 0 || h == 0 {
        return full;
    }
    let invs: Vec<[[f64; 3]; 2]> = sims.iter().filter(|s| **s != Sim::id()).map(|s| affine_inv(s.matrix(w, h))).collect();
    if invs.is_empty() {
        return full;
    }
    let (m, xmax, ymax) = (WARP_MARGIN, (w - 1) as f64 - WARP_MARGIN, (h - 1) as f64 - WARP_MARGIN);
    // per row, the x-interval [lo, hi] (inclusive pixel columns) covered by every frame
    let mut rows: Vec<(i64, i64)> = Vec::with_capacity(h);
    for y in 0..h {
        let (mut lo, mut hi) = (0f64, (w - 1) as f64);
        for inv in &invs {
            // source x and y are affine in the column: s = a·x + k
            for (a, k, smin, smax) in [(inv[0][0], inv[0][1] * y as f64 + inv[0][2], m, xmax), (inv[1][0], inv[1][1] * y as f64 + inv[1][2], m, ymax)] {
                if a.abs() < 1e-12 {
                    if k < smin || k > smax { lo = 1.0; hi = 0.0; }
                } else {
                    let (x1, x2) = ((smin - k) / a, (smax - k) / a);
                    lo = lo.max(x1.min(x2));
                    hi = hi.min(x1.max(x2));
                }
            }
        }
        rows.push(if lo <= hi { (lo.ceil() as i64, hi.floor() as i64) } else { (1, 0) });
    }
    // the largest rectangle over consecutive rows: for each top row, extend downwards
    // while narrowing to the rows' common interval
    let mut best = Rect { x: 0, y: 0, w: 0, h: 0 };
    for y0 in 0..h {
        let (mut lo, mut hi) = rows[y0];
        for y1 in y0..h {
            lo = lo.max(rows[y1].0);
            hi = hi.min(rows[y1].1);
            if lo > hi {
                break;
            }
            let (rw, rh) = ((hi - lo + 1) as usize, y1 - y0 + 1);
            if rw * rh > best.area() {
                best = Rect { x: lo as usize, y: y0, w: rw, h: rh };
            }
        }
    }
    if best.area() == 0 { full } else { best }
}

pub fn affine_inv(m: [[f64; 3]; 2]) -> [[f64; 3]; 2] {
    let (a, b, tx) = (m[0][0], m[0][1], m[0][2]);
    let (d, e, ty) = (m[1][0], m[1][1], m[1][2]);
    let det = a * e - b * d;
    let (ia, ib, id, ie) = (e / det, -b / det, -d / det, a / det);
    [[ia, ib, -(ia * tx + ib * ty)], [id, ie, -(id * tx + ie * ty)]]
}

/// 4-tap interpolating cubic (Spline4x4Kernel).
#[inline]
fn spline4(t: f64) -> [f64; 4] {
    [
        ((-1.0 / 3.0 * t + 0.8) * t - 0.46666667) * t,
        ((t - 1.8) * t - 0.2) * t + 1.0,
        ((1.2 - t) * t + 0.8) * t,
        ((1.0 / 3.0 * t - 0.2) * t - 0.13333334) * t,
    ]
}

/// Warp a single plane by `sim` into (ow x oh); edge-clamped, with a validity mask.
pub fn warp_plane(src: &[f32], w: usize, h: usize, sim: &Sim, ow: usize, oh: usize) -> (Vec<f32>, Vec<u8>) {
    let inv = affine_inv(sim.matrix(w, h));
    let mut out = vec![0f32; ow * oh];
    let mut valid = vec![0u8; ow * oh];
    out.par_chunks_mut(ow).zip(valid.par_chunks_mut(ow)).enumerate().for_each(|(y, (orow, vrow))| {
        for x in 0..ow {
            let sx = inv[0][0] * x as f64 + inv[0][1] * y as f64 + inv[0][2];
            let sy = inv[1][0] * x as f64 + inv[1][1] * y as f64 + inv[1][2];
            vrow[x] = (sx >= 0.0 && sx <= (w - 1) as f64 && sy >= 0.0 && sy <= (h - 1) as f64) as u8;
            let x0 = sx.floor() as isize;
            let y0 = sy.floor() as isize;
            let wx = spline4(sx - x0 as f64);
            let wy = spline4(sy - y0 as f64);
            let mut acc = 0f64;
            for j in 0..4 {
                let yy = (y0 + j as isize - 1).clamp(0, h as isize - 1) as usize;
                let mut r = 0f64;
                for i in 0..4 {
                    let xx = (x0 + i as isize - 1).clamp(0, w as isize - 1) as usize;
                    r += wx[i] * src[yy * w + xx] as f64;
                }
                acc += wy[j] * r;
            }
            orow[x] = acc as f32;
        }
    });
    (out, valid)
}

pub fn warp_img3(im: &Img3, sim: &Sim) -> (Img3, Vec<u8>) {
    let (a, valid) = warp_plane(&im.p[0], im.w, im.h, sim, im.w, im.h);
    let (b, _) = warp_plane(&im.p[1], im.w, im.h, sim, im.w, im.h);
    let (c, _) = warp_plane(&im.p[2], im.w, im.h, sim, im.w, im.h);
    (Img3 { w: im.w, h: im.h, p: [a, b, c] }, valid)
}

/// BT.601 luma plane of an RGB image.
pub fn luma(im: &Img3) -> Vec<f32> {
    let mut y = vec![0f32; im.w * im.h];
    for (i, o) in y.iter_mut().enumerate() {
        *o = 0.299 * im.p[0][i] + 0.587 * im.p[1][i] + 0.114 * im.p[2][i];
    }
    y
}

/// DC-removed RMS on Y over the valid region.
fn dc_removed_rms(a: &[f32], b: &[f32], valid: &[u8]) -> f64 {
    let (mut sa, mut sb, mut cnt) = (0f64, 0f64, 0usize);
    for i in 0..a.len() {
        if valid[i] != 0 {
            sa += a[i] as f64;
            sb += b[i] as f64;
            cnt += 1;
        }
    }
    if cnt < 16 {
        return 1e9;
    }
    let (ma, mb) = (sa / cnt as f64, sb / cnt as f64);
    let mut ss = 0f64;
    for i in 0..a.len() {
        if valid[i] != 0 {
            let d = (a[i] as f64 - ma) - (b[i] as f64 - mb);
            ss += d * d;
        }
    }
    (ss / cnt as f64).sqrt()
}

/// Burt generating kernel [c,b,a,b,c] with a = 0.33 (the registration pyramid).
#[inline]
fn burt_kernel() -> [f32; 5] {
    let a = 0.33f32;
    let b = 0.25f32;
    let c = 0.25f32 - a / 2.0;
    [c, b, a, b, c]
}

/// norm[i] = sum of in-bounds kernel taps at full-res position i (reduce denom).
fn norm_full(n: usize, k: &[f32; 5]) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut s = 0.0;
            for t in 0..5 {
                let x = i as isize + t as isize - 2;
                if x >= 0 && (x as usize) < n {
                    s += k[t];
                }
            }
            s
        })
        .collect()
}

/// Gaussian REDUCE (half-size, phase-0 subsample), border-renormalised.
fn reduce_burt(src: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let k = burt_kernel();
    let ow = (w + 1) / 2;
    let oh = (h + 1) / 2;
    let cn = norm_full(w, &k);
    let rn = norm_full(h, &k);
    // horizontal pass -> htmp (h x ow), unnormalized
    let mut htmp = vec![0f32; h * ow];
    for_rows(&mut htmp, ow, |y, row| {
        for oj in 0..ow {
            let cx = 2 * oj;
            let mut a = 0.0;
            for t in 0..5 {
                let s = cx as isize + t as isize - 2;
                if s >= 0 && (s as usize) < w {
                    a += k[t] * src[y * w + s as usize];
                }
            }
            row[oj] = a;
        }
    });
    // vertical pass -> out (oh x ow), normalized by rn*cn
    let mut out = vec![0f32; oh * ow];
    let htmp_ref = &htmp;
    for_rows(&mut out, ow, |oi, row| {
        let cy = 2 * oi;
        for oj in 0..ow {
            let mut a = 0.0;
            for t in 0..5 {
                let s = cy as isize + t as isize - 2;
                if s >= 0 && (s as usize) < h {
                    a += k[t] * htmp_ref[s as usize * ow + oj];
                }
            }
            row[oj] = a / (rn[cy] * cn[2 * oj]);
        }
    });
    (out, ow, oh)
}

pub type Lvl = (Vec<f32>, usize, usize);
pub fn gauss_pyramid(y: &[f32], w: usize, h: usize) -> Vec<Lvl> {
    let mut levels = vec![(y.to_vec(), w, h)];
    let (mut cw, mut ch) = (w, h);
    let mut cur = y.to_vec();
    while ch > 64 && cw > 8 {
        let (d, ow, oh) = reduce_burt(&cur, cw, ch);
        cur = d.clone();
        cw = ow;
        ch = oh;
        levels.push((d, ow, oh));
    }
    levels
}

/// Bounded Nelder-Mead. Minimizes f over the box [lo,hi]. n = 1..4.
pub fn nelder_mead<F: Fn(&[f64]) -> f64>(f: &F, x0: &[f64], lo: &[f64], hi: &[f64]) -> Vec<f64> {
    let n = x0.len();
    if n == 0 {
        return vec![];
    }
    let clamp = |v: &mut Vec<f64>| {
        for k in 0..n {
            v[k] = v[k].clamp(lo[k], hi[k]);
        }
    };
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for k in 0..n {
        let mut v = x0.to_vec();
        v[k] += 0.05 * (hi[k] - lo[k]).abs().max(1e-6);
        clamp(&mut v);
        simplex.push(v);
    }
    let mut fv: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);
    for _ in 0..200 {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| fv[a].partial_cmp(&fv[b]).unwrap());
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        fv = idx.iter().map(|&i| fv[i]).collect();
        // convergence
        if (fv[n] - fv[0]).abs() <= 1e-4 * (1.0 + fv[0].abs()) {
            let mut sz = 0.0;
            for k in 0..n {
                sz = f64::max(sz, (simplex[n][k] - simplex[0][k]).abs());
            }
            if sz <= 1e-4 {
                break;
            }
        }
        let mut c = vec![0.0; n];
        for i in 0..n {
            for k in 0..n {
                c[k] += simplex[i][k];
            }
        }
        for k in 0..n {
            c[k] /= n as f64;
        }
        let mut xr = vec![0.0; n];
        for k in 0..n {
            xr[k] = c[k] + alpha * (c[k] - simplex[n][k]);
        }
        clamp(&mut xr);
        let fr = f(&xr);
        if fr < fv[0] {
            let mut xe = vec![0.0; n];
            for k in 0..n {
                xe[k] = c[k] + gamma * (xr[k] - c[k]);
            }
            clamp(&mut xe);
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fv[n] = fe;
            } else {
                simplex[n] = xr;
                fv[n] = fr;
            }
        } else if fr < fv[n - 1] {
            simplex[n] = xr;
            fv[n] = fr;
        } else {
            let mut xc = vec![0.0; n];
            for k in 0..n {
                xc[k] = c[k] + rho * (simplex[n][k] - c[k]);
            }
            clamp(&mut xc);
            let fc = f(&xc);
            if fc < fv[n] {
                simplex[n] = xc;
                fv[n] = fc;
            } else {
                for i in 1..=n {
                    for k in 0..n {
                        simplex[i][k] = simplex[0][k] + sigma * (simplex[i][k] - simplex[0][k]);
                    }
                    clamp(&mut simplex[i]);
                    fv[i] = f(&simplex[i]);
                }
            }
        }
    }
    let mut best = 0;
    for i in 1..=n {
        if fv[i] < fv[best] {
            best = i;
        }
    }
    simplex[best].clone()
}

pub(crate) fn multiscale_align(
    rf: &[f32],
    tg: &[f32],
    w: usize,
    h: usize,
    init: Sim,
    free: [bool; 4],
    coarsen: usize,
) -> Sim {
    let pref = gauss_pyramid(rf, w, h);
    let ptgt = gauss_pyramid(tg, w, h);
    let n = pref.len().min(ptgt.len());
    let span = [0.10, 0.10, 0.10, 5.0f64.to_radians()];
    let iv = init.as_vec();
    let mut cur = iv;
    let free_idx: Vec<usize> = (0..4).filter(|&k| free[k]).collect();
    let lo_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] - span[k]).collect();
    let hi_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] + span[k]).collect();

    // Refine coarsest -> finest, but stop `coarsen` levels short of full res: the
    // Sim transform is resolution-independent (fractional offset + scale + angle),
    // so a fit at reduced res applies at full res — skipping the full-res warp+RMS
    // (the dominant cost) for a large speedup at sub-px accuracy. Keep >=1 level.
    let finest = coarsen.min(n.saturating_sub(1));
    for lvl in (finest..n).rev() {
        let a_d = &pref[lvl].0;
        let (aw, ah) = (pref[lvl].1, pref[lvl].2);
        let t_d = &ptgt[lvl].0;
        let (tw, th) = (ptgt[lvl].1, ptgt[lvl].2);
        let cur_snap = cur;
        let cost = |xf: &[f64]| -> f64 {
            let mut v = cur_snap;
            for (k, &idx) in free_idx.iter().enumerate() {
                v[idx] = xf[k];
            }
            let (bw, valid) = warp_plane(t_d, tw, th, &Sim::from_vec(&v), aw, ah);
            dc_removed_rms(a_d, &bw, &valid)
        };
        let x0: Vec<f64> = free_idx.iter().map(|&k| cur[k]).collect();
        let best = nelder_mead(&cost, &x0, &lo_f, &hi_f);
        for (k, &idx) in free_idx.iter().enumerate() {
            cur[idx] = best[k];
        }
    }
    Sim::from_vec(&cur)
}

/// Register all frames into frame-0 coordinates (sequential chaining).
/// `on_frame(i, sim)` fires as each frame lands; `cancel` is checked per frame.
pub fn align_stack(
    frames: &[Img3],
    allow_shift: bool,
    allow_scale: bool,
    allow_rotation: bool,
    coarsen: usize,
    cancel: &CancelToken,
    on_frame: &mut dyn FnMut(usize, Sim),
) -> Result<(Vec<Img3>, Vec<Sim>), Cancelled> {
    let (w, h) = (frames[0].w, frames[0].h);
    // Keep only the luma planes: a full three-plane copy of every frame would
    // hold an extra ~0.5 GB/frame (45 MP) for the whole alignment.
    let ys: Vec<Vec<f32>> = frames.iter().map(luma).collect();
    let free = [allow_shift, allow_shift, allow_scale, allow_rotation];

    let mut aligned = vec![frames[0].clone()];
    let mut params = vec![Sim::id()];
    let mut prev_ref = ys[0].clone();
    let mut guess = Sim::id();

    for i in 1..frames.len() {
        cancel.check()?;
        let sim = multiscale_align(&prev_ref, &ys[i], w, h, guess, free, coarsen);
        params.push(sim);
        let (mut wimg, valid) = warp_img3(&frames[i], &sim);
        for c in 0..3 {
            for p in 0..w * h {
                if valid[p] == 0 {
                    wimg.p[c][p] = frames[i].p[c][p];
                }
            }
        }
        aligned.push(wimg);
        let (pr, _) = warp_plane(&ys[i], w, h, &sim, w, h);
        prev_ref = pr;
        guess = sim;
        on_frame(i, sim);
    }
    Ok((aligned, params))
}

pub fn report(sim: &Sim, w: usize, h: usize) -> String {
    format!(
        "dx={:+7.2}px dy={:+7.2}px scale={:.5} rot={:+.3} deg",
        sim.xoff * w as f64,
        sim.yoff * h as f64,
        sim.scale,
        sim.rot.to_degrees()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_area_identity_is_full() {
        assert_eq!(common_area(&[Sim::id(), Sim::id()], 640, 480), Rect::full(640, 480));
        assert_eq!(common_area(&[], 640, 480), Rect::full(640, 480));
    }

    #[test]
    fn common_area_shift() {
        // frame 1 lands 10 px to the right: its left 10 columns are smeared edge, and
        // the warp's 2 px support trims the other sides
        let (w, h) = (640, 480);
        let s = Sim { xoff: 10.0 / w as f64, yoff: 0.0, scale: 1.0, rot: 0.0 };
        let r = common_area(&[Sim::id(), s], w, h);
        assert_eq!(r, Rect { x: 12, y: 2, w: w - 12, h: h - 4 });
    }

    #[test]
    fn common_area_rotation_is_inside_every_frame() {
        let (w, h) = (640, 480);
        let sims = [Sim::id(), Sim { xoff: -0.01, yoff: 0.02, scale: 1.03, rot: 0.02 }, Sim { xoff: 0.005, yoff: -0.01, scale: 0.98, rot: -0.015 }];
        let r = common_area(&sims, w, h);
        assert!(r.w > w / 2 && r.h > h / 2 && !r.is_full(w, h), "{r:?}");
        // every corner pixel of the rectangle maps inside every frame's sound area
        for s in &sims[1..] {
            let inv = affine_inv(s.matrix(w, h));
            for (x, y) in [(r.x, r.y), (r.x + r.w - 1, r.y), (r.x, r.y + r.h - 1), (r.x + r.w - 1, r.y + r.h - 1)] {
                let sx = inv[0][0] * x as f64 + inv[0][1] * y as f64 + inv[0][2];
                let sy = inv[1][0] * x as f64 + inv[1][1] * y as f64 + inv[1][2];
                assert!(sx >= WARP_MARGIN && sx <= (w - 1) as f64 - WARP_MARGIN && sy >= WARP_MARGIN && sy <= (h - 1) as f64 - WARP_MARGIN, "({x},{y}) -> ({sx:.1},{sy:.1})");
            }
        }
        // and it is maximal: one more row or column on any side breaks that for some frame
        let sound = |x: usize, y: usize| sims[1..].iter().all(|s| {
            let inv = affine_inv(s.matrix(w, h));
            let sx = inv[0][0] * x as f64 + inv[0][1] * y as f64 + inv[0][2];
            let sy = inv[1][0] * x as f64 + inv[1][1] * y as f64 + inv[1][2];
            sx >= WARP_MARGIN && sx <= (w - 1) as f64 - WARP_MARGIN && sy >= WARP_MARGIN && sy <= (h - 1) as f64 - WARP_MARGIN
        });
        let row_ok = |y: usize| (r.x..r.x + r.w).all(|x| sound(x, y));
        let col_ok = |x: usize| (r.y..r.y + r.h).all(|y| sound(x, y));
        assert!(r.y == 0 || !row_ok(r.y - 1));
        assert!(r.y + r.h == h || !row_ok(r.y + r.h));
        assert!(r.x == 0 || !col_ok(r.x - 1));
        assert!(r.x + r.w == w || !col_ok(r.x + r.w));
    }
}

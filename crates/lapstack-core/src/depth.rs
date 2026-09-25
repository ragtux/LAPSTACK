// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: AGPL-3.0-only

//! Depth from focus (DFF): a dense, sub-frame depth map from the aligned stack.
//!
//! The pipeline is the modern non-learned "focus volume" recipe, written from
//! the papers:
//!
//! 1. **Focus measure** per frame on luma at full resolution, taken as the
//!    frame is folded (`focus_slice`) so the stack is read once. Default is the
//!    *ring difference filter* of Jeon, Surh, Im & Kweon, "Ring Difference
//!    Filter for Fast and Noise Robust Depth From Focus", IEEE TIP 29 (2019):
//!    the magnitude of the difference between the mean of a small disk and the
//!    mean of the ring around it, a band-pass whose disk/ring support averages
//!    sensor noise away while staying local. The sum-modified Laplacian of
//!    Nayar & Nakagawa ("Shape from Focus", PAMI 1994) is available as the
//!    classical alternative.
//! 2. **Cost aggregation**: the measure is block-summed to a working grid
//!    (1/2^scale resolution) and each slice of the resulting focus volume is
//!    filtered with the *guided filter* (He, Sun & Tang, PAMI 2013) using the
//!    all-in-focus luma as guide — edge-aware aggregation as in fast
//!    cost-volume filtering (Hosni et al., PAMI 2013) and Jeon et al. 2019.
//! 3. **Peak search** per pixel over the frame axis, streamed (one slice at a
//!    time, O(1) memory in the stack size): the global peak, its two
//!    neighbours for the *Gaussian interpolation* of Nayar & Nakagawa 1994
//!    (sub-frame depth), the second-best local maximum for a peak-ratio
//!    confidence, and the profile mean for a prominence term.
//! 4. **Regularisation**: the sub-frame depth is smoothed / inpainted with an
//!    edge-aware *weighted least squares* energy (Farbman et al., SIGGRAPH
//!    2008) whose data term is weighted by the confidence — low-confidence
//!    pixels (flat, noisy, or ambiguous profiles) take their depth from
//!    confident neighbours along paths that do not cross image edges. The
//!    separable *fast global smoother* of Min et al. (IEEE TIP 2014) gives the
//!    initial guess and a conjugate gradient preconditioned by a multigrid
//!    V-cycle (`WlsSolver`) solves the 2-D system to a relative residual of
//!    1e-5. Regularising in the depth domain with confidence
//!    weights is the 2-D counterpart of the focus-volume WLS of Ali & Mahmood
//!    (Information Sciences, 2020).
//! 5. **Upsampling** back to full resolution with the guided filter (the
//!    "fast guided filter" coefficients of He & Sun 2015 evaluated on the
//!    full-resolution luma), so depth edges land on image edges.
//!
//! Output: a per-pixel frame index in `[0, n−1]` (fractional) plus a
//! confidence map in `[0, 1]`.

use crate::pyramid::{Img3, for_rows, half, reflect};
use rayon::prelude::*;
use crate::stack::FrameSource;
use std::time::Instant;

/// Which per-pixel focus measure to run on each frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FocusMeasure {
    /// Ring difference filter (Jeon et al. 2019): |mean(disk r≤r_in) − mean(ring r_in<r≤r_out)|.
    Rdf { r_in: usize, r_out: usize },
    /// Sum-modified Laplacian (Nayar & Nakagawa 1994) with sample step `step`.
    Sml { step: usize },
}

/// How the working-resolution depth is brought back to full resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Upsample {
    Bilinear,
    /// Guided-filter upsampling with the full-res luma as guide.
    Guided { radius: usize, eps: f32 },
}

#[derive(Clone, Debug)]
pub struct DepthParams {
    /// Working resolution is 1/2^scale (1 = half resolution).
    pub scale: usize,
    pub focus: FocusMeasure,
    /// Guided-filter aggregation radius (working-res pixels) and regulariser.
    pub agg_radius: usize,
    pub agg_eps: f32,
    /// WLS smoothness weight and guide-edge sensitivity (luma units).
    pub lambda: f32,
    pub sigma_c: f32,
    /// Conjugate-gradient iteration cap for the WLS solve.
    pub cg_iters: usize,
    /// 3×3 median on the raw sub-frame depth before regularisation.
    pub median: bool,
    pub upsample: Upsample,
    /// Noise gate: full confidence needs a peak ≥ (1 + gate) × the noise floor
    /// (median over pixels of the profile minimum). 0 = off.
    pub gate: f32,
    /// Robust data term: after the first WLS solve, data weights are scaled by
    /// a Huber factor `min(1, robust/|d − u|)` (frames) and the system is
    /// solved once more, so isolated outliers stop pulling. 0 = off.
    pub robust: f32,
}

impl Default for DepthParams {
    fn default() -> Self {
        DepthParams {
            scale: 1,
            focus: FocusMeasure::Rdf { r_in: 1, r_out: 3 },
            agg_radius: 3,
            agg_eps: 1e-4,
            lambda: 3.0,
            sigma_c: 0.04,
            cg_iters: 200,
            median: true,
            upsample: Upsample::Guided { radius: 2, eps: 1e-2 },
            gate: 1.0,
            robust: 1.0,
        }
    }
}

impl FocusMeasure {
    pub fn parse(s: &str) -> Option<FocusMeasure> {
        let mut it = s.split(':');
        match it.next()? {
            "rdf" => {
                let r_in = it.next().map_or(Some(1), |v| v.parse().ok())?;
                let r_out = it.next().map_or(Some(r_in + 2), |v| v.parse().ok())?;
                (r_out > r_in).then_some(FocusMeasure::Rdf { r_in, r_out })
            }
            "sml" => Some(FocusMeasure::Sml { step: it.next().map_or(Some(1), |v| v.parse().ok())? }),
            _ => None,
        }
    }
}

impl Upsample {
    pub fn parse(s: &str) -> Option<Upsample> {
        let mut it = s.split(':');
        match it.next()? {
            "bilinear" => Some(Upsample::Bilinear),
            "guided" => Some(Upsample::Guided {
                radius: it.next().map_or(Some(2), |v| v.parse().ok())?,
                eps: it.next().map_or(Some(1e-2), |v| v.parse().ok())?,
            }),
            _ => None,
        }
    }
}

/// Result of [`depth_from_focus`].
pub struct DepthMap {
    /// Full-resolution fractional frame index per pixel, in `[0, n−1]`.
    pub depth: Vec<f32>,
    /// Full-resolution confidence in `[0, 1]` (bilinear from the working grid).
    pub conf: Vec<f32>,
    pub w: usize,
    pub h: usize,
    /// Working-grid dimensions the volume was processed at.
    pub dw: usize,
    pub dh: usize,
    /// The noise floor of every working-grid cell: the least (aggregated)
    /// contrast any frame showed there — in a cell no frame is sharp in, the
    /// sensor noise's share of the measure. The weighted average (`wav.rs`)
    /// weighs the frames by their contrast above it.
    pub floor: Vec<f32>,
}

// ---------------------------------------------------------------- helpers

pub fn luma(img: &Img3) -> Vec<f32> {
    let (r, g, b) = (&img.p[0], &img.p[1], &img.p[2]);
    let mut y = vec![0f32; r.len()];
    y.par_iter_mut().enumerate().for_each(|(i, o)| *o = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]);
    y
}

/// Working-grid size for a full-res dimension and block size `k`.
#[inline]
pub fn blocks(n: usize, k: usize) -> usize {
    n.div_ceil(k)
}

/// Mean over `k×k` blocks (partial blocks at the far edges use their own count).
pub fn block_mean(src: &[f32], w: usize, h: usize, k: usize) -> (Vec<f32>, usize, usize) {
    let (dw, dh) = (blocks(w, k), blocks(h, k));
    if k == 1 {
        return (src.to_vec(), dw, dh);
    }
    let mut out = vec![0f32; dw * dh];
    for_rows(&mut out, dw, |oy, row| {
        let y0 = oy * k;
        let y1 = (y0 + k).min(h);
        for y in y0..y1 {
            let s = &src[y * w..y * w + w];
            for (ox, o) in row.iter_mut().enumerate() {
                let x0 = ox * k;
                let x1 = (x0 + k).min(w);
                *o += s[x0..x1].iter().sum::<f32>();
            }
        }
        for (ox, o) in row.iter_mut().enumerate() {
            let cnt = ((y1 - y0) * ((ox * k + k).min(w) - ox * k)) as f32;
            *o /= cnt;
        }
    });
    (out, dw, dh)
}

/// Bilinear resampling of a working-grid plane (block size `k`) to `w×h`.
/// Grid samples sit at block centres.
pub fn upsample_bilinear(g: &[f32], dw: usize, dh: usize, w: usize, h: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    let inv = 1.0 / k as f32;
    for_rows(&mut out, w, |y, row| {
        let fy = ((y as f32 + 0.5) * inv - 0.5).clamp(0.0, (dh - 1) as f32);
        let y0 = fy as usize;
        let y1 = (y0 + 1).min(dh - 1);
        let ty = fy - y0 as f32;
        let (r0, r1) = (&g[y0 * dw..y0 * dw + dw], &g[y1 * dw..y1 * dw + dw]);
        for (x, o) in row.iter_mut().enumerate() {
            let fx = ((x as f32 + 0.5) * inv - 0.5).clamp(0.0, (dw - 1) as f32);
            let x0 = fx as usize;
            let x1 = (x0 + 1).min(dw - 1);
            let tx = fx - x0 as f32;
            let top = r0[x0] + (r0[x1] - r0[x0]) * tx;
            let bot = r1[x0] + (r1[x1] - r1[x0]) * tx;
            *o = top + (bot - top) * ty;
        }
    });
    out
}

/// Pointer wrapper so disjoint column blocks of one buffer can be written from
/// rayon workers. Every use writes only columns `[x0, x1)` of its own block.
#[derive(Clone, Copy)]
struct ColPtr(*mut f32);
unsafe impl Send for ColPtr {}
unsafe impl Sync for ColPtr {}
impl ColPtr {
    /// The element `off` in; a method so a closure captures the whole
    /// (`Sync`) wrapper and not its raw pointer.
    #[inline]
    fn at(self, off: usize) -> *mut f32 {
        // SAFETY: callers index inside the buffer the wrapper was made from.
        unsafe { self.0.add(off) }
    }
}

const COL_BLOCK: usize = 64;

/// Run `f(x0, x1)` over disjoint column ranges in parallel; `f` receives a raw
/// pointer to `out` and may write only rows' elements inside its range.
fn par_col_blocks<F: Fn(usize, usize, ColPtr) + Sync>(out: &mut [f32], w: usize, f: F) {
    let p = ColPtr(out.as_mut_ptr());
    let nb = blocks(w, COL_BLOCK);
    (0..nb).into_par_iter().for_each(|b| {
        let x0 = b * COL_BLOCK;
        f(x0, (x0 + COL_BLOCK).min(w), p);
    });
}

/// Box filter (mean over a `(2r+1)²` window clipped at the image border).
///
/// It owns its scratch plane and writes into a caller's buffer: a fresh
/// working-grid plane is 45 MB at 45 MP, and faulting one in from a hundred
/// threads at once costs more than the sums do (170 ms against 20 ms), so
/// the depth pass's filters keep their planes from one slice to the next.
/// The moving sums run in f64: a running sum in f32 keeps the round-off of
/// every value that passed through it, and on a plane whose values span
/// many decades (the weighted average's weights: a sharp cell's against the
/// tiny even weight of a flat one) that drift outweighed the small values
/// and went negative — streaks along the rows and blocks down the columns.
pub struct BoxFilter {
    w: usize,
    h: usize,
    r: usize,
    inv_count: Vec<f32>,
    tmp: Vec<f32>,
}

impl BoxFilter {
    pub fn new(w: usize, h: usize, r: usize) -> BoxFilter {
        let cnt = |n: usize, i: usize| ((i + r + 1).min(n) - i.saturating_sub(r)) as f32;
        let mut inv_count = vec![0f32; w * h];
        for_rows(&mut inv_count, w, |y, row| {
            let cy = cnt(h, y);
            for (x, o) in row.iter_mut().enumerate() {
                *o = 1.0 / (cy * cnt(w, x));
            }
        });
        BoxFilter { w, h, r, inv_count, tmp: vec![0f32; w * h] }
    }

    /// Window sums (not normalised) into `out`.
    pub fn sum_into(&mut self, src: &[f32], out: &mut [f32]) {
        let (w, h, r) = (self.w, self.h, self.r);
        assert_eq!(src.len(), w * h);
        assert_eq!(out.len(), w * h);
        // horizontal moving sum
        let tmp = &mut self.tmp;
        for_rows(tmp, w, |y, row| {
            let s = &src[y * w..y * w + w];
            let mut acc: f64 = s[..(r + 1).min(w)].iter().map(|&v| v as f64).sum();
            for x in 0..w {
                row[x] = acc as f32;
                if x + r + 1 < w {
                    acc += s[x + r + 1] as f64;
                }
                if x >= r {
                    acc -= s[x - r] as f64;
                }
            }
        });
        // vertical moving sum, per column block
        let tmp: &[f32] = tmp;
        par_col_blocks(out, w, |x0, x1, p| {
            let bw = x1 - x0;
            let mut acc = [0f64; COL_BLOCK];
            let acc = &mut acc[..bw];
            for y in 0..(r + 1).min(h) {
                for (a, v) in acc.iter_mut().zip(&tmp[y * w + x0..y * w + x1]) {
                    *a += *v as f64;
                }
            }
            for y in 0..h {
                // SAFETY: this block owns columns x0..x1 of every row; no other
                // worker touches them, and `out` outlives the parallel loop.
                let dst = unsafe { std::slice::from_raw_parts_mut(p.0.add(y * w + x0), bw) };
                for (d, a) in dst.iter_mut().zip(acc.iter()) {
                    *d = *a as f32;
                }
                if y + r + 1 < h {
                    for (a, v) in acc.iter_mut().zip(&tmp[(y + r + 1) * w + x0..(y + r + 1) * w + x1]) {
                        *a += *v as f64;
                    }
                }
                if y >= r {
                    for (a, v) in acc.iter_mut().zip(&tmp[(y - r) * w + x0..(y - r) * w + x1]) {
                        *a -= *v as f64;
                    }
                }
            }
        });
    }

    /// Window means into `out`.
    pub fn mean_into(&mut self, src: &[f32], out: &mut [f32]) {
        self.sum_into(src, out);
        out.par_iter_mut().zip(&self.inv_count).for_each(|(v, c)| *v *= c);
    }

    pub fn mean(&mut self, src: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; src.len()];
        self.mean_into(src, &mut out);
        out
    }
}

/// Guided filter (He, Sun & Tang 2013) with a fixed grayscale guide. Made
/// once per guide and radius, it keeps the guide's statistics and the four
/// planes a filtering needs, so a slice costs only the arithmetic.
pub struct GuidedFilter {
    guide: Vec<f32>,
    mean_i: Vec<f32>,
    var_i: Vec<f32>,
    bf: BoxFilter,
    eps: f32,
    /// mean_p / mean_b, I·p / corr_Ip, a, b
    s: [Vec<f32>; 4],
}

impl GuidedFilter {
    pub fn new(guide: Vec<f32>, w: usize, h: usize, r: usize, eps: f32) -> GuidedFilter {
        let n = w * h;
        let mut bf = BoxFilter::new(w, h, r);
        let mut mean_i = vec![0f32; n];
        bf.mean_into(&guide, &mut mean_i);
        let mut var_i = vec![0f32; n];
        let mut ii = vec![0f32; n];
        ii.par_iter_mut().zip(&guide).for_each(|(o, v)| *o = v * v);
        bf.mean_into(&ii, &mut var_i);
        var_i.par_iter_mut().zip(&mean_i).for_each(|(v, m)| *v = (*v - m * m).max(0.0));
        let s = [ii, vec![0f32; n], vec![0f32; n], vec![0f32; n]];
        GuidedFilter { guide, mean_i, var_i, bf, eps, s }
    }

    /// The box-averaged linear coefficients, `q ≈ ā·I + b̄`, left in the
    /// scratch planes `s[1]` (ā) and `s[0]` (b̄).
    fn coeffs_into(&mut self, p: &[f32]) {
        let [s0, s1, s2, s3] = &mut self.s;
        let (eps, guide, mean_i, var_i) = (self.eps, &self.guide, &self.mean_i, &self.var_i);
        self.bf.mean_into(p, s0); // mean_p
        s1.par_iter_mut().zip(guide).zip(p).for_each(|((o, i), p)| *o = i * p);
        self.bf.mean_into(s1, s2); // corr_Ip
        s2.par_iter_mut().zip(s3.par_iter_mut()).zip(s0.par_iter()).enumerate().for_each(|(k, ((a, b), mp))| {
            let cov = *a - mean_i[k] * mp;
            *a = cov / (var_i[k] + eps); // a
            *b = mp - *a * mean_i[k]; // b
        });
        self.bf.mean_into(s2, s1); // ā
        self.bf.mean_into(s3, s0); // b̄
    }

    /// Linear coefficients `(ā, b̄)` such that `q ≈ ā·I + b̄` (already box-averaged).
    pub fn coeffs(&mut self, p: &[f32]) -> (Vec<f32>, Vec<f32>) {
        self.coeffs_into(p);
        (self.s[1].clone(), self.s[0].clone())
    }

    /// The filtered plane into `out`.
    pub fn filter_into(&mut self, p: &[f32], out: &mut [f32]) {
        self.coeffs_into(p);
        let (a, b) = (&self.s[1], &self.s[0]);
        out.par_iter_mut().enumerate().for_each(|(k, o)| *o = a[k] * self.guide[k] + b[k]);
    }

    pub fn filter(&mut self, p: &[f32]) -> Vec<f32> {
        let mut q = vec![0f32; p.len()];
        self.filter_into(p, &mut q);
        q
    }
}

// ---------------------------------------------------------- focus measures

/// Sparse 2-D convolution with reflect-101 borders; `taps` = (dy, dx, weight).
fn conv_sparse(src: &[f32], w: usize, h: usize, taps: &[(isize, isize, f32)]) -> Vec<f32> {
    let rad = taps.iter().map(|t| t.0.abs().max(t.1.abs())).max().unwrap_or(0) as usize;
    let mut rows: Vec<(isize, Vec<(isize, f32)>)> = Vec::new();
    for &(dy, dx, wt) in taps {
        match rows.iter_mut().find(|(d, _)| *d == dy) {
            Some((_, v)) => v.push((dx, wt)),
            None => rows.push((dy, vec![(dx, wt)])),
        }
    }
    let mut out = vec![0f32; w * h];
    let interior = 2 * rad < w;
    for_rows(&mut out, w, |y, acc| {
        for (dy, dxs) in &rows {
            let sy = reflect(y as isize + dy, h);
            let s = &src[sy * w..sy * w + w];
            for &(dx, wt) in dxs {
                if interior {
                    let (lo, hi) = (rad, w - rad);
                    let src_lo = (lo as isize + dx) as usize;
                    for (a, v) in acc[lo..hi].iter_mut().zip(&s[src_lo..src_lo + hi - lo]) {
                        *a += wt * v;
                    }
                    for x in (0..lo).chain(hi..w) {
                        acc[x] += wt * s[reflect(x as isize + dx, w)];
                    }
                } else {
                    for x in 0..w {
                        acc[x] += wt * s[reflect(x as isize + dx, w)];
                    }
                }
            }
        }
    });
    out
}

/// Ring difference filter taps: +1/|disk| for r ≤ r_in, −1/|ring| for r_in < r ≤ r_out.
pub fn rdf_taps(r_in: usize, r_out: usize) -> Vec<(isize, isize, f32)> {
    let (ri2, ro2) = ((r_in * r_in) as isize, (r_out * r_out) as isize);
    let mut disk = Vec::new();
    let mut ring = Vec::new();
    let r = r_out as isize;
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = dx * dx + dy * dy;
            if d2 <= ri2 {
                disk.push((dy, dx));
            } else if d2 <= ro2 {
                ring.push((dy, dx));
            }
        }
    }
    let (wd, wr) = (1.0 / disk.len() as f32, -1.0 / ring.len().max(1) as f32);
    disk.into_iter().map(|(y, x)| (y, x, wd)).chain(ring.into_iter().map(|(y, x)| (y, x, wr))).collect()
}

/// Per-pixel focus measure of a luma plane.
pub fn focus_measure(y: &[f32], w: usize, h: usize, fm: FocusMeasure) -> Vec<f32> {
    match fm {
        FocusMeasure::Rdf { r_in, r_out } => {
            let mut f = conv_sparse(y, w, h, &rdf_taps(r_in, r_out));
            f.par_iter_mut().for_each(|v| *v = v.abs());
            f
        }
        FocusMeasure::Sml { step } => {
            let s = step.max(1) as isize;
            let mut out = vec![0f32; w * h];
            for_rows(&mut out, w, |yy, row| {
                let (up, dn) = (reflect(yy as isize - s, h), reflect(yy as isize + s, h));
                let (c, u, d) = (&y[yy * w..yy * w + w], &y[up * w..up * w + w], &y[dn * w..dn * w + w]);
                for (x, o) in row.iter_mut().enumerate() {
                    let (l, r) = (c[reflect(x as isize - s, w)], c[reflect(x as isize + s, w)]);
                    *o = (2.0 * c[x] - l - r).abs() + (2.0 * c[x] - u[x] - d[x]).abs();
                }
            });
            out
        }
    }
}

// ------------------------------------------------------ streaming peak search

/// Per-pixel state of the focus profile, updated one slice at a time.
#[derive(Clone, Copy)]
struct Px {
    /// Best local maximum (value, left neighbour, right neighbour; −1 = none).
    c1: f32,
    l1: f32,
    r1: f32,
    /// Second-best local maximum (−1 = none).
    c2: f32,
    /// Last two slice values.
    prev: f32,
    prev2: f32,
    sum: f32,
    cmin: f32,
    i1: u32,
}

impl Px {
    #[inline]
    fn register(&mut self, val: f32, idx: u32, l: f32, r: f32) {
        if val > self.c1 {
            if self.c1 >= 0.0 {
                self.c2 = self.c1;
            }
            self.c1 = val;
            self.i1 = idx;
            self.l1 = l;
            self.r1 = r;
        } else if val > self.c2 {
            self.c2 = val;
        }
    }
}

pub struct PeakTracker {
    px: Vec<Px>,
    m: usize,
}

impl PeakTracker {
    pub fn new(n: usize) -> PeakTracker {
        PeakTracker {
            px: vec![Px { c1: -1.0, l1: -1.0, r1: -1.0, c2: -1.0, prev: 0.0, prev2: 0.0, sum: 0.0, cmin: f32::INFINITY, i1: 0 }; n],
            m: 0,
        }
    }

    /// Fold slice `m` (aggregated focus values ≥ 0).
    pub fn push(&mut self, c: &[f32]) {
        assert_eq!(c.len(), self.px.len());
        let m = self.m;
        self.px.par_iter_mut().zip(c).for_each(|(p, &v)| {
            if m >= 1 {
                let is_peak = (m == 1 || p.prev >= p.prev2) && p.prev > v;
                if is_peak {
                    let l = if m >= 2 { p.prev2 } else { -1.0 };
                    p.register(p.prev, (m - 1) as u32, l, v);
                }
            }
            p.sum += v;
            p.cmin = p.cmin.min(v);
            p.prev2 = p.prev;
            p.prev = v;
        });
        self.m += 1;
    }

    /// Close the profiles: sub-frame depth, raw confidence and the profile's
    /// minimum (the cell's noise floor) per pixel.
    pub fn finish(mut self, gate: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.m;
        assert!(n > 0, "no slices");
        self.px.par_iter_mut().for_each(|p| {
            if n == 1 || p.prev >= p.prev2 {
                let l = if n >= 2 { p.prev2 } else { -1.0 };
                p.register(p.prev, (n - 1) as u32, l, -1.0);
            }
        });
        // noise floor: median of the per-pixel profile minimum (subsampled)
        let mut mins: Vec<f32> = self.px.iter().step_by(7).map(|p| p.cmin).collect();
        let mid = mins.len() / 2;
        let floor = *mins.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
        let inv_n = 1.0 / n as f32;
        let mut depth = vec![0f32; self.px.len()];
        let mut conf = vec![0f32; self.px.len()];
        let mins: Vec<f32> = self.px.iter().map(|p| if p.cmin.is_finite() { p.cmin } else { 0.0 }).collect();
        depth.par_iter_mut().zip(conf.par_iter_mut()).zip(&self.px).for_each(|((d, c), p)| {
            let (c1, l, r) = (p.c1, p.l1, p.r1);
            let mut delta = 0.0f32;
            if l >= 0.0 && r >= 0.0 && c1 > 0.0 {
                // Gaussian interpolation (Nayar & Nakagawa 1994): parabola in log domain
                let (ll, lc, lr) = ((l.max(1e-12)).ln(), c1.ln(), (r.max(1e-12)).ln());
                let den = ll - 2.0 * lc + lr;
                if den < 0.0 {
                    delta = (0.5 * (ll - lr) / den).clamp(-0.5, 0.5);
                }
            }
            *d = p.i1 as f32 + delta;
            *c = if c1 > 0.0 {
                let mean = p.sum * inv_n;
                let prom = (1.0 - mean / c1).clamp(0.0, 1.0);
                // peak ratio: a rival local maximum of similar height means an
                // ambiguous profile; a unimodal profile is fully trusted here
                let pkr = if p.c2 >= 0.0 { (1.0 - p.c2 / c1).clamp(0.0, 1.0) } else { 1.0 };
                let g = if gate > 0.0 && floor > 0.0 { ((c1 - floor) / (gate * floor)).clamp(0.0, 1.0) } else { 1.0 };
                prom * pkr * g
            } else {
                0.0
            };
        });
        (depth, conf, mins)
    }
}

/// Scale the confidence so its 90th percentile maps to 1: the WLS `lambda`
/// then means the same thing on every stack. Returns the scale used.
pub fn normalize_conf(conf: &mut [f32]) -> f32 {
    let mut sample: Vec<f32> = conf.iter().step_by(7).copied().collect();
    let k = (sample.len() * 9 / 10).min(sample.len() - 1);
    let p90 = *sample.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1;
    if p90 > 1e-6 {
        conf.par_iter_mut().for_each(|c| *c = (*c / p90).min(1.0));
    }
    p90
}

// --------------------------------------------------------------- filtering

/// 3×3 median, reflect-101 borders.
pub fn median3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    for_rows(&mut out, w, |y, row| {
        let rows: [&[f32]; 3] = std::array::from_fn(|k| {
            let sy = reflect(y as isize + k as isize - 1, h);
            &src[sy * w..sy * w + w]
        });
        let mut v = [0f32; 9];
        for (x, o) in row.iter_mut().enumerate() {
            let xs = [reflect(x as isize - 1, w), x, reflect(x as isize + 1, w)];
            let mut k = 0;
            for r in &rows {
                for &xx in &xs {
                    v[k] = r[xx];
                    k += 1;
                }
            }
            v.sort_unstable_by(|a, b| a.total_cmp(b));
            *o = v[4];
        }
    });
    out
}

/// Edge weights between horizontal / vertical neighbours: exp(−|ΔI|/σ).
fn edge_weights(guide: &[f32], w: usize, h: usize, sigma: f32) -> (Vec<f32>, Vec<f32>) {
    let inv = -1.0 / sigma.max(1e-6);
    let mut ax = vec![0f32; w * h];
    let mut ay = vec![0f32; w * h];
    for_rows(&mut ax, w, |y, row| {
        let g = &guide[y * w..y * w + w];
        for x in 0..w - 1 {
            row[x] = ((g[x] - g[x + 1]).abs() * inv).exp();
        }
    });
    for_rows(&mut ay, w, |y, row| {
        if y + 1 < h {
            let (g0, g1) = (&guide[y * w..y * w + w], &guide[(y + 1) * w..(y + 1) * w + w]);
            for x in 0..w {
                row[x] = ((g0[x] - g1[x]).abs() * inv).exp();
            }
        }
    });
    (ax, ay)
}

/// Solve, for every row, the 1-D WLS system
/// `(wd_i + λ(a_{i−1} + a_i)) u_i − λ a_{i−1} u_{i−1} − λ a_i u_{i+1} = wd_i f_i`
/// by the Thomas algorithm. `a` holds the weight between `i` and `i+1`.
fn solve_rows(u: &mut [f32], f: &[f32], wd: &[f32], a: &[f32], w: usize, lam: f32) {
    for_rows(u, w, |y, row| {
        let o = y * w;
        let (f, wd, a) = (&f[o..o + w], &wd[o..o + w], &a[o..o + w]);
        let mut cp = vec![0f32; w];
        let mut dp = vec![0f32; w];
        // forward sweep
        let mut prev_a = 0.0f32;
        for i in 0..w {
            let ai = if i + 1 < w { a[i] } else { 0.0 };
            let diag = wd[i] + lam * (prev_a + ai);
            let lower = -lam * prev_a;
            let upper = -lam * ai;
            let m = diag - lower * if i > 0 { cp[i - 1] } else { 0.0 };
            let m = if m.abs() < 1e-12 { 1e-12 } else { m };
            cp[i] = upper / m;
            dp[i] = (wd[i] * f[i] - lower * if i > 0 { dp[i - 1] } else { 0.0 }) / m;
            prev_a = ai;
        }
        // back substitution
        row[w - 1] = dp[w - 1];
        for i in (0..w - 1).rev() {
            row[i] = dp[i] - cp[i] * row[i + 1];
        }
    });
}

/// Same as [`solve_rows`] along columns (`a` = weight between `y` and `y+1`),
/// processed in column blocks for cache locality. `scratch` holds two planes
/// (the sweep's `cp` and `dp`); each block uses its own columns of them.
fn solve_cols(u: &mut [f32], f: &[f32], wd: &[f32], a: &[f32], w: usize, h: usize, lam: f32, scratch: &mut [f32]) {
    let n = w * h;
    assert!(scratch.len() >= 2 * n);
    let sp = ColPtr(scratch.as_mut_ptr());
    par_col_blocks(u, w, |x0, x1, p| {
        let bw = x1 - x0;
        // SAFETY: this block owns columns x0..x1 of every row of `u` and of
        // both scratch planes; the row slices below are disjoint.
        let cp = |y: usize| unsafe { std::slice::from_raw_parts_mut(sp.at(y * w + x0), bw) };
        let dp = |y: usize| unsafe { std::slice::from_raw_parts_mut(sp.at(n + y * w + x0), bw) };
        for y in 0..h {
            let (cprev, dprev): (&[f32], &[f32]) = if y > 0 { (cp(y - 1), dp(y - 1)) } else { (&[], &[]) };
            let (cpy, dpy) = (cp(y), dp(y));
            for (j, x) in (x0..x1).enumerate() {
                let i = y * w + x;
                let prev_a = if y > 0 { a[i - w] } else { 0.0 };
                let ai = if y + 1 < h { a[i] } else { 0.0 };
                let diag = wd[i] + lam * (prev_a + ai);
                let lower = -lam * prev_a;
                let upper = -lam * ai;
                let (c0, d0) = if y > 0 { (cprev[j], dprev[j]) } else { (0.0, 0.0) };
                let m = diag - lower * c0;
                let m = if m.abs() < 1e-12 { 1e-12 } else { m };
                cpy[j] = upper / m;
                dpy[j] = (wd[i] * f[i] - lower * d0) / m;
            }
        }
        let col = |y: usize| unsafe { std::slice::from_raw_parts_mut(p.at(y * w + x0), bw) };
        col(h - 1).copy_from_slice(dp(h - 1));
        for y in (0..h - 1).rev() {
            // rows y and y+1 are disjoint memory
            let next = unsafe { std::slice::from_raw_parts(p.at((y + 1) * w + x0), bw) };
            let row = col(y);
            let (cpy, dpy) = (cp(y), dp(y));
            for j in 0..bw {
                row[j] = dpy[j] - cpy[j] * next[j];
            }
        }
    });
}

// ------------------------------------------------------------- WLS solver

/// The WLS operator `A = W + Σ a_pq` (the smoothness weight folded into the
/// edge weights) on one grid of the multigrid hierarchy, with the planes a
/// V-cycle needs there.
struct MgLevel {
    w: usize,
    h: usize,
    /// Data weight per cell; on the coarse grids the sum over the block.
    wd: Vec<f32>,
    /// λ·a between (x, y) and (x+1, y), and between (x, y) and (x, y+1); on
    /// the coarse grids the sum of the fine edges the block boundary cuts.
    ax: Vec<f32>,
    ay: Vec<f32>,
    /// 1 / diag(A).
    dinv: Vec<f32>,
    x: Vec<f32>,
    b: Vec<f32>,
    r: Vec<f32>,
}

/// Damped-Jacobi weight (4/5 is the smoothing optimum of the 5-point stencil).
pub const MG_OMEGA: f32 = 0.8;
/// Smoothing sweeps before and after the coarse correction, and on the
/// coarsest grid (where they are the solve).
pub const MG_PRE: usize = 1;
pub const MG_POST: usize = 1;
pub const MG_COARSE: usize = 32;
/// Coarsen until the grid is this small on its longer side.
pub const MG_MIN: usize = 16;

impl MgLevel {
    fn new(w: usize, h: usize) -> MgLevel {
        let n = w * h;
        MgLevel { w, h, wd: vec![0.0; n], ax: vec![0.0; n], ay: vec![0.0; n], dinv: vec![0.0; n], x: vec![0.0; n], b: vec![0.0; n], r: vec![0.0; n] }
    }

    /// `out_i = f(i, (A x)_i)` over the grid.
    fn apply<F: Fn(usize, f32) -> f32 + Sync>(&self, x: &[f32], out: &mut [f32], f: F) {
        let (w, h) = (self.w, self.h);
        let (wd, ax, ay) = (&self.wd, &self.ax, &self.ay);
        for_rows(out, w, |y, row| {
            let o = y * w;
            for i in 0..w {
                let k = o + i;
                let xk = x[k];
                let mut v = wd[k] * xk;
                if i > 0 {
                    v += ax[k - 1] * (xk - x[k - 1]);
                }
                if i + 1 < w {
                    v += ax[k] * (xk - x[k + 1]);
                }
                if y > 0 {
                    v += ay[k - w] * (xk - x[k - w]);
                }
                if y + 1 < h {
                    v += ay[k] * (xk - x[k + w]);
                }
                row[i] = f(k, v);
            }
        });
    }

    fn set_dinv(&mut self) {
        let (w, h) = (self.w, self.h);
        let (wd, ax, ay) = (&self.wd, &self.ax, &self.ay);
        for_rows(&mut self.dinv, w, |y, row| {
            for i in 0..w {
                let k = y * w + i;
                let mut v = wd[k];
                if i > 0 {
                    v += ax[k - 1];
                }
                if i + 1 < w {
                    v += ax[k];
                }
                if y > 0 {
                    v += ay[k - w];
                }
                if y + 1 < h {
                    v += ay[k];
                }
                row[i] = 1.0 / v.max(1e-12);
            }
        });
    }

    /// One damped-Jacobi sweep on `x` for `b`: x += ω D⁻¹ (b − A x).
    fn jacobi(&mut self, zero_start: bool) {
        let dinv = &self.dinv;
        if zero_start {
            self.x.par_iter_mut().zip(&self.b).zip(dinv).for_each(|((x, b), d)| *x = MG_OMEGA * d * b);
            return;
        }
        let (x, b) = (&self.x, &self.b);
        let mut r = std::mem::take(&mut self.r);
        self.apply(x, &mut r, |k, ax| x[k] + MG_OMEGA * dinv[k] * (b[k] - ax));
        self.r = std::mem::replace(&mut self.x, r);
    }

    /// The 2×2 aggregation of this grid: the Galerkin operator of a
    /// piecewise-constant prolongation (data weights summed over the block,
    /// the fine edges cut by a block boundary summed into the coarse edge).
    fn coarsen(&self) -> MgLevel {
        let (w, h) = (self.w, self.h);
        let (cw, ch) = (half(w), half(h));
        let mut c = MgLevel::new(cw, ch);
        let (wd, ax, ay) = (&self.wd, &self.ax, &self.ay);
        for_rows(&mut c.wd, cw, |yy, row| {
            for y in 2 * yy..(2 * yy + 2).min(h) {
                for (xx, o) in row.iter_mut().enumerate() {
                    *o += wd[y * w..y * w + w][2 * xx..(2 * xx + 2).min(w)].iter().sum::<f32>();
                }
            }
        });
        for_rows(&mut c.ax, cw, |yy, row| {
            for y in 2 * yy..(2 * yy + 2).min(h) {
                for (xx, o) in row.iter_mut().enumerate() {
                    if xx + 1 < cw {
                        *o += ax[y * w + 2 * xx + 1];
                    }
                }
            }
        });
        for_rows(&mut c.ay, cw, |yy, row| {
            if yy + 1 < ch {
                let y = 2 * yy + 1;
                for (xx, o) in row.iter_mut().enumerate() {
                    *o = ay[y * w..y * w + w][2 * xx..(2 * xx + 2).min(w)].iter().sum::<f32>();
                }
            }
        });
        c
    }

    /// The data weights of the coarse grid `c` from this grid's (after a
    /// reweighting; the edges do not change).
    fn coarsen_wd_into(&self, c: &mut MgLevel) {
        let (w, h, cw) = (self.w, self.h, c.w);
        let wd = &self.wd;
        for_rows(&mut c.wd, cw, |yy, row| {
            row.fill(0.0);
            for y in 2 * yy..(2 * yy + 2).min(h) {
                for (xx, o) in row.iter_mut().enumerate() {
                    *o += wd[y * w..y * w + w][2 * xx..(2 * xx + 2).min(w)].iter().sum::<f32>();
                }
            }
        });
    }
}

/// `x += P x_c`: the coarse correction injected (piecewise constant).
fn prolong_add(x: &mut [f32], w: usize, xc: &[f32], cw: usize) {
    for_rows(x, w, |y, row| {
        let c = &xc[(y / 2) * cw..(y / 2) * cw + cw];
        for (i, o) in row.iter_mut().enumerate() {
            *o += c[i / 2];
        }
    });
}

/// `b_c = Pᵀ r`: the residual summed over each block.
fn restrict(r: &[f32], w: usize, h: usize, bc: &mut [f32], cw: usize) {
    for_rows(bc, cw, |yy, row| {
        row.fill(0.0);
        for y in 2 * yy..(2 * yy + 2).min(h) {
            let s = &r[y * w..y * w + w];
            for (xx, o) in row.iter_mut().enumerate() {
                *o += s[2 * xx..(2 * xx + 2).min(w)].iter().sum::<f32>();
            }
        }
    });
}

/// One V-cycle from `levels[0]` (whose `b` is set) into its `x`.
fn vcycle(levels: &mut [MgLevel]) {
    let Some((top, rest)) = levels.split_first_mut() else { return };
    if rest.is_empty() {
        for k in 0..MG_COARSE {
            top.jacobi(k == 0);
        }
        return;
    }
    for k in 0..MG_PRE {
        top.jacobi(k == 0);
    }
    {
        let (x, b) = (&top.x, &top.b);
        let mut r = std::mem::take(&mut top.r);
        top.apply(x, &mut r, |k, ax| b[k] - ax);
        top.r = r;
    }
    restrict(&top.r, top.w, top.h, &mut rest[0].b, rest[0].w);
    vcycle(rest);
    prolong_add(&mut top.x, top.w, &rest[0].x, rest[0].w);
    for _ in 0..MG_POST {
        top.jacobi(false);
    }
}

/// Edge-aware WLS: argmin_u Σ w_p (u_p − d_p)² + λ Σ a_pq (u_p − u_q)², the
/// 2-D system `(W + λL) u = W d` solved by conjugate gradients from the
/// separable fast-global-smoother guess of Min et al. 2014, preconditioned
/// by one multigrid V-cycle (2×2 aggregation, damped Jacobi). A plain
/// Jacobi preconditioner left the low modes to the CG: on a 4140×2760 grid
/// it needed 300 iterations for a residual of 1e-5 (a few tens with the
/// V-cycle), and where the confidence is low over a large area (a flat
/// wall) it did not converge at all. The solver keeps its planes between
/// solves, so the robust reweighting's second solve costs no allocation.
pub struct WlsSolver {
    w: usize,
    h: usize,
    lambda: f32,
    /// Edge weights exp(−|ΔI|/σ) (without λ), for the separable sweeps.
    ex: Vec<f32>,
    ey: Vec<f32>,
    levels: Vec<MgLevel>,
    /// Column sweeps' scratch (two planes), CG vectors.
    cd: Vec<f32>,
    r: Vec<f32>,
    z: Vec<f32>,
    p: Vec<f32>,
    ap: Vec<f32>,
    f: Vec<f32>,
}

impl WlsSolver {
    pub fn new(guide: &[f32], w: usize, h: usize, lambda: f32, sigma_c: f32) -> WlsSolver {
        let n = w * h;
        let (ex, ey) = edge_weights(guide, w, h, sigma_c);
        let mut top = MgLevel::new(w, h);
        top.ax.par_iter_mut().zip(&ex).for_each(|(a, e)| *a = lambda * e);
        top.ay.par_iter_mut().zip(&ey).for_each(|(a, e)| *a = lambda * e);
        let mut levels = vec![top];
        while levels.last().unwrap().w.max(levels.last().unwrap().h) > MG_MIN {
            let c = levels.last().unwrap().coarsen();
            levels.push(c);
        }
        WlsSolver { w, h, lambda, ex, ey, levels, cd: vec![0.0; 2 * n], r: vec![0.0; n], z: vec![0.0; n], p: vec![0.0; n], ap: vec![0.0; n], f: vec![0.0; n] }
    }

    /// The data weights: the confidence (plus a floor so the system is
    /// definite where nothing is known), aggregated down the hierarchy.
    pub fn set_weights(&mut self, conf: &[f32]) {
        const EPS_DATA: f32 = 1e-4;
        self.levels[0].wd.par_iter_mut().zip(conf).for_each(|(w, c)| *w = c.max(0.0) + EPS_DATA);
        for l in 0..self.levels.len() {
            if l > 0 {
                let (fine, coarse) = self.levels.split_at_mut(l);
                fine[l - 1].coarsen_wd_into(&mut coarse[0]);
            }
            self.levels[l].set_dinv();
        }
    }

    /// Solve for the data `d` into `u`; returns the final relative residual
    /// and the CG iterations taken.
    pub fn solve(&mut self, d: &[f32], max_iters: usize, u: &mut [f32]) -> (f32, usize) {
        let (w, h, lambda) = (self.w, self.h, self.lambda);
        let wd = &self.levels[0].wd;
        if w == 1 || h == 1 {
            // degenerate: 1-D exact solve
            if h == 1 {
                solve_rows(u, d, wd, &self.ex, w, lambda);
            } else {
                solve_cols(u, d, wd, &self.ey, w, h, lambda, &mut self.cd);
            }
            return (0.0, 0);
        }

        // --- initial guess: separable FGS sweeps with the λ_t schedule.
        // Pass 1 uses the confidence-weighted data term so holes get filled by
        // interpolation; later passes carry the previous output (unit weight).
        const T: usize = 3;
        self.p.fill(1.0);
        let ones = &self.p; // all ones for the sweeps' unit data weight
        let f = &mut self.f;
        f.copy_from_slice(d);
        for t in 1..=T {
            let lam_t = 1.5 * lambda * 4f32.powi((T - t) as i32) / (4f32.powi(T as i32) - 1.0);
            let wt: &[f32] = if t == 1 { wd } else { ones };
            solve_rows(u, f, wt, &self.ex, w, lam_t);
            f.copy_from_slice(u);
            solve_cols(u, f, ones, &self.ey, w, h, lam_t, &mut self.cd);
            f.copy_from_slice(u);
        }

        // --- multigrid-preconditioned CG on (W + λL) u = W d
        let dot = |a: &[f32], b: &[f32]| -> f64 { a.par_iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum() };
        let b = &mut self.f;
        b.par_iter_mut().zip(wd).zip(d).for_each(|((b, w), d)| *b = w * d);
        let bnorm = dot(b, b).sqrt().max(1e-30);
        let top = &self.levels[0];
        top.apply(u, &mut self.r, |k, au| b[k] - au);
        let precond = |levels: &mut Vec<MgLevel>, r: &[f32], z: &mut [f32]| {
            levels[0].b.copy_from_slice(r);
            vcycle(levels);
            z.copy_from_slice(&levels[0].x);
        };
        precond(&mut self.levels, &self.r, &mut self.z);
        self.p.copy_from_slice(&self.z);
        let mut rz = dot(&self.r, &self.z);
        let mut rel = (dot(&self.r, &self.r).sqrt() / bnorm) as f32;
        let mut iters = 0;
        for _ in 0..max_iters {
            if rel < 1e-5 {
                break;
            }
            iters += 1;
            self.levels[0].apply(&self.p, &mut self.ap, |_, v| v);
            let pap = dot(&self.p, &self.ap);
            if pap <= 0.0 {
                break;
            }
            let alpha = (rz / pap) as f32;
            u.par_iter_mut().zip(&self.p).for_each(|(u, p)| *u += alpha * p);
            self.r.par_iter_mut().zip(&self.ap).for_each(|(r, ap)| *r -= alpha * ap);
            rel = (dot(&self.r, &self.r).sqrt() / bnorm) as f32;
            if rel < 1e-5 {
                break;
            }
            precond(&mut self.levels, &self.r, &mut self.z);
            let rz_new = dot(&self.r, &self.z);
            let beta = (rz_new / rz) as f32;
            rz = rz_new;
            self.p.par_iter_mut().zip(&self.z).for_each(|(p, z)| *p = z + beta * *p);
        }
        (rel, iters)
    }
}

/// [`WlsSolver`] in one call: the solution and its final relative residual.
pub fn wls_solve(d: &[f32], conf: &[f32], guide: &[f32], w: usize, h: usize, lambda: f32, sigma_c: f32, max_iters: usize) -> (Vec<f32>, f32) {
    let mut s = WlsSolver::new(guide, w, h, lambda, sigma_c);
    s.set_weights(conf);
    let mut u = vec![0f32; w * h];
    let (rel, _) = s.solve(d, max_iters, &mut u);
    (u, rel)
}

// ---------------------------------------------------------------- pipeline

/// A frame's slice of the focus volume: the focus measure of its luma,
/// block-averaged to the working grid (1/2^scale). It needs nothing but the
/// frame, so the fold takes it as each frame passes (`stack::fuse_range`)
/// and the depth pass never decodes or warps the frames again; a slice is
/// `dw × dh` floats (45 MB per 45 MP frame at the default half grid), the
/// one thing kept per frame.
pub fn focus_slice(img: &Img3, p: &DepthParams) -> Vec<f32> {
    let y = luma(img);
    block_mean(&focus_measure(&y, img.w, img.h, p.focus), img.w, img.h, 1 << p.scale).0
}

/// Depth from focus over `src` (aligned frames), guided by the fused
/// all-in-focus image. Streams the frames once, taking each one's
/// `focus_slice`; `depth_from_slices` when the fold took them already.
pub fn depth_from_focus(src: &mut dyn FrameSource, fused: &Img3, p: &DepthParams, log: &mut dyn FnMut(String)) -> Result<DepthMap, String> {
    let (w, h) = src.dims();
    let n = src.len();
    if fused.w != w || fused.h != h {
        return Err("depth: fused image size differs from the frames".into());
    }
    let mut slices = (0..n).map(|m| src.get(m).map(|f| focus_slice(&f, p)));
    depth_from_slices(&mut slices, n, fused, p, log)
}

/// Depth from focus over the `n` frames' focus slices (`focus_slice`, in
/// frame order), guided by the fused all-in-focus image: each slice is
/// aggregated with the guided filter and fed to the peak search as it comes,
/// then the sub-frame depth is regularised and upsampled.
pub fn depth_from_slices(
    slices: &mut dyn Iterator<Item = Result<Vec<f32>, String>>,
    n: usize,
    fused: &Img3,
    p: &DepthParams,
    log: &mut dyn FnMut(String),
) -> Result<DepthMap, String> {
    let (w, h) = (fused.w, fused.h);
    let k = 1usize << p.scale;
    let t = Instant::now();
    let y_full = luma(fused);
    let (guide, dw, dh) = block_mean(&y_full, w, h, k);
    let mut gf = GuidedFilter::new(guide.clone(), dw, dh, p.agg_radius, p.agg_eps);
    let mut agg = vec![0f32; dw * dh];
    log(format!(
        "depth: {n} frames, focus {:?}, working grid {dw}x{dh} (1/{k}), aggregation r={} eps={}",
        p.focus, p.agg_radius, p.agg_eps
    ));
    let mut tracker = PeakTracker::new(dw * dh);
    for m in 0..n {
        let c = slices.next().ok_or_else(|| format!("depth: slice {m} of {n} missing"))??;
        if c.len() != dw * dh {
            return Err(format!("depth: slice {m} has {} cells, the working grid {dw}x{dh}", c.len()));
        }
        // the guided filter can undershoot; the profile statistics assume ≥ 0
        if p.agg_radius > 0 {
            gf.filter_into(&c, &mut agg);
            agg.par_iter_mut().for_each(|v| *v = v.max(0.0));
            tracker.push(&agg);
        } else {
            tracker.push(&c);
        }
    }
    drop(agg);
    log(format!("depth: {n} slices aggregated  ({:.1}s)", t.elapsed().as_secs_f64()));
    let (mut depth_w, mut conf, floor) = tracker.finish(p.gate);
    if p.median {
        depth_w = median3(&depth_w, dw, dh);
    }
    let p90 = normalize_conf(&mut conf);
    let mean_conf = conf.iter().map(|&c| c as f64).sum::<f64>() / conf.len() as f64;
    log(format!("depth: peaks found, confidence p90 {p90:.3}, mean (normalised) {mean_conf:.3}  ({:.1}s)", t.elapsed().as_secs_f64()));
    let (depth_w, rel, iters) = if p.lambda > 0.0 {
        let mut solver = WlsSolver::new(&guide, dw, dh, p.lambda, p.sigma_c);
        solver.set_weights(&conf);
        let mut u = vec![0f32; dw * dh];
        let (rel, iters) = solver.solve(&depth_w, p.cg_iters, &mut u);
        if p.robust > 0.0 {
            // one IRLS step with a Huber loss on the data residual
            let w2: Vec<f32> = conf.par_iter().zip(&depth_w).zip(&u).map(|((c, d), u)| c * (p.robust / (d - u).abs()).min(1.0)).collect();
            solver.set_weights(&w2);
            let (rel2, iters2) = solver.solve(&depth_w, p.cg_iters, &mut u);
            log(format!("depth: robust reweighting (tau={} frames), first solve {iters} iterations, residual {rel:.1e}", p.robust));
            (u, rel2, iters2)
        } else {
            (u, rel, iters)
        }
    } else {
        (depth_w, 0.0, 0)
    };
    log(format!("depth: WLS lambda={} sigma_c={} solved in {iters} iterations, residual {rel:.1e}  ({:.1}s)", p.lambda, p.sigma_c, t.elapsed().as_secs_f64()));
    let max_d = (n - 1) as f32;
    let mut depth = match p.upsample {
        Upsample::Bilinear => upsample_bilinear(&depth_w, dw, dh, w, h, k),
        Upsample::Guided { radius, eps } => {
            if k == 1 {
                GuidedFilter::new(guide.clone(), dw, dh, radius, eps).filter(&depth_w)
            } else {
                let (a, b) = GuidedFilter::new(guide.clone(), dw, dh, radius, eps).coeffs(&depth_w);
                let a = upsample_bilinear(&a, dw, dh, w, h, k);
                let b = upsample_bilinear(&b, dw, dh, w, h, k);
                let mut d = vec![0f32; w * h];
                d.par_iter_mut().enumerate().for_each(|(i, o)| *o = a[i] * y_full[i] + b[i]);
                d
            }
        }
    };
    depth.par_iter_mut().for_each(|v| *v = v.clamp(0.0, max_d));
    let conf_full = if k == 1 { conf } else { upsample_bilinear(&conf, dw, dh, w, h, k) };
    log(format!("depth: upsampled to {w}x{h} ({:?})  ({:.1}s)", p.upsample, t.elapsed().as_secs_f64()));
    Ok(DepthMap { depth, conf: conf_full, w, h, dw, dh, floor })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) % 10000) as f32 / 10000.0
    }

    #[test]
    fn box_filter_matches_naive() {
        let (w, h, r) = (23, 17, 3);
        let mut s = 5u64;
        let src: Vec<f32> = (0..w * h).map(|_| lcg(&mut s)).collect();
        let mut bf = BoxFilter::new(w, h, r);
        let m = bf.mean(&src);
        for y in 0..h {
            for x in 0..w {
                let (mut acc, mut cnt) = (0.0, 0.0);
                for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                    for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                        acc += src[yy * w + xx];
                        cnt += 1.0;
                    }
                }
                assert!((m[y * w + x] - acc / cnt).abs() < 1e-4, "({x},{y})");
            }
        }
    }

    #[test]
    fn guided_filter_flat_guide_is_box_mean() {
        let (w, h) = (40, 30);
        let mut s = 9u64;
        let p: Vec<f32> = (0..w * h).map(|_| lcg(&mut s)).collect();
        let mut gf = GuidedFilter::new(vec![0.5; w * h], w, h, 2, 1e-3);
        let q = gf.filter(&p);
        let mut bf = BoxFilter::new(w, h, 2);
        let m = bf.mean(&p);
        let m = bf.mean(&m); // a = 0, b = mean_p, q = mean(b)
        for i in 0..w * h {
            assert!((q[i] - m[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn rdf_taps_are_zero_mean() {
        let taps = rdf_taps(1, 3);
        let s: f32 = taps.iter().map(|t| t.2).sum();
        assert!(s.abs() < 1e-6);
        assert_eq!(taps.iter().filter(|t| t.2 > 0.0).count(), 5);
        let y = vec![0.3f32; 20 * 20];
        assert!(focus_measure(&y, 20, 20, FocusMeasure::Rdf { r_in: 1, r_out: 3 }).iter().all(|v| v.abs() < 1e-6));
    }

    fn track(profile: &[f32]) -> (f32, f32) {
        let mut t = PeakTracker::new(1);
        for &v in profile {
            t.push(&[v]);
        }
        let (d, c, _) = t.finish(2.0);
        (d[0], c[0])
    }

    #[test]
    fn peak_tracker_profiles() {
        // symmetric peak at index 2, no sub-frame offset
        let (d, c) = track(&[0.1, 0.5, 1.0, 0.5, 0.1]);
        assert!((d - 2.0).abs() < 1e-5, "{d}");
        assert!(c > 0.5, "{c}");
        // asymmetric: right neighbour higher -> peak shifts right
        let (d, _) = track(&[0.1, 0.4, 1.0, 0.8, 0.1]);
        assert!(d > 2.0 && d < 2.5, "{d}");
        // peak at the first / last frame
        assert!((track(&[1.0, 0.5, 0.1]).0).abs() < 1e-6);
        assert!((track(&[0.1, 0.5, 1.0]).0 - 2.0).abs() < 1e-6);
        // plateau: the interpolation lands between the two equal samples
        assert!((track(&[0.1, 1.0, 1.0, 0.1]).0 - 1.5).abs() < 1e-5);
        // two peaks of similar height -> low confidence; single sharp peak -> high
        let (_, c2) = track(&[0.1, 1.0, 0.1, 0.95, 0.1]);
        let (_, c1) = track(&[0.1, 1.0, 0.1, 0.1, 0.1]);
        assert!(c2 < 0.2 && c1 > 0.5, "{c2} {c1}");
        // flat profile -> zero confidence
        assert!(track(&[0.3, 0.3, 0.3, 0.3]).1 < 1e-6);
        // single frame
        assert_eq!(track(&[0.7]).0, 0.0);
    }

    #[test]
    fn wls_keeps_confident_data_and_fills_holes() {
        let (w, h) = (32, 24);
        let mut d = vec![0f32; w * h];
        let mut conf = vec![1f32; w * h];
        // ramp in x, with a hole (zero confidence, garbage data) in the middle
        for y in 0..h {
            for x in 0..w {
                d[y * w + x] = x as f32;
                if (10..22).contains(&x) && (6..18).contains(&y) {
                    conf[y * w + x] = 0.0;
                    d[y * w + x] = 100.0;
                }
            }
        }
        let guide = vec![0.5; w * h];
        let (u, rel) = wls_solve(&d, &conf, &guide, w, h, 5.0, 0.05, 500);
        assert!(rel < 1e-4, "residual {rel}");
        // (the natural boundary condition bends the ramp within ~sqrt(lambda)
        // px of the image border, so only the interior is checked)
        for y in 4..h - 4 {
            for x in 4..w - 4 {
                let e = (u[y * w + x] - x as f32).abs();
                assert!(e < 0.6, "({x},{y}): {} vs {}", u[y * w + x], x);
            }
        }
        // lambda = 0 with full confidence returns the data
        let (u0, _) = wls_solve(&d, &vec![1.0; w * h], &guide, w, h, 0.0, 0.05, 10);
        for i in 0..w * h {
            assert!((u0[i] - d[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn wls_respects_guide_edges() {
        // two flat depth plates with a luma edge between them: no bleed across
        let (w, h) = (40, 20);
        let mut d = vec![0f32; w * h];
        let mut guide = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let right = x >= w / 2;
                d[y * w + x] = if right { 10.0 } else { 0.0 };
                guide[y * w + x] = if right { 0.9 } else { 0.1 };
            }
        }
        let (u, _) = wls_solve(&d, &vec![0.2; w * h], &guide, w, h, 50.0, 0.02, 300);
        assert!(u[10 * w + w / 2 - 1] < 0.5 && u[10 * w + w / 2] > 9.5, "{} {}", u[10 * w + w / 2 - 1], u[10 * w + w / 2]);
        // same without the guide edge: heavily blended at the seam
        let (v, _) = wls_solve(&d, &vec![0.2; w * h], &vec![0.5; w * h], w, h, 50.0, 0.02, 300);
        assert!(v[10 * w + w / 2 - 1] > 2.0 && v[10 * w + w / 2] < 8.0);
    }

    fn checker(w: usize, h: usize, cell: usize) -> Img3 {
        let mut im = Img3::zeros(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if ((x / cell) + (y / cell)) % 2 == 0 { 0.2 } else { 0.8 };
                for c in 0..3 {
                    im.p[c][y * w + x] = v;
                }
            }
        }
        im
    }

    fn blur(im: &Img3, passes: usize) -> Img3 {
        let mut o = im.clone();
        for _ in 0..passes {
            for c in 0..3 {
                o.p[c] = crate::fuse::window_sum(&o.p[c], o.w, o.h, &crate::pyramid::KERNEL);
            }
        }
        o
    }

    #[test]
    fn end_to_end_two_frames() {
        let (w, h) = (128, 96);
        let sharp = checker(w, h, 5);
        let soft = blur(&sharp, 6);
        let (mut a, mut b) = (sharp.clone(), sharp.clone());
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                for c in 0..3 {
                    if x >= w / 2 {
                        a.p[c][i] = soft.p[c][i];
                    } else {
                        b.p[c][i] = soft.p[c][i];
                    }
                }
            }
        }
        let frames = vec![a, b];
        let mut src: &[Img3] = &frames;
        let p = DepthParams { lambda: 5.0, ..Default::default() };
        let dm = depth_from_focus(&mut src, &sharp, &p, &mut |_| {}).unwrap();
        assert_eq!((dm.w, dm.h), (w, h));
        let mut bad = 0;
        for y in 0..h {
            for x in 0..w {
                if (x as isize - w as isize / 2).abs() < 6 {
                    continue;
                }
                let want = if x < w / 2 { 0.0 } else { 1.0 };
                if (dm.depth[y * w + x] - want).abs() > 0.25 {
                    bad += 1;
                }
            }
        }
        assert!(bad < w * h / 100, "{bad} bad pixels");
        // two-frame profiles cap the prominence term at 0.5
        let mean_conf = dm.conf.iter().sum::<f32>() / (w * h) as f32;
        assert!(mean_conf > 0.2, "mean confidence {mean_conf}");
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// `cargo test --release -p lapstack-core bench_depth -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_depth_stages() {
        let (w, h) = (4140usize, 2760usize);
        let n = w * h;
        let mut s = 1u64;
        let mut lcg = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 10000) as f32 / 10000.0
        };
        let guide: Vec<f32> = (0..n).map(|i| ((i % w) as f32 / w as f32) * 0.5 + 0.25 * lcg()).collect();
        let slice: Vec<f32> = (0..n).map(|_| lcg()).collect();
        let t = Instant::now();
        let mut gf = GuidedFilter::new(guide.clone(), w, h, 3, 1e-4);
        eprintln!("gf new: {:.3}s", t.elapsed().as_secs_f64());
        for _ in 0..3 {
            let t = Instant::now();
            let q = gf.filter(&slice);
            let t1 = t.elapsed().as_secs_f64();
            let c: Vec<f32> = q.into_par_iter().map(|v| v.max(0.0)).collect();
            eprintln!("gf filter: {t1:.3}s  clamp {:.3}s", t.elapsed().as_secs_f64() - t1);
            let _ = c;
        }
        let mut bf = BoxFilter::new(w, h, 3);
        for _ in 0..3 {
            let t = Instant::now();
            let _ = bf.mean(&slice);
            eprintln!("box mean: {:.3}s", t.elapsed().as_secs_f64());
        }
        let mut tr = PeakTracker::new(n);
        for m in 0..4 {
            let t = Instant::now();
            tr.push(&slice);
            eprintln!("push {m}: {:.3}s", t.elapsed().as_secs_f64());
        }
        let (d, mut conf, _) = tr.finish(1.0);
        normalize_conf(&mut conf);
        let d: Vec<f32> = d.iter().enumerate().map(|(i, v)| v + 3.0 * guide[i] + lcg()).collect();
        let t = Instant::now();
        let m = median3(&d, w, h);
        eprintln!("median3: {:.3}s", t.elapsed().as_secs_f64());
        let t = Instant::now();
        let mut solver = WlsSolver::new(&guide, w, h, 3.0, 0.04);
        eprintln!("wls new: {:.3}s ({} levels)", t.elapsed().as_secs_f64(), solver.levels.len());
        let t = Instant::now();
        solver.set_weights(&conf);
        eprintln!("wls weights: {:.3}s", t.elapsed().as_secs_f64());
        let mut u = vec![0f32; n];
        let t = Instant::now();
        let (rel, it) = solver.solve(&m, 200, &mut u);
        eprintln!("wls: {:.3}s rel {rel:.2e} in {it} iterations", t.elapsed().as_secs_f64());
        let w2: Vec<f32> = conf.par_iter().zip(&m).zip(&u).map(|((c, d), u)| c * (1.0 / (d - u).abs()).min(1.0)).collect();
        let t = Instant::now();
        solver.set_weights(&w2);
        let (rel, it) = solver.solve(&m, 200, &mut u);
        eprintln!("wls 2: {:.3}s rel {rel:.2e} in {it} iterations", t.elapsed().as_secs_f64());
        // a confident, smooth field: the regime of real data
        let conf2: Vec<f32> = (0..n).map(|i| if (i / w / 200 + i % w / 200) % 3 == 0 { 0.0 } else { 0.5 }).collect();
        solver.set_weights(&conf2);
        let t = Instant::now();
        let (rel, it) = solver.solve(&m, 200, &mut u);
        eprintln!("wls (patchy confidence): {:.3}s rel {rel:.2e} in {it} iterations", t.elapsed().as_secs_f64());
    }
}

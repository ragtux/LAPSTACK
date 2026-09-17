// Copyright (c) 2026 MATCHMUSEUM.COM
// INTERNAL USE ONLY

//! Depth from focus (DFF): a dense, sub-frame depth map from the aligned stack.
//!
//! The pipeline is the modern non-learned "focus volume" recipe, written from
//! the papers:
//!
//! 1. **Focus measure** per frame on luma at full resolution. Default is the
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
//!    initial guess and a Jacobi-preconditioned conjugate gradient solves the
//!    2-D system exactly. Regularising in the depth domain with confidence
//!    weights is the 2-D counterpart of the focus-volume WLS of Ali & Mahmood
//!    (Information Sciences, 2020).
//! 5. **Upsampling** back to full resolution with the guided filter (the
//!    "fast guided filter" coefficients of He & Sun 2015 evaluated on the
//!    full-resolution luma), so depth edges land on image edges.
//!
//! Output: a per-pixel frame index in `[0, n−1]` (fractional) plus a
//! confidence map in `[0, 1]`.

use crate::pyramid::{Img3, for_rows, reflect};
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
pub struct BoxFilter {
    w: usize,
    h: usize,
    r: usize,
    inv_count: Vec<f32>,
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
        BoxFilter { w, h, r, inv_count }
    }

    /// Window sums (not normalised).
    pub fn sum(&self, src: &[f32]) -> Vec<f32> {
        let (w, h, r) = (self.w, self.h, self.r);
        // horizontal moving sum
        let mut tmp = vec![0f32; w * h];
        for_rows(&mut tmp, w, |y, row| {
            let s = &src[y * w..y * w + w];
            let mut acc: f32 = s[..(r + 1).min(w)].iter().sum();
            for x in 0..w {
                row[x] = acc;
                if x + r + 1 < w {
                    acc += s[x + r + 1];
                }
                if x >= r {
                    acc -= s[x - r];
                }
            }
        });
        // vertical moving sum, per column block
        let mut out = vec![0f32; w * h];
        par_col_blocks(&mut out, w, |x0, x1, p| {
            let bw = x1 - x0;
            let mut acc = vec![0f32; bw];
            for y in 0..(r + 1).min(h) {
                for (a, v) in acc.iter_mut().zip(&tmp[y * w + x0..y * w + x1]) {
                    *a += v;
                }
            }
            for y in 0..h {
                // SAFETY: this block owns columns x0..x1 of every row; no other
                // worker touches them, and `out` outlives the parallel loop.
                let dst = unsafe { std::slice::from_raw_parts_mut(p.0.add(y * w + x0), bw) };
                dst.copy_from_slice(&acc);
                if y + r + 1 < h {
                    for (a, v) in acc.iter_mut().zip(&tmp[(y + r + 1) * w + x0..(y + r + 1) * w + x1]) {
                        *a += v;
                    }
                }
                if y >= r {
                    for (a, v) in acc.iter_mut().zip(&tmp[(y - r) * w + x0..(y - r) * w + x1]) {
                        *a -= v;
                    }
                }
            }
        });
        out
    }

    pub fn mean(&self, src: &[f32]) -> Vec<f32> {
        let mut s = self.sum(src);
        s.par_iter_mut().zip(&self.inv_count).for_each(|(v, c)| *v *= c);
        s
    }
}

/// Guided filter (He, Sun & Tang 2013) with a fixed grayscale guide.
pub struct GuidedFilter {
    guide: Vec<f32>,
    mean_i: Vec<f32>,
    var_i: Vec<f32>,
    bf: BoxFilter,
    eps: f32,
}

impl GuidedFilter {
    pub fn new(guide: Vec<f32>, w: usize, h: usize, r: usize, eps: f32) -> GuidedFilter {
        let bf = BoxFilter::new(w, h, r);
        let mean_i = bf.mean(&guide);
        let ii: Vec<f32> = guide.par_iter().map(|v| v * v).collect();
        let mut var_i = bf.mean(&ii);
        var_i.par_iter_mut().zip(&mean_i).for_each(|(v, m)| *v = (*v - m * m).max(0.0));
        GuidedFilter { guide, mean_i, var_i, bf, eps }
    }

    /// Linear coefficients `(a, b)` such that `q ≈ a·I + b` (already box-averaged).
    pub fn coeffs(&self, p: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mean_p = self.bf.mean(p);
        let ip: Vec<f32> = self.guide.par_iter().zip(p).map(|(i, p)| i * p).collect();
        let corr_ip = self.bf.mean(&ip);
        let mut a = vec![0f32; p.len()];
        let mut b = vec![0f32; p.len()];
        a.par_iter_mut().zip(b.par_iter_mut()).enumerate().for_each(|(k, (a, b))| {
            let cov = corr_ip[k] - self.mean_i[k] * mean_p[k];
            *a = cov / (self.var_i[k] + self.eps);
            *b = mean_p[k] - *a * self.mean_i[k];
        });
        (self.bf.mean(&a), self.bf.mean(&b))
    }

    pub fn filter(&self, p: &[f32]) -> Vec<f32> {
        let (a, b) = self.coeffs(p);
        let mut q = vec![0f32; p.len()];
        q.par_iter_mut().enumerate().for_each(|(k, o)| *o = a[k] * self.guide[k] + b[k]);
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

    /// Close the profiles: sub-frame depth and raw confidence per pixel.
    pub fn finish(mut self, gate: f32) -> (Vec<f32>, Vec<f32>) {
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
        (depth, conf)
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
/// processed in column blocks for cache locality.
fn solve_cols(u: &mut [f32], f: &[f32], wd: &[f32], a: &[f32], w: usize, h: usize, lam: f32) {
    par_col_blocks(u, w, |x0, x1, p| {
        let bw = x1 - x0;
        let mut cp = vec![0f32; h * bw];
        let mut dp = vec![0f32; h * bw];
        for y in 0..h {
            for (j, x) in (x0..x1).enumerate() {
                let i = y * w + x;
                let prev_a = if y > 0 { a[i - w] } else { 0.0 };
                let ai = if y + 1 < h { a[i] } else { 0.0 };
                let diag = wd[i] + lam * (prev_a + ai);
                let lower = -lam * prev_a;
                let upper = -lam * ai;
                let (cprev, dprev) = if y > 0 { (cp[(y - 1) * bw + j], dp[(y - 1) * bw + j]) } else { (0.0, 0.0) };
                let m = diag - lower * cprev;
                let m = if m.abs() < 1e-12 { 1e-12 } else { m };
                cp[y * bw + j] = upper / m;
                dp[y * bw + j] = (wd[i] * f[i] - lower * dprev) / m;
            }
        }
        // SAFETY: this block owns columns x0..x1 of every row of `u`.
        let col = |y: usize| unsafe { std::slice::from_raw_parts_mut(p.0.add(y * w + x0), bw) };
        col(h - 1).copy_from_slice(&dp[(h - 1) * bw..h * bw]);
        for y in (0..h - 1).rev() {
            // rows y and y+1 are disjoint memory
            let next = unsafe { std::slice::from_raw_parts(p.0.add((y + 1) * w + x0), bw) };
            let row = col(y);
            for j in 0..bw {
                row[j] = dp[y * bw + j] - cp[y * bw + j] * next[j];
            }
        }
    });
}

/// Edge-aware WLS: argmin_u Σ w_p (u_p − d_p)² + λ Σ a_pq (u_p − u_q)².
/// Fast-global-smoother initial guess (Min et al. 2014) + preconditioned CG.
/// Returns the solution and the final relative residual.
pub fn wls_solve(d: &[f32], conf: &[f32], guide: &[f32], w: usize, h: usize, lambda: f32, sigma_c: f32, max_iters: usize) -> (Vec<f32>, f32) {
    let n = w * h;
    let (ax, ay) = edge_weights(guide, w, h, sigma_c);
    const EPS_DATA: f32 = 1e-4;
    let wd: Vec<f32> = conf.iter().map(|c| c.max(0.0) + EPS_DATA).collect();
    if w == 1 || h == 1 {
        // degenerate: 1-D exact solve
        let mut u = vec![0f32; n];
        if h == 1 {
            solve_rows(&mut u, d, &wd, &ax, w, lambda);
        } else {
            solve_cols(&mut u, d, &wd, &ay, w, h, lambda);
        }
        return (u, 0.0);
    }

    // --- initial guess: separable FGS sweeps with the λ_t schedule.
    // Pass 1 uses the confidence-weighted data term so holes get filled by
    // interpolation; later passes carry the previous output (unit weight).
    const T: usize = 3;
    let ones = vec![1f32; n];
    let mut u = vec![0f32; n];
    let mut f = d.to_vec();
    for t in 1..=T {
        let lam_t = 1.5 * lambda * 4f32.powi((T - t) as i32) / (4f32.powi(T as i32) - 1.0);
        let wt: &[f32] = if t == 1 { &wd } else { &ones };
        solve_rows(&mut u, &f, wt, &ax, w, lam_t);
        f.copy_from_slice(&u);
        solve_cols(&mut u, &f, &ones, &ay, w, h, lam_t);
        f.copy_from_slice(&u);
    }

    // --- Jacobi-PCG on (W + λ L) u = W d
    let matvec = |x: &[f32], out: &mut [f32]| {
        for_rows(out, w, |y, row| {
            let o = y * w;
            for i in 0..w {
                let k = o + i;
                let mut v = wd[k] * x[k];
                if i > 0 {
                    v += lambda * ax[k - 1] * (x[k] - x[k - 1]);
                }
                if i + 1 < w {
                    v += lambda * ax[k] * (x[k] - x[k + 1]);
                }
                if y > 0 {
                    v += lambda * ay[k - w] * (x[k] - x[k - w]);
                }
                if y + 1 < h {
                    v += lambda * ay[k] * (x[k] - x[k + w]);
                }
                row[i] = v;
            }
        });
    };
    let dot = |a: &[f32], b: &[f32]| -> f64 { a.par_iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum() };
    let mut diag = vec![0f32; n];
    for_rows(&mut diag, w, |y, row| {
        for i in 0..w {
            let k = y * w + i;
            let mut v = wd[k];
            if i > 0 {
                v += lambda * ax[k - 1];
            }
            if i + 1 < w {
                v += lambda * ax[k];
            }
            if y > 0 {
                v += lambda * ay[k - w];
            }
            if y + 1 < h {
                v += lambda * ay[k];
            }
            row[i] = 1.0 / v.max(1e-12);
        }
    });
    let b: Vec<f32> = wd.par_iter().zip(d).map(|(w, d)| w * d).collect();
    let bnorm = dot(&b, &b).sqrt().max(1e-30);
    let mut r = vec![0f32; n];
    matvec(&u, &mut r);
    r.par_iter_mut().zip(&b).for_each(|(r, b)| *r = b - *r);
    let mut z: Vec<f32> = r.par_iter().zip(&diag).map(|(r, d)| r * d).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![0f32; n];
    let mut rel = (dot(&r, &r).sqrt() / bnorm) as f32;
    for _ in 0..max_iters {
        if rel < 1e-5 {
            break;
        }
        matvec(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            break;
        }
        let alpha = (rz / pap) as f32;
        u.par_iter_mut().zip(&p).for_each(|(u, p)| *u += alpha * p);
        r.par_iter_mut().zip(&ap).for_each(|(r, ap)| *r -= alpha * ap);
        z.par_iter_mut().zip(&r).zip(&diag).for_each(|((z, r), d)| *z = r * d);
        let rz_new = dot(&r, &z);
        let beta = (rz_new / rz) as f32;
        rz = rz_new;
        p.par_iter_mut().zip(&z).for_each(|(p, z)| *p = z + beta * *p);
        rel = (dot(&r, &r).sqrt() / bnorm) as f32;
    }
    (u, rel)
}

// ---------------------------------------------------------------- pipeline

/// Depth from focus over `src` (aligned frames), guided by the fused
/// all-in-focus image. Streams the frames once.
pub fn depth_from_focus(src: &mut dyn FrameSource, fused: &Img3, p: &DepthParams, log: &mut dyn FnMut(String)) -> Result<DepthMap, String> {
    let (w, h) = src.dims();
    let n = src.len();
    if fused.w != w || fused.h != h {
        return Err("depth: fused image size differs from the frames".into());
    }
    let k = 1usize << p.scale;
    let t = Instant::now();
    let y_full = luma(fused);
    let (guide, dw, dh) = block_mean(&y_full, w, h, k);
    let gf = GuidedFilter::new(guide.clone(), dw, dh, p.agg_radius, p.agg_eps);
    log(format!(
        "depth: {n} frames, focus {:?}, working grid {dw}x{dh} (1/{k}), aggregation r={} eps={}",
        p.focus, p.agg_radius, p.agg_eps
    ));
    let mut tracker = PeakTracker::new(dw * dh);
    for m in 0..n {
        let (c, _, _) = {
            let f = src.get(m)?;
            let y = luma(&f);
            block_mean(&focus_measure(&y, w, h, p.focus), w, h, k)
        };
        let c = if p.agg_radius > 0 { gf.filter(&c) } else { c };
        // the guided filter can undershoot; the profile statistics assume ≥ 0
        let c: Vec<f32> = c.into_par_iter().map(|v| v.max(0.0)).collect();
        tracker.push(&c);
        log(format!("  frame {:>3}/{n} measured  ({:.1}s)", m + 1, t.elapsed().as_secs_f64()));
    }
    let (mut depth_w, mut conf) = tracker.finish(p.gate);
    if p.median {
        depth_w = median3(&depth_w, dw, dh);
    }
    let p90 = normalize_conf(&mut conf);
    let mean_conf = conf.iter().map(|&c| c as f64).sum::<f64>() / conf.len() as f64;
    log(format!("depth: peaks found, confidence p90 {p90:.3}, mean (normalised) {mean_conf:.3}  ({:.1}s)", t.elapsed().as_secs_f64()));
    let (depth_w, rel) = if p.lambda > 0.0 {
        let (u, rel) = wls_solve(&depth_w, &conf, &guide, dw, dh, p.lambda, p.sigma_c, p.cg_iters);
        if p.robust > 0.0 {
            // one IRLS step with a Huber loss on the data residual
            let w2: Vec<f32> = conf.par_iter().zip(&depth_w).zip(&u).map(|((c, d), u)| c * (p.robust / (d - u).abs()).min(1.0)).collect();
            let (u2, rel2) = wls_solve(&depth_w, &w2, &guide, dw, dh, p.lambda, p.sigma_c, p.cg_iters);
            log(format!("depth: robust reweighting (tau={} frames), first residual {rel:.1e}", p.robust));
            (u2, rel2)
        } else {
            (u, rel)
        }
    } else {
        (depth_w, 0.0)
    };
    log(format!("depth: WLS lambda={} sigma_c={} solved, residual {rel:.1e}  ({:.1}s)", p.lambda, p.sigma_c, t.elapsed().as_secs_f64()));
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
    Ok(DepthMap { depth, conf: conf_full, w, h, dw, dh })
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
        let bf = BoxFilter::new(w, h, r);
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
        let gf = GuidedFilter::new(vec![0.5; w * h], w, h, 2, 1e-3);
        let q = gf.filter(&p);
        let bf = BoxFilter::new(w, h, 2);
        let m = bf.mean(&bf.mean(&p)); // a = 0, b = mean_p, q = mean(b)
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
        let (d, c) = t.finish(2.0);
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

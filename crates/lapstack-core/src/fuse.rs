// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

//! Fusion rules of Wang & Chang 2011 (§III), generalized to N frames.
//!
//! * Band-pass levels `0 ≤ l < N` — **maximum region energy** (eq. 13–14):
//!   `RE_l(i,j) = Σ ω(m,n) L_l(i+m, j+n)²` over a small window, and the fused
//!   coefficient is the one from the frame with the largest `RE`. With a
//!   window radius of 0 this degenerates to Adelson et al. 1984 eq. (7):
//!   pick the node with the larger |L|.
//! * Residual level `N` — **region information** (eq. 10–12): local deviation
//!   `D` and local entropy `E`. For two frames the paper takes A when A wins
//!   both, B when B wins both, and averages otherwise. For N frames we keep
//!   exactly that shape as a Pareto rule: a frame is *dominated* if another
//!   frame is at least as good on both D and E and strictly better on one;
//!   the fused value is the mean of the non-dominated frames. (With two
//!   frames this reproduces eq. 12, except that an exact tie on one measure is
//!   resolved by the other instead of being averaged.)
//!
//! Decisions are made on luminance (Y = .299R + .587G + .114B of the
//! coefficients — the pyramid is linear, so that *is* the luma pyramid) and
//! applied to all three channels, so color never splits at a selection edge.
//!
//! * **Halo control** (`halo` > 0) — the levels coarser than `depth_level`
//!   do not pick their own winners. Each level's winner-take-all is blind to
//!   the others, and a bright object's defocused copy, spread over the
//!   background in the frames that focus behind it, carries strong coarse
//!   energy where the sharp frame has none, so the coarse levels collect the
//!   glow from one frame after another while the fine levels take the sharp
//!   background — the halo of the pyramid method. With halo control the
//!   level `depth_level` (the guide) still selects by region energy, and every
//!   coarser level, the residual included, is the mean of the frames weighed
//!   by `w = ((RE_guide + ε) / ρ)^halo`, the guide's region energy raised to the
//!   hardness `halo` and REDUCEd to the level's size (the Gaussian pyramid of a
//!   weight mask, as a multiresolution spline blends with it). The coarse
//!   structure then follows the frames the guide found sharp, and where none
//!   is (a flat area) the frames average. `halo` = 1 weighs by the energy
//!   itself, higher values approach a hard pick; 0 is off.
//!
//! The accumulator folds frames in one at a time: only the running fused
//! pyramid, one best-energy (or weight-sum) plane per level and the tiny
//! per-frame residuals are held, so memory does not grow with the stack size.

use crate::pyramid::{self, Img3, for_rows, reflect};
use rayon::prelude::*;

/// Rule for the residual (top) level.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TopRule {
    /// Wang & Chang eq. (12): deviation + entropy, Pareto-averaged (default).
    DevEntropy,
    /// Largest local deviation alone.
    Deviation,
    /// Plain mean of all frames (Burt's choice: low-pass content is shared).
    Average,
}

impl TopRule {
    pub fn parse(s: &str) -> Option<TopRule> {
        match s {
            "de" | "dev-entropy" => Some(TopRule::DevEntropy),
            "dev" | "deviation" => Some(TopRule::Deviation),
            "avg" | "average" => Some(TopRule::Average),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FuseParams {
    /// Band-pass levels; `None` = as many as keep the residual ≥ 32 px.
    pub levels: Option<usize>,
    /// Region-energy window radius for the band-pass levels (1 = 3×3, binomial
    /// weights). 0 = per-node |L| max.
    pub energy_radius: usize,
    pub top_rule: TopRule,
    /// Window radius for the residual's D / E (2 = 5×5).
    pub top_radius: usize,
    /// Gray levels used for the entropy histogram (the paper's `L`; 8-bit → 256).
    pub entropy_bins: usize,
    /// Energy from R²+G²+B² instead of Y².
    pub use_chroma: bool,
    /// Pyramid level whose winner map is reported as the depth map (0 = the
    /// finest, which is noise wherever the scene is flat; 2 = quarter
    /// resolution, a usable index map). Clamped to the last band-pass level.
    /// With halo control it is also the guide level.
    pub depth_level: usize,
    /// Halo control: hardness of the weights the levels coarser than
    /// `depth_level` blend with (the guide's region energy to this power);
    /// 0 = off, every level picks its own winner.
    pub halo: f32,
}

impl Default for FuseParams {
    fn default() -> Self {
        FuseParams {
            levels: None,
            energy_radius: 1,
            top_rule: TopRule::DevEntropy,
            top_radius: 2,
            entropy_bins: 256,
            use_chroma: false,
            depth_level: 2,
            halo: 0.0,
        }
    }
}

/// Reference energy the halo-control weights are taken relative to (a
/// well-textured level-2 band), so that `(RE / ρ)^p` stays within f32 for
/// hardness up to `HALO_MAX`.
pub const HALO_REF: f32 = 1e-4;
/// Floor added to the energy before weighing: below it the frames average.
pub const HALO_FLOOR: f32 = 1e-8;
/// Largest useful hardness (the weights' exponent is clamped to ±60, a span
/// of 1e52, which hardness 8 uses over the practical range of energies).
pub const HALO_MAX: f32 = 8.0;

/// Halo control: the guide level `depth_level` and the hardness, when on.
pub fn halo_guide(params: &FuseParams, levels: usize) -> Option<(usize, f32)> {
    (params.halo > 0.0).then(|| (params.depth_level.min(levels - 1), params.halo.min(HALO_MAX)))
}

/// The weight a coarse coefficient gets from the guide's region energy `re`
/// at hardness `p`: `((re + floor) / ref)^p`, its exponent clamped to ±60.
#[inline]
pub fn halo_weight(re: f32, p: f32) -> f32 {
    (p * ((re + HALO_FLOOR) / HALO_REF).ln()).clamp(-60.0, 60.0).exp()
}

/// Plane `w` of halo weights from the region energy `re`.
pub fn halo_weights(re: &[f32], p: f32) -> Vec<f32> {
    let mut w = vec![0f32; re.len()];
    w.par_iter_mut().zip(re.par_iter()).for_each(|(o, &e)| *o = halo_weight(e, p));
    w
}

/// Binomial window weights of radius `r` (row 2r of Pascal's triangle / 4^r).
pub fn binomial(r: usize) -> Vec<f32> {
    let n = 2 * r + 1;
    let mut row = vec![1u64; n];
    for i in 1..n {
        for j in (1..i).rev() {
            row[j] += row[j - 1];
        }
    }
    let s = row.iter().sum::<u64>() as f32;
    row.into_iter().map(|v| v as f32 / s).collect()
}

/// Per-node energy of a band: Y² (or R²+G²+B²).
fn node_energy(l: &Img3, use_chroma: bool) -> Vec<f32> {
    let n = l.w * l.h;
    let mut e = vec![0f32; n];
    let (r, g, b) = (&l.p[0], &l.p[1], &l.p[2]);
    if use_chroma {
        e.par_iter_mut().enumerate().for_each(|(i, o)| *o = r[i] * r[i] + g[i] * g[i] + b[i] * b[i]);
    } else {
        e.par_iter_mut().enumerate().for_each(|(i, o)| {
            let y = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
            *o = y * y;
        });
    }
    e
}

/// Weighted window sum with separable weights `wt` (odd length), reflect-101 borders.
pub fn window_sum(src: &[f32], w: usize, h: usize, wt: &[f32]) -> Vec<f32> {
    let r = wt.len() / 2;
    if r == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0f32; w * h];
    for_rows(&mut tmp, w, |y, row| {
        let s = &src[y * w..y * w + w];
        for (j, o) in row.iter_mut().enumerate() {
            let mut a = 0.0;
            for (t, k) in wt.iter().enumerate() {
                a += k * s[reflect(j as isize + t as isize - r as isize, w)];
            }
            *o = a;
        }
    });
    let mut out = vec![0f32; w * h];
    let tmp = &tmp;
    for_rows(&mut out, w, |y, row| {
        row.fill(0.0);
        for (t, k) in wt.iter().enumerate() {
            let sy = reflect(y as isize + t as isize - r as isize, h);
            let s = &tmp[sy * w..sy * w + w];
            for j in 0..w {
                row[j] += k * s[j];
            }
        }
    });
    out
}

/// Region energy of a band-pass level, eq. (13).
pub fn region_energy(l: &Img3, radius: usize, use_chroma: bool) -> Vec<f32> {
    let e = node_energy(l, use_chroma);
    window_sum(&e, l.w, l.h, &binomial(radius))
}

/// Local deviation (eq. 10, box window) and local entropy (eq. 11, `bins`
/// gray levels) of a plane, reflect-101 borders.
pub fn deviation_entropy(y: &[f32], w: usize, h: usize, radius: usize, bins: usize) -> (Vec<f32>, Vec<f32>) {
    let n = (2 * radius + 1) * (2 * radius + 1);
    let inv_n = 1.0 / n as f32;
    let bins = bins.clamp(2, 65536);
    let qscale = (bins - 1) as f32;
    let (mut dev, mut ent) = (vec![0f32; w * h], vec![0f32; w * h]);
    let r = radius as isize;
    dev.par_chunks_mut(w).zip(ent.par_chunks_mut(w)).enumerate().for_each(|(i, (drow, erow))| {
        let mut hist = vec![0u32; bins];
        let mut touched: Vec<usize> = Vec::with_capacity(n);
        for j in 0..w {
            let (mut s, mut s2) = (0f32, 0f32);
            touched.clear();
            for dy in -r..=r {
                let yy = reflect(i as isize + dy, h);
                for dx in -r..=r {
                    let v = y[yy * w + reflect(j as isize + dx, w)];
                    s += v;
                    s2 += v * v;
                    let q = (v.clamp(0.0, 1.0) * qscale + 0.5) as usize;
                    if hist[q] == 0 {
                        touched.push(q);
                    }
                    hist[q] += 1;
                }
            }
            let mean = s * inv_n;
            drow[j] = (s2 * inv_n - mean * mean).max(0.0);
            let mut e = 0f32;
            for &q in &touched {
                let p = hist[q] as f32 * inv_n;
                e -= p * p.ln();
                hist[q] = 0;
            }
            erow[j] = e;
        }
    });
    (dev, ent)
}

/// Fuse the frames' residual levels per `TopRule` (eq. 12, Pareto form).
pub fn fuse_residuals(tops: &[Img3], params: &FuseParams) -> Img3 {
    let t0 = &tops[0];
    let (w, h, n) = (t0.w, t0.h, tops.len());
    let sz = w * h;
    let mut out = Img3::zeros(w, h);
    if n == 1 || params.top_rule == TopRule::Average {
        let inv = 1.0 / n as f32;
        for t in tops {
            for c in 0..3 {
                for i in 0..sz {
                    out.p[c][i] += t.p[c][i] * inv;
                }
            }
        }
        return out;
    }
    let luma = |t: &Img3| -> Vec<f32> {
        (0..sz).map(|i| 0.299 * t.p[0][i] + 0.587 * t.p[1][i] + 0.114 * t.p[2][i]).collect()
    };
    let measures: Vec<(Vec<f32>, Vec<f32>)> = tops
        .par_iter()
        .map(|t| deviation_entropy(&luma(t), w, h, params.top_radius, params.entropy_bins))
        .collect();
    let use_entropy = params.top_rule == TopRule::DevEntropy;
    let mut keep = vec![false; n];
    for i in 0..sz {
        let mut kept = 0usize;
        for a in 0..n {
            let (da, ea) = (measures[a].0[i], measures[a].1[i]);
            let mut dominated = false;
            for b in 0..n {
                if a == b {
                    continue;
                }
                let (db, eb) = (measures[b].0[i], measures[b].1[i]);
                let d = if use_entropy {
                    db >= da && eb >= ea && (db > da || eb > ea)
                } else {
                    db > da || (db == da && b < a)
                };
                if d {
                    dominated = true;
                    break;
                }
            }
            keep[a] = !dominated;
            kept += !dominated as usize;
        }
        let inv = 1.0 / kept.max(1) as f32;
        for (a, t) in tops.iter().enumerate() {
            if keep[a] {
                for c in 0..3 {
                    out.p[c][i] += t.p[c][i] * inv;
                }
            }
        }
    }
    out
}

/// What share of the detail each frame won: per frame, the fraction of the
/// winner map's cells it won among those with detail — a winning region
/// energy above `DETAIL_FLOOR` of the plane's largest (a flat area's winner is
/// noise, and would hand every frame its 1/N whatever it holds). A frame that
/// won next to nothing is redundant to the pyramid: a near-duplicate of its
/// neighbors, or focused on empty space. Sums to 1 over the frames when any
/// cell has detail; all zeros when none has.
pub fn winner_shares<T: Copy + Into<f32>>(win: &[T], best: &[f32], frames: usize) -> Vec<f32> {
    let floor = DETAIL_FLOOR * best.iter().cloned().fold(0f32, f32::max);
    let mut counts = vec![0usize; frames];
    let mut total = 0usize;
    for (w, &b) in win.iter().zip(best) {
        if b > floor && floor > 0.0 {
            let i = (*w).into().round() as usize;
            if i < frames {
                counts[i] += 1;
                total += 1;
            }
        }
    }
    counts.into_iter().map(|c| if total > 0 { c as f32 / total as f32 } else { 0.0 }).collect()
}

/// `winner_shares`: a cell has detail when its winning energy is above this
/// share of the plane's largest.
pub const DETAIL_FLOOR: f32 = 0.01;

/// Nearest-neighbor upsample of a level-`level` index map to `w×h`, as f32.
pub fn upsample_index<T: Copy + Into<f32> + Sync>(win: &[T], dw: usize, dh: usize, w: usize, h: usize, level: usize) -> Vec<f32> {
    let scale = 1usize << level;
    let mut depth = vec![0f32; w * h];
    for_rows(&mut depth, w, |y, row| {
        let sy = (y / scale).min(dh - 1);
        for (x, o) in row.iter_mut().enumerate() {
            *o = win[sy * dw + (x / scale).min(dw - 1)].into();
        }
    });
    depth
}

/// Incremental N-frame fusion accumulator.
pub struct Fuser {
    pub w: usize,
    pub h: usize,
    pub levels: usize,
    params: FuseParams,
    /// Fused band-pass levels `L_0 … L_{N-1}` (empty until the first push);
    /// with halo control also the residual `G_N`, as `Σ w·G_N`.
    acc: Vec<Img3>,
    /// Winning region energy per level; with halo control, at the levels
    /// coarser than the guide (the residual included), the weight sum `Σ w`.
    best: Vec<Vec<f32>>,
    /// Winning frame index at level `depth_level`.
    winner: Vec<u16>,
    depth_level: usize,
    /// Every frame's residual `G_N` (tiny); unused with halo control.
    tops: Vec<Img3>,
    /// Halo control: (guide level, hardness).
    halo: Option<(usize, f32)>,
    count: usize,
}

impl Fuser {
    pub fn new(w: usize, h: usize, params: FuseParams) -> Fuser {
        let levels = params.levels.unwrap_or_else(|| pyramid::auto_levels(w, h, 32)).max(1);
        let depth_level = params.depth_level.min(levels - 1);
        let halo = halo_guide(&params, levels);
        Fuser {
            w,
            h,
            levels,
            params,
            acc: Vec::new(),
            best: Vec::new(),
            winner: Vec::new(),
            depth_level,
            tops: Vec::new(),
            halo,
            count: 0,
        }
    }

    /// The band-pass levels that pick their own winner: all of them, or up
    /// to the guide with halo control.
    fn select_levels(&self) -> usize {
        self.halo.map_or(self.levels, |(g, _)| g + 1)
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Fold one aligned RGB frame (planes in [0,1]) into the accumulator.
    pub fn push(&mut self, frame: &Img3) {
        assert!(frame.w == self.w && frame.h == self.h, "frame size mismatch");
        let mut pyr = pyramid::build(frame, self.levels);
        let nsel = self.select_levels();
        let idx = self.count as u16;
        let first = self.count == 0;
        // region energy of the levels that select (with halo control the guide's also makes the weights)
        let energies: Vec<Vec<f32>> = pyr[..nsel].par_iter().map(|l| region_energy(l, self.params.energy_radius, self.params.use_chroma)).collect();
        if first {
            let d = &pyr[self.depth_level];
            self.winner = vec![0; d.w * d.h];
            self.best = energies.clone();
        } else {
            for (li, (l, en)) in pyr.iter().zip(&energies).enumerate() {
                let (w, best, acc) = (l.w, &mut self.best[li], &mut self.acc[li]);
                let [a0, a1, a2] = &mut acc.p;
                let [n0, n1, n2] = &l.p;
                let rows = a0
                    .par_chunks_mut(w)
                    .zip(a1.par_chunks_mut(w))
                    .zip(a2.par_chunks_mut(w))
                    .zip(best.par_chunks_mut(w))
                    .enumerate();
                if li == self.depth_level {
                    let win = &mut self.winner;
                    rows.zip(win.par_chunks_mut(w)).for_each(|((y, (((r0, r1), r2), rb)), rw)| {
                        let o = y * w;
                        for j in 0..w {
                            if en[o + j] > rb[j] {
                                rb[j] = en[o + j];
                                r0[j] = n0[o + j];
                                r1[j] = n1[o + j];
                                r2[j] = n2[o + j];
                                rw[j] = idx;
                            }
                        }
                    });
                } else {
                    rows.for_each(|(y, (((r0, r1), r2), rb))| {
                        let o = y * w;
                        for j in 0..w {
                            if en[o + j] > rb[j] {
                                rb[j] = en[o + j];
                                r0[j] = n0[o + j];
                                r1[j] = n1[o + j];
                                r2[j] = n2[o + j];
                            }
                        }
                    });
                }
            }
        }
        if let Some((guide, p)) = self.halo {
            // the coarser levels (residual included): Σ w·L and Σ w, the guide's
            // weights REDUCEd down to each level's size
            let mut wgt = halo_weights(&energies[guide], p);
            let (mut ww, mut wh) = (pyr[guide].w, pyr[guide].h);
            for li in guide + 1..=self.levels {
                (wgt, ww, wh) = pyramid::reduce(&wgt, ww, wh);
                debug_assert!(ww == pyr[li].w && wh == pyr[li].h);
                if first {
                    for c in 0..3 {
                        pyr[li].p[c].par_iter_mut().zip(&wgt).for_each(|(a, &k)| *a *= k);
                    }
                    self.best.push(wgt.clone());
                } else {
                    self.best[li].par_iter_mut().zip(&wgt).for_each(|(b, &k)| *b += k);
                    for c in 0..3 {
                        self.acc[li].p[c].par_iter_mut().zip(&pyr[li].p[c]).zip(&wgt).for_each(|((a, &v), &k)| *a += k * v);
                    }
                }
            }
            if first {
                self.acc = pyr;
            }
        } else {
            let top = pyr.pop().unwrap();
            if first {
                self.acc = pyr;
            }
            self.tops.push(top);
        }
        self.count += 1;
    }

    /// `winner_shares` of the frames folded so far, from the winner map of
    /// `depth_level`; empty when that level does not select (halo control
    /// with a guide finer than it).
    pub fn shares(&self) -> Vec<f32> {
        if self.count == 0 || self.depth_level >= self.select_levels() {
            return Vec::new();
        }
        winner_shares(&self.winner, &self.best[self.depth_level], self.count)
    }

    /// Finish: fuse the residuals, collapse the pyramid. Returns the fused
    /// RGB image (clamped to [0,1]) and the depth map: the winning frame index
    /// at `depth_level`, nearest-upsampled to full resolution, as f32.
    pub fn finish(mut self) -> (Img3, Vec<f32>) {
        assert!(self.count > 0, "no frames pushed");
        let (dw, dh) = (self.acc[self.depth_level].w, self.acc[self.depth_level].h);
        let mut pyr = std::mem::take(&mut self.acc);
        match self.halo {
            Some((guide, _)) => {
                // the weighted means: Σ w·L / Σ w
                for li in guide + 1..=self.levels {
                    for c in 0..3 {
                        pyr[li].p[c].par_iter_mut().zip(&self.best[li]).for_each(|(a, &k)| *a /= k);
                    }
                }
            }
            None => pyr.push(fuse_residuals(&self.tops, &self.params)),
        }
        let mut img = pyramid::collapse(pyr);
        for c in 0..3 {
            img.p[c].par_iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
        }
        let depth = upsample_index(&self.winner, dw, dh, self.w, self.h, self.depth_level);
        (img, depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winner_shares_count_detail_cells_only() {
        // frame 0 wins the flat cells (noise-level energy), frame 1 the detail
        let win: Vec<u16> = vec![0, 0, 0, 1, 1, 2];
        let best = vec![0.001, 0.002, 0.001, 1.0, 0.5, 0.2];
        let s = winner_shares(&win, &best, 3);
        assert_eq!(s, vec![0.0, 2.0 / 3.0, 1.0 / 3.0]);
        assert!(winner_shares(&win, &[0.0; 6], 3).iter().all(|&v| v == 0.0), "no detail at all");
        // a three-frame stack of near-duplicates: the first frame wins ties, the others nothing
        let (w, h) = (64, 64);
        let img = checker(w, h, 8);
        let mut f = Fuser::new(w, h, FuseParams::default());
        for _ in 0..3 {
            f.push(&img);
        }
        let s = f.shares();
        assert_eq!(s.len(), 3);
        assert!((s[0] - 1.0).abs() < 1e-6 && s[1] == 0.0 && s[2] == 0.0, "{s:?}");
    }

    fn checker(w: usize, h: usize, cell: usize) -> Img3 {
        let mut im = Img3::zeros(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if ((x / cell) + (y / cell)) % 2 == 0 { 0.2 } else { 0.8 };
                im.p[0][y * w + x] = v;
                im.p[1][y * w + x] = v * 0.9;
                im.p[2][y * w + x] = v * 0.5;
            }
        }
        im
    }

    /// Blur a plane with the 5-tap kernel `passes` times (no decimation).
    fn blur(im: &Img3, passes: usize) -> Img3 {
        let mut o = im.clone();
        for _ in 0..passes {
            for c in 0..3 {
                o.p[c] = window_sum(&o.p[c], o.w, o.h, &pyramid::KERNEL);
            }
        }
        o
    }

    fn rms(a: &Img3, b: &Img3) -> f32 {
        let mut s = 0f64;
        for c in 0..3 {
            for (x, y) in a.p[c].iter().zip(&b.p[c]) {
                s += ((x - y) * (x - y)) as f64;
            }
        }
        (s / (3 * a.w * a.h) as f64).sqrt() as f32
    }

    #[test]
    fn binomial_weights() {
        assert_eq!(binomial(0), vec![1.0]);
        assert_eq!(binomial(1), vec![0.25, 0.5, 0.25]);
        let b2 = binomial(2);
        for (a, k) in b2.iter().zip(&pyramid::KERNEL) {
            assert!((a - k).abs() < 1e-6);
        }
    }

    #[test]
    fn identical_frames_are_a_fixed_point() {
        let im = checker(70, 50, 7);
        let mut f = Fuser::new(70, 50, FuseParams::default());
        for _ in 0..3 {
            f.push(&im);
        }
        let (out, depth) = f.finish();
        assert!(rms(&out, &im) < 1e-5);
        assert!(depth.iter().all(|&d| d == 0.0), "ties keep the first frame");
    }

    #[test]
    fn picks_the_sharp_half_from_each_frame() {
        let (w, h) = (96, 64);
        let sharp = checker(w, h, 6);
        let soft = blur(&sharp, 6);
        // frame A: sharp on the left, blurred on the right; frame B: the opposite
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
        let base = FuseParams { depth_level: 0, ..Default::default() };
        for params in [
            base.clone(),
            FuseParams { energy_radius: 0, ..base.clone() },
            FuseParams { top_rule: TopRule::Average, ..base.clone() },
        ] {
            let mut f = Fuser::new(w, h, params.clone());
            f.push(&a);
            f.push(&b);
            let (out, depth) = f.finish();
            let err = rms(&out, &sharp);
            assert!(err < 0.6 * rms(&a, &sharp), "{params:?}: fused rms {err}");
            // finest-level selection follows the sharp side away from the seam
            let n = depth.len();
            let left = depth.iter().enumerate().filter(|(i, _)| i % w < w / 2 - 8).filter(|&(_, &d)| d == 0.0).count();
            let right = depth.iter().enumerate().filter(|(i, _)| i % w >= w / 2 + 8).filter(|&(_, &d)| d == 1.0).count();
            assert!(left as f32 > 0.9 * (n / 2 - 8 * h) as f32, "{params:?}: left {left}");
            assert!(right as f32 > 0.9 * (n / 2 - 8 * h) as f32, "{params:?}: right {right}");
        }
    }

    #[test]
    fn halo_weight_is_monotonic_and_bounded() {
        assert!((halo_weight(HALO_REF - HALO_FLOOR, 3.0) - 1.0).abs() < 1e-5);
        assert!(halo_weight(1e-3, 2.0) > halo_weight(1e-4, 2.0));
        assert!(halo_weight(1e-4, 4.0) / halo_weight(1e-5, 4.0) > 9e3);
        assert!(halo_weight(0.0, 8.0) > 0.0 && halo_weight(0.0, 8.0).is_finite());
        assert!(halo_weight(1e3, 8.0).is_finite());
        assert_eq!(halo_guide(&FuseParams::default(), 5), None);
        assert_eq!(halo_guide(&FuseParams { halo: 2.0, ..Default::default() }, 2), Some((1, 2.0)));
    }

    #[test]
    fn halo_control_keeps_identical_frames() {
        let im = checker(70, 50, 7);
        let p = FuseParams { halo: 2.0, levels: Some(3), depth_level: 1, ..Default::default() };
        let mut f = Fuser::new(70, 50, p);
        for _ in 0..3 {
            f.push(&im);
        }
        let (out, depth) = f.finish();
        assert!(rms(&out, &im) < 1e-5);
        assert!(depth.iter().all(|&d| d == 0.0));
    }

    /// The halo mechanism in one piece: frame A is sharp fine texture; frame
    /// B is the texture defocused with a broad bright bump on it — coarse
    /// structure only, as a defocused copy of a bright object has. Every level
    /// picking its own winner takes the bump (B alone has energy at the coarse
    /// levels); with halo control the guide finds A sharp everywhere, so the
    /// coarse levels follow A and the bump stays out.
    #[test]
    fn halo_control_coarse_levels_follow_the_guide() {
        let (w, h) = (192, 144);
        let a = checker(w, h, 6); // a 12 px period: the guide (level 2, scale 4 px) sees it
        let mut b = blur(&a, 64); // σ = 8 px: the texture is gone
        for y in 0..h {
            for x in 0..w {
                let d2 = (x as f32 - 96.0).powi(2) + (y as f32 - 72.0).powi(2);
                let bump = 0.35 * (-d2 / (2.0 * 20.0f32.powi(2))).exp();
                for c in 0..3 {
                    b.p[c][y * w + x] = (b.p[c][y * w + x] + bump).min(1.0);
                }
            }
        }
        let fuse = |halo: f32, order: [&Img3; 2]| {
            let p = FuseParams { levels: Some(5), depth_level: 2, halo, ..Default::default() };
            let mut f = Fuser::new(w, h, p);
            f.push(order[0]);
            f.push(order[1]);
            f.finish().0
        };
        let off = rms(&fuse(0.0, [&a, &b]), &a);
        assert!(off > 0.05, "without halo control the bump gets in: rms {off}");
        for halo in [1.0, 2.0, 4.0, 8.0] {
            let on = rms(&fuse(halo, [&a, &b]), &a);
            assert!(on < 0.15 * off, "halo {halo}: rms {on} vs {off} without");
        }
        // the frames' order makes no difference
        assert!(rms(&fuse(2.0, [&b, &a]), &a) < 0.15 * off);
    }

    #[test]
    fn top_rule_two_frames_matches_eq12() {
        // Frame 0 residual wins D and E everywhere → fused top equals it.
        let (w, h) = (40, 40);
        let a = checker(w, h, 4);
        let mut b = a.clone();
        for c in 0..3 {
            for v in b.p[c].iter_mut() {
                *v = 0.5; // flat: zero deviation, zero entropy
            }
        }
        let p = FuseParams { levels: Some(1), ..Default::default() };
        let mut f = Fuser::new(w, h, p);
        f.push(&a);
        f.push(&b);
        let top = fuse_residuals(&f.tops, &f.params);
        let ta = &f.tops[0];
        assert!(rms(&top, ta) < 1e-6);
    }
}

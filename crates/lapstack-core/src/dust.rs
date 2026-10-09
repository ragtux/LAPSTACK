// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

//! Dust map removal: sensor dust shows in every
//! frame of a stack at the same place, as a soft dark spot, and the fusion
//! rule keeps it — worse, the region-energy rule sees the spot's edge as
//! detail and picks it, so the spot is sharper in the result than in any
//! frame. A frame of an evenly lit, featureless surface, shot out of focus
//! (a white wall, the sky, a sheet of paper; the same aperture as the stack,
//! since a spot's size and darkness follow the aperture) shows nothing but
//! the dust, and that frame is the dust map.
//!
//! `detect` finds the spots in it: the luma at half resolution (one REDUCE,
//! which also tames the noise) is divided by its own large-scale background
//! — a plane fitted under each cell four pyramid levels up, cells of 32
//! pixels, and expanded back, so the illumination's falloff is followed, out
//! to the frame's edges, but a spot is not — and a pixel darker
//! than the background by more than `threshold` is dust. The background is
//! estimated twice, the second time with the first pass's spots left out, so
//! a big spot does not pull its own background down. The connected
//! components of that mask, dilated by `margin` pixels (the soft edge of the
//! spot, and the little the spot moves between apertures), are the spots;
//! specks below `min_area` pixels are noise and are dropped, and a blob wider
//! than a quarter of the frame is not dust.
//!
//! `apply` takes the spots out of a frame, before it is aligned (the dust
//! sits on the sensor, so it is fixed in the frame as shot — and a fixed
//! pattern in every frame is exactly what pulls a registration toward zero
//! shift). `Fill` interpolates each spot from its surroundings with the
//! pull-push of Gortler et al. (the masked window's pyramid is built with the
//! spot's pixels weighted out, and on the way back down every hole takes the
//! coarser level's value: a smooth patch that meets its edges).
//! `Flat` instead divides the spot by the dust map's own attenuation
//! (the ratio the detection measured), a flat-field correction that keeps
//! whatever detail lies under the spot — right when the map was shot at the
//! stack's aperture and lighting, wrong (a ring) when it was not.

use crate::pyramid::{Img3, expand, half, reduce};
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DustMode {
    /// Interpolate each spot from its surroundings (pull-push).
    Fill,
    /// Divide each spot by the dust map's attenuation (flat-field).
    Flat,
}

impl DustMode {
    pub fn parse(s: &str) -> Option<DustMode> {
        match s {
            "fill" => Some(DustMode::Fill),
            "flat" => Some(DustMode::Flat),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            DustMode::Fill => "fill",
            DustMode::Flat => "flat",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DustParams {
    /// A pixel darker than its background by more than this fraction is dust.
    pub threshold: f32,
    /// Pixels the spots are grown by, at full resolution.
    pub margin: usize,
    /// Smaller components (full-resolution pixels) are noise, not dust.
    pub min_area: usize,
    pub mode: DustMode,
}

impl Default for DustParams {
    fn default() -> Self {
        DustParams { threshold: 0.03, margin: 3, min_area: 16, mode: DustMode::Fill }
    }
}

/// One dust spot: a window of the frame, its mask (1 = dust), and the
/// attenuation the dust map measured in it (≥ 1; for `Flat`).
#[derive(Clone)]
pub struct Spot {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    /// Dust pixels in the window (before the margin).
    pub area: usize,
    pub mask: Vec<u8>,
    pub gain: Vec<f32>,
}

impl Spot {
    /// The longer side of the spot's window, less the margin: its size as seen.
    pub fn size(&self, margin: usize) -> usize {
        self.w.max(self.h).saturating_sub(2 * margin)
    }
}

/// The dust map of a frame size: its spots, and how they are removed.
#[derive(Clone)]
pub struct DustMap {
    pub w: usize,
    pub h: usize,
    pub spots: Vec<Spot>,
    pub params: DustParams,
    /// Components too large to be dust, dropped.
    pub rejected: usize,
}

/// Luma of a frame, Y = .299R + .587G + .114B.
pub fn luma(img: &Img3) -> Vec<f32> {
    let (r, g, b) = (&img.p[0], &img.p[1], &img.p[2]);
    (0..img.w * img.h).into_par_iter().map(|i| 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]).collect()
}

/// Pyramid levels above `w×h` until a cell is about `cell` pixels across
/// (the background must follow the illumination, not a spot).
fn coarse_levels(w: usize, h: usize, cell: usize) -> usize {
    let (mut n, mut c, mut cw, mut ch) = (0, 1, w, h);
    while c < cell && half(cw).min(half(ch)) >= 2 {
        cw = half(cw);
        ch = half(ch);
        c *= 2;
        n += 1;
    }
    n
}

/// Large-scale background of `v` (width `w`, height `h`) with the weights
/// `wt` in [0, 1]: at each cell of the pyramid level `levels` up, a plane is
/// fitted to the pixels under the cell's binomial window (a first-order
/// normalized convolution: the nine weighted moments of v, x and y are
/// REDUCEd, and the 3×3 normal equations solved per cell), and the plane's
/// value at the cell's center is EXPANDed back. A plain mean would sit too
/// high at the frame's edges wherever the illumination falls off toward
/// them (a one-sided window's mean lies inward of its center), and dust
/// would be seen along every edge; the plane follows the slope out.
fn background(v: &[f32], wt: &[f32], w: usize, h: usize, levels: usize) -> Vec<f32> {
    // coordinates normalized to about [-1/2, 1/2], so the moments stay O(1) in f32
    let (sx, sy) = (1.0 / w.max(h) as f32, 1.0 / w.max(h) as f32);
    let xs: Vec<f32> = (0..w * h).map(|i| ((i % w) as f32 - w as f32 / 2.0) * sx).collect();
    let ys: Vec<f32> = (0..w * h).map(|i| ((i / w) as f32 - h as f32 / 2.0) * sy).collect();
    let prod = |f: &dyn Fn(usize) -> f32| -> Vec<f32> { (0..w * h).map(|i| wt[i] * f(i)).collect() };
    let mut m: Vec<Vec<f32>> = vec![
        prod(&|_| 1.0),                    // S
        prod(&|i| xs[i]),                  // Sx
        prod(&|i| ys[i]),                  // Sy
        prod(&|i| v[i]),                   // Sv
        prod(&|i| xs[i] * v[i]),           // Sxv
        prod(&|i| ys[i] * v[i]),           // Syv
        prod(&|i| xs[i] * xs[i]),          // Sxx
        prod(&|i| xs[i] * ys[i]),          // Sxy
        prod(&|i| ys[i] * ys[i]),          // Syy
    ];
    let mut dims = vec![(w, h)];
    for _ in 0..levels {
        let &(cw, ch) = dims.last().unwrap();
        let mut ow = 0;
        let mut oh = 0;
        for k in m.iter_mut() {
            let (r, a, b) = reduce(k, cw, ch);
            *k = r;
            ow = a;
            oh = b;
        }
        dims.push((ow, oh));
    }
    let &(cw, ch) = dims.last().unwrap();
    let step = (1usize << levels) as f32;
    let mut bg = vec![f32::NAN; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            let k = j * cw + i;
            let s = m[0][k] as f64;
            if s < 1e-3 {
                continue;   // (almost) no weight under this cell: filled from its neighbors below
            }
            // the cell's center (coarse sample i sits on fine pixel i·2^levels), and the moments about it
            let xc = ((i as f32 * step) - w as f32 / 2.0) as f64 * sx as f64;
            let yc = ((j as f32 * step) - h as f32 / 2.0) as f64 * sy as f64;
            let (sxg, syg, sv, sxv, syv, sxx, sxy, syy) = (m[1][k] as f64, m[2][k] as f64, m[3][k] as f64, m[4][k] as f64, m[5][k] as f64, m[6][k] as f64, m[7][k] as f64, m[8][k] as f64);
            let dx = sxg - xc * s;
            let dy = syg - yc * s;
            let dxx = sxx - 2.0 * xc * sxg + xc * xc * s;
            let dxy = sxy - xc * syg - yc * sxg + xc * yc * s;
            let dyy = syy - 2.0 * yc * syg + yc * yc * s;
            let dxv = sxv - xc * sv;
            let dyv = syv - yc * sv;
            // normal equations [s dx dy; dx dxx dxy; dy dxy dyy] [a b c] = [sv dxv dyv]; a is the value at the center
            let det = s * (dxx * dyy - dxy * dxy) - dx * (dx * dyy - dxy * dy) + dy * (dx * dxy - dxx * dy);
            let scale = s * (dxx * dyy).abs().max(1e-30);
            bg[k] = if det.abs() > 1e-9 * scale {
                let a = (sv * (dxx * dyy - dxy * dxy) - dx * (dxv * dyy - dxy * dyv) + dy * (dxv * dxy - dxx * dyv)) / det;
                a as f32
            } else {
                (sv / s) as f32
            };
        }
    }
    // a coarse cell with no weight (all dust, or all outside) takes its neighbors' mean
    if bg.iter().any(|v| v.is_nan()) {
        let mean = {
            let (s, n) = bg.iter().filter(|v| !v.is_nan()).fold((0.0f64, 0usize), |(s, n), &v| (s + v as f64, n + 1));
            if n > 0 { (s / n as f64) as f32 } else { 1.0 }
        };
        for _ in 0..(cw.max(ch)) {
            let prev = bg.clone();
            let mut changed = false;
            for y in 0..ch {
                for x in 0..cw {
                    if !prev[y * cw + x].is_nan() {
                        continue;
                    }
                    let (mut s, mut n) = (0.0, 0);
                    for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                        let (nx, ny) = (x as isize + dx, y as isize + dy);
                        if nx >= 0 && ny >= 0 && (nx as usize) < cw && (ny as usize) < ch && !prev[ny as usize * cw + nx as usize].is_nan() {
                            s += prev[ny as usize * cw + nx as usize];
                            n += 1;
                        }
                    }
                    if n > 0 {
                        bg[y * cw + x] = s / n as f32;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for v in bg.iter_mut() {
            if v.is_nan() {
                *v = mean;
            }
        }
    }
    for l in (0..levels).rev() {
        let (ow, oh) = dims[l];
        let (cw, ch) = dims[l + 1];
        bg = expand_linear(&bg, cw, ch, ow, oh);
    }
    bg
}

/// EXPAND with the borders extrapolated linearly instead of reflected: the
/// plane pads itself by one sample each side (x[-1] = 2x[0] - x[1]), is
/// expanded, and the padding is cut. The pyramid's reflect-101 folds a slope
/// back at the frame's edge, which would leave the last cell of a falling
/// background too high.
fn expand_linear(coarse: &[f32], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<f32> {
    let (pw, ph) = (cw + 2, ch + 2);
    let mut pad = vec![0f32; pw * ph];
    let at = |x: isize, y: isize| -> f32 {
        let ext = |i: isize, n: usize| -> (usize, usize, f32) {
            // sample index, its neighbor, and the extrapolation factor (0 inside)
            if i < 0 { (0, 1.min(n - 1), 1.0) } else if i as usize >= n { (n - 1, n.saturating_sub(2), 1.0) } else { (i as usize, i as usize, 0.0) }
        };
        let (x0, x1, fx) = ext(x, cw);
        let (y0, y1, fy) = ext(y, ch);
        let v = |xx: usize, yy: usize| coarse[yy * cw + xx];
        // linear extrapolation along each axis that is outside (2·edge − next)
        let row = |yy: usize| v(x0, yy) + fx * (v(x0, yy) - v(x1, yy));
        row(y0) + fy * (row(y0) - row(y1))
    };
    for y in 0..ph {
        for x in 0..pw {
            pad[y * pw + x] = at(x as isize - 1, y as isize - 1);
        }
    }
    let big = expand(&pad, pw, ph, 2 * pw, 2 * ph);
    let mut out = vec![0f32; ow * oh];
    for y in 0..oh {
        out[y * ow..y * ow + ow].copy_from_slice(&big[(y + 2) * 2 * pw + 2..(y + 2) * 2 * pw + 2 + ow]);
    }
    out
}

/// Chebyshev dilation of a 0/1 mask by `r` pixels (separable running max).
fn dilate(mask: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return mask.to_vec();
    }
    let mut tmp = vec![0u8; w * h];
    for y in 0..h {
        let row = &mask[y * w..y * w + w];
        let out = &mut tmp[y * w..y * w + w];
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            out[x] = row[lo..=hi].iter().copied().max().unwrap_or(0);
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = (y + r).min(h - 1);
        for x in 0..w {
            let mut m = 0;
            for yy in lo..=hi {
                m = m.max(tmp[yy * w + x]);
            }
            out[y * w + x] = m;
        }
    }
    out
}

/// Find the dust spots in the luma `y` (width `w`, height `h`, in [0, 1]) of
/// a dust-map frame.
pub fn detect(y: &[f32], w: usize, h: usize, p: &DustParams) -> DustMap {
    if w < 4 || h < 4 {
        return DustMap { w, h, spots: Vec::new(), params: *p, rejected: 0 };
    }
    let (ys, hw, hh) = half_luma(y, w, h);
    detect_half(&ys, hw, hh, w, h, p)
}

/// The luma at half resolution, as `detect_half` takes it: one REDUCE blurs
/// the noise away and halves the work (and is all a browser need keep of the
/// map frame to find the spots again with other settings).
pub fn half_luma(y: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    reduce(y, w, h)
}

/// `detect` from the half-resolution luma `ys` (`hw×hh`) of a `w×h` frame.
pub fn detect_half(ys: &[f32], hw: usize, hh: usize, w: usize, h: usize, p: &DustParams) -> DustMap {
    let mut map = DustMap { w, h, spots: Vec::new(), params: *p, rejected: 0 };
    if w < 4 || h < 4 || ys.len() != hw * hh {
        return map;
    }
    let levels = coarse_levels(hw, hh, 16);
    let thr = 1.0 - p.threshold.clamp(0.001, 0.9);
    let dark = |bg: &[f32]| -> Vec<u8> { ys.iter().zip(bg).map(|(v, b)| (*b > 1e-6 && v / b < thr) as u8).collect() };
    // pass 1 against the plain background, pass 2 with those spots weighted out
    let bg1 = background(&ys, &vec![1.0; hw * hh], hw, hh, levels);
    let m1 = dilate(&dark(&bg1), hw, hh, 2);
    let wt: Vec<f32> = m1.iter().map(|&m| 1.0 - m as f32).collect();
    let bg = background(&ys, &wt, hw, hh, levels);
    let mask = dark(&bg);
    // connected components (8-connected) of the half-resolution mask
    let mut label = vec![0u32; hw * hh];
    let mut stack: Vec<usize> = Vec::new();
    let mut next = 1u32;
    let short = hw.min(hh);
    for start in 0..hw * hh {
        if mask[start] == 0 || label[start] != 0 {
            continue;
        }
        let id = next;
        next += 1;
        label[start] = id;
        stack.push(start);
        let mut pixels: Vec<usize> = Vec::new();
        let (mut x0, mut y0, mut x1, mut y1) = (hw, hh, 0usize, 0usize);
        while let Some(i) = stack.pop() {
            pixels.push(i);
            let (x, yy) = (i % hw, i / hw);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(yy);
            y1 = y1.max(yy);
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let (nx, ny) = (x as isize + dx, yy as isize + dy);
                    if nx < 0 || ny < 0 || nx as usize >= hw || ny as usize >= hh {
                        continue;
                    }
                    let j = ny as usize * hw + nx as usize;
                    if mask[j] != 0 && label[j] == 0 {
                        label[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
        // half-resolution pixels are 4 full-resolution ones
        if pixels.len() * 4 < p.min_area.max(1) {
            continue;
        }
        if (x1 - x0 + 1).max(y1 - y0 + 1) * 4 > short {
            map.rejected += 1;
            continue;
        }
        // the spot's window at full resolution, grown by the margin
        let m = p.margin;
        let (wx0, wy0) = ((2 * x0).saturating_sub(m), (2 * y0).saturating_sub(m));
        let (wx1, wy1) = ((2 * (x1 + 1) + m).min(w), (2 * (y1 + 1) + m).min(h));
        let (sw, sh) = (wx1 - wx0, wy1 - wy0);
        let mut smask = vec![0u8; sw * sh];
        for &i in &pixels {
            let (x, yy) = (i % hw, i / hw);
            for (fx, fy) in [(2 * x, 2 * yy), (2 * x + 1, 2 * yy), (2 * x, 2 * yy + 1), (2 * x + 1, 2 * yy + 1)] {
                if fx >= wx0 && fx < wx1 && fy >= wy0 && fy < wy1 {
                    smask[(fy - wy0) * sw + (fx - wx0)] = 1;
                }
            }
        }
        let smask = dilate(&smask, sw, sh, m);
        // the attenuation under the spot, for the flat-field mode: ≥ 1, capped at ×10,
        // the half-resolution ratio sampled bilinearly (half-res pixel k is centered on 2k)
        let ratio = |hx: usize, hy: usize| -> f32 {
            let (v, b) = (ys[hy * hw + hx], bg[hy * hw + hx]);
            if v > 1e-6 && b > v { (b / v).min(10.0) } else { 1.0 }
        };
        let mut gain = vec![1.0f32; sw * sh];
        for sy in 0..sh {
            for sx in 0..sw {
                if smask[sy * sw + sx] == 0 {
                    continue;
                }
                let (fx, fy) = ((wx0 + sx) as f32 / 2.0, (wy0 + sy) as f32 / 2.0);
                let (x0f, y0f) = (fx.floor().min((hw - 1) as f32), fy.floor().min((hh - 1) as f32));
                let (tx, ty) = (fx - x0f, fy - y0f);
                let (ix, iy) = (x0f as usize, y0f as usize);
                let (ix1, iy1) = ((ix + 1).min(hw - 1), (iy + 1).min(hh - 1));
                let g = (1.0 - ty) * ((1.0 - tx) * ratio(ix, iy) + tx * ratio(ix1, iy)) + ty * ((1.0 - tx) * ratio(ix, iy1) + tx * ratio(ix1, iy1));
                gain[sy * sw + sx] = g;
            }
        }
        map.spots.push(Spot { x: wx0, y: wy0, w: sw, h: sh, area: pixels.len() * 4, mask: smask, gain });
    }
    map.spots.sort_by_key(|s| std::cmp::Reverse(s.area));
    map
}

impl DustMap {
    pub fn is_empty(&self) -> bool {
        self.spots.is_empty()
    }

    /// Dust pixels in all, counting the margins.
    pub fn covered(&self) -> usize {
        self.spots.iter().map(|s| s.mask.iter().filter(|&&m| m != 0).count()).sum()
    }

    /// "23 dust spots, 12–61 px across, 0.4 % of the frame" (and the rejects).
    pub fn describe(&self) -> String {
        let n = self.spots.len();
        let mut s = if n == 0 {
            "no dust spots".to_string()
        } else {
            let m = self.params.margin;
            let (lo, hi) = self.spots.iter().fold((usize::MAX, 0), |(a, b), s| (a.min(s.size(m)), b.max(s.size(m))));
            format!(
                "{n} dust spot{}, {} px across, {:.2} % of the frame",
                if n == 1 { "" } else { "s" },
                if lo == hi { format!("{lo}") } else { format!("{lo}–{hi}") },
                100.0 * self.covered() as f64 / (self.w * self.h).max(1) as f64
            )
        };
        if self.rejected > 0 {
            s.push_str(&format!(" ({} blob{} wider than a quarter of the frame left out: not dust)", self.rejected, if self.rejected == 1 { "" } else { "s" }));
        }
        s
    }

    /// The mask as a plane, 1 where dust (to save and look at).
    pub fn plane(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.w * self.h];
        for s in &self.spots {
            for sy in 0..s.h {
                for sx in 0..s.w {
                    if s.mask[sy * s.w + sx] != 0 {
                        out[(s.y + sy) * self.w + s.x + sx] = 1.0;
                    }
                }
            }
        }
        out
    }

    /// Take the dust out of a frame (planar float).
    pub fn apply(&self, frame: &mut Img3) {
        if self.spots.is_empty() || frame.w != self.w || frame.h != self.h {
            return;
        }
        let w = self.w;
        frame.p.par_iter_mut().for_each(|plane| self.apply_plane(&mut F32Plane { p: plane, w }));
    }

    /// Take the dust out of an interleaved RGB u16 frame (`w*h*3` samples).
    pub fn apply_rgb16(&self, rgb: &mut [u16]) {
        if self.spots.is_empty() || rgb.len() < self.w * self.h * 3 {
            return;
        }
        let w = self.w;
        for c in 0..3 {
            self.apply_plane(&mut U16Plane { rgb, w, c });
        }
    }

    /// One plane: every spot is filled (or flattened) in turn, each from a
    /// window of the plane around it in which every spot's pixels (its own
    /// and its neighbors') are weighted out.
    fn apply_plane(&self, plane: &mut dyn Plane) {
        for s in &self.spots {
            match self.params.mode {
                DustMode::Flat => {
                    for sy in 0..s.h {
                        for sx in 0..s.w {
                            if s.mask[sy * s.w + sx] != 0 {
                                let (x, y) = (s.x + sx, s.y + sy);
                                plane.set(x, y, plane.get(x, y) * s.gain[sy * s.w + sx]);
                            }
                        }
                    }
                }
                DustMode::Fill => {
                    // the window: the spot and as much again around it
                    let pad = (s.w.max(s.h) / 2).max(6);
                    let (x0, y0) = (s.x.saturating_sub(pad), s.y.saturating_sub(pad));
                    let (x1, y1) = ((s.x + s.w + pad).min(self.w), (s.y + s.h + pad).min(self.h));
                    let (ww, wh) = (x1 - x0, y1 - y0);
                    let mut v = vec![0f32; ww * wh];
                    let mut wt = vec![1f32; ww * wh];
                    for y in 0..wh {
                        for x in 0..ww {
                            v[y * ww + x] = plane.get(x0 + x, y0 + y);
                        }
                    }
                    for o in self.spots.iter().filter(|o| o.x < x1 && o.x + o.w > x0 && o.y < y1 && o.y + o.h > y0) {
                        for sy in 0..o.h {
                            for sx in 0..o.w {
                                let (x, y) = (o.x + sx, o.y + sy);
                                if o.mask[sy * o.w + sx] != 0 && x >= x0 && x < x1 && y >= y0 && y < y1 {
                                    wt[(y - y0) * ww + (x - x0)] = 0.0;
                                }
                            }
                        }
                    }
                    let filled = pull_push(&v, &wt, ww, wh);
                    for sy in 0..s.h {
                        for sx in 0..s.w {
                            if s.mask[sy * s.w + sx] != 0 {
                                let (x, y) = (s.x + sx, s.y + sy);
                                plane.set(x, y, filled[(y - y0) * ww + (x - x0)]);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// One channel of a frame, however it is stored.
trait Plane {
    fn get(&self, x: usize, y: usize) -> f32;
    fn set(&mut self, x: usize, y: usize, v: f32);
}

struct F32Plane<'a> {
    p: &'a mut [f32],
    w: usize,
}

impl Plane for F32Plane<'_> {
    fn get(&self, x: usize, y: usize) -> f32 {
        self.p[y * self.w + x]
    }
    fn set(&mut self, x: usize, y: usize, v: f32) {
        self.p[y * self.w + x] = v;
    }
}

/// Channel `c` of interleaved RGB u16.
struct U16Plane<'a> {
    rgb: &'a mut [u16],
    w: usize,
    c: usize,
}

impl Plane for U16Plane<'_> {
    fn get(&self, x: usize, y: usize) -> f32 {
        self.rgb[3 * (y * self.w + x) + self.c] as f32 / 65535.0
    }
    fn set(&mut self, x: usize, y: usize, v: f32) {
        self.rgb[3 * (y * self.w + x) + self.c] = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

/// Pull-push interpolation (Gortler et al. 1996): `v` with weights `wt` in
/// [0, 1] is REDUCEd as v·wt and wt to a single cell, and on the way back
/// every pixel takes its own value where it has weight and the coarser
/// level's where it has none, blending in between.
pub fn pull_push(v: &[f32], wt: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut nums = vec![v.iter().zip(wt).map(|(a, b)| a * b).collect::<Vec<f32>>()];
    let mut dens = vec![wt.to_vec()];
    let mut dims = vec![(w, h)];
    while dims.last().map_or(false, |&(cw, ch)| cw > 1 || ch > 1) && dims.len() < 24 {
        let &(cw, ch) = dims.last().unwrap();
        let (n2, ow, oh) = reduce(nums.last().unwrap(), cw, ch);
        let (d2, _, _) = reduce(dens.last().unwrap(), cw, ch);
        nums.push(n2);
        dens.push(d2);
        dims.push((ow, oh));
    }
    let top = dims.len() - 1;
    let mut f: Vec<f32> = nums[top].iter().zip(&dens[top]).map(|(n, d)| if *d > 1e-9 { n / d } else { 0.0 }).collect();
    for l in (0..top).rev() {
        let (ow, oh) = dims[l];
        let (cw, ch) = dims[l + 1];
        let up = expand(&f, cw, ch, ow, oh);
        f = nums[l]
            .iter()
            .zip(&dens[l])
            .zip(&up)
            .map(|((n, d), u)| {
                let wc = d.clamp(0.0, 1.0);
                if *d > 1e-9 { wc * (n / d) + (1.0 - wc) * u } else { *u }
            })
            .collect();
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A softly lit white frame with soft dark spots at the given centers and radii.
    fn dust_frame(w: usize, h: usize, spots: &[(f32, f32, f32, f32)]) -> Img3 {
        let mut im = Img3::zeros(w, h);
        for y in 0..h {
            for x in 0..w {
                // vignetting: 0.9 in the middle down to 0.85 in the corners (a real lens
                // falls off further, but over ten times as many pixels)
                let (u, v) = (x as f32 / w as f32 - 0.5, y as f32 / h as f32 - 0.5);
                let mut l = 0.9 - 0.1 * (u * u + v * v);
                for &(cx, cy, r, depth) in spots {
                    let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / r;
                    if d < 1.0 {
                        l *= 1.0 - depth * (1.0 - d * d).powi(2);   // a soft bowl, its edge without a kink
                    }
                }
                for c in 0..3 {
                    im.p[c][y * w + x] = l * [1.0, 0.98, 0.95][c];
                }
            }
        }
        im
    }

    fn scene(w: usize, h: usize) -> Img3 {
        let mut im = Img3::zeros(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                im.p[0][i] = 0.3 + 0.3 * ((x as f32 / 23.0).sin() * 0.5 + 0.5);
                im.p[1][i] = 0.2 + 0.4 * (y as f32 / h as f32);
                im.p[2][i] = 0.5 + 0.2 * ((x + y) as f32 / 37.0).cos();
            }
        }
        im
    }

    #[test]
    fn spots_are_found_where_they_are() {
        let spots = [(100.0, 80.0, 12.0, 0.4), (300.0, 200.0, 20.0, 0.15), (40.0, 220.0, 8.0, 0.25)];
        let map = detect(&luma(&dust_frame(400, 300, &spots)), 400, 300, &DustParams::default());
        assert_eq!(map.spots.len(), 3, "{}", map.describe());
        assert_eq!(map.rejected, 0);
        for &(cx, cy, r, _) in &spots {
            let s = map.spots.iter().find(|s| (s.x as f32) < cx && cx < (s.x + s.w) as f32 && (s.y as f32) < cy && cy < (s.y + s.h) as f32).expect("a spot covers each center");
            let seen = s.size(map.params.margin) as f32;
            assert!(seen > r && seen < 2.6 * r, "spot at {cx},{cy} r={r}: window {}x{} seen {seen}", s.w, s.h);
            assert!(s.mask[(cy as usize - s.y) * s.w + (cx as usize - s.x)] != 0, "the center is masked");
        }
        let plane = map.plane();
        assert_eq!(plane.iter().filter(|&&v| v > 0.0).count(), map.covered());
    }

    #[test]
    fn a_clean_frame_has_no_spots_and_a_speck_is_noise() {
        let map = detect(&luma(&dust_frame(300, 200, &[])), 300, 200, &DustParams::default());
        assert!(map.is_empty(), "{}", map.describe());
        assert_eq!(map.describe(), "no dust spots");
        // a 1-px speck: below min_area
        let mut f = dust_frame(300, 200, &[]);
        f.p[0][100 * 300 + 150] = 0.0;
        f.p[1][100 * 300 + 150] = 0.0;
        f.p[2][100 * 300 + 150] = 0.0;
        let map = detect(&luma(&f), 300, 200, &DustParams { min_area: 16, ..DustParams::default() });
        assert!(map.is_empty(), "{}", map.describe());
    }

    #[test]
    fn a_steep_falloff_towards_an_edge_is_not_dust() {
        // lit from the left: 0.9 down to 0.55 at the right edge, 0.1 % per pixel, and a spot near that edge
        let (w, h) = (400, 300);
        let mut im = dust_frame(w, h, &[(370.0, 150.0, 10.0, 0.3)]);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    im.p[c][y * w + x] *= 1.0 - 0.4 * x as f32 / w as f32;
                }
            }
        }
        let map = detect(&luma(&im), w, h, &DustParams::default());
        assert_eq!(map.spots.len(), 1, "{}", map.describe());
        assert_eq!(map.rejected, 0);
        assert!(map.spots[0].x > 340 && map.spots[0].x + map.spots[0].w < 400);
    }

    #[test]
    fn a_blob_wider_than_a_quarter_of_the_frame_is_not_dust() {
        let map = detect(&luma(&dust_frame(400, 300, &[(200.0, 150.0, 60.0, 0.5)])), 400, 300, &DustParams::default());
        assert_eq!(map.rejected, 1);
        assert!(map.is_empty());
    }

    #[test]
    fn fill_restores_a_smooth_scene_and_flat_restores_a_textured_one() {
        let (w, h) = (400, 300);
        let spots = [(100.0, 80.0, 12.0, 0.4), (300.0, 200.0, 20.0, 0.15)];
        let map = detect(&luma(&dust_frame(w, h, &spots)), w, h, &DustParams::default());
        assert_eq!(map.spots.len(), 2);
        // the dust on a smooth scene: the fill brings it back to within a few counts
        let clean = scene(w, h);
        let mut dusty = clean.clone();
        let shadow = dust_frame(w, h, &spots);
        let flat = dust_frame(w, h, &[]);
        for c in 0..3 {
            for i in 0..w * h {
                dusty.p[c][i] *= shadow.p[c][i] / flat.p[c][i];
            }
        }
        let err = |a: &Img3, b: &Img3| a.p.iter().zip(&b.p).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs())).fold(0f32, f32::max);
        let before = err(&dusty, &clean);
        assert!(before > 0.1, "the dust darkens the scene by {before}");
        let mut filled = dusty.clone();
        map.apply(&mut filled);
        let after = err(&filled, &clean);
        // (the pull-push patch is smooth: what it cannot follow is the scene's curvature across the hole)
        assert!(after < 0.07 && after < before / 3.0, "fill: max error {after} (was {before})");
        // flat-field: the same frame divided by the attenuation, closer still under the spot
        let mut fmap = map.clone();
        fmap.params.mode = DustMode::Flat;
        let mut flattened = dusty.clone();
        fmap.apply(&mut flattened);
        let after_flat = err(&flattened, &clean);
        assert!(after_flat < 0.02, "flat: max error {after_flat}");
        // the interleaved u16 path agrees with the planar one
        let mut rgb = vec![0u16; w * h * 3];
        for i in 0..w * h {
            for c in 0..3 {
                rgb[3 * i + c] = (dusty.p[c][i] * 65535.0 + 0.5) as u16;
            }
        }
        map.apply_rgb16(&mut rgb);
        let mut worst = 0f32;
        for i in 0..w * h {
            for c in 0..3 {
                worst = worst.max((rgb[3 * i + c] as f32 / 65535.0 - filled.p[c][i]).abs());
            }
        }
        assert!(worst < 2.0 / 65535.0, "u16 vs planar: {worst}");
        // a frame of another size is left alone
        let mut other = scene(200, 100);
        let copy = other.clone();
        map.apply(&mut other);
        assert!(err(&other, &copy) == 0.0);
    }

    #[test]
    fn pull_push_interpolates_a_hole() {
        let (w, h) = (32, 24);
        let v: Vec<f32> = (0..w * h).map(|i| 0.2 + 0.5 * (i % w) as f32 / w as f32).collect();   // a ramp
        let wt: Vec<f32> = (0..w * h).map(|i| { let (x, y) = (i % w, i / w); if (10..16).contains(&x) && (8..14).contains(&y) { 0.0 } else { 1.0 } }).collect();
        let f = pull_push(&v, &wt, w, h);
        for i in 0..w * h {
            if wt[i] > 0.0 {
                assert!((f[i] - v[i]).abs() < 1e-5);
            } else {
                assert!((f[i] - v[i]).abs() < 0.02, "hole pixel {i}: {} vs {}", f[i], v[i]);
            }
        }
        assert_eq!(DustMode::parse("flat"), Some(DustMode::Flat));
        assert_eq!(DustMode::parse("x"), None);
    }
}

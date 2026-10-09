// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

//! Alignment — similarity, affine or projective registration
//! (`AlignModel`: 4, 6 or 8 parameters). Direct intensity-based,
//! coarse-to-fine, DC-removed-RMS on luminance, one frame against the
//! previous aligned one (`multiscale_align`; the chaining to frame 0 is
//! `stack::AlignedFrames`, which streams the frames, on the CUDA device
//! with `gpu::GpuFrames`). Uses a bounded
//! Nelder-Mead optimiser. The search resamples with Spline4x4; the aligned
//! frames are resampled with the kernel of the user's choice (`Interp`,
//! Spline4x4 by default).
//!
//! The registration search runs on its own Gaussian pyramid (Burt's
//! generating kernel with a = 0.33, border-renormalised, halved while
//! `h > 64 && w > 8`), independent of the fusion pyramid in `pyramid.rs`.
//! At each level the simplex starts one pixel of that level wide and stops
//! at a tenth of one (`level_steps`: the same schedule drives the CUDA and
//! WebGPU searches), so the coarse levels only hand the next one a start
//! within its pixel and the finest level searched settles sub-pixel; every
//! cost evaluation is one row-parallel pass that warps and reduces at once
//! (`warp_cost`), nothing allocated.

use crate::pyramid::{Img3, for_rows};
use rayon::prelude::*;

/// Where the fusion, the aligner's cost search and the depth pass run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    Cpu,
    /// `gpu.rs`: CUDA (the `gpu` build feature).
    Cuda,
    /// `wg::engine`: the browser app's kernels on Vulkan, Metal or DX12 (the `wgpu` build feature).
    Wgpu,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Cuda => "CUDA",
            Backend::Wgpu => "wgpu",
        }
    }
    pub fn is_gpu(self) -> bool {
        self != Backend::Cpu
    }
}

#[derive(Clone, Copy)]
pub struct AlignParams {
    pub shift: bool,
    pub scale: bool,
    pub rotation: bool,
    /// Similarity (the four above), affine (+ aspect, shear) or projective (+ perspective).
    pub model: AlignModel,
    /// Skip the N finest pyramid levels during the fit (0 = full res).
    pub coarsen: usize,
    /// Where the cost search's evaluations run.
    pub backend: Backend,
    /// The kernel the aligned frames are resampled with.
    pub interp: Interp,
}

impl Default for AlignParams {
    fn default() -> AlignParams {
        AlignParams { shift: true, scale: true, rotation: true, model: AlignModel::default(), coarsen: 0, backend: Backend::Cpu, interp: Interp::default() }
    }
}

impl AlignParams {
    /// Which of `Sim`'s parameters the search may move.
    pub fn free(&self) -> [bool; Sim::N] {
        free_mask(self.shift, self.scale, self.rotation, self.model)
    }
}

/// The transform searched for: how much a frame may be deformed to land on
/// the previous one. Similarity is the usual model (and what a focus
/// rail or a focus ring produces: the image breathes, shifts, turns a
/// little). Affine adds an aspect ratio and a shear; projective the two
/// perspective terms, for a stack whose camera tilted against the subject as
/// it stepped (keystone). More parameters take longer to search, and on a
/// stack that needs none, the extra ones only fit noise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AlignModel {
    #[default]
    Similarity,
    Affine,
    Projective,
}

impl AlignModel {
    pub fn parse(s: &str) -> Option<AlignModel> {
        match s.trim().to_ascii_lowercase().as_str() {
            "similarity" | "sim" => Some(AlignModel::Similarity),
            "affine" => Some(AlignModel::Affine),
            "projective" | "perspective" | "homography" => Some(AlignModel::Projective),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            AlignModel::Similarity => "similarity",
            AlignModel::Affine => "affine",
            AlignModel::Projective => "projective",
        }
    }
}

/// The parameters of `Sim` a search over `model` may move: x and y shift,
/// scale, rotation, then aspect and shear (affine and up), then the two
/// perspective terms (projective).
pub fn free_mask(shift: bool, scale: bool, rotation: bool, model: AlignModel) -> [bool; Sim::N] {
    let affine = model != AlignModel::Similarity;
    let projective = model == AlignModel::Projective;
    [shift, shift, scale, rotation, affine, affine, projective, projective]
}

/// The interpolation kernel of the warp: how an aligned frame's pixel is read
/// from between its source's. All separable, all interpolating (a pixel-centered
/// sample comes back as it is; the sum of the weights is 1 — Lanczos's are
/// normalized to make it so). The wider the kernel, the sharper the fine
/// detail survives a fractional shift, and the more the noise and the
/// ringing at hard edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interp {
    /// The nearest source pixel: no blur, no ringing, and jagged sub-pixel
    /// shifts. For test renders and stacks already aligned to the pixel.
    Nearest,
    /// The 2×2 linear blend: soft.
    Bilinear,
    /// Keys' cubic convolution (a = −0.5), 4×4.
    Bicubic,
    /// Panorama Tools' spline16, 4×4: our default.
    #[default]
    Spline4x4,
    /// Panorama Tools' spline36, 6×6: sharper, a little ringing.
    Spline6x6,
    /// Lanczos, 3 lobes, 6×6: the sharpest, and the most ringing.
    Lanczos3,
}

impl Interp {
    pub const ALL: [Interp; 6] = [Interp::Nearest, Interp::Bilinear, Interp::Bicubic, Interp::Spline4x4, Interp::Spline6x6, Interp::Lanczos3];

    pub fn parse(s: &str) -> Option<Interp> {
        match s.trim().to_ascii_lowercase().as_str() {
            "nearest" | "nn" => Some(Interp::Nearest),
            "bilinear" | "linear" => Some(Interp::Bilinear),
            "bicubic" | "cubic" => Some(Interp::Bicubic),
            "spline4x4" | "spline16" => Some(Interp::Spline4x4),
            "spline6x6" | "spline36" => Some(Interp::Spline6x6),
            "lanczos3" | "lanczos" => Some(Interp::Lanczos3),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Interp::Nearest => "nearest",
            Interp::Bilinear => "bilinear",
            Interp::Bicubic => "bicubic",
            Interp::Spline4x4 => "spline4x4",
            Interp::Spline6x6 => "spline6x6",
            Interp::Lanczos3 => "lanczos3",
        }
    }

    /// The kernel's number in the GPU warp kernels (`p.klen` of the browser's
    /// `warp`): the order of `ALL`.
    pub fn id(self) -> u32 {
        Interp::ALL.iter().position(|k| *k == self).unwrap() as u32
    }

    /// Taps along each axis.
    pub fn taps(self) -> usize {
        match self {
            Interp::Nearest => 1,
            Interp::Bilinear => 2,
            Interp::Bicubic | Interp::Spline4x4 => 4,
            Interp::Spline6x6 | Interp::Lanczos3 => 6,
        }
    }

    /// The kernel's support each way, in source pixels: a destination pixel
    /// is only sound when its source point lies this far inside the frame
    /// (outside, the warp repeats the edge).
    pub fn margin(self) -> f64 {
        (self.taps() / 2) as f64
    }
}

/// A frame's transform onto the previous aligned frame, about the frame's
/// center: a similarity (shift, scale, rotation); with `aspect` and `shear`
/// an affine transform; with `px` and `py` a projective one (`AlignModel`).
/// Frame 0 sits at `id()`. `matrix` gives it as a 3×3 homography.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Sim {
    pub xoff: f64, // fraction of width
    pub yoff: f64, // fraction of height
    pub scale: f64,
    pub rot: f64, // radians
    /// Vertical over horizontal scale (1 = isotropic).
    pub aspect: f64,
    /// Horizontal shear per unit of height (0 = none).
    pub shear: f64,
    /// Perspective: the divisor 1 + px·u/w + py·v/h over the centered pixel (u, v).
    pub px: f64,
    pub py: f64,
}

impl Sim {
    /// The number of parameters (the length of `as_vec`).
    pub const N: usize = 8;
    /// The search box's half-width around a guess, per parameter: 10 % of
    /// the frame in shift, 10 % in scale, 5° in rotation, 5 % in aspect and
    /// shear, 5 % in each perspective term.
    pub const SPAN: [f64; Sim::N] = [0.10, 0.10, 0.10, 0.087266462599716474, 0.05, 0.05, 0.05, 0.05];

    pub fn id() -> Sim {
        Sim { xoff: 0.0, yoff: 0.0, scale: 1.0, rot: 0.0, aspect: 1.0, shear: 0.0, px: 0.0, py: 0.0 }
    }
    pub fn from_vec(v: &[f64]) -> Sim {
        Sim { xoff: v[0], yoff: v[1], scale: v[2], rot: v[3], aspect: v[4], shear: v[5], px: v[6], py: v[7] }
    }
    pub fn as_vec(&self) -> [f64; Sim::N] {
        [self.xoff, self.yoff, self.scale, self.rot, self.aspect, self.shear, self.px, self.py]
    }
    /// No perspective: the matrix is affine.
    pub fn is_affine(&self) -> bool {
        self.px == 0.0 && self.py == 0.0
    }

    /// Forward 3×3 homography mapping SOURCE(target) -> REFERENCE pixel
    /// coordinates: (x, y) ↦ (X/W, Y/W) with [X, Y, W]ᵀ = M·[x, y, 1]ᵀ. The
    /// linear part is scale · R(rot) · [[1, shear], [0, aspect]] about the
    /// center, the perspective terms act on the centered pixel; without them
    /// the last row is exactly [0, 0, 1].
    pub fn matrix(&self, w: usize, h: usize) -> [[f64; 3]; 3] {
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let (c, s) = (self.rot.cos(), self.rot.sin());
        let sc = self.scale;
        let (a, b, d, e) = (sc * c, sc * (c * self.shear - s * self.aspect), sc * s, sc * (s * self.shear + c * self.aspect));
        let (gx, gy) = (self.px / w as f64, self.py / h as f64);
        // T(c) · [[a, b, t], [d, e, t'], [gx, gy, 1]] · T(−c), written out
        let (a2, b2, d2, e2) = (a + cx * gx, b + cx * gy, d + cy * gx, e + cy * gy);
        let tx = cx + self.xoff * w as f64 - (a2 * cx + b2 * cy);
        let ty = cy + self.yoff * h as f64 - (d2 * cx + e2 * cy);
        [[a2, b2, tx], [d2, e2, ty], [gx, gy, 1.0 - gx * cx - gy * cy]]
    }
}

/// The inverse of a 3×3 homography, scaled so its last entry is 1; an affine
/// matrix (last row [0, 0, 1]) inverts to one with the same last row, by the
/// 2×3 formulas.
pub fn inverse(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    if m[2] == [0.0, 0.0, 1.0] {
        let (a, b, tx) = (m[0][0], m[0][1], m[0][2]);
        let (d, e, ty) = (m[1][0], m[1][1], m[1][2]);
        let det = a * e - b * d;
        let (ia, ib, id, ie) = (e / det, -b / det, -d / det, a / det);
        return [[ia, ib, -(ia * tx + ib * ty)], [id, ie, -(id * tx + ie * ty)], [0.0, 0.0, 1.0]];
    }
    let c = |i: usize, j: usize| {
        // cofactor (i, j)
        let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
        let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    // adjugate: the transposed cofactors; the determinant cancels in the normalization
    let mut inv = [[0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            inv[i][j] = c(j, i);
        }
    }
    let k = inv[2][2];
    for row in &mut inv {
        for v in row.iter_mut() {
            *v /= k;
        }
    }
    inv
}

/// The source point of destination pixel (x, y) under the (normalized)
/// inverse `inv`: the division is by exactly 1 when `inv` is affine.
#[inline]
pub fn map(inv: &[[f64; 3]; 3], x: f64, y: f64) -> (f64, f64) {
    let d = inv[2][0] * x + inv[2][1] * y + inv[2][2];
    ((inv[0][0] * x + inv[0][1] * y + inv[0][2]) / d, (inv[1][0] * x + inv[1][1] * y + inv[1][2]) / d)
}

/// `inv` for an output moved `dx` pixels to the right: its source is taken
/// `dx` to the left (inv · T(−dx)), normalized again.
pub fn shifted(inv: [[f64; 3]; 3], dx: f64) -> [[f64; 3]; 3] {
    let mut m = inv;
    for row in &mut m {
        row[2] -= row[0] * dx;
    }
    let k = m[2][2];
    for row in &mut m {
        for v in row.iter_mut() {
            *v /= k;
        }
    }
    m
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

/// The largest axis-aligned rectangle every aligned frame covers with real
/// pixels: the area to crop the result to, so no smeared border of any warped
/// frame is left in it. For each frame, a destination pixel is covered when its
/// source point (the frame's inverse transform of it) lies the kernel's
/// support (`interp.margin()`) inside the source; that is a convex
/// quadrilateral (a projective image of a rectangle too), and its cut with a pixel row is
/// one interval, so each row's common interval is an intersection over frames
/// and the best rectangle is the largest one spanning consecutive rows. Frames
/// at the identity are sampled straight and cover everything. Rows are pixel
/// centers, rectangles pixel-aligned; the full frame comes back when nothing
/// is cut. The whole frame when the frames leave no common area.
pub fn common_area(sims: &[Sim], w: usize, h: usize, interp: Interp) -> Rect {
    let full = Rect::full(w, h);
    if w == 0 || h == 0 {
        return full;
    }
    let invs: Vec<[[f64; 3]; 3]> = sims.iter().filter(|s| **s != Sim::id()).map(|s| inverse(s.matrix(w, h))).collect();
    if invs.is_empty() {
        return full;
    }
    let m = interp.margin();
    let (xmax, ymax) = ((w - 1) as f64 - m, (h - 1) as f64 - m);
    // per row, the x-interval [lo, hi] (inclusive pixel columns) covered by every frame
    let mut rows: Vec<(i64, i64)> = Vec::with_capacity(h);
    for y in 0..h {
        let (mut lo, mut hi) = (0f64, (w - 1) as f64);
        let yf = y as f64;
        for inv in &invs {
            // the homography's divisor over the row, g·x + d: positive across the
            // frame (it is 1 for an affine transform), or the row is given up
            let (g, d) = (inv[2][0], inv[2][1] * yf + inv[2][2]);
            if d <= 0.0 || g * (w - 1) as f64 + d <= 0.0 {
                lo = 1.0;
                hi = 0.0;
                break;
            }
            // source x and y are rational in the column, s = (a·x + k) / (g·x + d), so
            // s ≥ smin ⇔ (a − smin·g)·x + (k − smin·d) ≥ 0, and s ≤ smax alike
            for (a, k, smin, smax) in [(inv[0][0], inv[0][1] * yf + inv[0][2], m, xmax), (inv[1][0], inv[1][1] * yf + inv[1][2], m, ymax)] {
                for (p, q) in [(a - smin * g, k - smin * d), (smax * g - a, smax * d - k)] {
                    if p.abs() < 1e-12 {
                        if q < 0.0 { lo = 1.0; hi = 0.0; }
                    } else if p > 0.0 {
                        lo = lo.max(-q / p);
                    } else {
                        hi = hi.min(-q / p);
                    }
                }
            }
        }
        rows.push(if lo <= hi { (lo.ceil() as i64, hi.floor() as i64) } else { (1, 0) });
    }
    // the largest rectangle over consecutive rows: for each top row, extend downward
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

/// Panorama Tools' spline16 (the 4×4 spline kernel): the 4 weights at
/// fraction `t`, taps at −1, 0, +1, +2 from the floor.
#[inline]
fn spline4(t: f64) -> [f64; 4] {
    [
        ((-1.0 / 3.0 * t + 0.8) * t - 0.46666667) * t,
        ((t - 1.8) * t - 0.2) * t + 1.0,
        ((1.2 - t) * t + 0.8) * t,
        ((1.0 / 3.0 * t - 0.2) * t - 0.13333334) * t,
    ]
}

/// Keys' cubic convolution, a = −0.5, at distance `d`.
#[inline]
fn keys(d: f64) -> f64 {
    if d < 1.0 { (1.5 * d - 2.5) * d * d + 1.0 } else if d < 2.0 { ((-0.5 * d + 2.5) * d - 4.0) * d + 2.0 } else { 0.0 }
}

/// Panorama Tools' spline36 at distance `d`.
#[inline]
fn spline36(d: f64) -> f64 {
    if d < 1.0 {
        ((13.0 / 11.0 * d - 453.0 / 209.0) * d - 3.0 / 209.0) * d + 1.0
    } else if d < 2.0 {
        let u = d - 1.0;
        ((-6.0 / 11.0 * u + 270.0 / 209.0) * u - 156.0 / 209.0) * u
    } else if d < 3.0 {
        let u = d - 2.0;
        ((1.0 / 11.0 * u - 45.0 / 209.0) * u + 26.0 / 209.0) * u
    } else {
        0.0
    }
}

/// The 3-lobe Lanczos window at distance `d` (not yet normalized).
#[inline]
fn lanczos3(d: f64) -> f64 {
    if d < 1e-9 {
        1.0
    } else if d < 3.0 {
        let (a, b) = (std::f64::consts::PI * d, std::f64::consts::PI * d / 3.0);
        a.sin() / a * (b.sin() / b)
    } else {
        0.0
    }
}

/// The weights of an `N`-tap kernel `k(d)` at fraction `t`: taps at
/// `1 − N/2 ..` from the floor, normalized to sum 1.
#[inline]
fn taps_of<const N: usize>(k: impl Fn(f64) -> f64, t: f64) -> [f64; N] {
    let mut w = [0f64; N];
    let mut sum = 0.0;
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = k((t - (i as f64 + 1.0 - (N / 2) as f64)).abs());
        sum += *wi;
    }
    for wi in &mut w {
        *wi /= sum;
    }
    w
}

/// Warp a single plane by `sim` into (ow x oh) with the `N`-tap kernel
/// `weights` (taps at `1 − N/2 ..` from the floor of the source point);
/// edge-clamped, with a validity mask (1 where the source point is inside the
/// frame).
/// The source points along output row `y` under `inv`: `(x0, y0, dx, dy)`
/// such that pixel `x` reads `(x0 + dx·x, y0 + dy·x)` — the case of an affine
/// `inv`, whose divisor is exactly 1. `None` for a projective one, which
/// `map` handles pixel by pixel.
#[inline]
fn affine_row(inv: &[[f64; 3]; 3], y: f64) -> Option<(f64, f64, f64, f64)> {
    (inv[2] == [0.0, 0.0, 1.0]).then(|| (inv[0][1] * y + inv[0][2], inv[1][1] * y + inv[1][2], inv[0][0], inv[1][0]))
}

fn warp_with<const N: usize>(
    src: &[f32], w: usize, h: usize, sim: &Sim, ow: usize, oh: usize, weights: impl Fn(f64) -> [f64; N] + Sync,
) -> (Vec<f32>, Vec<u8>) {
    let inv = inverse(sim.matrix(w, h));
    let start = 1 - (N / 2) as isize;
    let (xmax, ymax) = ((w - 1) as f64, (h - 1) as f64);
    let mut out = vec![0f32; ow * oh];
    let mut valid = vec![0u8; ow * oh];
    out.par_chunks_mut(ow).zip(valid.par_chunks_mut(ow)).enumerate().for_each(|(y, (orow, vrow))| {
        let yf = y as f64;
        let aff = affine_row(&inv, yf);
        for x in 0..ow {
            let (sx, sy) = match aff {
                Some((x0, y0, dx, dy)) => (x0 + dx * x as f64, y0 + dy * x as f64),
                None => map(&inv, x as f64, yf),
            };
            vrow[x] = (sx >= 0.0 && sx <= xmax && sy >= 0.0 && sy <= ymax) as u8;
            let x0 = sx.floor() as isize;
            let y0 = sy.floor() as isize;
            let wx = weights(sx - x0 as f64).map(|v| v as f32);
            let wy = weights(sy - y0 as f64).map(|v| v as f32);
            let mut acc = 0f32;
            let (xs, ys) = (x0 + start, y0 + start);
            if xs >= 0 && xs + N as isize <= w as isize && ys >= 0 && ys + N as isize <= h as isize {
                // the footprint lies inside the frame: no clamps
                let (xs, ys) = (xs as usize, ys as usize);
                for j in 0..N {
                    let s = &src[(ys + j) * w + xs..(ys + j) * w + xs + N];
                    let mut r = 0f32;
                    for i in 0..N {
                        r += wx[i] * s[i];
                    }
                    acc += wy[j] * r;
                }
            } else {
                for j in 0..N {
                    let yy = (ys + j as isize).clamp(0, h as isize - 1) as usize;
                    let mut r = 0f32;
                    for i in 0..N {
                        let xx = (xs + i as isize).clamp(0, w as isize - 1) as usize;
                        r += wx[i] * src[yy * w + xx];
                    }
                    acc += wy[j] * r;
                }
            }
            orow[x] = acc;
        }
    });
    (out, valid)
}

/// Warp a single plane by `sim` into (ow x oh) with the kernel `interp`;
/// edge-clamped, with a validity mask.
pub fn warp_plane(src: &[f32], w: usize, h: usize, sim: &Sim, ow: usize, oh: usize, interp: Interp) -> (Vec<f32>, Vec<u8>) {
    match interp {
        // the nearer of the two neighbors, in the bilinear frame
        Interp::Nearest => warp_with::<2>(src, w, h, sim, ow, oh, |t| if t < 0.5 { [1.0, 0.0] } else { [0.0, 1.0] }),
        Interp::Bilinear => warp_with::<2>(src, w, h, sim, ow, oh, |t| [1.0 - t, t]),
        Interp::Bicubic => warp_with::<4>(src, w, h, sim, ow, oh, |t| taps_of::<4>(keys, t)),
        Interp::Spline4x4 => warp_with::<4>(src, w, h, sim, ow, oh, spline4),
        Interp::Spline6x6 => warp_with::<6>(src, w, h, sim, ow, oh, |t| taps_of::<6>(spline36, t)),
        Interp::Lanczos3 => warp_with::<6>(src, w, h, sim, ow, oh, |t| taps_of::<6>(lanczos3, t)),
    }
}

pub fn warp_img3(im: &Img3, sim: &Sim, interp: Interp) -> (Img3, Vec<u8>) {
    let (a, valid) = warp_plane(&im.p[0], im.w, im.h, sim, im.w, im.h, interp);
    let (b, _) = warp_plane(&im.p[1], im.w, im.h, sim, im.w, im.h, interp);
    let (c, _) = warp_plane(&im.p[2], im.w, im.h, sim, im.w, im.h, interp);
    (Img3 { w: im.w, h: im.h, p: [a, b, c] }, valid)
}

/// BT.601 luma plane of an RGB image.
pub fn luma(im: &Img3) -> Vec<f32> {
    let mut y = vec![0f32; im.w * im.h];
    let w = im.w.max(1);
    y.par_chunks_mut(w).zip(im.p[0].par_chunks(w)).zip(im.p[1].par_chunks(w)).zip(im.p[2].par_chunks(w)).for_each(|(((o, r), g), b)| {
        for i in 0..o.len() {
            o[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
        }
    });
    y
}

/// The search's objective: `tgt` (tw × th) warped by `sim` onto `rf`'s grid
/// (aw wide) with Spline4x4, and the RMS of the difference with its mean
/// removed, over the pixels whose source point lies inside `tgt`. One pass,
/// row-parallel, nothing allocated: each row folds Σd, Σd² and the count of
/// d = ref − warped into three sums, and rms = √((Σd² − (Σd)²/n) / n) — the
/// CUDA and WGSL `cost` kernels reduce the same way. Coordinates in f64 (an
/// f32 source point is ~0.001 px off at 8K), taps and weights in f32. 1e9
/// when fewer than 16 pixels overlap.
fn warp_cost(rf: &[f32], aw: usize, tgt: &[f32], tw: usize, th: usize, sim: &Sim) -> f64 {
    let inv = inverse(sim.matrix(tw, th));
    let (xmax, ymax) = ((tw - 1) as f64, (th - 1) as f64);
    let (sd, sd2, cnt) = rf
        .par_chunks(aw)
        .enumerate()
        .map(|(y, row)| {
            let (mut sd, mut sd2, mut cnt) = (0f64, 0f64, 0usize);
            let yf = y as f64;
            let aff = affine_row(&inv, yf);
            for (x, &r) in row.iter().enumerate() {
                let (sx, sy) = match aff {
                    Some((x0, y0, dx, dy)) => (x0 + dx * x as f64, y0 + dy * x as f64),
                    None => map(&inv, x as f64, yf),
                };
                if !(sx >= 0.0 && sx <= xmax && sy >= 0.0 && sy <= ymax) {
                    continue;
                }
                let x0 = sx.floor() as isize;
                let y0 = sy.floor() as isize;
                let wx = spline4f((sx - x0 as f64) as f32);
                let wy = spline4f((sy - y0 as f64) as f32);
                let mut acc = 0f32;
                if x0 >= 1 && x0 + 2 < tw as isize && y0 >= 1 && y0 + 2 < th as isize {
                    // the 4×4 footprint lies inside the frame: no clamps
                    let (x0, y0) = (x0 as usize, y0 as usize);
                    for j in 0..4 {
                        let s = &tgt[(y0 + j - 1) * tw + x0 - 1..(y0 + j - 1) * tw + x0 + 3];
                        acc += wy[j] * (wx[0] * s[0] + wx[1] * s[1] + wx[2] * s[2] + wx[3] * s[3]);
                    }
                } else {
                    for j in 0..4 {
                        let yy = (y0 + j as isize - 1).clamp(0, th as isize - 1) as usize;
                        let mut r = 0f32;
                        for i in 0..4 {
                            let xx = (x0 + i as isize - 1).clamp(0, tw as isize - 1) as usize;
                            r += wx[i] * tgt[yy * tw + xx];
                        }
                        acc += wy[j] * r;
                    }
                }
                let d = (r - acc) as f64;
                sd += d;
                sd2 += d * d;
                cnt += 1;
            }
            (sd, sd2, cnt)
        })
        .reduce(|| (0.0, 0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    if cnt < 16 {
        return 1e9;
    }
    let n = cnt as f64;
    ((sd2 - sd * sd / n) / n).max(0.0).sqrt()
}

/// `spline4` in f32, for the cost's inner loop.
#[inline]
fn spline4f(t: f32) -> [f32; 4] {
    [
        ((-1.0 / 3.0 * t + 0.8) * t - 0.46666667) * t,
        ((t - 1.8) * t - 0.2) * t + 1.0,
        ((1.2 - t) * t + 0.8) * t,
        ((1.0 / 3.0 * t - 0.2) * t - 0.13333334) * t,
    ]
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
    // the column norms on the output grid, and the output columns whose 5 taps all lie inside
    let cn2: Vec<f32> = (0..ow).map(|oj| cn[2 * oj]).collect();
    let (j0, j1) = (1.min(ow), if w >= 5 { (w - 3) / 2 + 1 } else { 0 });
    // horizontal pass -> htmp (h x ow), unnormalized
    let mut htmp = vec![0f32; h * ow];
    for_rows(&mut htmp, ow, |y, row| {
        let s = &src[y * w..(y + 1) * w];
        let edge = |oj: usize| {
            let cx = 2 * oj;
            let mut a = 0.0;
            for t in 0..5 {
                let i = cx as isize + t as isize - 2;
                if i >= 0 && (i as usize) < w {
                    a += k[t] * s[i as usize];
                }
            }
            a
        };
        for oj in (0..j0).chain(j1.max(j0)..ow) {
            row[oj] = edge(oj);
        }
        for oj in j0..j1 {
            let c = 2 * oj;
            let mut a = 0.0;
            a += k[0] * s[c - 2];
            a += k[1] * s[c - 1];
            a += k[2] * s[c];
            a += k[3] * s[c + 1];
            a += k[4] * s[c + 2];
            row[oj] = a;
        }
    });
    // vertical pass -> out (oh x ow), normalized by rn*cn
    let mut out = vec![0f32; oh * ow];
    let htmp_ref = &htmp;
    for_rows(&mut out, ow, |oi, row| {
        let cy = 2 * oi;
        let r = rn[cy];
        row.fill(0.0);
        for t in 0..5 {
            let s = cy as isize + t as isize - 2;
            if s >= 0 && (s as usize) < h {
                let kt = k[t];
                let hrow = &htmp_ref[s as usize * ow..(s as usize + 1) * ow];
                for oj in 0..ow {
                    row[oj] += kt * hrow[oj];
                }
            }
        }
        for oj in 0..ow {
            row[oj] /= r * cn2[oj];
        }
    });
    (out, ow, oh)
}

pub type Lvl = (Vec<f32>, usize, usize);

/// The registration pyramid's levels above the full-resolution plane `y`:
/// `[0]` is half size, and so on while `h > 64 && w > 8`. `level` reads any
/// level, the plane itself as level 0.
pub fn gauss_levels(y: &[f32], w: usize, h: usize) -> Vec<Lvl> {
    let mut levels: Vec<Lvl> = Vec::new();
    let (mut cw, mut ch) = (w, h);
    while ch > 64 && cw > 8 {
        let (d, ow, oh) = match levels.last() {
            None => reduce_burt(y, cw, ch),
            Some((p, _, _)) => reduce_burt(p, cw, ch),
        };
        cw = ow;
        ch = oh;
        levels.push((d, ow, oh));
    }
    levels
}

/// Level `l` of `y`'s pyramid: the plane itself at 0, else `levels[l − 1]`.
pub fn level<'a>(y: &'a [f32], w: usize, h: usize, levels: &'a [Lvl], l: usize) -> (&'a [f32], usize, usize) {
    if l == 0 { (y, w, h) } else { (&levels[l - 1].0, levels[l - 1].1, levels[l - 1].2) }
}

/// Full-resolution pixels a point at the frame's edge moves per unit of each
/// of `Sim`'s parameters (the worst case over the frame): the width and
/// height for the shifts, the half-frame for scale and rotation, the
/// half-height for aspect and shear, a quarter-frame for the perspective
/// terms (their divisor changes by px/2 at the edge, moving it by w/4 · px).
pub fn param_gain(w: usize, h: usize) -> [f64; Sim::N] {
    let (w, h) = (w as f64, h as f64);
    let r = w.max(h) / 2.0;
    [w, h, r, r, h / 2.0, h / 2.0, w / 4.0, h / 4.0]
}

/// The simplex's first step at a level: one pixel of that level.
pub const STEP_PX: f64 = 1.0;
/// The simplex's stopping size at a level: a tenth of a pixel of that level,
/// so the search at the finest level it runs on is sub-pixel there, and the
/// coarser levels only hand the next one a start within its pixel.
pub const TOL_PX: f64 = 0.1;

/// The search's schedule at pyramid level `lvl` (a pixel there is `2^lvl`
/// full-resolution pixels), over the parameters `free_idx`: the first step
/// and the stopping size of the simplex, per parameter, in the parameter's
/// units — `STEP_PX` and `TOL_PX` pixels of the level moved at the frame's
/// edge (`param_gain`). The same for the CPU, CUDA and WebGPU searches.
pub fn level_steps(free_idx: &[usize], lvl: usize, w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let gain = param_gain(w, h);
    let px = (1u64 << lvl) as f64;
    let step = free_idx.iter().map(|&k| STEP_PX * px / gain[k]).collect();
    let tol = free_idx.iter().map(|&k| TOL_PX * px / gain[k]).collect();
    (step, tol)
}

/// Bounded Nelder-Mead: minimizes `f` over the box `[lo, hi]` from `x0`,
/// the first simplex `x0` moved by `step[k]` along each axis, until the
/// vertices agree to `1e-4` relative in `f` and lie within `tol[k]` of the
/// best along every axis (or 200 iterations). `n = 1..8`.
pub fn nelder_mead<F: Fn(&[f64]) -> f64>(f: &F, x0: &[f64], lo: &[f64], hi: &[f64], step: &[f64], tol: &[f64]) -> Vec<f64> {
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
        v[k] += step[k];
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
        if converged(&simplex, &fv, tol) {
            break;
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

/// Nelder-Mead's stopping rule on a simplex sorted by `fv`: the values agree
/// to 1e-4 relative and every vertex lies within `tol[k]` of the best along
/// each axis.
pub fn converged(simplex: &[Vec<f64>], fv: &[f64], tol: &[f64]) -> bool {
    let n = tol.len();
    if (fv[n] - fv[0]).abs() > 1e-4 * (1.0 + fv[0].abs()) {
        return false;
    }
    simplex[1..].iter().all(|v| (0..n).all(|k| (v[k] - simplex[0][k]).abs() <= tol[k]))
}

pub(crate) fn multiscale_align(
    rf: &[f32],
    tg: &[f32],
    w: usize,
    h: usize,
    init: Sim,
    free: [bool; Sim::N],
    coarsen: usize,
) -> Sim {
    let pref = gauss_levels(rf, w, h);
    let ptgt = gauss_levels(tg, w, h);
    let n = pref.len().min(ptgt.len()) + 1;
    let span = Sim::SPAN;
    let iv = init.as_vec();
    let mut cur = iv;
    let free_idx: Vec<usize> = (0..Sim::N).filter(|&k| free[k]).collect();
    let lo_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] - span[k]).collect();
    let hi_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] + span[k]).collect();

    // Refine coarsest -> finest, but stop `coarsen` levels short of full res: the
    // Sim transform is resolution-independent (fractional offset + scale + angle),
    // so a fit at reduced res applies at full res — skipping the full-res warp+RMS
    // (the dominant cost) for a large speedup at sub-px accuracy. Keep >=1 level.
    // Each level's simplex starts a pixel of that level wide and stops at a tenth
    // of one (`level_steps`).
    let finest = coarsen.min(n.saturating_sub(1));
    for lvl in (finest..n).rev() {
        let (a_d, aw, _) = level(rf, w, h, &pref, lvl);
        let (t_d, tw, th) = level(tg, w, h, &ptgt, lvl);
        let cur_snap = cur;
        let cost = |xf: &[f64]| -> f64 {
            let mut v = cur_snap;
            for (k, &idx) in free_idx.iter().enumerate() {
                v[idx] = xf[k];
            }
            // the search's own kernel, whatever the frames are resampled with
            warp_cost(a_d, aw, t_d, tw, th, &Sim::from_vec(&v))
        };
        let x0: Vec<f64> = free_idx.iter().map(|&k| cur[k]).collect();
        let (step, tol) = level_steps(&free_idx, lvl, w, h);
        let best = nelder_mead(&cost, &x0, &lo_f, &hi_f, &step, &tol);
        for (k, &idx) in free_idx.iter().enumerate() {
            cur[idx] = best[k];
        }
    }
    Sim::from_vec(&cur)
}

pub fn report(sim: &Sim, w: usize, h: usize) -> String {
    let mut s = format!(
        "dx={:+7.2}px dy={:+7.2}px scale={:.5} rot={:+.3} deg",
        sim.xoff * w as f64,
        sim.yoff * h as f64,
        sim.scale,
        sim.rot.to_degrees()
    );
    if sim.aspect != 1.0 || sim.shear != 0.0 {
        s += &format!(" aspect={:.5} shear={:+.5}", sim.aspect, sim.shear);
    }
    if !sim.is_affine() {
        s += &format!(" persp={:+.5}/{:+.5}", sim.px, sim.py);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REDUCE is border-renormalised: a constant plane stays constant at every
    /// width, the edge columns and rows included (widths below the 5 taps go
    /// through the edge path only).
    #[test]
    fn reduce_keeps_a_constant_plane() {
        for (w, h) in [(1, 1), (2, 3), (3, 4), (4, 5), (5, 5), (6, 7), (9, 65), (10, 66), (33, 130)] {
            let src = vec![0.75f32; w * h];
            let (d, ow, oh) = reduce_burt(&src, w, h);
            assert_eq!((ow, oh), ((w + 1) / 2, (h + 1) / 2));
            assert!(d.iter().all(|v| (v - 0.75).abs() < 1e-6), "{w}x{h}: {d:?}");
        }
        let src = vec![0.25f32; 40 * 300];
        let lv = gauss_levels(&src, 40, 300);
        assert_eq!(lv.iter().map(|l| (l.1, l.2)).collect::<Vec<_>>(), vec![(20, 150), (10, 75), (5, 38)]);
        assert!(lv.iter().all(|l| l.0.iter().all(|v| (v - 0.25).abs() < 1e-6)));
        assert_eq!(level(&src, 40, 300, &lv, 0).1, 40);
        assert_eq!(level(&src, 40, 300, &lv, 3).1, 5);
    }

    /// The fused cost is the DC-removed RMS of the warped difference over the
    /// valid pixels, as the two-pass definition gives it.
    #[test]
    fn cost_matches_the_two_pass_definition() {
        let (w, h) = (96, 72);
        let a: Vec<f32> = (0..w * h).map(|i| ((i * 7919) % 1000) as f32 / 1000.0).collect();
        let b: Vec<f32> = (0..w * h).map(|i| ((i * 104729 + 13) % 1000) as f32 / 1000.0 + 0.2).collect();
        for s in [Sim::id(), Sim { xoff: 0.03, yoff: -0.02, scale: 1.02, rot: 0.01, ..Sim::id() }, Sim { px: 0.03, py: -0.02, shear: 0.01, ..Sim::id() }] {
            let (bw, valid) = warp_plane(&b, w, h, &s, w, h, Interp::Spline4x4);
            let (mut sa, mut sb, mut n) = (0f64, 0f64, 0usize);
            for i in 0..w * h {
                if valid[i] != 0 {
                    sa += a[i] as f64;
                    sb += bw[i] as f64;
                    n += 1;
                }
            }
            let (ma, mb) = (sa / n as f64, sb / n as f64);
            let ss: f64 = (0..w * h).filter(|&i| valid[i] != 0).map(|i| ((a[i] as f64 - ma) - (bw[i] as f64 - mb)).powi(2)).sum();
            let want = (ss / n as f64).sqrt();
            let got = warp_cost(&a, w, &b, w, h, &s);
            assert!((got - want).abs() < 1e-5 * want, "{s:?}: {got} vs {want}");
        }
        // identical planes at the identity: nothing left
        assert!(warp_cost(&a, w, &a, w, h, &Sim::id()) < 1e-9);
        // no overlap: the sentinel
        assert_eq!(warp_cost(&a, w, &a, w, h, &Sim { xoff: 2.0, ..Sim::id() }), 1e9);
    }

    /// The schedule: a level's step and tolerance move the frame's edge by
    /// `STEP_PX` and `TOL_PX` of its pixels, whatever the parameter.
    #[test]
    fn level_steps_move_the_edge_by_a_level_pixel() {
        let (w, h) = (8000, 6000);
        let all: Vec<usize> = (0..Sim::N).collect();
        let (step, tol) = level_steps(&all, 3, w, h);
        let gain = param_gain(w, h);
        for k in 0..Sim::N {
            assert!((step[k] * gain[k] - 8.0 * STEP_PX).abs() < 1e-9);
            assert!((tol[k] * gain[k] - 8.0 * TOL_PX).abs() < 1e-9);
        }
        // a unit of scale moves the far corner's edge by half the frame, of xoff by the width
        assert_eq!(step[0] * w as f64, 8.0 * STEP_PX);
        assert_eq!(step[2] * w as f64 / 2.0, 8.0 * STEP_PX);
        let (s2, _) = level_steps(&[0, 3], 0, w, h);
        assert_eq!(s2.len(), 2);
        assert_eq!(s2[0], STEP_PX / w as f64);
    }

    #[test]
    fn common_area_identity_is_full() {
        assert_eq!(common_area(&[Sim::id(), Sim::id()], 640, 480, Interp::Spline4x4), Rect::full(640, 480));
        assert_eq!(common_area(&[], 640, 480, Interp::Spline4x4), Rect::full(640, 480));
    }

    #[test]
    fn common_area_shift() {
        // frame 1 lands 10 px to the right: its left 10 columns are smeared edge, and
        // the warp's 2 px support trims the other sides
        let (w, h) = (640, 480);
        let s = Sim { xoff: 10.0 / w as f64, ..Sim::id() };
        let r = common_area(&[Sim::id(), s], w, h, Interp::Spline4x4);
        assert_eq!(r, Rect { x: 12, y: 2, w: w - 12, h: h - 4 });
        // a wider kernel trims more, a narrower one less, nearest nothing beyond the shift
        assert_eq!(common_area(&[Sim::id(), s], w, h, Interp::Lanczos3), Rect { x: 13, y: 3, w: w - 13, h: h - 6 });
        assert_eq!(common_area(&[Sim::id(), s], w, h, Interp::Bilinear), Rect { x: 11, y: 1, w: w - 11, h: h - 2 });
        assert_eq!(common_area(&[Sim::id(), s], w, h, Interp::Nearest), Rect { x: 10, y: 0, w: w - 10, h });
    }

    /// A shift of a whole pixel reads the source pixels straight, with every
    /// kernel; the identity comes back as it is.
    #[test]
    fn every_kernel_interpolates() {
        let (w, h) = (40, 30);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 7919) % 1000) as f32 / 1000.0).collect();
        for k in Interp::ALL {
            let (o, _) = warp_plane(&src, w, h, &Sim::id(), w, h, k);
            assert!(o.iter().zip(&src).all(|(a, b)| (a - b).abs() < 1e-6), "{k:?} identity");
            let s = Sim { xoff: 3.0 / w as f64, yoff: -2.0 / h as f64, ..Sim::id() };
            let (o, valid) = warp_plane(&src, w, h, &s, w, h, k);
            for y in 6..h - 6 {
                for x in 6..w - 6 {
                    // the output moved 3 right and 2 up: its (x, y) reads the source's (x − 3, y + 2)
                    assert!(valid[y * w + x] == 1);
                    assert!((o[y * w + x] - src[(y + 2) * w + x - 3]).abs() < 1e-5, "{k:?} at ({x},{y})");
                }
            }
        }
    }

    /// Every kernel reproduces a plane (partition of unity and first-order
    /// accuracy) at fractional shifts, and nearest picks the nearer neighbor.
    /// Lanczos is the exception: a windowed sinc is not first-order accurate,
    /// and a ramp comes back with a ripple of about a percent of a step.
    #[test]
    fn kernels_reproduce_a_ramp() {
        let (w, h) = (48, 40);
        let ramp: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 + 0.5 * (i / w) as f32).collect();
        let s = Sim { xoff: 2.3 / w as f64, yoff: 1.6 / h as f64, ..Sim::id() };
        for k in Interp::ALL {
            let (o, _) = warp_plane(&ramp, w, h, &s, w, h, k);
            for y in 8..h - 8 {
                for x in 8..w - 8 {
                    let (sx, sy) = (x as f32 - 2.3, y as f32 - 1.6);
                    let want = if k == Interp::Nearest { sx.round() + 0.5 * sy.round() } else { sx + 0.5 * sy };
                    let tol = if k == Interp::Lanczos3 { 0.03 } else { 1e-3 };
                    assert!((o[y * w + x] - want).abs() < tol, "{k:?} at ({x},{y}): {} vs {want}", o[y * w + x]);
                }
            }
        }
    }

    /// The weights of each kernel sum to 1 and the kernels interpolate (a
    /// pixel-centered sample takes only that pixel).
    #[test]
    fn kernel_weights() {
        for t in [0.0, 0.1, 0.5, 0.9] {
            let all: Vec<Vec<f64>> = vec![
                spline4(t).to_vec(),
                taps_of::<4>(keys, t).to_vec(),
                taps_of::<6>(spline36, t).to_vec(),
                taps_of::<6>(lanczos3, t).to_vec(),
            ];
            for ws in all {
                assert!((ws.iter().sum::<f64>() - 1.0).abs() < 1e-6, "{ws:?}");
                if t == 0.0 {
                    let c = ws.len() / 2 - 1;
                    assert!((ws[c] - 1.0).abs() < 1e-6 && ws.iter().enumerate().all(|(i, w)| i == c || w.abs() < 1e-6), "{ws:?}");
                }
            }
        }
        // spline36 unnormalised is already a partition of unity
        assert!((spline36(0.5) * 2.0 + spline36(1.5) * 2.0 + spline36(2.5) * 2.0 - 1.0).abs() < 1e-9);
        assert_eq!(Interp::parse("Spline36"), Some(Interp::Spline6x6));
        assert!(Interp::ALL.iter().all(|k| Interp::parse(k.name()) == Some(*k) && Interp::ALL[k.id() as usize] == *k));
    }

    #[test]
    fn common_area_rotation_is_inside_every_frame() {
        let (w, h) = (640, 480);
        let sims = [Sim::id(), Sim { xoff: -0.01, yoff: 0.02, scale: 1.03, rot: 0.02, ..Sim::id() }, Sim { xoff: 0.005, yoff: -0.01, scale: 0.98, rot: -0.015, ..Sim::id() }];
        let m = Interp::Spline4x4.margin();
        let r = common_area(&sims, w, h, Interp::Spline4x4);
        assert!(r.w > w / 2 && r.h > h / 2 && !r.is_full(w, h), "{r:?}");
        // every corner pixel of the rectangle maps inside every frame's sound area
        for s in &sims[1..] {
            let inv = inverse(s.matrix(w, h));
            for (x, y) in [(r.x, r.y), (r.x + r.w - 1, r.y), (r.x, r.y + r.h - 1), (r.x + r.w - 1, r.y + r.h - 1)] {
                let (sx, sy) = map(&inv, x as f64, y as f64);
                assert!(sx >= m && sx <= (w - 1) as f64 - m && sy >= m && sy <= (h - 1) as f64 - m, "({x},{y}) -> ({sx:.1},{sy:.1})");
            }
        }
        // and it is maximal: one more row or column on any side breaks that for some frame
        let sound = |x: usize, y: usize| sims[1..].iter().all(|s| {
            let inv = inverse(s.matrix(w, h));
            let (sx, sy) = map(&inv, x as f64, y as f64);
            sx >= m && sx <= (w - 1) as f64 - m && sy >= m && sy <= (h - 1) as f64 - m
        });
        let row_ok = |y: usize| (r.x..r.x + r.w).all(|x| sound(x, y));
        let col_ok = |x: usize| (r.y..r.y + r.h).all(|y| sound(x, y));
        assert!(r.y == 0 || !row_ok(r.y - 1));
        assert!(r.y + r.h == h || !row_ok(r.y + r.h));
        assert!(r.x == 0 || !col_ok(r.x - 1));
        assert!(r.x + r.w == w || !col_ok(r.x + r.w));
    }

    /// The homography and its inverse round-trip, and the affine parameters
    /// do what they say: aspect scales y, shear moves x with y, perspective
    /// keystones (a square's top wider than its bottom).
    #[test]
    fn homography_round_trip_and_meaning() {
        let (w, h) = (640, 480);
        let s = Sim { xoff: 0.01, yoff: -0.02, scale: 1.03, rot: 0.02, aspect: 1.02, shear: 0.01, px: 0.03, py: -0.02 };
        let m = s.matrix(w, h);
        let inv = inverse(m);
        assert!(!s.is_affine() && inv[2][2] == 1.0);
        for (x, y) in [(0.0, 0.0), (100.0, 37.0), (639.0, 479.0), (320.0, 240.0)] {
            let (fx, fy) = map(&m, x, y);
            let (bx, by) = map(&inv, fx, fy);
            assert!((bx - x).abs() < 1e-9 && (by - y).abs() < 1e-9, "({x},{y}) -> ({fx},{fy}) -> ({bx},{by})");
        }
        // the center stays put (up to the shift); affine ones exactly
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let a = Sim { aspect: 1.5, shear: 0.2, ..Sim::id() };
        assert_eq!(map(&a.matrix(w, h), cx, cy), (cx, cy));
        assert_eq!(map(&a.matrix(w, h), cx, cy + 10.0), (cx + 2.0, cy + 15.0));
        // py > 0 divides the top half by less than 1: the top widens, the bottom narrows
        let p = Sim { px: 0.0, py: 0.1, ..Sim::id() };
        let (top_l, _) = map(&p.matrix(w, h), cx - 100.0, cy - 100.0);
        let (bot_l, _) = map(&p.matrix(w, h), cx - 100.0, cy + 100.0);
        assert!(top_l < cx - 100.0 && bot_l > cx - 100.0, "keystone: {top_l} {bot_l}");
        // affine transforms invert by the exact 2×3 route
        assert_eq!(inverse(Sim::id().matrix(w, h)), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(shifted(inv, 0.0), inv);
        let sh = shifted(inv, 5.0);
        let (x1, y1) = map(&sh, 105.0, 37.0);
        let (x0, y0) = map(&inv, 100.0, 37.0);
        assert!((x1 - x0).abs() < 1e-9 && (y1 - y0).abs() < 1e-9);
    }

    /// A projective warp reads the source where the forward map says, and
    /// the common area of a keystoned frame is inside its sound quad.
    #[test]
    fn projective_warp_and_common_area() {
        let (w, h) = (64, 48);
        let ramp: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 + 0.5 * (i / w) as f32).collect();
        let s = Sim { px: 0.04, py: -0.03, shear: 0.01, ..Sim::id() };
        let (o, valid) = warp_plane(&ramp, w, h, &s, w, h, Interp::Bilinear);
        let inv = inverse(s.matrix(w, h));
        for y in 4..h - 4 {
            for x in 4..w - 4 {
                let (sx, sy) = map(&inv, x as f64, y as f64);
                assert_eq!(valid[y * w + x], 1);
                assert!((o[y * w + x] as f64 - (sx + 0.5 * sy)).abs() < 1e-3, "({x},{y})");
            }
        }
        let r = common_area(&[Sim::id(), s], w, h, Interp::Bilinear);
        assert!(!r.is_full(w, h) && r.w > w / 2 && r.h > h / 2, "{r:?}");
        let m = Interp::Bilinear.margin();
        for (x, y) in [(r.x, r.y), (r.x + r.w - 1, r.y), (r.x, r.y + r.h - 1), (r.x + r.w - 1, r.y + r.h - 1)] {
            let (sx, sy) = map(&inv, x as f64, y as f64);
            assert!(sx >= m && sx <= (w - 1) as f64 - m && sy >= m && sy <= (h - 1) as f64 - m, "({x},{y}) -> ({sx:.2},{sy:.2})");
        }
        assert_eq!(AlignModel::parse("perspective"), Some(AlignModel::Projective));
        assert_eq!(free_mask(true, true, false, AlignModel::Affine), [true, true, true, false, true, true, false, false]);
        assert_eq!(free_mask(true, true, true, AlignModel::Projective), [true; 8]);
    }
}

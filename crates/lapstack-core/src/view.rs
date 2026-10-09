// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

//! Synthetic stereo and rocking: views of the stacked image from a little to
//! the side.
//!
//! The depth map makes the result a relief — a textured surface over the
//! image plane — and a camera moved sideways sees that surface sheared: every
//! pixel slides horizontally in proportion to its depth. The same picture can
//! be had by shifting each *frame* by its index before stacking (the shift
//! between the two ends of the stack, a percentage of the width, is then the
//! parameter), or by projecting a textured 3D model built from the depth
//! map. Here the stacked image and its depth map are sheared in one pass,
//! with that same parameter: a [`View`] moves the far end of the stack `shift`
//! × width sideways relative to the near end, about the `pivot` depth that
//! stays put. Two views at ∓`shift` are a stereo pair; a sequence of views
//! with the shift sweeping ±A is a rocking animation.
//!
//! The shear is a forward warp per row ([`shear_row`]): consecutive samples
//! whose destinations are less than [`MAX_SPAN`] apart form a patch of the
//! surface, rasterized with a nearness test so the nearer surface wins where
//! two overlap (a foreground edge sliding over the background); a larger gap
//! is a depth discontinuity, and the hole it opens is filled from the farther
//! of its two neighbors (the background shows through, the foreground is not
//! stretched). Pixels are sampled linearly between source columns, so a view
//! is as smooth as the image.

use crate::align::Rect;
use rayon::prelude::*;

/// A view from the side: see the module docs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// The far end of the stack moves this far sideways relative to the near
    /// end, as a fraction of the image width (0.03 = 3 %). Positive
    /// is the view from the right (near things slide left), negative from
    /// the left.
    pub shift: f32,
    /// The depth that stays where it is, as a fraction of the stack
    /// (0.5 = the middle of the stack, on the "window" plane of a stereo pair).
    pub pivot: f32,
    /// Frame 0 is the near end (the focus traveled front to back). This
    /// decides which surface wins where two overlap, and which way a view
    /// turns for a given shift.
    pub near_first: bool,
}

impl View {
    pub fn new(shift: f32, near_first: bool) -> View {
        View { shift, pivot: 0.5, near_first }
    }
}

/// Destinations of two consecutive samples more than this far apart are a
/// depth discontinuity (a hole to fill), not a patch of surface to stretch.
pub const MAX_SPAN: f32 = 2.0;

/// The layouts of a stereo pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layout {
    /// left | right, for parallel (wall-eyed) viewing and stereo viewers
    SideBySide,
    /// right | left, for cross-eyed viewing
    CrossEyed,
    /// red from the left view, green and blue from the right (red–cyan glasses)
    Anaglyph,
}

impl Layout {
    pub fn parse(s: &str) -> Option<Layout> {
        match s {
            "sbs" | "parallel" => Some(Layout::SideBySide),
            "cross" | "crossed" => Some(Layout::CrossEyed),
            "anaglyph" => Some(Layout::Anaglyph),
            _ => None,
        }
    }
}

/// A pixel sample the renderer can interpolate: `u16` images (the engine's
/// masters, depth maps with 65535 = last frame) and `f32` (the native path).
pub trait Sample: Copy + Send + Sync {
    fn to_f(self) -> f32;
    fn from_f(v: f32) -> Self;
}
impl Sample for f32 {
    #[inline]
    fn to_f(self) -> f32 {
        self
    }
    #[inline]
    fn from_f(v: f32) -> f32 {
        v
    }
}
impl Sample for u16 {
    #[inline]
    fn to_f(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_f(v: f32) -> u16 {
        (v + 0.5).clamp(0.0, 65535.0) as u16
    }
}
impl Sample for u8 {
    #[inline]
    fn to_f(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_f(v: f32) -> u8 {
        (v + 0.5).clamp(0.0, 255.0) as u8
    }
}

/// The source column each pixel of one row takes in the sheared view.
///
/// `z` is the row's depth in [0, 1] (0 = frame 0); `k` the shift in pixels
/// per unit of depth (the sign decides the side the view is from); `near` is
/// +1 when a larger z is nearer and −1 when frame 0 is nearest. On return
/// `src[c]` is the fractional source column of output column `c` (every
/// column is filled); `zb` is scratch of the same length.
pub fn shear_row(z: &[f32], k: f32, pivot: f32, near: f32, src: &mut [f32], zb: &mut [f32]) {
    let w = z.len();
    src[..w].fill(f32::NAN);
    zb[..w].fill(f32::NEG_INFINITY);
    let dest = |x: usize| x as f32 + k * (z[x] - pivot);
    let mut put = |c: i64, s: f32, n: f32| {
        if c >= 0 && (c as usize) < w {
            let c = c as usize;
            if n > zb[c] {
                zb[c] = n;
                src[c] = s;
            }
        }
    };
    for x in 0..w {
        let d0 = dest(x);
        let n0 = near * z[x];
        if x + 1 < w {
            // the patch of surface to the next sample, when it is continuous and not turned away
            let d1 = dest(x + 1);
            let span = d1 - d0;
            if span > 0.0 && span <= MAX_SPAN {
                let n1 = near * z[x + 1];
                let (c0, c1) = (d0.ceil() as i64, d1.ceil() as i64); // the columns in [d0, d1)
                for c in c0..c1 {
                    let t = (c as f32 - d0) / span;
                    put(c, x as f32 + t, n0 + t * (n1 - n0));
                }
                continue;
            }
        }
        put(d0.round() as i64, x as f32, n0);
    }
    // a row whose every sample left the image (a shift wider than the row): nothing sensible to show
    if src[..w].iter().all(|v| v.is_nan()) {
        for (i, v) in src[..w].iter_mut().enumerate() {
            *v = i as f32;
        }
        return;
    }
    // holes: the farther of the two neighbors reaches across; at the row's
    // ends the one neighbor there does
    let mut c = 0;
    while c < w {
        if !src[c].is_nan() {
            c += 1;
            continue;
        }
        let start = c;
        while c < w && src[c].is_nan() {
            c += 1;
        }
        let left = (start > 0).then(|| start - 1);
        let right = (c < w).then_some(c);
        let from = match (left, right) {
            (Some(l), Some(r)) => if zb[l] <= zb[r] { l } else { r },
            (Some(l), None) => l,
            (None, Some(r)) => r,
            (None, None) => unreachable!("a row with a sample in it"),
        };
        let (s, n) = (src[from], zb[from]);
        for i in start..c {
            src[i] = s;
            zb[i] = n;
        }
    }
}

/// Render `view` of a `w`×`h` image with `ch` interleaved channels (`src`)
/// and depth `z` (same size, one value per pixel, `z * zscale` in [0, 1])
/// into the window of `out` that starts at column `x0` of rows `out_w` wide.
pub fn render<T: Sample, Z: Sample>(src: &[T], w: usize, h: usize, ch: usize, z: &[Z], zscale: f32, view: &View, out: &mut [T], out_w: usize, x0: usize) {
    assert!(src.len() >= w * h * ch && z.len() >= w * h && out.len() >= out_w * h * ch && x0 + w <= out_w);
    let k = view.shift * w as f32 * if view.near_first { 1.0 } else { -1.0 };
    let near = if view.near_first { -1.0 } else { 1.0 };
    out[..out_w * h * ch].par_chunks_mut(out_w * ch).enumerate().for_each(|(y, orow)| {
        let zr: Vec<f32> = z[y * w..(y + 1) * w].iter().map(|v| (v.to_f() * zscale).clamp(0.0, 1.0)).collect();
        let mut sx = vec![0f32; w];
        let mut zb = vec![0f32; w];
        shear_row(&zr, k, view.pivot, near, &mut sx, &mut zb);
        let srow = &src[y * w * ch..(y + 1) * w * ch];
        let orow = &mut orow[x0 * ch..(x0 + w) * ch];
        for c in 0..w {
            let s = sx[c].clamp(0.0, (w - 1) as f32);
            let i0 = s as usize;
            let i1 = (i0 + 1).min(w - 1);
            let f = s - i0 as f32;
            for k in 0..ch {
                let a = srow[i0 * ch + k].to_f();
                let b = srow[i1 * ch + k].to_f();
                orow[c * ch + k] = T::from_f(a + f * (b - a));
            }
        }
    });
}

/// One view as a new image.
pub fn render_new<T: Sample, Z: Sample>(src: &[T], w: usize, h: usize, ch: usize, z: &[Z], zscale: f32, view: &View) -> Vec<T> {
    let mut out = vec![T::from_f(0.0); w * h * ch];
    render(src, w, h, ch, z, zscale, view, &mut out, w, 0);
    out
}

/// The stereo pair of views at ∓`shift` (left, right) in `layout`:
/// (pixels, width, height). The anaglyph needs `ch == 3`.
pub fn stereo<T: Sample, Z: Sample>(src: &[T], w: usize, h: usize, ch: usize, z: &[Z], zscale: f32, shift: f32, near_first: bool, layout: Layout) -> (Vec<T>, usize, usize) {
    let left = render_new(src, w, h, ch, z, zscale, &View::new(-shift, near_first));
    let right = render_new(src, w, h, ch, z, zscale, &View::new(shift, near_first));
    compose_pair(&left, &right, w, h, ch, layout)
}

/// Two rendered views (`w`×`h`, `ch` channels) as a stereo pair in `layout`:
/// (pixels, width, height). The anaglyph needs `ch == 3`.
pub fn compose_pair<T: Sample>(left: &[T], right: &[T], w: usize, h: usize, ch: usize, layout: Layout) -> (Vec<T>, usize, usize) {
    match layout {
        Layout::SideBySide | Layout::CrossEyed => {
            let (a, b) = if layout == Layout::SideBySide { (left, right) } else { (right, left) };
            let mut out = vec![T::from_f(0.0); 2 * w * h * ch];
            out.par_chunks_mut(2 * w * ch).enumerate().for_each(|(y, row)| {
                row[..w * ch].copy_from_slice(&a[y * w * ch..(y + 1) * w * ch]);
                row[w * ch..].copy_from_slice(&b[y * w * ch..(y + 1) * w * ch]);
            });
            (out, 2 * w, h)
        }
        Layout::Anaglyph => {
            assert_eq!(ch, 3, "an anaglyph needs RGB");
            let mut out = left[..w * h * 3].to_vec();
            out.par_chunks_mut(3).zip(right[..w * h * 3].par_chunks(3)).for_each(|(o, p)| {
                o[1] = p[1];
                o[2] = p[2];
            });
            (out, w, h)
        }
    }
}

/// The shifts of one cycle of a rocking animation with `n` frames: a sine
/// sweep of ±`amplitude`, so the motion eases at the ends and the loop
/// closes on itself.
pub fn rocking_shifts(amplitude: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| amplitude * (2.0 * std::f32::consts::PI * i as f32 / n.max(1) as f32).sin()).collect()
}

/// Area-average the window `r` of a `stride`-wide image with `ch`
/// interleaved channels down to `ow`×`oh` (the views of an animation are
/// rendered at the animation's size, not shrunk after).
pub fn shrink<T: Sample>(src: &[T], stride: usize, r: &Rect, ch: usize, ow: usize, oh: usize) -> Vec<T> {
    let mut out = vec![T::from_f(0.0); ow * oh * ch];
    let bounds = |i: usize, n: usize, total: usize| -> (usize, usize) {
        let lo = i * total / n;
        let hi = ((i + 1) * total / n).max(lo + 1).min(total);
        (lo, hi)
    };
    out.par_chunks_mut(ow * ch).enumerate().for_each(|(oy, orow)| {
        let (y0, y1) = bounds(oy, oh, r.h);
        let mut acc = vec![0f32; ch];
        for ox in 0..ow {
            let (x0, x1) = bounds(ox, ow, r.w);
            acc.fill(0.0);
            for y in y0..y1 {
                let row = &src[((r.y + y) * stride + r.x + x0) * ch..((r.y + y) * stride + r.x + x1) * ch];
                for px in row.chunks_exact(ch) {
                    for k in 0..ch {
                        acc[k] += px[k].to_f();
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as f32;
            for k in 0..ch {
                orow[ox * ch + k] = T::from_f(acc[k] / n);
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: usize, h: usize) -> Vec<f32> {
        (0..w * h).map(|i| (i % w) as f32).collect()
    }

    #[test]
    fn a_flat_scene_on_the_pivot_plane_does_not_move() {
        let (w, h) = (50, 3);
        let img = ramp(w, h);
        let z = vec![0.5f32; w * h];
        let out = render_new(&img, w, h, 1, &z, 1.0, &View::new(0.1, true));
        assert_eq!(out, img);
    }

    #[test]
    fn a_flat_scene_off_the_pivot_slides_whole_and_repeats_its_edge() {
        // z = 1 (the far end), pivot 0.5, shift 10 % of 100 px, frame 0 nearest: k = +10, s = +5
        let (w, h) = (100, 2);
        let img = ramp(w, h);
        let z = vec![1f32; w * h];
        let out = render_new(&img, w, h, 1, &z, 1.0, &View::new(0.1, true));
        for c in 5..w {
            assert!((out[c] - img[c - 5]).abs() < 1e-4, "column {c}: {} vs {}", out[c], img[c - 5]);
        }
        for c in 0..5 {
            assert_eq!(out[c], img[0], "the vacated edge repeats column 0");
        }
        // the far end seen from the left slides the other way; frame 0 farthest flips it back
        let out = render_new(&img, w, h, 1, &z, 1.0, &View::new(-0.1, true));
        assert!((out[10] - img[15]).abs() < 1e-4);
        let out = render_new(&img, w, h, 1, &z, 1.0, &View::new(0.1, false));
        assert!((out[10] - img[15]).abs() < 1e-4);
    }

    #[test]
    fn the_near_surface_covers_the_far_one_and_the_hole_fills_from_the_far_side() {
        // a near block (z = 0, value 1000) on a far background (z = 1, value = column), frame 0 nearest,
        // shift 10 % of 100 px about the middle: the block slides 5 px left, the background 5 px right
        let (w, h) = (100, 1);
        let mut img = ramp(w, h);
        let mut z = vec![1f32; w];
        for x in 40..60 {
            img[x] = 1000.0;
            z[x] = 0.0;
        }
        let out = render_new(&img, w, h, 1, &z, 1.0, &View::new(0.1, true));
        // the block now covers 35..55
        for c in 35..55 {
            assert_eq!(out[c], 1000.0, "column {c} is the block");
        }
        // left of it the background slid right by 5: column 34 shows source 29
        assert!((out[34] - 29.0).abs() < 1e-4, "{}", out[34]);
        // the strip the block vacated, 55..65, is a hole: the far neighbor (background from column 60 → 65) fills it, not the block
        for c in 55..65 {
            assert_eq!(out[c], 60.0, "column {c} shows the background, got {}", out[c]);
        }
        assert!((out[65] - 60.0).abs() < 1e-4);
        assert!((out[70] - 65.0).abs() < 1e-4);
    }

    #[test]
    fn stereo_layouts() {
        let (w, h) = (20, 2);
        let img: Vec<f32> = (0..w * h * 3).map(|i| i as f32).collect();
        let z = vec![0.5f32; w * h];
        let (sbs, pw, ph) = stereo(&img, w, h, 3, &z, 1.0, 0.1, true, Layout::SideBySide);
        assert_eq!((pw, ph), (2 * w, h));
        // on the pivot plane both views are the image itself
        for y in 0..h {
            assert_eq!(&sbs[y * 2 * w * 3..y * 2 * w * 3 + w * 3], &img[y * w * 3..(y + 1) * w * 3]);
            assert_eq!(&sbs[y * 2 * w * 3 + w * 3..(y + 1) * 2 * w * 3], &img[y * w * 3..(y + 1) * w * 3]);
        }
        let (ana, aw, ah) = stereo(&img, w, h, 3, &z, 1.0, 0.1, true, Layout::Anaglyph);
        assert_eq!((aw, ah), (w, h));
        assert_eq!(ana, img);
    }

    #[test]
    fn u16_depth_scales_and_rocking_closes_its_loop() {
        let (w, h) = (30, 1);
        let img: Vec<u16> = (0..w).map(|i| (i * 1000) as u16).collect();
        let z = vec![65535u16; w];
        let out = render_new(&img, w, h, 1, &z, 1.0 / 65535.0, &View::new(0.1, true));
        assert_eq!(out[10], 8500); // k = 3 px, s = 1.5: column 10 samples source 8.5, between 8000 and 9000
        let s = rocking_shifts(0.03, 8);
        assert!((s[0]).abs() < 1e-6 && (s[2] - 0.03).abs() < 1e-6 && (s[6] + 0.03).abs() < 1e-6);
    }

    #[test]
    fn shrink_averages_boxes() {
        let (w, h) = (4, 2);
        let img: Vec<f32> = vec![0.0, 2.0, 4.0, 6.0, 0.0, 2.0, 4.0, 6.0];
        let out = shrink(&img, w, &Rect { x: 0, y: 0, w, h }, 1, 2, 1);
        assert_eq!(out, vec![1.0, 5.0]);
        let out = shrink(&img, w, &Rect { x: 2, y: 0, w: 2, h: 2 }, 1, 1, 1);
        assert_eq!(out, vec![5.0]);
    }
}

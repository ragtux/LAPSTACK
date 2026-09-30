// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

//! Frame preparation: what a decoded frame goes through before it is aligned
//! and folded, beyond the dust map — turned by quarter turns (`--rotate`),
//! brought to frame 0's size when a frame of another size slips into a stack
//! (`resize`), and block-averaged for a draft run (`--draft`).

use crate::align::{Interp, Sim, warp_plane};
use crate::pyramid::Img3;
use rayon::prelude::*;

/// `img` turned by `quarters` quarter turns clockwise (0..3), as the camera
/// held sideways would have it; `flip_h` mirrors it first (the EXIF
/// orientations with a mirror).
pub fn rotate(img: &Img3, quarters: u8, flip_h: bool) -> Img3 {
    let (w, h) = (img.w, img.h);
    let q = quarters % 4;
    if q == 0 && !flip_h {
        return img.clone();
    }
    let (ow, oh) = if q % 2 == 1 { (h, w) } else { (w, h) };
    let mut out = Img3::zeros(ow, oh);
    for c in 0..3 {
        let src = &img.p[c];
        out.p[c].par_chunks_mut(ow).enumerate().for_each(|(oy, row)| {
            for (ox, o) in row.iter_mut().enumerate() {
                // the source pixel of output (ox, oy): undo the turn, then the mirror
                let (sx, sy) = match q {
                    0 => (ox, oy),
                    1 => (oy, h - 1 - ox),   // 90° cw: output x runs down the source's rows from the bottom
                    2 => (w - 1 - ox, h - 1 - oy),
                    _ => (w - 1 - oy, ox),   // 270° cw
                };
                let sx = if flip_h { w - 1 - sx } else { sx };
                *o = src[sy * w + sx];
            }
        });
    }
    out
}

/// `img` resampled to `w × h` with the kernel `interp`, each axis scaled on
/// its own (a frame of another aspect is stretched, not cut: the alignment
/// takes care of the rest, and a stack should not have such a frame anyway).
pub fn resize(img: &Img3, w: usize, h: usize, interp: Interp) -> Img3 {
    if img.w == w && img.h == h {
        return img.clone();
    }
    // the warp maps the source through a transform about its own centre (w/2, the edge
    // between the middle pixels) onto the reference grid: scale it so the source spans the
    // output, and shift it so that pixel centres correspond — output pixel o reads source
    // (o + ½)/k − ½, which the transform gives when its offset is (k − 1)(w + 1)/2
    let (sw, sh) = (img.w as f64, img.h as f64);
    let (kx, ky) = (w as f64 / sw, h as f64 / sh);
    let sim = Sim { xoff: (kx - 1.0) * (sw + 1.0) / (2.0 * sw), yoff: (ky - 1.0) * (sh + 1.0) / (2.0 * sh), scale: kx, aspect: ky / kx, ..Sim::id() };
    let mut out = Img3::zeros(w, h);
    for c in 0..3 {
        out.p[c] = warp_plane(&img.p[c], img.w, img.h, &sim, w, h, interp).0;
    }
    out
}

/// `img` block-averaged by `2^levels` (a draft run's frames): every block of
/// `k × k` pixels becomes their mean, the ragged last row and column
/// averaging what is there. Fast, alias-free enough for a preview, and the
/// depth pass's own block mean.
pub fn reduce(img: &Img3, levels: usize) -> Img3 {
    if levels == 0 {
        return img.clone();
    }
    let k = 1usize << levels;
    let (w, h) = (img.w, img.h);
    let (ow, oh) = (w.div_ceil(k).max(1), h.div_ceil(k).max(1));
    let mut out = Img3::zeros(ow, oh);
    for c in 0..3 {
        let src = &img.p[c];
        out.p[c].par_chunks_mut(ow).enumerate().for_each(|(oy, row)| {
            let (y0, y1) = (oy * k, ((oy + 1) * k).min(h));
            for (ox, o) in row.iter_mut().enumerate() {
                let (x0, x1) = (ox * k, ((ox + 1) * k).min(w));
                let mut s = 0f64;
                for y in y0..y1 {
                    for &v in &src[y * w + x0..y * w + x1] {
                        s += v as f64;
                    }
                }
                *o = (s / ((y1 - y0) * (x1 - x0)) as f64) as f32;
            }
        });
    }
    out
}

/// Quarter turns clockwise from `--rotate`'s degrees.
pub fn quarters_of(degrees: i32) -> Option<u8> {
    match degrees.rem_euclid(360) {
        0 => Some(0),
        90 => Some(1),
        180 => Some(2),
        270 => Some(3),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: usize, h: usize) -> Img3 {
        let mut im = Img3::zeros(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    im.p[c][y * w + x] = (x + 10 * y + 100 * c) as f32;
                }
            }
        }
        im
    }

    #[test]
    fn quarter_turns_compose_and_undo() {
        let im = ramp(5, 3);
        let r1 = rotate(&im, 1, false);
        assert_eq!((r1.w, r1.h), (3, 5));
        // 90° cw: the top-left of the source becomes the top-right of the output
        assert_eq!(r1.p[0][2], im.p[0][0]);
        assert_eq!(r1.p[0][0], im.p[0][2 * 5]);   // the bottom-left of the source goes to the top-left
        let r2 = rotate(&r1, 1, false);
        assert_eq!(r2.p, rotate(&im, 2, false).p);
        let r4 = rotate(&rotate(&r2, 1, false), 1, false);
        assert_eq!(r4.p, im.p);
        let back = rotate(&r1, 3, false);
        assert_eq!(back.p, im.p);
        let m = rotate(&im, 0, true);
        assert_eq!(m.p[1][0], im.p[1][4]);
        assert_eq!(quarters_of(90), Some(1));
        assert_eq!(quarters_of(-90), Some(3));
        assert_eq!(quarters_of(45), None);
    }

    #[test]
    fn resize_keeps_a_constant_and_lands_on_the_size_asked() {
        let mut im = Img3::zeros(7, 5);
        for c in 0..3 { im.p[c].fill(0.25 * (c + 1) as f32); }
        let out = resize(&im, 13, 9, Interp::Spline4x4);
        assert_eq!((out.w, out.h), (13, 9));
        for c in 0..3 {
            assert!(out.p[c].iter().all(|&v| (v - 0.25 * (c + 1) as f32).abs() < 1e-5));
        }
        // a linear ramp is resampled onto its own line: the centre keeps its value
        let mut ramp = Img3::zeros(21, 3);
        for y in 0..3 { for x in 0..21 { ramp.p[0][y * 21 + x] = x as f32 / 20.0; } }
        let r = resize(&ramp, 41, 3, Interp::Bilinear);
        assert!((r.p[0][20] - 0.5).abs() < 1e-3, "{}", r.p[0][20]);
        assert!(r.p[0][0] < 0.05 && r.p[0][40] > 0.95);
    }

    #[test]
    fn reduce_averages_blocks_and_the_ragged_edge() {
        let im = ramp(5, 3);
        let r = reduce(&im, 1);
        assert_eq!((r.w, r.h), (3, 2));
        // block (0,0): pixels (0,0) (1,0) (0,1) (1,1) = 0, 1, 10, 11
        assert!((r.p[0][0] - 5.5).abs() < 1e-6);
        // the last column is one pixel wide: (4,0) (4,1) = 4, 14
        assert!((r.p[0][2] - 9.0).abs() < 1e-6);
        // the last row is one pixel high: (0,2) (1,2) = 20, 21
        assert!((r.p[0][3] - 20.5).abs() < 1e-6);
        assert_eq!(reduce(&im, 0).p, im.p);
    }
}

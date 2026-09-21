// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY
//
// Brightness (flicker) normalisation: every frame is brought to the brightness
// of frame 0 by one gain per channel before it is fused.
//
// Flash recycling, mains-powered lights, a shutter that is not quite repeatable
// and focus breathing all make the frames of a stack differ in exposure by a
// percent or two, and the fusion rule sees that directly: region energy grows
// with the square of the gain, so a brighter frame wins ties it should not,
// and the seams between winning frames show as patches of different tone.
//
// The correction compares means, not pixels: a frame out of focus is a blurred
// copy of the scene, and a blur leaves the mean of an area alone while any
// least-squares fit of one frame's pixels against another's would slope
// towards zero with the blur. So each channel's mean is taken over the pixels
// the frame's warp actually covers (its smeared edge left out), in both the
// frame and frame 0, and the gain is their ratio; per channel, so a light
// whose colour flickers is corrected too, and an exposure flicker gives the
// same gain three times. Gains are clamped to [1/4, 4]: a larger difference
// is not flicker.

use crate::align::{Sim, affine_inv};
use crate::pyramid::Img3;
use rayon::prelude::*;

pub const GAIN_MIN: f32 = 0.25;
pub const GAIN_MAX: f32 = 4.0;
/// Pixels are sampled on this grid: the mean of a 45 MP frame needs no more.
const STEP: usize = 4;

/// Per-channel sums and the sample count of `frame` over the pixels that `sim`
/// maps inside the frame (all of them at the identity), on the sampling grid;
/// `other`, when given, is summed over the same pixels.
fn sums(frame: &Img3, other: Option<&Img3>, sim: &Sim) -> ([f64; 3], [f64; 3], usize) {
    let (w, h) = (frame.w, frame.h);
    let inv = (*sim != Sim::id()).then(|| affine_inv(sim.matrix(w, h)));
    let (mut sf, mut so, mut n) = ([0f64; 3], [0f64; 3], 0usize);
    for y in (0..h).step_by(STEP) {
        for x in (0..w).step_by(STEP) {
            if let Some(inv) = &inv {
                let sx = inv[0][0] * x as f64 + inv[0][1] * y as f64 + inv[0][2];
                let sy = inv[1][0] * x as f64 + inv[1][1] * y as f64 + inv[1][2];
                if !(sx >= 0.0 && sx <= (w - 1) as f64 && sy >= 0.0 && sy <= (h - 1) as f64) {
                    continue;
                }
            }
            let i = y * w + x;
            for c in 0..3 {
                sf[c] += frame.p[c][i] as f64;
                if let Some(o) = other {
                    so[c] += o.p[c][i] as f64;
                }
            }
            n += 1;
        }
    }
    (sf, so, n)
}

/// Gains from a reference's and a frame's channel sums over the same pixels.
fn ratio(reference: [f64; 3], frame: [f64; 3], n: usize) -> [f32; 3] {
    if n < 64 {
        return [1.0; 3];
    }
    [0, 1, 2].map(|c| if frame[c] > 1e-9 { (reference[c] / frame[c]) as f32 } else { 1.0 }.clamp(GAIN_MIN, GAIN_MAX))
}

/// The per-channel gains that bring `frame`, already warped by `sim`, to the
/// brightness of `reference` over the pixels the warp covers.
pub fn gains(reference: &Img3, frame: &Img3, sim: &Sim) -> [f32; 3] {
    let (sf, sr, n) = sums(frame, Some(reference), sim);
    ratio(sr, sf, n)
}

/// Channel means of a whole (unwarped) frame, for `gains_to`.
pub fn means(frame: &Img3) -> [f64; 3] {
    let (s, _, n) = sums(frame, None, &Sim::id());
    if n == 0 { [0.0; 3] } else { s.map(|v| v / n as f64) }
}

/// The gains that bring a whole frame to the reference means (frames that were
/// not warped: every pixel counts).
pub fn gains_to(reference: [f64; 3], frame: &Img3) -> [f32; 3] {
    let m = means(frame);
    ratio(reference, m, usize::MAX)
}

pub fn is_unity(g: [f32; 3]) -> bool {
    g.iter().all(|v| (v - 1.0).abs() < 1e-6)
}

/// Multiply the frame's channels by the gains (nothing when they are all 1).
pub fn apply(frame: &mut Img3, g: [f32; 3]) {
    if is_unity(g) {
        return;
    }
    for c in 0..3 {
        if (g[c] - 1.0).abs() > 1e-6 {
            frame.p[c].par_iter_mut().for_each(|v| *v *= g[c]);
        }
    }
}

/// "×0.983", or "×0.98/1.01/0.97" when the channels differ.
pub fn describe(g: [f32; 3]) -> String {
    let spread = g.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    if spread.1 - spread.0 > 0.005 {
        format!("×{:.2}/{:.2}/{:.2}", g[0], g[1], g[2])
    } else {
        format!("×{:.3}", (g[0] + g[1] + g[2]) / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(w: usize, h: usize) -> Img3 {
        let mut im = Img3::zeros(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let t = ((x * 7 + y * 3) % 23) as f32 / 23.0;
                im.p[0][i] = 0.2 + 0.5 * t;
                im.p[1][i] = 0.3 + 0.3 * (x as f32 / w as f32);
                im.p[2][i] = 0.1 + 0.6 * (y as f32 / h as f32);
            }
        }
        im
    }

    fn blur5(im: &Img3) -> Img3 {
        let (w, h) = (im.w, im.h);
        let mut out = Img3::zeros(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let mut s = 0.0;
                    for dy in -2i64..=2 {
                        for dx in -2i64..=2 {
                            let xx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                            let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                            s += im.p[c][yy * w + xx];
                        }
                    }
                    out.p[c][y * w + x] = s / 25.0;
                }
            }
        }
        out
    }

    #[test]
    fn same_frame_is_unity() {
        let a = scene(160, 120);
        assert_eq!(gains(&a, &a, &Sim::id()), [1.0; 3]);
        assert_eq!(gains(&a, &a, &Sim { xoff: 0.05, yoff: -0.02, scale: 1.01, rot: 0.01 }), [1.0; 3]);
    }

    #[test]
    fn darker_blurred_frame_is_brought_back() {
        let a = scene(160, 120);
        let mut b = blur5(&a);
        apply(&mut b, [0.8, 0.9, 1.1]);
        let g = gains(&a, &b, &Sim::id());
        for (got, want) in g.iter().zip([1.25, 1.0 / 0.9, 1.0 / 1.1]) {
            assert!((got / want - 1.0).abs() < 0.01, "{g:?}");
        }
        apply(&mut b, g);
        let m = means(&b);
        for (x, y) in m.iter().zip(means(&a)) {
            assert!((x / y - 1.0).abs() < 0.01);
        }
        assert_eq!(gains_to(means(&a), &a), [1.0; 3]);
    }

    #[test]
    fn clamped_and_described() {
        assert_eq!(ratio([1.0; 3], [0.01; 3], 1000), [GAIN_MAX; 3]);
        assert_eq!(describe([0.983, 0.984, 0.982]), "×0.983");
        assert_eq!(describe([0.98, 1.01, 0.97]), "×0.98/1.01/0.97");
    }
}

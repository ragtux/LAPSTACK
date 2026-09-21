// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! The weighted average — Helicon Focus's method A: every frame's pixels are
//! averaged with weights that follow their local contrast, so the frame in
//! focus at a pixel counts most and the rest fade in with their sharpness. No
//! pixel is ever picked outright: the seams and halos of a winner-take-all
//! rule cannot arise, and where nothing is sharp (a flat area) the frames
//! simply average and the noise drops by the square root of their number —
//! at the cost of some softness where a hard pick would have kept a single
//! frame's detail. It suits short, smooth, low-contrast or noisy stacks; the
//! pyramid (`fuse.rs`) is the sharper tool.
//!
//! The contrast is the depth pass's focus measure (`depth.rs`: the ring
//! difference filter on the luma), block-averaged to the depth pass's working
//! grid, optionally box-smoothed there (`smooth`, in grid pixels — Helicon's
//! "smoothing"), raised to `power` (1 = plain contrast weighting, higher =
//! a keener pick of the sharpest frame) and taken back to full resolution
//! bilinearly. A floor of (1e-4)^power keeps a flat pixel from dividing by
//! zero: there, every frame weighs the same.

use crate::depth::{BoxFilter, DepthParams, block_mean, blocks, focus_measure, luma, upsample_bilinear};
use crate::pyramid::Img3;
use crate::stack::FrameSource;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WavParams {
    /// The contrast raised to this before weighing (1 = contrast itself).
    pub power: f32,
    /// Box radius of the weight map on the working grid, 0 = none.
    pub smooth: usize,
}

impl Default for WavParams {
    fn default() -> Self {
        WavParams { power: 2.0, smooth: 1 }
    }
}

/// The frames of `src` (aligned, equalised) averaged by their contrast, the
/// measure and working grid those of `dp`.
pub fn weighted_average(src: &mut dyn FrameSource, dp: &DepthParams, wp: &WavParams, log: &mut dyn FnMut(String)) -> Result<Img3, String> {
    let (w, h) = src.dims();
    let n = src.len();
    let k = 1usize << dp.scale;
    let (dw, dh) = (blocks(w, k), blocks(h, k));
    let power = wp.power.max(0.0);
    let floor = 1e-4f32.powf(power);
    let bf = (wp.smooth > 0).then(|| BoxFilter::new(dw, dh, wp.smooth));
    log(format!("weighted average: {n} frames, contrast {:?} on the {dw}x{dh} grid, power {power}, smoothing {}", dp.focus, wp.smooth));
    let t = Instant::now();
    let mut acc = Img3::zeros(w, h);
    let mut wsum = vec![0f32; w * h];
    for m in 0..n {
        let f = src.get(m)?;
        let y = luma(&f);
        let (g, _, _) = block_mean(&focus_measure(&y, w, h, dp.focus), w, h, k);
        let g = match &bf {
            Some(b) => b.mean(&g),
            None => g,
        };
        let g: Vec<f32> = g.iter().map(|v| v.max(0.0).powf(power) + floor).collect();
        let wf = upsample_bilinear(&g, dw, dh, w, h, k);
        for c in 0..3 {
            acc.p[c].par_chunks_mut(w).zip(f.p[c].par_chunks(w)).zip(wf.par_chunks(w)).for_each(|((a, s), ww)| {
                for x in 0..w {
                    a[x] += ww[x] * s[x];
                }
            });
        }
        wsum.par_iter_mut().zip(&wf).for_each(|(a, b)| *a += b);
        if (m + 1) % 10 == 0 || m + 1 == n {
            log(format!("  frame {:>3}/{n} weighed in  ({:.1}s)", m + 1, t.elapsed().as_secs_f64()));
        }
    }
    for c in 0..3 {
        acc.p[c].par_iter_mut().zip(&wsum).for_each(|(a, s)| *a = (*a / s).clamp(0.0, 1.0));
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two frames, each sharp on one half: the average takes each half from
    /// the frame that is sharp there, and a flat area averages the two.
    #[test]
    fn takes_the_sharper_frame() {
        let (w, h) = (64, 64);
        let mut a = Img3::zeros(w, h);
        let mut b = Img3::zeros(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let fine = if (x / 2 + y / 2) % 2 == 0 { 0.8 } else { 0.2 };   // a checkerboard: contrast
                let (va, vb) = if x < 32 { (fine, 0.5) } else { (0.5, fine) };   // a sharp on the left, b on the right
                for c in 0..3 {
                    a.p[c][i] = va;
                    b.p[c][i] = vb;
                }
            }
        }
        let frames = vec![a.clone(), b.clone()];
        let mut src: &[Img3] = &frames;
        let dp = DepthParams { scale: 1, ..Default::default() };
        let out = weighted_average(&mut src, &dp, &WavParams { power: 2.0, smooth: 0 }, &mut |_| {}).unwrap();
        // well inside each half, the average is the sharp frame's checkerboard
        let i = 16 * w + 8;
        assert!((out.p[0][i] - a.p[0][i]).abs() < 0.05, "left: {} vs {}", out.p[0][i], a.p[0][i]);
        let j = 16 * w + 56;
        assert!((out.p[0][j] - b.p[0][j]).abs() < 0.05, "right: {} vs {}", out.p[0][j], b.p[0][j]);
    }
}

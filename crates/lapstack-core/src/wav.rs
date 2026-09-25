// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: AGPL-3.0-only

//! The weighted average: every frame's pixels are
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
//! grid. What weighs is the contrast *above the cell's noise floor*: the
//! measure of sensor noise alone is not zero (it is the mean magnitude of the
//! noise's ring difference, the same in every frame), and at full resolution
//! it is of the order of the sharp frame's own texture — the lemon peel of a
//! 45 MP frame reads 2.7× its out-of-focus frames, so raised to 2 and shared
//! with two dozen frames of noise the sharp frame counted for a fifth and the
//! average came out as soft as no stacking at all. The depth pass tracks the
//! least contrast any frame shows at every cell (`DepthMap::floor`; the frames
//! farthest out of focus leave only the noise) and `gate` says how far above
//! that floor a contrast must be to mean anything: the weight is
//! `max(c − (1 + gate) · floor, 0)^power`, so the frames the noise alone could
//! explain weigh nothing (a fade instead of the cut let two dozen frames of
//! noise back in for a third of the weight); where nothing at all is above
//! the floor a tiny even weight, `(floor / 100)^power`, makes the frames a
//! plain average. The floor is a minimum over the frames of an aggregated
//! (guided-filtered) measure, so it sits a little under the noise's mean, and
//! with the contrast smoothed over the default 7×7 window its spread is a few
//! per cent of that mean: half the floor (the default gate) clears it with
//! room, while the depth pass's gate of a whole floor asked for twice the
//! noise and a smooth tomato skin, at twice, fell either side of it from one
//! cell to the next — and on the CPU and the GPU differently.
//! `power` 1 is plain contrast weighting, higher a keener pick of the sharpest
//! frame. The contrast is box-smoothed on the grid before the cut (`smooth`,
//! in grid pixels; the depth pass aggregates its
//! slices the same way before it takes their statistics): a cell's pick is a
//! region's, not its own — one cell's measure strays over the gate by chance
//! and picks one noisy frame where a flat area should average them all, and
//! along a silhouette, where one frame holds the edge and the other the
//! blurred halo over it, neighbouring cells picked different frames and the
//! 2 px bilinear ramp between them showed as jagged speckle along every depth
//! edge. The weights are box-smoothed by the same radius after the cut and
//! the power, so a region's border is a cross-fade over the window and not
//! a step wherever a cell's contrast sits at the gate (the CPU and the GPU
//! put those steps a cell apart, and a band of them along the tomato's edge
//! came out blotched on one and smooth on the other). Then the weights are
//! taken back to full resolution bilinearly.

use crate::depth::{BoxFilter, DepthParams, block_mean, blocks, focus_measure, luma, upsample_bilinear};
use crate::pyramid::Img3;
use crate::stack::FrameSource;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WavParams {
    /// The contrast raised to this before weighing (1 = contrast itself).
    pub power: f32,
    /// Box radius the contrast, and then the weights, are smoothed by on the
    /// working grid, 0 = none.
    pub smooth: usize,
    /// The margin over the noise floor: a contrast counts above (1 + gate) ×
    /// the cell's floor.
    pub gate: f32,
}

impl Default for WavParams {
    fn default() -> Self {
        WavParams { power: 2.0, smooth: 3, gate: 0.5 }
    }
}

/// One cell's weight: the contrast `c` above (1 + `gate`) × the cell's noise
/// floor `f`, raised to `power`, plus the tiny even weight that makes a cell
/// nothing rises above a plain average.
#[inline]
pub fn weight(c: f32, f: f32, gate: f32, power: f32) -> f32 {
    (c - (1.0 + gate) * f).max(0.0).powf(power) + (1e-2 * f).powf(power) + 1e-30
}

/// The frames of `src` (aligned, equalised) averaged by their contrast above
/// the noise floor, the measure and working grid those of `dp`, `floor` the
/// depth pass's per-cell floor on that grid (`DepthMap::floor`).
pub fn weighted_average(src: &mut dyn FrameSource, dp: &DepthParams, wp: &WavParams, floor: &[f32], log: &mut dyn FnMut(String)) -> Result<Img3, String> {
    let (w, h) = src.dims();
    let n = src.len();
    let k = 1usize << dp.scale;
    let (dw, dh) = (blocks(w, k), blocks(h, k));
    if floor.len() != dw * dh {
        return Err(format!("weighted average: the noise floor has {} cells, the working grid {dw}x{dh}", floor.len()));
    }
    let power = wp.power.max(0.0);
    let gate = wp.gate.max(0.0);
    let mut bf = (wp.smooth > 0).then(|| BoxFilter::new(dw, dh, wp.smooth));
    let median = {
        let mut s: Vec<f32> = floor.iter().step_by(7).copied().collect();
        let mid = s.len() / 2;
        *s.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
    };
    log(format!(
        "weighted average: {n} frames, contrast {:?} on the {dw}x{dh} grid above the noise floor (median {median:.2e}, gate {gate}), power {power}, smoothing {}",
        dp.focus, wp.smooth
    ));
    let t = Instant::now();
    let mut acc = Img3::zeros(w, h);
    let mut wsum = vec![0f32; w * h];
    for m in 0..n {
        let f = src.get(m)?;
        let y = luma(&f);
        let (g, _, _) = block_mean(&focus_measure(&y, w, h, dp.focus), w, h, k);
        let g = match &mut bf {
            Some(b) => b.mean(&g),
            None => g,
        };
        let g: Vec<f32> = g.iter().zip(floor).map(|(&c, &f)| weight(c, f, gate, power)).collect();
        let g = match &mut bf {
            Some(b) => b.mean(&g),
            None => g,
        };
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
        let floor = vec![0f32; blocks(w, 2) * blocks(h, 2)];
        let out = weighted_average(&mut src, &dp, &WavParams { power: 2.0, smooth: 0, gate: 0.5 }, &floor, &mut |_| {}).unwrap();
        // well inside each half, the average is the sharp frame's checkerboard
        let i = 16 * w + 8;
        assert!((out.p[0][i] - a.p[0][i]).abs() < 0.05, "left: {} vs {}", out.p[0][i], a.p[0][i]);
        let j = 16 * w + 56;
        assert!((out.p[0][j] - b.p[0][j]).abs() < 0.05, "right: {} vs {}", out.p[0][j], b.p[0][j]);
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) % 10000) as f32 / 10000.0
    }

    /// Noise in every frame, a faint texture sharp in one of many: with the
    /// depth pass's noise floor the sharp frame's texture survives the average
    /// where it is, and a flat area averages the frames (its noise drops).
    /// Without the floor, weighed by the contrast itself, the texture drowns.
    #[test]
    fn noise_floor_keeps_the_sharp_frame() {
        let (w, h, n) = (96, 64, 12);
        let mut seed = 7u64;
        let mut frames: Vec<Img3> = (0..n)
            .map(|_| {
                let mut f = Img3::zeros(w, h);
                for c in 0..3 {
                    for v in f.p[c].iter_mut() {
                        *v = 0.5 + 0.06 * (lcg(&mut seed) - 0.5);
                    }
                }
                f
            })
            .collect();
        // frame 5 carries a checkerboard on the left half, 0.05 above / below the mean
        for y in 0..h {
            for x in 0..w / 2 {
                let t = if (x / 2 + y / 2) % 2 == 0 { 0.05 } else { -0.05 };
                for c in 0..3 {
                    frames[5].p[c][y * w + x] += t;
                }
            }
        }
        let dp = DepthParams { scale: 1, ..Default::default() };
        let fused = frames[5].clone();
        let mut slices = frames.iter().map(|f| Ok(crate::depth::focus_slice(f, &dp)));
        let dm = crate::depth::depth_from_slices(&mut slices, n, &fused, &dp, &mut |_| {}).unwrap();
        let mut src: &[Img3] = &frames;
        let wp = WavParams { power: 2.0, smooth: 2, gate: 0.5 };
        let out = weighted_average(&mut src, &dp, &wp, &dm.floor, &mut |_| {}).unwrap();
        // the texture: correlation of the output with frame 5's checkerboard, well inside the left half
        let corr = |img: &Img3| {
            let mut s = 0.0;
            for y in 8..h - 8 {
                for x in 8..w / 2 - 8 {
                    let t = if (x / 2 + y / 2) % 2 == 0 { 1.0 } else { -1.0 };
                    s += t * (img.p[0][y * w + x] - 0.5);
                }
            }
            s / ((h - 16) * (w / 2 - 16)) as f32
        };
        let kept = corr(&out) / corr(&frames[5]);
        assert!(kept > 0.8, "the sharp frame's texture kept: {kept}");
        // the flat right half: the noise is well below one frame's
        let sd = |img: &Img3| {
            let px: Vec<f32> = (8..h - 8).flat_map(|y| (w / 2 + 8..w - 8).map(move |x| (x, y))).map(|(x, y)| img.p[0][y * w + x]).collect();
            let m = px.iter().sum::<f32>() / px.len() as f32;
            (px.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / px.len() as f32).sqrt()
        };
        let ratio = sd(&out) / sd(&frames[0]);
        assert!(ratio < 0.5, "flat-area noise vs one frame: {ratio}");
        // the contrast itself as the weight (no floor): the texture drowns among the noisy frames
        let none = vec![0f32; dm.floor.len()];
        let plain = weighted_average(&mut src, &dp, &wp, &none, &mut |_| {}).unwrap();
        let drowned = corr(&plain) / corr(&frames[5]);
        assert!(drowned < 0.5, "without the floor: {drowned}");
    }
}

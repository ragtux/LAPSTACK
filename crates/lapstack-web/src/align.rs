//! Streaming alignment on the GPU: the Nelder-Mead control loop runs here
//! (async), every cost evaluation is one `cost` dispatch (Spline4x4 warp +
//! DC-removed RMS partial sums) and a small readback. Same model, search and
//! per-level schedule (`lapstack_core::align::level_steps`) as lapstack-core's
//! aligner; FP32 like its CUDA path. A readback is the expensive part in the
//! browser (a round trip through Chrome's GPU process), so the points an
//! iteration may need — the reflection, and the expansion or the contraction
//! that follows it — are evaluated in one submit and read back together, and
//! the decisions are made from the values as the sequential search makes them.

use crate::gpu::{Gpu, P, grid2};
pub use lapstack_core::align::{Sim, inverse};
use lapstack_core::align::{converged, level_steps};
use std::future::Future;

/// Bounded Nelder-Mead with an async batch cost (the sequential search of
/// lapstack-core, its evaluations grouped): `f` takes several points and
/// returns their costs in order. From `x0`, the first simplex `x0` moved by
/// `step[k]` along each axis, until `converged` within `tol` (or 200
/// iterations). With `speculate` an iteration evaluates the reflection with
/// the expansion and the contraction it may lead to in one batch (three
/// dispatches for one round trip); without, the reflection alone, then the
/// one it calls for — for a level whose cost dispatch outweighs a round trip.
pub async fn nelder_mead<F, Fut>(f: F, x0: &[f64], lo: &[f64], hi: &[f64], step: &[f64], tol: &[f64], speculate: bool) -> Vec<f64>
where
    F: Fn(Vec<Vec<f64>>) -> Fut,
    Fut: Future<Output = Vec<f64>>,
{
    let n = x0.len();
    let clamp = |x: &mut Vec<f64>| {
        for k in 0..n {
            x[k] = x[k].clamp(lo[k], hi[k]);
        }
    };
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for k in 0..n {
        let mut v = x0.to_vec();
        v[k] += step[k];
        clamp(&mut v);
        simplex.push(v);
    }
    let mut fv = f(simplex.clone()).await;
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
        let mut xe = vec![0.0; n];
        let mut xc = vec![0.0; n];
        for k in 0..n {
            xr[k] = c[k] + alpha * (c[k] - simplex[n][k]);
        }
        clamp(&mut xr);
        for k in 0..n {
            xe[k] = c[k] + gamma * (xr[k] - c[k]);
            xc[k] = c[k] + rho * (simplex[n][k] - c[k]);
        }
        clamp(&mut xe);
        clamp(&mut xc);
        // the three costs: all at once, or the reflection first and then the one it calls for
        let (fr, fe, fc) = if speculate {
            let r = f(vec![xr.clone(), xe.clone(), xc.clone()]).await;
            (r[0], Some(r[1]), Some(r[2]))
        } else {
            (f(vec![xr.clone()]).await[0], None, None)
        };
        if fr < fv[0] {
            let fe = match fe {
                Some(v) => v,
                None => f(vec![xe.clone()]).await[0],
            };
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
            let fc = match fc {
                Some(v) => v,
                None => f(vec![xc.clone()]).await[0],
            };
            if fc < fv[n] {
                simplex[n] = xc;
                fv[n] = fc;
            } else {
                for i in 1..=n {
                    for k in 0..n {
                        simplex[i][k] = simplex[0][k] + sigma * (simplex[i][k] - simplex[0][k]);
                    }
                    clamp(&mut simplex[i]);
                }
                let r = f(simplex[1..].to_vec()).await;
                fv[1..].copy_from_slice(&r);
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

/// A luma pyramid on the device: level 0 is full res, then halved while
/// `h > 64 && w > 8` (the native aligner's rule).
pub struct LumaPyr {
    pub lv: Vec<(wgpu::Buffer, usize, usize)>,
}

pub fn pyr_dims(w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut d = vec![(w, h)];
    let (mut cw, mut ch) = (w, h);
    while ch > 64 && cw > 8 {
        cw = cw.div_ceil(2);
        ch = ch.div_ceil(2);
        d.push((cw, ch));
    }
    d
}

impl LumaPyr {
    pub fn new(gpu: &Gpu, w: usize, h: usize, label: &str) -> LumaPyr {
        let lv = pyr_dims(w, h)
            .into_iter()
            .map(|(lw, lh)| (gpu.buffer_f32(&format!("{label} luma"), lw * lh), lw, lh))
            .collect();
        LumaPyr { lv }
    }
    /// Reduce level 0 (already filled) down the chain. `tmp` ≥ half(w)*h floats.
    pub fn reduce_chain(&self, rec: &mut crate::gpu::Rec<'_>, tmp: &wgpu::Buffer) {
        for l in 0..self.lv.len() - 1 {
            let (fw, fh) = (self.lv[l].1, self.lv[l].2);
            let (cw, ch) = (self.lv[l + 1].1, self.lv[l + 1].2);
            let p = P { w: fw as u32, h: fh as u32, ow: cw as u32, oh: ch as u32, ..Default::default() };
            rec.dispatch("red_h", [Some(&self.lv[l].0), Some(tmp), None, None, None, None], p, grid2(cw, fh));
            rec.dispatch("red_v", [None, Some(tmp), Some(&self.lv[l + 1].0), None, None, None], p, grid2(cw, ch));
        }
    }
}

/// The most points one batch evaluates: the first simplex (`Sim::N + 1`
/// vertices), a shrink (`Sim::N`), or a speculative iteration (3).
const BATCH: usize = Sim::N + 1;

/// A level's cost dispatch below this many pixels is cheaper than a round
/// trip, and its iterations are speculated (`nelder_mead`).
const SPECULATE_PX: usize = 8 << 20;

pub struct Aligner {
    /// `BATCH` regions of (sd, sd2, n) triples, one per workgroup.
    partials: wgpu::Buffer,
    /// The translation and perspective terms of each point's inverse map.
    aff: Vec<wgpu::Buffer>,
    /// Number of triples one region holds (the workgroups at full resolution).
    cap: usize,
}

impl Aligner {
    pub fn new(gpu: &Gpu, w: usize, h: usize) -> Aligner {
        let g = grid2(w, h);
        let cap = (g.0 * g.1) as usize;
        Aligner {
            partials: gpu.buffer_f32("cost partials", cap * 3 * BATCH),
            aff: (0..BATCH).map(|_| gpu.buffer_init("affine t", bytemuck::cast_slice(&[0f32; 4]))).collect(),
            cap,
        }
    }

    /// The costs of several points at one level, in one submit and one
    /// readback: each warps the `tgt` level into the `rf` level with its
    /// transform and reduces the DC-removed RMS to per-workgroup partials in
    /// its own region of `partials`.
    async fn costs(&self, gpu: &Gpu, rf: &(wgpu::Buffer, usize, usize), tgt: &(wgpu::Buffer, usize, usize), vs: &[Vec<f64>]) -> Vec<f64> {
        debug_assert!(vs.len() <= BATCH);
        let (aw, ah) = (rf.1, rf.2);
        let (tw, th) = (tgt.1, tgt.2);
        let g = grid2(aw, ah);
        let n = ((g.0 * g.1) as usize).min(self.cap);
        let mut rec = gpu.rec();
        for (b, v) in vs.iter().enumerate() {
            let inv = inverse(Sim::from_vec(v).matrix(tw, th));
            gpu.queue.write_buffer(&self.aff[b], 0, bytemuck::cast_slice(&[inv[0][2] as f32, inv[1][2] as f32, inv[2][0] as f32, inv[2][1] as f32]));
            let p = P {
                w: aw as u32,
                h: ah as u32,
                ow: tw as u32,
                oh: th as u32,
                off_out: (b * n * 3) as u32,
                f0: inv[0][0] as f32,
                f1: inv[0][1] as f32,
                f2: inv[1][0] as f32,
                f3: inv[1][1] as f32,
                ..Default::default()
            };
            rec.dispatch("cost", [Some(&tgt.0), Some(&rf.0), None, Some(&self.partials), Some(&self.aff[b]), None], p, g);
        }
        rec.submit();
        let r = match gpu.read_f32(&self.partials, vs.len() * n * 3).await {
            Ok(r) => r,
            Err(_) => return vec![1e9; vs.len()],
        };
        (0..vs.len())
            .map(|b| {
                let (mut sd, mut sd2, mut cnt) = (0f64, 0f64, 0f64);
                for k in b * n..(b + 1) * n {
                    sd += r[3 * k] as f64;
                    sd2 += r[3 * k + 1] as f64;
                    cnt += r[3 * k + 2] as f64;
                }
                if cnt < 16.0 {
                    return 1e9;
                }
                ((sd2 - sd * sd / cnt) / cnt).max(0.0).sqrt()
            })
            .collect()
    }

    /// Coarse-to-fine search (stopping `coarsen` levels short of full res),
    /// exactly the native `multiscale_align` schedule.
    pub async fn align(&self, gpu: &Gpu, rf: &LumaPyr, tg: &LumaPyr, init: Sim, free: [bool; Sim::N], coarsen: usize) -> Sim {
        let n = rf.lv.len().min(tg.lv.len());
        let (w, h) = (rf.lv[0].1, rf.lv[0].2);
        let span = Sim::SPAN;
        let iv = init.as_vec();
        let mut cur = iv;
        let free_idx: Vec<usize> = (0..Sim::N).filter(|&k| free[k]).collect();
        if free_idx.is_empty() {
            return init;
        }
        let lo_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] - span[k]).collect();
        let hi_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] + span[k]).collect();
        let finest = coarsen.min(n.saturating_sub(1));
        for lvl in (finest..n).rev() {
            let cur_snap = cur;
            let fi = &free_idx;
            let cost = |xs: Vec<Vec<f64>>| async move {
                let vs: Vec<Vec<f64>> = xs
                    .iter()
                    .map(|xf| {
                        let mut v = cur_snap;
                        for (k, &idx) in fi.iter().enumerate() {
                            v[idx] = xf[k];
                        }
                        v.to_vec()
                    })
                    .collect();
                self.costs(gpu, &rf.lv[lvl], &tg.lv[lvl], &vs).await
            };
            let x0: Vec<f64> = free_idx.iter().map(|&k| cur[k]).collect();
            let (step, tol) = level_steps(&free_idx, lvl, w, h);
            let speculate = rf.lv[lvl].1 * rf.lv[lvl].2 <= SPECULATE_PX;
            let best = nelder_mead(cost, &x0, &lo_f, &hi_f, &step, &tol, speculate).await;
            for (k, &idx) in free_idx.iter().enumerate() {
                cur[idx] = best[k];
            }
        }
        Sim::from_vec(&cur)
    }
}

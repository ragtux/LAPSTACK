//! Streaming 4-DOF similarity alignment on the GPU: the Nelder-Mead control
//! loop runs here (async), every cost evaluation is one `cost` dispatch
//! (Spline4x4 warp + DC-removed RMS partial sums) and a small readback.
//! Same model and search as lapstack-core's aligner; FP32 like its CUDA path.

use crate::gpu::{Gpu, P, grid2};
pub use lapstack_core::align::{Sim, affine_inv};
use std::future::Future;

/// Bounded Nelder-Mead with an async cost (transcribed from lapstack-core).
pub async fn nelder_mead<F, Fut>(f: F, x0: &[f64], lo: &[f64], hi: &[f64]) -> Vec<f64>
where
    F: Fn(Vec<f64>) -> Fut,
    Fut: Future<Output = f64>,
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
        v[k] += (hi[k] - lo[k]) * 0.05;
        clamp(&mut v);
        simplex.push(v);
    }
    let mut fv = Vec::with_capacity(n + 1);
    for s in &simplex {
        fv.push(f(s.clone()).await);
    }
    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);
    for _ in 0..200 {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| fv[a].partial_cmp(&fv[b]).unwrap());
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        fv = idx.iter().map(|&i| fv[i]).collect();
        let spread = (fv[n] - fv[0]).abs();
        let mut size = 0.0f64;
        for i in 1..=n {
            for k in 0..n {
                size = size.max((simplex[i][k] - simplex[0][k]).abs() / (hi[k] - lo[k]).max(1e-12));
            }
        }
        if spread < 1e-7 && size < 1e-4 {
            break;
        }
        let mut c = vec![0.0; n];
        for i in 0..n {
            for k in 0..n {
                c[k] += simplex[i][k] / n as f64;
            }
        }
        let mut xr = vec![0.0; n];
        for k in 0..n {
            xr[k] = c[k] + alpha * (c[k] - simplex[n][k]);
        }
        clamp(&mut xr);
        let fr = f(xr.clone()).await;
        if fr < fv[0] {
            let mut xe = vec![0.0; n];
            for k in 0..n {
                xe[k] = c[k] + gamma * (xr[k] - c[k]);
            }
            clamp(&mut xe);
            let fe = f(xe.clone()).await;
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
            let fc = f(xc.clone()).await;
            if fc < fv[n] {
                simplex[n] = xc;
                fv[n] = fc;
            } else {
                for i in 1..=n {
                    for k in 0..n {
                        simplex[i][k] = simplex[0][k] + sigma * (simplex[i][k] - simplex[0][k]);
                    }
                    clamp(&mut simplex[i]);
                    fv[i] = f(simplex[i].clone()).await;
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

pub struct Aligner {
    partials: wgpu::Buffer,
    aff: wgpu::Buffer,
    /// Number of (sd, sd2, n) triples the partials buffer can hold.
    cap: usize,
}

impl Aligner {
    pub fn new(gpu: &Gpu, w: usize, h: usize) -> Aligner {
        let g = grid2(w, h);
        let cap = (g.0 * g.1) as usize;
        Aligner {
            partials: gpu.buffer_f32("cost partials", cap * 3),
            aff: gpu.buffer_init("affine t", bytemuck::cast_slice(&[0f32; 4])),
            cap,
        }
    }

    /// One cost evaluation: warp `tgt` level into `rf` level with `sim`, return DC-removed RMS.
    async fn cost(&self, gpu: &Gpu, rf: &(wgpu::Buffer, usize, usize), tgt: &(wgpu::Buffer, usize, usize), v: &[f64]) -> f64 {
        let (aw, ah) = (rf.1, rf.2);
        let (tw, th) = (tgt.1, tgt.2);
        let inv = affine_inv(Sim::from_vec(v).matrix(tw, th));
        gpu.queue.write_buffer(&self.aff, 0, bytemuck::cast_slice(&[inv[0][2] as f32, inv[1][2] as f32, 0.0, 0.0]));
        let p = P {
            w: aw as u32,
            h: ah as u32,
            ow: tw as u32,
            oh: th as u32,
            f0: inv[0][0] as f32,
            f1: inv[0][1] as f32,
            f2: inv[1][0] as f32,
            f3: inv[1][1] as f32,
            ..Default::default()
        };
        let g = grid2(aw, ah);
        let n = (g.0 * g.1) as usize;
        let mut rec = gpu.rec();
        rec.dispatch("cost", [Some(&tgt.0), Some(&rf.0), None, Some(&self.partials), Some(&self.aff), None], p, g);
        rec.submit();
        let r = match gpu.read_f32(&self.partials, n.min(self.cap) * 3).await {
            Ok(r) => r,
            Err(_) => return 1e9,
        };
        let (mut sd, mut sd2, mut cnt) = (0f64, 0f64, 0f64);
        for k in 0..n.min(self.cap) {
            sd += r[3 * k] as f64;
            sd2 += r[3 * k + 1] as f64;
            cnt += r[3 * k + 2] as f64;
        }
        if cnt < 16.0 {
            return 1e9;
        }
        ((sd2 - sd * sd / cnt) / cnt).max(0.0).sqrt()
    }

    /// Coarse-to-fine search (stopping `coarsen` levels short of full res),
    /// exactly the native `multiscale_align` schedule.
    pub async fn align(&self, gpu: &Gpu, rf: &LumaPyr, tg: &LumaPyr, init: Sim, free: [bool; 4], coarsen: usize) -> Sim {
        let n = rf.lv.len().min(tg.lv.len());
        let span = [0.10, 0.10, 0.10, 5.0f64.to_radians()];
        let iv = init.as_vec();
        let mut cur = iv;
        let free_idx: Vec<usize> = (0..4).filter(|&k| free[k]).collect();
        if free_idx.is_empty() {
            return init;
        }
        let lo_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] - span[k]).collect();
        let hi_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] + span[k]).collect();
        let finest = coarsen.min(n.saturating_sub(1));
        for lvl in (finest..n).rev() {
            let cur_snap = cur;
            let fi = &free_idx;
            let cost = |xf: Vec<f64>| async move {
                let mut v = cur_snap;
                for (k, &idx) in fi.iter().enumerate() {
                    v[idx] = xf[k];
                }
                self.cost(gpu, &rf.lv[lvl], &tg.lv[lvl], &v).await
            };
            let x0: Vec<f64> = free_idx.iter().map(|&k| cur[k]).collect();
            let best = nelder_mead(cost, &x0, &lo_f, &hi_f).await;
            for (k, &idx) in free_idx.iter().enumerate() {
                cur[idx] = best[k];
            }
        }
        Sim::from_vec(&cur)
    }
}

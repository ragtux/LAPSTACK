// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

//! Depth from focus on WebGPU — the pipeline of `crate::depth`
//! (ring difference filter, guided-filter aggregation, streamed sub-frame peak
//! search, confidence-weighted WLS with a Huber reweight — the conjugate
//! gradient preconditioned by a multigrid V-cycle — guided upsampling)
//! as WGSL kernels (`shaders.wgsl`, "Depth from focus" section).
//!
//! During the run every frame's focus measure is block-averaged to the
//! working grid on the GPU and kept on the CPU side as a quantized u16 slice
//! (≈ 2 bytes per working-grid pixel per frame). After fusion the slices are
//! streamed back through the guided filter — the aggregation is guided by the
//! *fused* luma, which does not exist before the end of the run — into the
//! peak tracker, and the rest of the pass runs entirely on the device.

use super::gpu::{Gpu, P, Rec, grid1, grid2};
use crate::depth::{DepthParams, FocusMeasure, MG_COARSE, MG_MIN, MG_OMEGA, MG_POST, MG_PRE, Upsample, rdf_taps};

const EPS_DATA: f32 = 1e-4;

pub struct DepthGpu {
    pub k: usize,
    pub dw: usize,
    pub dh: usize,
    pub params: DepthParams,
    taps: wgpu::Buffer,
    ntaps: u32,
    slice: wgpu::Buffer,
    /// Per frame: the block-averaged focus measure, quantized (values, scale).
    pub slices: Vec<(Vec<u16>, f32)>,
    /// Working-grid planes for the weighted average's smoothed weight map
    /// (four: the guided filter needs its coefficients beside the input).
    wtmp: wgpu::Buffer,
    wgrid: wgpu::Buffer,
    wa: wgpu::Buffer,
    wb: wgpu::Buffer,
    /// The fused luma on the working grid, set by `finish`: the guide of
    /// the weighted average's edge-aware smoothing, with its box statistics
    /// `[mean | variance]` (two planes) made at the smoothing radius.
    guide: wgpu::Buffer,
    gstats: wgpu::Buffer,
    /// After `finish`: the noise floor of every working-grid cell, the least
    /// aggregated contrast any frame showed there (`DepthMap::floor` in core).
    floor: Option<wgpu::Buffer>,
}

/// One grid of the WLS multigrid hierarchy (`crate::depth::MgLevel`):
/// `[wd | dinv]` and `[ax | ay]` (two planes each, λ in the edges), and the
/// V-cycle's solution, right-hand side and residual planes.
struct Lv {
    w: usize,
    h: usize,
    x: wgpu::Buffer,
    b: wgpu::Buffer,
    r: wgpu::Buffer,
    wdd: wgpu::Buffer,
    e2: wgpu::Buffer,
}

/// Working-grid buffers for the finish stage (allocated only then).
struct Work {
    t: Vec<wgpu::Buffer>, // 8 planes
    rt: wgpu::Buffer,     // box-filter row sums
    x2b: wgpu::Buffer,    // 2 planes
    /// The WLS system's multigrid hierarchy, the working grid first.
    levels: Vec<Lv>,
    state: wgpu::Buffer,  // 9 planes (peak tracker)
    guide: wgpu::Buffer,
    d: wgpu::Buffer,
    wd: wgpu::Buffer,
    u: wgpu::Buffer,
    ones: wgpu::Buffer,
    partials: wgpu::Buffer,
    scal: wgpu::Buffer,
    npart: u32,
}

impl DepthGpu {
    pub fn new(g: &Gpu, w: usize, h: usize, params: DepthParams) -> Result<DepthGpu, String> {
        let k = 1usize << params.scale;
        let (dw, dh) = (w.div_ceil(k), h.div_ceil(k));
        let taps = match params.focus {
            FocusMeasure::Rdf { r_in, r_out } => rdf_taps(r_in, r_out),
            FocusMeasure::Sml { .. } => return Err("the browser depth pass supports the rdf focus measure only".into()),
        };
        let flat: Vec<f32> = taps.iter().flat_map(|&(dy, dx, w)| [dy as f32, dx as f32, w]).collect();
        Ok(DepthGpu {
            k,
            dw,
            dh,
            params,
            taps: g.buffer_init("rdf taps", bytemuck::cast_slice(&flat)),
            ntaps: taps.len() as u32,
            slice: g.buffer_f32("depth slice", dw * dh),
            slices: Vec::new(),
            wtmp: g.buffer_f32("wav tmp", dw * dh),
            wgrid: g.buffer_f32("wav grid", dw * dh),
            wa: g.buffer_f32("wav a", dw * dh),
            wb: g.buffer_f32("wav b", dw * dh),
            guide: g.buffer_f32("wav guide", dw * dh),
            gstats: g.buffer_f32("wav guide stats", 2 * dw * dh),
            floor: None,
        })
    }

    /// Record the focus measure of the frame whose luma is in `luma` (w×h,
    /// f32) using `tmp` (w×h) as scratch; the slice lands in `self.slice`.
    pub fn record_measure(&self, rec: &mut Rec<'_>, luma: &wgpu::Buffer, tmp: &wgpu::Buffer, w: usize, h: usize) {
        rec.dispatch(
            "conv_taps",
            [Some(luma), None, Some(tmp), None, Some(&self.taps), None],
            P { w: w as u32, h: h as u32, klen: self.ntaps, flag: 1, ..Default::default() },
            grid2(w, h),
        );
        rec.dispatch(
            "down1",
            [Some(tmp), None, Some(&self.slice), None, None, None],
            P { w: w as u32, h: h as u32, ow: self.dw as u32, oh: self.dh as u32, klen: self.k as u32, ..Default::default() },
            grid2(self.dw, self.dh),
        );
    }

    /// The weighted average's weight map for the frame in `luma` (w×h f32): its
    /// contrast on the working grid, box-smoothed by `smooth` grid pixels, above
    /// (1 + `gate`) × the cell's noise floor and raised to `power` (`wav_weight`; the
    /// twin of core `wav.rs`), the weights smoothed by the same window — with
    /// `edge`, by the guided filter with the fused luma as guide (`WavParams::edge`:
    /// the cross-fade stops at a silhouette), else by the box mean — in the
    /// buffer returned, for `wav_acc`. `tmp` is a w×h f32 scratch. Needs the
    /// pass finished (the floor and the guide).
    #[allow(clippy::too_many_arguments)]
    pub fn record_weight<'a>(&'a self, rec: &mut Rec<'_>, luma: &wgpu::Buffer, tmp: &wgpu::Buffer, w: usize, h: usize, power: f32, smooth: u32, gate: f32, edge: bool) -> Result<&'a wgpu::Buffer, String> {
        let floor = self.floor.as_ref().ok_or("the weighted average needs the depth pass finished")?;
        self.record_measure(rec, luma, tmp, w, h);
        let n = self.dw * self.dh;
        let pg = P { w: self.dw as u32, h: self.dh as u32, klen: smooth, ..Default::default() };
        if smooth == 0 {
            rec.dispatch("wav_weight", [Some(&self.slice), None, Some(&self.wgrid), None, Some(floor), None], P { f0: power.max(0.0), f1: gate.max(0.0), ..pg }, grid1(n));
            return Ok(&self.wgrid);
        }
        let boxf = |rec: &mut Rec<'_>, src: &wgpu::Buffer, via: &wgpu::Buffer, dst: &wgpu::Buffer| {
            rec.dispatch("box_h", [Some(src), Some(via), None, None, None, None], pg, grid2(self.dw, self.dh));
            rec.dispatch("box_v", [None, Some(via), Some(dst), None, None, None], pg, grid2(self.dw, self.dh));
        };
        // the contrast smoothed, weighed: the weight p in wtmp
        boxf(rec, &self.slice, &self.wtmp, &self.wgrid);
        rec.dispatch("wav_weight", [Some(&self.wgrid), None, Some(&self.wtmp), None, Some(floor), None], P { f0: power.max(0.0), f1: gate.max(0.0), ..pg }, grid1(n));
        if !edge {
            boxf(rec, &self.wtmp, &self.wgrid, &self.wa);
            return Ok(&self.wa);
        }
        // the guide's statistics at this radius: [mean_I | var_I] (cheap on the grid, so per frame)
        boxf(rec, &self.guide, &self.wb, &self.wa);
        rec.dispatch("mul", [Some(&self.guide), None, Some(&self.wgrid), None, Some(&self.guide), None], pg, grid1(n));
        boxf(rec, &self.wgrid, &self.wb, &self.wgrid);
        rec.dispatch("gf_var", [Some(&self.wa), None, None, Some(&self.gstats), Some(&self.wgrid), None], pg, grid1(n));
        // the guided filter of p (twin of core `GuidedFilter`): corr_Ip, mean_p, the coefficients a, b, their means, q = ā·I + b̄
        rec.dispatch("mul", [Some(&self.guide), None, Some(&self.wgrid), None, Some(&self.wtmp), None], pg, grid1(n));
        boxf(rec, &self.wgrid, &self.wb, &self.wgrid);
        boxf(rec, &self.wtmp, &self.wb, &self.wa);
        rec.dispatch("gf_ab", [Some(&self.wa), Some(&self.wtmp), Some(&self.wb), Some(&self.gstats), Some(&self.wgrid), None], P { f0: self.params.agg_eps, ..pg }, grid1(n));
        boxf(rec, &self.wtmp, &self.wgrid, &self.wa);
        boxf(rec, &self.wb, &self.wgrid, &self.wtmp);
        // an undershoot at an edge is clamped: a weight stays a weight (flag 1)
        rec.dispatch("gf_apply", [Some(&self.wa), None, Some(&self.wgrid), Some(&self.guide), Some(&self.wtmp), None], P { flag: 1, ..pg }, grid1(n));
        Ok(&self.wgrid)
    }

    /// The noise floor of every working-grid cell, after `finish`
    /// (`DepthMap::floor` in core).
    pub async fn floor_values(&self, g: &Gpu) -> Result<Vec<f32>, String> {
        let floor = self.floor.as_ref().ok_or("the floor comes with the pass finished")?;
        g.read_f32(floor, self.dw * self.dh).await
    }

    /// Read the recorded slice back and keep it (quantized).
    pub async fn take_slice(&mut self, g: &Gpu) -> Result<(), String> {
        let v = g.read_f32(&self.slice, self.dw * self.dh).await?;
        let max = v.iter().cloned().fold(0f32, f32::max).max(1e-20);
        let q = 65535.0 / max;
        self.slices.push((v.iter().map(|&x| (x.max(0.0) * q + 0.5) as u16).collect(), max / 65535.0));
        Ok(())
    }

    /// The finish stage. `luma_full` holds the fused luma (w×h, f32);
    /// `full_tmp` and `full_out` are w×h f32 scratch buffers; `up16` is a
    /// buffer of at least dw*dh/2 u32 for slice uploads. Returns the
    /// working-grid depth (fractional frame index, dw×dh), the working-grid
    /// confidence normalized like the WLS data weight (`min(1, c / p90)`,
    /// dw×dh) and the full-res depth quantized to u16 (65535 = last frame).
    pub async fn finish(
        &mut self,
        g: &Gpu,
        w: usize,
        h: usize,
        luma_full: &wgpu::Buffer,
        full_tmp: &wgpu::Buffer,
        full_out: &wgpu::Buffer,
        up16: &wgpu::Buffer,
        log: &dyn Fn(&str),
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<u16>), String> {
        let (dw, dh, n) = (self.dw, self.dh, self.dw * self.dh);
        let slices = std::mem::take(&mut self.slices);
        let nf = slices.len();
        if nf == 0 {
            return Err("no frames measured".into());
        }
        let p = self.params.clone();
        let (gx, gy) = grid1(n);
        let mut wk = Work {
            t: (0..8).map(|i| g.buffer_f32(&format!("dff t{i}"), n)).collect(),
            rt: g.buffer_f32("dff rowsum", n),
            x2b: g.buffer_f32("dff x2b", 2 * n),
            levels: {
                let mut levels = Vec::new();
                let (mut lw, mut lh) = (dw, dh);
                loop {
                    let m = lw * lh;
                    let l = levels.len();
                    levels.push(Lv {
                        w: lw,
                        h: lh,
                        x: g.buffer_f32(&format!("mg x{l}"), m),
                        b: g.buffer_f32(&format!("mg b{l}"), m),
                        r: g.buffer_f32(&format!("mg r{l}"), m),
                        wdd: g.buffer_f32(&format!("mg wdd{l}"), 2 * m),
                        e2: g.buffer_f32(&format!("mg e{l}"), 2 * m),
                    });
                    if lw.max(lh) <= MG_MIN {
                        break;
                    }
                    (lw, lh) = (lw.div_ceil(2), lh.div_ceil(2));
                }
                levels
            },
            state: g.buffer_f32("dff state", 9 * n),
            guide: g.buffer_f32("dff guide", n),
            d: g.buffer_f32("dff d", n),
            wd: g.buffer_f32("dff wd", n),
            u: g.buffer_f32("dff u", n),
            ones: g.buffer_init("dff ones", bytemuck::cast_slice(&vec![1f32; n])),
            partials: g.buffer_f32("dff partials", (gx * gy) as usize),
            scal: g.buffer_init("dff scal", bytemuck::cast_slice(&[0f32; 8])),
            npart: gx * gy,
        };
        let pw = P { w: dw as u32, h: dh as u32, ..Default::default() };
        let t = &wk.t;

        // ---- guide = block mean of the fused luma; guided-filter statistics
        let mut rec = g.rec();
        rec.dispatch(
            "down1",
            [Some(luma_full), None, Some(&wk.guide), None, None, None],
            P { w: w as u32, h: h as u32, ow: dw as u32, oh: dh as u32, klen: self.k as u32, ..Default::default() },
            grid2(dw, dh),
        );
        // the same guide kept for the weighted average's edge-aware weights
        rec.dispatch(
            "down1",
            [Some(luma_full), None, Some(&self.guide), None, None, None],
            P { w: w as u32, h: h as u32, ow: dw as u32, oh: dh as u32, klen: self.k as u32, ..Default::default() },
            grid2(dw, dh),
        );
        let r_agg = p.agg_radius;
        let guide_stats = |rec: &mut Rec<'_>, wk: &Work, r: usize| {
            let t = &wk.t;
            self.boxf(rec, wk, &wk.guide, &t[1], r);
            rec.dispatch("mul", [Some(&wk.guide), None, Some(&t[2]), None, Some(&wk.guide), None], pw, grid1(n));
            self.boxf(rec, wk, &t[2], &t[3], r);
            rec.dispatch("gf_var", [Some(&t[1]), None, None, Some(&wk.x2b), Some(&t[3]), None], pw, grid1(n));
        };
        guide_stats(&mut rec, &wk, r_agg);
        rec.dispatch("peak_init", [None, None, Some(&wk.state), None, None, None], pw, grid1(n));
        rec.submit();

        // ---- aggregate every slice with the guided filter, fold into the tracker
        for m in 0..nf {
            let (q, scale) = &slices[m];
            let mut packed = vec![0u32; n.div_ceil(2)];
            for (i, &v) in q.iter().enumerate() {
                packed[i >> 1] |= (v as u32) << (16 * (i & 1));
            }
            g.queue.write_buffer(up16, 0, bytemuck::cast_slice(&packed));
            let mut rec = g.rec();
            rec.dispatch("unpack_u16", [None, None, Some(&t[0]), None, None, Some(up16)], P { f0: *scale, ..pw }, grid1(n));
            if r_agg > 0 {
                self.boxf(&mut rec, &wk, &t[0], &t[1], r_agg); // mean_p
                rec.dispatch("mul", [Some(&wk.guide), None, Some(&t[2]), None, Some(&t[0]), None], pw, grid1(n));
                self.boxf(&mut rec, &wk, &t[2], &t[3], r_agg); // corr_Ip
                rec.dispatch("gf_ab", [Some(&t[1]), Some(&t[4]), Some(&t[5]), Some(&wk.x2b), Some(&t[3]), None], P { f0: p.agg_eps, ..pw }, grid1(n));
                self.boxf(&mut rec, &wk, &t[4], &t[6], r_agg);
                self.boxf(&mut rec, &wk, &t[5], &t[2], r_agg);
                rec.dispatch("gf_apply", [Some(&t[6]), None, Some(&t[7]), Some(&wk.guide), Some(&t[2]), None], P { flag: 1, ..pw }, grid1(n));
                rec.dispatch("peak_push", [Some(&t[7]), None, Some(&wk.state), None, None, None], P { klen: m as u32, ..pw }, grid1(n));
            } else {
                rec.dispatch("peak_push", [Some(&t[0]), None, Some(&wk.state), None, None, None], P { klen: m as u32, ..pw }, grid1(n));
            }
            rec.submit();
        }
        drop(slices);

        // ---- noise floor (median of the profile minima), close the profiles; the
        // per-cell minima stay for the weighted average
        let cmin = g.read_range_f32(&wk.state, 7 * n, n).await?;
        let mut mins: Vec<f32> = cmin.iter().step_by(7).copied().collect();
        let floor_buf = g.buffer_init("dff floor", bytemuck::cast_slice(&cmin));
        drop(cmin);
        let mid = mins.len() / 2;
        let floor = *mins.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
        let mut rec = g.rec();
        rec.dispatch(
            "peak_finish",
            [None, Some(&t[0]), Some(&wk.state), Some(&t[1]), None, None],
            P { klen: nf as u32, f0: floor, f1: p.gate, ..pw },
            grid1(n),
        );
        if p.median {
            rec.dispatch("median3", [Some(&t[0]), None, Some(&wk.d), None, None, None], pw, grid2(dw, dh));
        } else {
            rec.copy(&t[0], 0, &wk.d, 0, (n * 4) as u64);
        }
        rec.submit();
        let conf = g.read_f32(&t[1], n).await?;
        let mut sample: Vec<f32> = conf.iter().step_by(7).copied().collect();
        let k90 = (sample.len() * 9 / 10).min(sample.len() - 1);
        let p90 = *sample.select_nth_unstable_by(k90, |a, b| a.total_cmp(b)).1;
        // the confidence the viewer shows and the save writes: the WLS data weight before its epsilon
        let conf_w: Vec<f32> = conf.iter().map(|&c| (c / p90.max(1e-6)).min(1.0)).collect();
        let mean_conf = conf_w.iter().map(|&c| c as f64).sum::<f64>() / n as f64;
        log(&format!("[lapstack] depth: {nf} slices folded on the {dw}x{dh} grid, confidence p90 {p90:.3}, mean {mean_conf:.3}"));

        // ---- WLS (+ one robust reweight): the normalized confidence (t0), the
        // data weight with its floor, the edges with lambda in them
        let mut rec = g.rec();
        rec.dispatch("scale_clamp", [Some(&wk.t[1]), None, Some(&wk.t[0]), None, None, None], P { f0: 1.0 / p90.max(1e-6), f1: 0.0, ..pw }, grid1(n));
        rec.dispatch("scale_clamp", [Some(&wk.t[0]), None, Some(&wk.wd), None, None, None], P { f0: 1.0, f1: EPS_DATA, ..pw }, grid1(n));
        if p.lambda > 0.0 {
            let l0 = &wk.levels[0];
            rec.dispatch("edge_w", [Some(&wk.guide), None, Some(&l0.e2), None, None, None], P { f0: p.sigma_c, f1: p.lambda, ..pw }, grid2(dw, dh));
            rec.copy(&wk.wd, 0, &l0.wdd, 0, (n * 4) as u64);
            Self::set_weights(&mut rec, &wk);
            rec.submit();
            let (rel, iters) = self.wls(g, &mut wk, &p).await?;
            let (rel, iters) = if p.robust > 0.0 {
                // one IRLS step with a Huber loss on the data residual
                let mut rec = g.rec();
                rec.dispatch("robust_w", [Some(&wk.d), None, Some(&wk.levels[0].wdd), Some(&wk.u), Some(&wk.t[0]), None], P { f0: p.robust, f1: EPS_DATA, ..pw }, grid1(n));
                Self::set_weights(&mut rec, &wk);
                rec.submit();
                let (rel2, iters2) = self.wls(g, &mut wk, &p).await?;
                log(&format!("[lapstack] depth: robust reweighting (tau={} frames), first solve {iters} iterations, residual {rel:.1e}", p.robust));
                (rel2, iters2)
            } else {
                (rel, iters)
            };
            log(&format!("[lapstack] depth: WLS lambda={} sigma_c={} solved in {iters} iterations, residual {rel:.1e}", p.lambda, p.sigma_c));
        } else {
            rec.copy(&wk.d, 0, &wk.u, 0, (n * 4) as u64);
            rec.submit();
        }
        let max_d = (nf - 1) as f32;
        let mut depth_w = g.read_f32(&wk.u, n).await?;
        for v in depth_w.iter_mut() {
            *v = v.clamp(0.0, max_d);
        }

        // ---- guided upsampling to full resolution
        let t = &wk.t;
        let mut rec = g.rec();
        match p.upsample {
            Upsample::Guided { radius, eps } => {
                guide_stats(&mut rec, &wk, radius);
                self.boxf(&mut rec, &wk, &wk.u, &t[1], radius);
                rec.dispatch("mul", [Some(&wk.guide), None, Some(&t[2]), None, Some(&wk.u), None], pw, grid1(n));
                self.boxf(&mut rec, &wk, &t[2], &t[3], radius);
                rec.dispatch("gf_ab", [Some(&t[1]), Some(&t[4]), Some(&t[5]), Some(&wk.x2b), Some(&t[3]), None], P { f0: eps, ..pw }, grid1(n));
                self.boxf(&mut rec, &wk, &t[4], &t[6], radius);
                self.boxf(&mut rec, &wk, &t[5], &t[2], radius);
            }
            Upsample::Bilinear => {
                // a = 0, b = u
                rec.dispatch("scale_clamp", [Some(&wk.u), None, Some(&t[6]), None, None, None], P { f0: 0.0, f1: 0.0, ..pw }, grid1(n));
                rec.copy(&wk.u, 0, &t[2], 0, (n * 4) as u64);
            }
        }
        rec.dispatch(
            "up_apply",
            [Some(&t[6]), None, Some(full_out), Some(luma_full), Some(&t[2]), None],
            P { w: w as u32, h: h as u32, ow: dw as u32, oh: dh as u32, klen: self.k as u32, f0: max_d, ..Default::default() },
            grid2(w, h),
        );
        let nfull = w * h;
        rec.dispatch(
            "pack_u16",
            [Some(full_out), None, Some(full_tmp), None, None, None],
            P { w: w as u32, h: h as u32, f0: 65535.0 / max_d.max(1e-6), ..Default::default() },
            grid1(nfull.div_ceil(2)),
        );
        rec.submit();
        let packed = g.read(full_tmp, (nfull.div_ceil(2) * 4) as u64).await?;
        let mut full: Vec<u16> = bytemuck::cast_slice::<u8, u16>(&packed).to_vec();
        full.truncate(nfull);
        self.floor = Some(floor_buf);
        Ok((depth_w, conf_w, full))
    }

    /// Box mean with border clipping: src -> dst (radius r), row sums in wk.rt.
    fn boxf(&self, rec: &mut Rec<'_>, wk: &Work, src: &wgpu::Buffer, dst: &wgpu::Buffer, r: usize) {
        let pw = P { w: self.dw as u32, h: self.dh as u32, klen: r as u32, ..Default::default() };
        rec.dispatch("box_h", [Some(src), Some(&wk.rt), None, None, None, None], pw, grid2(self.dw, self.dh));
        rec.dispatch("box_v", [None, Some(&wk.rt), Some(dst), None, None, None], pw, grid2(self.dw, self.dh));
    }

    /// The data weights are in `levels[0]`'s `wd` half: aggregate them (with
    /// the edges) down the hierarchy and take every grid's diagonal.
    fn set_weights(rec: &mut Rec<'_>, wk: &Work) {
        for l in 0..wk.levels.len() {
            let c = &wk.levels[l];
            if l > 0 {
                let f = &wk.levels[l - 1];
                let pc = P { w: f.w as u32, h: f.h as u32, ow: c.w as u32, oh: c.h as u32, ..Default::default() };
                rec.dispatch("mg_coarsen", [Some(&f.wdd), Some(&c.wdd), Some(&c.e2), None, Some(&f.e2), None], pc, grid2(c.w, c.h));
            }
            rec.dispatch("mg_dinv", [None, Some(&c.wdd), None, Some(&c.e2), None, None], P { w: c.w as u32, h: c.h as u32, ..Default::default() }, grid2(c.w, c.h));
        }
    }

    /// `o = f(A x)` on grid `lv` by `mg_stencil`'s flag (0: A x, 1: rhs − A x,
    /// 2: the damped-Jacobi sweep).
    fn stencil(rec: &mut Rec<'_>, lv: &Lv, x: &wgpu::Buffer, rhs: &wgpu::Buffer, o: &wgpu::Buffer, flag: u32) {
        let pc = P { w: lv.w as u32, h: lv.h as u32, flag, f0: MG_OMEGA, ..Default::default() };
        rec.dispatch("mg_stencil", [Some(x), Some(&lv.wdd), Some(o), Some(&lv.e2), Some(rhs), None], pc, grid2(lv.w, lv.h));
    }

    /// One damped-Jacobi sweep on `lv.x` for `lv.b` (from zero when `zero_start`).
    fn jacobi(rec: &mut Rec<'_>, lv: &mut Lv, zero_start: bool) {
        if zero_start {
            let pc = P { w: lv.w as u32, h: lv.h as u32, f0: MG_OMEGA, ..Default::default() };
            rec.dispatch("mg_jac0", [Some(&lv.b), Some(&lv.wdd), Some(&lv.x), None, None, None], pc, grid1(lv.w * lv.h));
        } else {
            Self::stencil(rec, lv, &lv.x, &lv.b, &lv.r, 2);
            std::mem::swap(&mut lv.x, &mut lv.r);
        }
    }

    /// One V-cycle from `levels[0]` (its `b` set) into its `x`.
    fn vcycle(rec: &mut Rec<'_>, levels: &mut [Lv]) {
        let Some((top, rest)) = levels.split_first_mut() else { return };
        if rest.is_empty() {
            for k in 0..MG_COARSE {
                Self::jacobi(rec, top, k == 0);
            }
            return;
        }
        for k in 0..MG_PRE {
            Self::jacobi(rec, top, k == 0);
        }
        let pc = P { w: top.w as u32, h: top.h as u32, ow: rest[0].w as u32, oh: rest[0].h as u32, ..Default::default() };
        Self::stencil(rec, top, &top.x, &top.b, &top.r, 1);
        rec.dispatch("mg_restrict", [Some(&top.r), None, Some(&rest[0].b), None, None, None], pc, grid2(rest[0].w, rest[0].h));
        Self::vcycle(rec, rest);
        rec.dispatch("mg_prolong", [Some(&rest[0].x), None, Some(&top.x), None, None, None], pc, grid2(top.w, top.h));
        for _ in 0..MG_POST {
            Self::jacobi(rec, top, false);
        }
    }

    /// `scal` by `reduce_scal`'s mode from the dot product of two planes.
    fn dot(rec: &mut Rec<'_>, wk: &Work, a: &wgpu::Buffer, b: &wgpu::Buffer, mode: u32) {
        let pw = P { w: wk.levels[0].w as u32, h: wk.levels[0].h as u32, ..Default::default() };
        rec.dispatch("dot_partial", [Some(a), None, Some(&wk.partials), None, Some(b), None], pw, grid1(wk.levels[0].w * wk.levels[0].h));
        rec.dispatch("reduce_scal", [Some(&wk.partials), None, Some(&wk.scal), None, None, None], P { klen: wk.npart, flag: mode, ..pw }, (1, 1));
    }

    /// WLS solve of (W + λL) u = W d, the data weights in `levels[0]`
    /// (`crate::depth::WlsSolver`): the FGS guess (3 alternating
    /// row/column sweeps), then CG preconditioned by one multigrid V-cycle,
    /// the residual read back every few iterations. Result in `wk.u`; the
    /// final relative residual and the iterations taken.
    async fn wls(&self, g: &Gpu, wk: &mut Work, p: &DepthParams) -> Result<(f32, usize), String> {
        let (dw, dh, n) = (self.dw, self.dh, self.dw * self.dh);
        let pw = P { w: dw as u32, h: dh as u32, ..Default::default() };
        let bytes = (n * 4) as u64;
        let mut rec = g.rec();
        // FGS: f = t2, scratch [cp|dp] = x2b; the schedule's lambda_t as a ratio of the
        // lambda in the edges
        const T: usize = 3;
        {
            let (t, l0) = (&wk.t, &wk.levels[0]);
            let f = &t[2];
            rec.copy(&wk.d, 0, f, 0, bytes);
            for it in 1..=T {
                let lam_t = 1.5 * 4f32.powi((T - it) as i32) / (4f32.powi(T as i32) - 1.0);
                let wt: &wgpu::Buffer = if it == 1 { &l0.wdd } else { &wk.ones };
                rec.dispatch("fgs_rows", [Some(f), Some(&wk.u), Some(&wk.x2b), Some(&l0.e2), Some(wt), None], P { f0: lam_t, ..pw }, (dh.div_ceil(64) as u32, 1));
                rec.copy(&wk.u, 0, f, 0, bytes);
                rec.dispatch("fgs_cols", [Some(f), Some(&wk.u), Some(&wk.x2b), Some(&l0.e2), Some(&wk.ones), None], P { f0: lam_t, ..pw }, (dw.div_ceil(64) as u32, 1));
                rec.copy(&wk.u, 0, f, 0, bytes);
            }
            // CG: bvec = t3, r = t4, p = t5, Ap = t6; z is the V-cycle's output, levels[0].x
            rec.dispatch("mul", [Some(&wk.d), None, Some(&t[3]), None, Some(&l0.wdd), None], pw, grid1(n));
        }
        Self::dot(&mut rec, wk, &wk.t[3], &wk.t[3], 3);
        rec.submit();
        let bnorm = g.read_range_f32(&wk.scal, 4, 1).await?[0].sqrt().max(1e-30);
        let mut rec = g.rec();
        Self::stencil(&mut rec, &wk.levels[0], &wk.u, &wk.t[3], &wk.t[4], 1);
        rec.copy(&wk.t[4], 0, &wk.levels[0].b, 0, bytes);
        Self::vcycle(&mut rec, &mut wk.levels);
        rec.copy(&wk.levels[0].x, 0, &wk.t[5], 0, bytes);
        Self::dot(&mut rec, wk, &wk.t[4], &wk.levels[0].x, 0);
        Self::dot(&mut rec, wk, &wk.t[4], &wk.t[4], 3);
        rec.submit();
        let mut rel = g.read_range_f32(&wk.scal, 4, 1).await?[0].sqrt() / bnorm;
        let mut iters = 0;
        let mut rec = g.rec();
        for _ in 0..p.cg_iters {
            if rel < 1e-5 {
                break;
            }
            iters += 1;
            Self::stencil(&mut rec, &wk.levels[0], &wk.t[5], &wk.t[3], &wk.t[6], 0);
            Self::dot(&mut rec, wk, &wk.t[5], &wk.t[6], 1);
            rec.dispatch("cg_axpy_u", [Some(&wk.t[5]), Some(&wk.u), None, None, None, Some(&wk.scal)], P { flag: 0, ..pw }, grid1(n));
            rec.dispatch("cg_axpy_u", [Some(&wk.t[6]), Some(&wk.t[4]), None, None, None, Some(&wk.scal)], P { flag: 1, ..pw }, grid1(n));
            Self::dot(&mut rec, wk, &wk.t[4], &wk.t[4], 3);
            if iters % 8 == 0 {
                rec.submit();
                rel = g.read_range_f32(&wk.scal, 4, 1).await?[0].sqrt() / bnorm;
                rec = g.rec();
                if rel < 1e-5 {
                    break;
                }
            }
            rec.copy(&wk.t[4], 0, &wk.levels[0].b, 0, bytes);
            Self::vcycle(&mut rec, &mut wk.levels);
            Self::dot(&mut rec, wk, &wk.t[4], &wk.levels[0].x, 2);
            rec.dispatch("cg_update_p", [Some(&wk.levels[0].x), Some(&wk.t[5]), None, None, None, Some(&wk.scal)], pw, grid1(n));
        }
        rec.submit();
        rel = g.read_range_f32(&wk.scal, 4, 1).await?[0].sqrt() / bnorm;
        Ok((rel, iters))
    }
}

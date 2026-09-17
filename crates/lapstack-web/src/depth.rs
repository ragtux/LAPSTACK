// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! Depth from focus on WebGPU — the pipeline of `lapstack_core::depth`
//! (ring difference filter, guided-filter aggregation, streamed sub-frame peak
//! search, confidence-weighted WLS with a Huber reweight, guided upsampling)
//! as WGSL kernels (`shaders.wgsl`, "Depth from focus" section).
//!
//! During the run every frame's focus measure is block-averaged to the
//! working grid on the GPU and kept on the CPU side as a quantised u16 slice
//! (≈ 2 bytes per working-grid pixel per frame). After fusion the slices are
//! streamed back through the guided filter — the aggregation is guided by the
//! *fused* luma, which does not exist before the end of the run — into the
//! peak tracker, and the rest of the pass runs entirely on the device.

use crate::gpu::{Gpu, P, Rec, grid1, grid2};
use lapstack_core::depth::{DepthParams, FocusMeasure, Upsample, rdf_taps};

const EPS_DATA: f32 = 1e-4;

pub struct DepthGpu {
    pub k: usize,
    pub dw: usize,
    pub dh: usize,
    pub params: DepthParams,
    taps: wgpu::Buffer,
    ntaps: u32,
    slice: wgpu::Buffer,
    /// Per frame: the block-averaged focus measure, quantised (values, scale).
    pub slices: Vec<(Vec<u16>, f32)>,
}

/// Working-grid buffers for the finish stage (allocated only then).
struct Work {
    t: Vec<wgpu::Buffer>, // 8 planes
    rt: wgpu::Buffer,     // box-filter row sums
    x2a: wgpu::Buffer,    // 2 planes
    x2b: wgpu::Buffer,    // 2 planes
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

    /// Read the recorded slice back and keep it (quantised).
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
    /// working-grid depth (fractional frame index, dw×dh) and the full-res
    /// depth quantised to u16 (65535 = last frame).
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
    ) -> Result<(Vec<f32>, Vec<u16>), String> {
        let (dw, dh, n) = (self.dw, self.dh, self.dw * self.dh);
        let slices = std::mem::take(&mut self.slices);
        let nf = slices.len();
        if nf == 0 {
            return Err("no frames measured".into());
        }
        let p = self.params.clone();
        let (gx, gy) = grid1(n);
        let wk = Work {
            t: (0..8).map(|i| g.buffer_f32(&format!("dff t{i}"), n)).collect(),
            rt: g.buffer_f32("dff rowsum", n),
            x2a: g.buffer_f32("dff x2a", 2 * n),
            x2b: g.buffer_f32("dff x2b", 2 * n),
            state: g.buffer_f32("dff state", 9 * n),
            guide: g.buffer_f32("dff guide", n),
            d: g.buffer_f32("dff d", n),
            wd: g.buffer_f32("dff wd", n),
            u: g.buffer_f32("dff u", n),
            ones: g.buffer_init("dff ones", bytemuck::cast_slice(&vec![1f32; n])),
            partials: g.buffer_f32("dff partials", (gx * gy) as usize),
            scal: g.buffer_init("dff scal", bytemuck::cast_slice(&[0f32; 4])),
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
        let r_agg = p.agg_radius;
        let guide_stats = |rec: &mut Rec<'_>, r: usize| {
            self.boxf(rec, &wk, &wk.guide, &t[1], r);
            rec.dispatch("mul", [Some(&wk.guide), None, Some(&t[2]), None, Some(&wk.guide), None], pw, grid1(n));
            self.boxf(rec, &wk, &t[2], &t[3], r);
            rec.dispatch("gf_var", [Some(&t[1]), None, None, Some(&wk.x2b), Some(&t[3]), None], pw, grid1(n));
        };
        guide_stats(&mut rec, r_agg);
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

        // ---- noise floor (median of the profile minima), close the profiles
        let cmin = g.read_range_f32(&wk.state, 7 * n, n).await?;
        let mut mins: Vec<f32> = cmin.iter().step_by(7).copied().collect();
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
        let mean_conf = conf.iter().map(|&c| (c / p90.max(1e-6)).min(1.0) as f64).sum::<f64>() / n as f64;
        log(&format!("[lapstack] depth: {nf} slices folded on the {dw}x{dh} grid, confidence p90 {p90:.3}, mean {mean_conf:.3}"));

        // ---- WLS (+ one robust reweight)
        let mut rec = g.rec();
        rec.dispatch("scale_clamp", [Some(&t[1]), None, Some(&wk.wd), None, None, None], P { f0: 1.0 / p90.max(1e-6), f1: EPS_DATA, ..pw }, grid1(n));
        if p.lambda > 0.0 {
            rec.dispatch("edge_w", [Some(&wk.guide), None, Some(&wk.x2a), None, None, None], P { f0: p.sigma_c, ..pw }, grid2(dw, dh));
            self.wls(&mut rec, &wk, &wk.wd, &p);
            if p.robust > 0.0 {
                rec.dispatch("robust_w", [Some(&wk.d), None, Some(&t[7]), Some(&wk.u), Some(&wk.wd), None], P { f0: p.robust, f1: EPS_DATA, ..pw }, grid1(n));
                self.wls(&mut rec, &wk, &t[7], &p);
            }
        } else {
            rec.copy(&wk.d, 0, &wk.u, 0, (n * 4) as u64);
        }
        rec.submit();
        let max_d = (nf - 1) as f32;
        let mut depth_w = g.read_f32(&wk.u, n).await?;
        for v in depth_w.iter_mut() {
            *v = v.clamp(0.0, max_d);
        }

        // ---- guided upsampling to full resolution
        let mut rec = g.rec();
        match p.upsample {
            Upsample::Guided { radius, eps } => {
                guide_stats(&mut rec, radius);
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
        Ok((depth_w, full))
    }

    /// Box mean with border clipping: src -> dst (radius r), row sums in wk.rt.
    fn boxf(&self, rec: &mut Rec<'_>, wk: &Work, src: &wgpu::Buffer, dst: &wgpu::Buffer, r: usize) {
        let pw = P { w: self.dw as u32, h: self.dh as u32, klen: r as u32, ..Default::default() };
        rec.dispatch("box_h", [Some(src), Some(&wk.rt), None, None, None, None], pw, grid2(self.dw, self.dh));
        rec.dispatch("box_v", [None, Some(&wk.rt), Some(dst), None, None, None], pw, grid2(self.dw, self.dh));
    }

    /// WLS solve of (W + λL) u = W d with data weights `wd`: FGS initial guess
    /// (3 alternating row/column sweeps) + Jacobi-PCG. Result in wk.u.
    fn wls(&self, rec: &mut Rec<'_>, wk: &Work, wd: &wgpu::Buffer, p: &DepthParams) {
        let (dw, dh, n) = (self.dw, self.dh, self.dw * self.dh);
        let pw = P { w: dw as u32, h: dh as u32, ..Default::default() };
        let t = &wk.t;
        let f = &t[2];
        let bytes = (n * 4) as u64;
        // FGS
        const T: usize = 3;
        rec.copy(&wk.d, 0, f, 0, bytes);
        for it in 1..=T {
            let lam_t = 1.5 * p.lambda * 4f32.powi((T - it) as i32) / (4f32.powi(T as i32) - 1.0);
            let wt: &wgpu::Buffer = if it == 1 { wd } else { &wk.ones };
            rec.dispatch("fgs_rows", [Some(f), Some(&wk.u), Some(&wk.x2b), Some(&wk.x2a), Some(wt), None], P { f0: lam_t, ..pw }, (dh.div_ceil(64) as u32, 1));
            rec.copy(&wk.u, 0, f, 0, bytes);
            rec.dispatch("fgs_cols", [Some(f), Some(&wk.u), Some(&wk.x2b), Some(&wk.x2a), Some(&wk.ones), None], P { f0: lam_t, ..pw }, (dw.div_ceil(64) as u32, 1));
            rec.copy(&wk.u, 0, f, 0, bytes);
        }
        // PCG: rdiag = x2b, z = t4, pp = t5, ap = t6
        let (rdiag, z, pp, ap) = (&wk.x2b, &t[4], &t[5], &t[6]);
        let pl = P { f0: p.lambda, ..pw };
        rec.dispatch("cg_resid", [Some(&wk.u), Some(&wk.d), Some(rdiag), Some(&wk.x2a), Some(wd), None], pl, grid2(dw, dh));
        rec.dispatch("cg_zp", [Some(rdiag), Some(z), Some(pp), None, None, None], pw, grid1(n));
        rec.dispatch("dot_partial", [Some(rdiag), None, Some(&wk.partials), None, Some(z), None], pw, grid1(n));
        rec.dispatch("reduce_scal", [Some(&wk.partials), None, Some(&wk.scal), None, None, None], P { klen: wk.npart, flag: 0, ..pw }, (1, 1));
        for _ in 0..p.cg_iters {
            rec.dispatch("cg_matvec", [Some(pp), Some(ap), None, Some(&wk.x2a), Some(wd), None], pl, grid2(dw, dh));
            rec.dispatch("dot_partial", [Some(pp), None, Some(&wk.partials), None, Some(ap), None], pw, grid1(n));
            rec.dispatch("reduce_scal", [Some(&wk.partials), None, Some(&wk.scal), None, None, None], P { klen: wk.npart, flag: 1, ..pw }, (1, 1));
            rec.dispatch("cg_axpy_u", [Some(pp), Some(&wk.u), None, None, None, Some(&wk.scal)], pw, grid1(n));
            rec.dispatch("cg_update_rz", [Some(ap), Some(rdiag), Some(z), None, None, Some(&wk.scal)], pw, grid1(n));
            rec.dispatch("dot_partial", [Some(rdiag), None, Some(&wk.partials), None, Some(z), None], pw, grid1(n));
            rec.dispatch("reduce_scal", [Some(&wk.partials), None, Some(&wk.scal), None, None, None], P { klen: wk.npart, flag: 2, ..pw }, (1, 1));
            rec.dispatch("cg_update_p", [Some(z), Some(pp), None, None, None, Some(&wk.scal)], pw, grid1(n));
        }
    }
}

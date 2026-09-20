// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! lapstack-web — the lapstack pipeline for the browser: frames are decoded in
//! WASM, aligned and fused on WebGPU (`shaders.wgsl`), one frame at a time, so
//! memory stays at a couple of frames regardless of stack size. Driven from a
//! Web Worker (`web/worker.js`); the residual rule and PNG encoding run on the
//! CPU side in WASM via lapstack-core. The depth-from-focus pass (`depth.rs`)
//! also runs on WebGPU, after fusion.

mod align;
mod decode;
mod depth;
mod gpu;

use align::{Aligner, LumaPyr, Sim, affine_inv};
use gpu::{Gpu, P, grid1, grid2};
use depth::DepthGpu;
use lapstack_core::depth::{DepthParams, FocusMeasure, Upsample};
use lapstack_core::fuse::{FuseParams, TopRule, binomial, fuse_residuals, upsample_index};
use lapstack_core::pyramid::{Img3, auto_levels, half};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

fn log(s: &str) {
    web_sys::console::log_1(&JsValue::from_str(s));
}
fn now() -> f64 {
    js_sys::Date::now()
}

#[derive(Deserialize, Clone)]
#[serde(default)]
pub struct Params {
    pub levels: Option<usize>,
    pub energy_radius: usize,
    pub top: String,
    pub top_radius: usize,
    pub entropy_bins: usize,
    pub use_chroma: bool,
    pub depth_level: usize,
    pub align: bool,
    pub shift: bool,
    pub scale: bool,
    pub rotation: bool,
    pub coarsen: usize,
    pub proxy_edge: usize,
    /// "dff" = depth from focus (default) | "winner" = pyramid winner map of `depth_level`.
    pub depth: String,
    /// Depth-from-focus tuning (see lapstack_core::depth::DepthParams).
    pub depth_scale: usize,
    pub depth_rdf: [usize; 2],
    pub depth_agg: usize,
    pub depth_agg_eps: f32,
    pub depth_lambda: f32,
    pub depth_sigma: f32,
    pub depth_cg: usize,
    pub depth_median: bool,
    pub depth_gate: f32,
    pub depth_robust: f32,
    pub depth_up: [f32; 2],
}

impl Params {
    fn depth_params(&self) -> DepthParams {
        DepthParams {
            scale: self.depth_scale,
            focus: FocusMeasure::Rdf { r_in: self.depth_rdf[0], r_out: self.depth_rdf[1].max(self.depth_rdf[0] + 1) },
            agg_radius: self.depth_agg,
            agg_eps: self.depth_agg_eps,
            lambda: self.depth_lambda,
            sigma_c: self.depth_sigma,
            cg_iters: self.depth_cg,
            median: self.depth_median,
            upsample: if self.depth_up[0] < 0.0 { Upsample::Bilinear } else { Upsample::Guided { radius: self.depth_up[0] as usize, eps: self.depth_up[1] } },
            gate: self.depth_gate,
            robust: self.depth_robust,
        }
    }
}

impl Default for Params {
    fn default() -> Self {
        let d = DepthParams::default();
        let (r_in, r_out) = match d.focus { FocusMeasure::Rdf { r_in, r_out } => (r_in, r_out), _ => (1, 3) };
        let up = match d.upsample { Upsample::Guided { radius, eps } => [radius as f32, eps], Upsample::Bilinear => [-1.0, 0.0] };
        Params {
            levels: None,
            energy_radius: 1,
            top: "de".into(),
            top_radius: 2,
            entropy_bins: 256,
            use_chroma: false,
            depth_level: 2,
            align: true,
            shift: true,
            scale: true,
            rotation: true,
            coarsen: 2,
            proxy_edge: 1400,
            depth: "dff".into(),
            // quarter resolution: the browser pass trades a little accuracy for
            // ~4x less device memory and time than the native default (half)
            depth_scale: 2,
            depth_rdf: [r_in, r_out],
            depth_agg: d.agg_radius,
            depth_agg_eps: d.agg_eps,
            depth_lambda: d.lambda,
            depth_sigma: d.sigma_c,
            depth_cg: d.cg_iters,
            depth_median: d.median,
            depth_gate: d.gate,
            depth_robust: d.robust,
            depth_up: up,
        }
    }
}

struct Run {
    w: usize,
    h: usize,
    bits: u32,
    fp: FuseParams,
    params: Params,
    levels: usize,
    depth_level: usize,
    dims: Vec<(usize, usize)>,
    cur: Vec<wgpu::Buffer>,
    acc: Vec<wgpu::Buffer>,
    best: Vec<wgpu::Buffer>,
    tmp_half: wgpu::Buffer,
    tmp_full: wgpu::Buffer,
    en: wgpu::Buffer,
    en2: wgpu::Buffer,
    wt: wgpu::Buffer,
    klen: u32,
    up: wgpu::Buffer,
    aff: wgpu::Buffer,
    proxy: (wgpu::Buffer, usize, usize, usize),
    /// Focus-peaking map: region energy of the level `peak.4` band, area-averaged
    /// to (peak.1 x peak.2) by factor peak.3.
    peak: (wgpu::Buffer, usize, usize, usize, usize),
    ref_pyr: Option<LumaPyr>,
    tgt_pyr: Option<LumaPyr>,
    aligner: Option<Aligner>,
    tops: Vec<Img3>,
    sims: Vec<Sim>,
    guess: Sim,
    count: usize,
    fused_rgb16: Option<Vec<u16>>,
    /// Depth map at reduced resolution (values, w, h): the DFF working grid
    /// (fractional frame index) or the pyramid winner map of `depth_level`.
    depth_small: Option<(Vec<f32>, usize, usize)>,
    /// Pyramid winner map of level `depth_level` (values, w, h): free by-product of fusion.
    winner_small: Option<(Vec<f32>, usize, usize)>,
    /// Depth from focus: full-resolution depth, 65535 = last frame.
    depth_full: Option<Vec<u16>>,
    dff: Option<DepthGpu>,
    /// Depth-map rendering (second pass): frames folded so far, and the result.
    render_count: usize,
    dmap_rgb16: Option<Vec<u16>>,
    /// Retouch: the currently loaded aligned source frame (index, RGB u16).
    src_rgb16: Option<(usize, Vec<u16>)>,
    /// The source frame currently warped into `cur[0]` (`load_source`); In focus renders from it.
    src_gpu: Option<usize>,
    undo: Vec<Patch>,
    redo: Vec<Patch>,
    undo_bytes: usize,
}

/// One retouch stroke's effect on the fused image: the bbox and its pixels
/// before and after (RGB u16, row-major within the bbox).
struct Patch {
    /// Which result was painted: false = pyramid, true = depth-map rendering.
    dmap: bool,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    before: Vec<u16>,
    after: Vec<u16>,
}

/// Undo memory cap (before + after copies); oldest strokes are dropped first.
const UNDO_CAP: usize = 600 << 20;

#[wasm_bindgen]
pub struct Engine {
    gpu: Gpu,
    run: Option<Run>,
}

/// Decode a frame on the CPU and area-average it to `edge` px on the long side:
/// {w, h, bits, proxy_w, proxy_h, proxy: Uint8Array (RGBA8)}. Used for the
/// filmstrip / Source view before a run (the browser cannot decode TIFF).
#[wasm_bindgen]
pub fn thumbnail(bytes: &[u8], edge: usize) -> Result<JsValue, JsValue> {
    let f = decode::decode(bytes).map_err(|e| JsValue::from_str(&e))?;
    let (w, h) = (f.w, f.h);
    let pf = (w.max(h)).div_ceil(edge.max(64)).max(1);
    let (pw, ph) = (w.div_ceil(pf), h.div_ceil(pf));
    let mut out = vec![0u8; pw * ph * 4];
    let inv = 1.0 / 65535.0;
    for py in 0..ph {
        for px in 0..pw {
            let (mut r, mut g, mut b, mut n) = (0f32, 0f32, 0f32, 0f32);
            for y in py * pf..((py + 1) * pf).min(h) {
                let row = &f.rgb[y * w * 3..(y + 1) * w * 3];
                for x in px * pf..((px + 1) * pf).min(w) {
                    r += row[3 * x] as f32;
                    g += row[3 * x + 1] as f32;
                    b += row[3 * x + 2] as f32;
                    n += 1.0;
                }
            }
            let k = 255.0 * inv / n.max(1.0);
            let o = 4 * (py * pw + px);
            out[o] = (r * k + 0.5) as u8;
            out[o + 1] = (g * k + 0.5) as u8;
            out[o + 2] = (b * k + 0.5) as u8;
            out[o + 3] = 255;
        }
    }
    let o = js_sys::Object::new();
    set(&o, "w", w as u32);
    set(&o, "h", h as u32);
    set(&o, "bits", f.bits);
    set(&o, "proxy_w", pw as u32);
    set(&o, "proxy_h", ph as u32);
    set(&o, "proxy", js_sys::Uint8Array::from(&out[..]));
    Ok(o.into())
}

#[wasm_bindgen]
pub async fn create_engine() -> Result<Engine, JsValue> {
    console_error_panic_hook::set_once();
    let gpu = Gpu::new().await.map_err(|e| JsValue::from_str(&e))?;
    Ok(Engine { gpu, run: None })
}

fn set(obj: &js_sys::Object, k: &str, v: impl Into<JsValue>) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(k), &v.into());
}

/// {target, x, y, w, h, rgba: Uint8Array} of a bbox of an RGB u16 image.
fn patch_obj(img: &[u16], iw: usize, dmap: bool, x: usize, y: usize, w: usize, h: usize) -> JsValue {
    let mut rgba = vec![0u8; w * h * 4];
    for r in 0..h {
        let src = &img[((y + r) * iw + x) * 3..((y + r) * iw + x + w) * 3];
        let dst = &mut rgba[r * w * 4..(r + 1) * w * 4];
        for c in 0..w {
            for k in 0..3 {
                dst[4 * c + k] = (src[3 * c + k] as f32 / 65535.0 * 255.0 + 0.5) as u8;
            }
            dst[4 * c + 3] = 255;
        }
    }
    let o = js_sys::Object::new();
    set(&o, "target", if dmap { "dmap" } else { "fused" });
    set(&o, "x", x as u32);
    set(&o, "y", y as u32);
    set(&o, "w", w as u32);
    set(&o, "h", h as u32);
    set(&o, "rgba", js_sys::Uint8Array::from(&rgba[..]));
    o.into()
}

#[wasm_bindgen]
impl Engine {
    /// Adapter description + the limits that matter, as JSON.
    pub fn info(&self) -> String {
        let l = &self.gpu.limits;
        format!(
            "{{\"vendor\":{:?},\"device\":{:?},\"backend\":{:?},\"max_buffer_mb\":{},\"max_storage_mb\":{}}}",
            self.gpu.info.vendor,
            self.gpu.info.name,
            format!("{:?}", self.gpu.info.backend),
            l.max_buffer_size / (1 << 20),
            l.max_storage_buffer_binding_size / (1 << 20)
        )
    }

    pub fn reset(&mut self) {
        self.run = None;
    }

    /// Decode, align (chained to frame 0) and fold one frame. The first call
    /// sets up the run from `params_json`. Returns {index, w, h, bits,
    /// proxy_w, proxy_h, proxy: Uint8Array (RGBA8), sim: [dx_px, dy_px, scale, rot_deg], ms}.
    pub async fn push(&mut self, bytes: &[u8], params_json: &str) -> Result<JsValue, JsValue> {
        let t0 = now();
        let frame = decode::decode(bytes).map_err(|e| JsValue::from_str(&e))?;
        let t_dec = now();
        if self.run.is_none() {
            let params: Params = serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&format!("params: {e}")))?;
            self.run = Some(self.setup(frame.w, frame.h, frame.bits, params).map_err(|e| JsValue::from_str(&e))?);
        }
        let g = &self.gpu;
        let run = self.run.as_mut().unwrap();
        if frame.w != run.w || frame.h != run.h {
            return Err(JsValue::from_str(&format!(
                "frame is {}x{} but the stack is {}x{}; frames must share one size",
                frame.w, frame.h, run.w, run.h
            )));
        }
        let (w, h, n) = (run.w, run.h, run.w * run.h);
        g.upload(&run.up, bytemuck::cast_slice(&frame.rgb)).await.map_err(|e| JsValue::from_str(&e))?;
        drop(frame);

        // ---- align: luma pyramid of the new frame, NM search against the previous (warped) frame
        let mut sim = Sim::id();
        let mut t_align = now();
        if run.params.align && run.count > 0 {
            let tgt = run.tgt_pyr.as_ref().unwrap();
            let mut rec = g.rec();
            rec.dispatch("luma_u16", [None, None, Some(&tgt.lv[0].0), None, None, Some(&run.up)], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(n));
            tgt.reduce_chain(&mut rec, &run.tmp_half);
            rec.submit();
            let free = [run.params.shift, run.params.shift, run.params.scale, run.params.rotation];
            sim = run.aligner.as_ref().unwrap().align(g, run.ref_pyr.as_ref().unwrap(), tgt, run.guess, free, run.params.coarsen).await;
            t_align = now();
        }

        // ---- warp into cur[0], proxy, reference luma pyramid for the next frame, Laplacian pyramid
        let mut rec = g.rec();
        let identity = !run.params.align || run.count == 0;
        let mut p = P { w: w as u32, h: h as u32, flag: identity as u32, ..Default::default() };
        if !identity {
            let inv = affine_inv(sim.matrix(w, h));
            g.queue.write_buffer(&run.aff, 0, bytemuck::cast_slice(&[inv[0][2] as f32, inv[1][2] as f32, 0.0, 0.0]));
            p.f0 = inv[0][0] as f32;
            p.f1 = inv[0][1] as f32;
            p.f2 = inv[1][0] as f32;
            p.f3 = inv[1][1] as f32;
        }
        rec.dispatch("warp", [None, None, Some(&run.cur[0]), None, Some(&run.aff), Some(&run.up)], p, grid2(w, h));
        let (pw, ph, pf) = (run.proxy.1, run.proxy.2, run.proxy.3);
        rec.dispatch(
            "proxy",
            [Some(&run.cur[0]), None, Some(&run.proxy.0), None, None, None],
            P { w: w as u32, h: h as u32, ow: pw as u32, oh: ph as u32, klen: pf as u32, ..Default::default() },
            grid2(pw, ph),
        );
        if run.params.align {
            let rf = run.ref_pyr.as_ref().unwrap();
            rec.dispatch("luma_f32", [Some(&run.cur[0]), None, Some(&rf.lv[0].0), None, None, None], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(n));
            rf.reduce_chain(&mut rec, &run.tmp_half);
        }
        // depth from focus: this frame's focus measure, block-averaged to the working grid
        if let Some(dff) = &run.dff {
            rec.dispatch("luma_f32", [Some(&run.cur[0]), None, Some(&run.tmp_full), None, None, None], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(n));
            dff.record_measure(&mut rec, &run.tmp_full, &run.en, w, h);
        }
        // build: L_l = G_l - EXPAND(REDUCE(G_l)), per plane
        for l in 0..run.levels {
            let (fw, fh) = run.dims[l];
            let (cw, ch) = run.dims[l + 1];
            for c in 0..3 {
                let pr = P { w: fw as u32, h: fh as u32, ow: cw as u32, oh: ch as u32, off_in: (c * fw * fh) as u32, off_out: (c * cw * ch) as u32, ..Default::default() };
                rec.dispatch("red_h", [Some(&run.cur[l]), Some(&run.tmp_half), None, None, None, None], pr, grid2(cw, fh));
                rec.dispatch("red_v", [None, Some(&run.tmp_half), Some(&run.cur[l + 1]), None, None, None], pr, grid2(cw, ch));
                let pe = P { w: fw as u32, h: fh as u32, ow: cw as u32, oh: ch as u32, off_in: (c * cw * ch) as u32, off_out: (c * fw * fh) as u32, flag: 1, ..Default::default() };
                rec.dispatch("exp_h", [Some(&run.cur[l + 1]), Some(&run.tmp_half), None, None, None, None], pe, grid2(fw, ch));
                rec.dispatch("exp_v", [None, Some(&run.tmp_half), Some(&run.cur[l]), None, None, None], pe, grid2(fw, fh));
            }
        }
        // region energy + winner-take-all per band-pass level
        for l in 0..run.levels {
            let (lw, lh) = run.dims[l];
            let ln = lw * lh;
            let pl = P { w: lw as u32, h: lh as u32, klen: run.klen, flag: run.fp.use_chroma as u32, ..Default::default() };
            rec.dispatch("energy", [Some(&run.cur[l]), None, None, Some(&run.en), None, None], pl, grid1(ln));
            if run.klen > 1 {
                rec.dispatch("win_h", [None, Some(&run.en2), None, Some(&run.en), Some(&run.wt), None], pl, grid2(lw, lh));
                rec.dispatch("win_v", [None, Some(&run.en2), None, Some(&run.en), Some(&run.wt), None], pl, grid2(lw, lh));
            }
            if l == run.peak.4 {
                let (kw, kh, kf) = (run.peak.1, run.peak.2, run.peak.3);
                rec.dispatch(
                    "down1",
                    [Some(&run.en), None, Some(&run.peak.0), None, None, None],
                    P { w: lw as u32, h: lh as u32, ow: kw as u32, oh: kh as u32, klen: kf as u32, ..Default::default() },
                    grid2(kw, kh),
                );
            }
            let ps = P { w: lw as u32, h: lh as u32, flag: (l == run.depth_level) as u32, f0: run.count as f32, ..Default::default() };
            rec.dispatch("sel", [Some(&run.cur[l]), Some(&run.best[l]), Some(&run.acc[l]), Some(&run.en), None, None], ps, grid1(ln));
        }
        rec.submit();
        // residual (tiny) to the host; proxy to the caller
        let (tw, th) = run.dims[run.levels];
        let top = g.read_f32(&run.cur[run.levels], 3 * tw * th).await.map_err(|e| JsValue::from_str(&e))?;
        run.tops.push(Img3 { w: tw, h: th, p: [top[..tw * th].to_vec(), top[tw * th..2 * tw * th].to_vec(), top[2 * tw * th..].to_vec()] });
        let proxy = g.read(&run.proxy.0, (pw * ph * 4) as u64).await.map_err(|e| JsValue::from_str(&e))?;
        let (kw, kh) = (run.peak.1, run.peak.2);
        let peak = g.read_f32(&run.peak.0, kw * kh).await.map_err(|e| JsValue::from_str(&e))?;
        if let Some(dff) = &mut run.dff {
            dff.take_slice(g).await.map_err(|e| JsValue::from_str(&e))?;
        }
        run.sims.push(sim);
        run.guess = sim;
        let idx = run.count;
        run.count += 1;
        let t1 = now();
        log(&format!(
            "[lapstack] frame {idx}: decode {:.0} ms, align {:.0} ms, fuse {:.0} ms",
            t_dec - t0,
            t_align - t_dec,
            t1 - t_align
        ));
        let o = js_sys::Object::new();
        set(&o, "index", idx as u32);
        set(&o, "w", w as u32);
        set(&o, "h", h as u32);
        set(&o, "bits", run.bits);
        set(&o, "proxy_w", pw as u32);
        set(&o, "proxy_h", ph as u32);
        set(&o, "proxy", js_sys::Uint8Array::from(&proxy[..]));
        set(&o, "peak_w", kw as u32);
        set(&o, "peak_h", kh as u32);
        set(&o, "peak", js_sys::Float32Array::from(&peak[..]));
        let sv = js_sys::Array::new();
        for v in [sim.xoff * w as f64, sim.yoff * h as f64, sim.scale, sim.rot.to_degrees()] {
            sv.push(&JsValue::from_f64(v));
        }
        set(&o, "sim", sv);
        set(&o, "ms", t1 - t0);
        Ok(o.into())
    }

    /// Fuse the residuals, collapse, and read back the result. Returns
    /// {w, h, bits, frames, rgba: Uint8Array, depth_w, depth_h, depth: Float32Array,
    /// winner_w, winner_h, winner: Float32Array, ms} — `depth` is the depth-from-focus
    /// map on its working grid, `winner` the pyramid winner index of `depth_level`.
    pub async fn finish(&mut self) -> Result<JsValue, JsValue> {
        let t0 = now();
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no frames pushed"))?;
        let (w, h, n) = (run.w, run.h, run.w * run.h);
        let top = fuse_residuals(&run.tops, &run.fp);
        let (tw, th) = run.dims[run.levels];
        let mut flat = Vec::with_capacity(3 * tw * th);
        for c in 0..3 {
            flat.extend_from_slice(&top.p[c]);
        }
        g.queue.write_buffer(&run.acc[run.levels], 0, bytemuck::cast_slice(&flat));
        let mut rec = g.rec();
        for l in (0..run.levels).rev() {
            let (fw, fh) = run.dims[l];
            let (cw, ch) = run.dims[l + 1];
            for c in 0..3 {
                let pe = P { w: fw as u32, h: fh as u32, ow: cw as u32, oh: ch as u32, off_in: (c * cw * ch) as u32, off_out: (c * fw * fh) as u32, flag: 2, ..Default::default() };
                rec.dispatch("exp_h", [Some(&run.acc[l + 1]), Some(&run.tmp_half), None, None, None, None], pe, grid2(fw, ch));
                rec.dispatch("exp_v", [None, Some(&run.tmp_half), Some(&run.acc[l]), None, None, None], pe, grid2(fw, fh));
            }
        }
        let pw = P { w: w as u32, h: h as u32, ..Default::default() };
        rec.dispatch("clamp01", [None, None, Some(&run.acc[0]), None, None, None], pw, grid1(3 * n));
        rec.dispatch("to_rgba8", [Some(&run.acc[0]), None, Some(&run.tmp_full), None, None, None], pw, grid1(n));
        let rgb16 = g.buffer("rgb16 out", ((3 * n).div_ceil(2) * 4) as u64);
        rec.dispatch("to_rgb16", [Some(&run.acc[0]), None, Some(&rgb16), None, None, None], pw, grid1((3 * n).div_ceil(2)));
        rec.submit();
        let rgba = g.read(&run.tmp_full, (n * 4) as u64).await.map_err(|e| JsValue::from_str(&e))?;
        let r16 = g.read(&rgb16, ((3 * n).div_ceil(2) * 4) as u64).await.map_err(|e| JsValue::from_str(&e))?;
        drop(rgb16);
        let mut v16: Vec<u16> = bytemuck::cast_slice(&r16).to_vec();
        v16.truncate(3 * n);
        run.fused_rgb16 = Some(v16);
        let t_fuse = now();
        // the pyramid winner map (plane 3 of the depth level's accumulator) is always there
        let (ww, wh) = run.dims[run.depth_level];
        let wn = ww * wh;
        let winner = g.read_f32(&run.acc[run.depth_level], 4 * wn).await.map_err(|e| JsValue::from_str(&e))?[3 * wn..].to_vec();
        let (depth, dw, dh) = if let Some(dff) = &mut run.dff {
            // depth from focus, guided by the fused luma
            let mut rec = g.rec();
            rec.dispatch("luma_f32", [Some(&run.acc[0]), None, Some(&run.en), None, None, None], pw, grid1(n));
            rec.submit();
            let (dw, dh) = (dff.dw, dff.dh);
            let (depth_w, full) = dff
                .finish(g, w, h, &run.en, &run.tmp_full, &run.en2, &run.up, &|s| log(s))
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            run.depth_full = Some(full);
            (depth_w, dw, dh)
        } else {
            // the renderer reads the full-res depth from en2
            let full = upsample_index(&winner, ww, wh, w, h, run.depth_level);
            g.queue.write_buffer(&run.en2, 0, bytemuck::cast_slice(&full));
            (winner.clone(), ww, wh)
        };
        run.depth_small = Some((depth.clone(), dw, dh));
        run.winner_small = Some((winner.clone(), ww, wh));
        let t1 = now();
        log(&format!("[lapstack] finish: {:.0} ms (collapse {:.0} ms, depth {:.0} ms)", t1 - t0, t_fuse - t0, t1 - t_fuse));
        let o = js_sys::Object::new();
        set(&o, "w", w as u32);
        set(&o, "h", h as u32);
        set(&o, "bits", run.bits);
        set(&o, "frames", run.count as u32);
        set(&o, "rgba", js_sys::Uint8Array::from(&rgba[..]));
        set(&o, "depth_w", dw as u32);
        set(&o, "depth_h", dh as u32);
        set(&o, "depth", js_sys::Float32Array::from(&depth[..]));
        set(&o, "winner_w", ww as u32);
        set(&o, "winner_h", wh as u32);
        set(&o, "winner", js_sys::Float32Array::from(&winner[..]));
        set(&o, "ms", t1 - t0);
        Ok(o.into())
    }

    /// Depth-map rendering, pass 2 (the DMAP renderer): fold one frame,
    /// decoded again and warped with the run's registration, into the
    /// accumulator with weight `1 − |index − depth|` per pixel. The first call
    /// starts the pass. Returns {index, ms}.
    pub async fn render_push(&mut self, index: usize, bytes: &[u8]) -> Result<JsValue, JsValue> {
        let t0 = now();
        let frame = decode::decode(bytes).map_err(|e| JsValue::from_str(&e))?;
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        if run.fused_rgb16.is_none() {
            return Err(JsValue::from_str("finish the run first"));
        }
        if frame.w != run.w || frame.h != run.h {
            return Err(JsValue::from_str("frame size differs from the run"));
        }
        let sim = *run.sims.get(index).ok_or_else(|| JsValue::from_str("unknown frame index"))?;
        let (w, h, n) = (run.w, run.h, run.w * run.h);
        g.queue.write_buffer(&run.up, 0, bytemuck::cast_slice(&frame.rgb));
        drop(frame);
        let mut rec = g.rec();
        if run.render_count == 0 {
            // accumulator = acc[0] (the pyramid result was read back already), weights = best[0]
            rec.clear(&run.acc[0]);
            rec.clear(&run.best[0]);
        }
        let identity = !run.params.align || index == 0;
        let mut p = P { w: w as u32, h: h as u32, flag: identity as u32, ..Default::default() };
        if !identity {
            let inv = affine_inv(sim.matrix(w, h));
            g.queue.write_buffer(&run.aff, 0, bytemuck::cast_slice(&[inv[0][2] as f32, inv[1][2] as f32, 0.0, 0.0]));
            p.f0 = inv[0][0] as f32;
            p.f1 = inv[0][1] as f32;
            p.f2 = inv[1][0] as f32;
            p.f3 = inv[1][1] as f32;
        }
        rec.dispatch("warp", [None, None, Some(&run.cur[0]), None, Some(&run.aff), Some(&run.up)], p, grid2(w, h));
        run.src_gpu = None;
        rec.dispatch(
            "dmap_acc",
            [Some(&run.cur[0]), Some(&run.acc[0]), Some(&run.best[0]), None, Some(&run.en2), None],
            P { w: w as u32, h: h as u32, f0: index as f32, ..Default::default() },
            grid1(n),
        );
        rec.submit();
        run.render_count += 1;
        let o = js_sys::Object::new();
        set(&o, "index", index as u32);
        set(&o, "ms", now() - t0);
        Ok(o.into())
    }

    /// Normalise the depth-map rendering and read it back: {w, h, rgba, ms}.
    pub async fn render_finish(&mut self) -> Result<JsValue, JsValue> {
        let t0 = now();
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        if run.render_count == 0 {
            return Err(JsValue::from_str("no frames rendered"));
        }
        let (w, h, n) = (run.w, run.h, run.w * run.h);
        let pw = P { w: w as u32, h: h as u32, ..Default::default() };
        let mut rec = g.rec();
        rec.dispatch("dmap_norm", [None, Some(&run.acc[0]), Some(&run.best[0]), None, None, None], pw, grid1(n));
        rec.dispatch("to_rgba8", [Some(&run.acc[0]), None, Some(&run.tmp_full), None, None, None], pw, grid1(n));
        let rgb16 = g.buffer("dmap rgb16", ((3 * n).div_ceil(2) * 4) as u64);
        rec.dispatch("to_rgb16", [Some(&run.acc[0]), None, Some(&rgb16), None, None, None], pw, grid1((3 * n).div_ceil(2)));
        rec.submit();
        let rgba = g.read(&run.tmp_full, (n * 4) as u64).await.map_err(|e| JsValue::from_str(&e))?;
        let r16 = g.read(&rgb16, ((3 * n).div_ceil(2) * 4) as u64).await.map_err(|e| JsValue::from_str(&e))?;
        let mut v16: Vec<u16> = bytemuck::cast_slice(&r16).to_vec();
        v16.truncate(3 * n);
        run.dmap_rgb16 = Some(v16);
        run.render_count = 0;
        log(&format!("[lapstack] depth-map rendering finished ({:.0} ms)", now() - t0));
        let o = js_sys::Object::new();
        set(&o, "w", w as u32);
        set(&o, "h", h as u32);
        set(&o, "rgba", js_sys::Uint8Array::from(&rgba[..]));
        set(&o, "ms", now() - t0);
        Ok(o.into())
    }

    /// Decode frame `index` again and warp it with the registration found
    /// during the run into `cur[0]`, the GPU-resident source frame that
    /// `source_focus` renders from. With `readback` it is also copied to the
    /// CPU as the retouch brush source and returned for display, {index, w, h,
    /// rgba: Uint8Array} (full-resolution RGBA8); without, the reply is just
    /// {index, w, h} and the two full-frame readbacks are skipped.
    pub async fn load_source(&mut self, index: usize, bytes: &[u8], readback: bool) -> Result<JsValue, JsValue> {
        let t0 = now();
        let frame = decode::decode(bytes).map_err(|e| JsValue::from_str(&e))?;
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        if run.fused_rgb16.is_none() {
            return Err(JsValue::from_str("finish the run first"));
        }
        if frame.w != run.w || frame.h != run.h {
            return Err(JsValue::from_str("frame size differs from the run"));
        }
        let sim = *run.sims.get(index).ok_or_else(|| JsValue::from_str("unknown frame index"))?;
        let (w, h) = (run.w, run.h);
        g.queue.write_buffer(&run.up, 0, bytemuck::cast_slice(&frame.rgb));
        drop(frame);
        let identity = !run.params.align || index == 0;
        let mut p = P { w: w as u32, h: h as u32, flag: identity as u32, ..Default::default() };
        if !identity {
            let inv = affine_inv(sim.matrix(w, h));
            g.queue.write_buffer(&run.aff, 0, bytemuck::cast_slice(&[inv[0][2] as f32, inv[1][2] as f32, 0.0, 0.0]));
            p.f0 = inv[0][0] as f32;
            p.f1 = inv[0][1] as f32;
            p.f2 = inv[1][0] as f32;
            p.f3 = inv[1][1] as f32;
        }
        let mut rec = g.rec();
        rec.dispatch("warp", [None, None, Some(&run.cur[0]), None, Some(&run.aff), Some(&run.up)], p, grid2(w, h));
        rec.submit();
        run.src_gpu = Some(index);
        if !readback {
            log(&format!("[lapstack] source {index} warped ({:.0} ms)", now() - t0));
            let o = js_sys::Object::new();
            set(&o, "index", index as u32);
            set(&o, "w", w as u32);
            set(&o, "h", h as u32);
            return Ok(o.into());
        }
        let o = self.source_readback().await?;
        log(&format!("[lapstack] retouch source {index} loaded ({:.0} ms)", now() - t0));
        Ok(o)
    }

    /// Retouch: read the GPU-resident source frame back as the 16-bit brush
    /// source, and return it for display: {index, w, h, rgba: Uint8Array}.
    pub async fn source_readback(&mut self) -> Result<JsValue, JsValue> {
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let index = run.src_gpu.ok_or_else(|| JsValue::from_str("no source loaded"))?;
        let (w, h, n) = (run.w, run.h, run.w * run.h);
        let pw = P { w: w as u32, h: h as u32, ..Default::default() };
        let mut rec = g.rec();
        rec.dispatch("to_rgba8", [Some(&run.cur[0]), None, Some(&run.tmp_full), None, None, None], pw, grid1(n));
        let rgb16 = g.buffer("src rgb16", ((3 * n).div_ceil(2) * 4) as u64);
        rec.dispatch("to_rgb16", [Some(&run.cur[0]), None, Some(&rgb16), None, None, None], pw, grid1((3 * n).div_ceil(2)));
        rec.submit();
        let rgba = g.read(&run.tmp_full, (n * 4) as u64).await.map_err(|e| JsValue::from_str(&e))?;
        let r16 = g.read(&rgb16, ((3 * n).div_ceil(2) * 4) as u64).await.map_err(|e| JsValue::from_str(&e))?;
        let mut v16: Vec<u16> = bytemuck::cast_slice(&r16).to_vec();
        v16.truncate(3 * n);
        run.src_rgb16 = Some((index, v16));
        let o = js_sys::Object::new();
        set(&o, "index", index as u32);
        set(&o, "w", w as u32);
        set(&o, "h", h as u32);
        set(&o, "rgba", js_sys::Uint8Array::from(&rgba[..]));
        Ok(o.into())
    }

    /// Index of the source frame held on the CPU as the retouch brush source, or -1.
    pub fn source_index(&self) -> i32 {
        self.run.as_ref().and_then(|r| r.src_rgb16.as_ref()).map_or(-1, |(i, _)| *i as i32)
    }

    /// Index of the source frame currently warped into `cur[0]` on the GPU, or -1.
    pub fn source_gpu_index(&self) -> i32 {
        self.run.as_ref().and_then(|r| r.src_gpu).map_or(-1, |i| i as i32)
    }

    /// The GPU-resident source frame rendered as the "In focus" layer: every
    /// pixel is darkened by how far, in frames, the full-resolution depth map
    /// (`en2`, a frame index per pixel) puts it from that frame — weight
    /// w = clamp((w1 − |depth − index|) / (w1 − w0), 0, 1), full inside ±w0
    /// frames, none beyond ±w1 — and the out-of-focus part keeps its outlines:
    /// out = rgb·(w + (1 − w)·dim) + (1 − w)·soft(tex·|luma − box(luma)|),
    /// the box being (2r+1)² with r = w/1000 px and soft(t) = cap·(1 − e^(−t/cap)),
    /// cap = 0.4, so hard edges stay light grey. Three kernels (luma, its row
    /// sums, the `focus_out` combine) and one RGBA8 readback; `en` and
    /// `tmp_full` are scratch. Returns {index, w, h, rgba}.
    pub async fn source_focus(&self, dim: f32, w0: f32, w1: f32, tex: f32) -> Result<JsValue, JsValue> {
        let t0 = now();
        let g = &self.gpu;
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no run"))?;
        let index = run.src_gpu.ok_or_else(|| JsValue::from_str("no source loaded"))?;
        if run.fused_rgb16.is_none() {
            return Err(JsValue::from_str("finish the run first"));
        }
        let (w, h, n) = (run.w, run.h, run.w * run.h);
        let r = (w / 1000).max(1) as u32;
        let pw = P { w: w as u32, h: h as u32, ..Default::default() };
        let mut rec = g.rec();
        rec.dispatch("luma_f32", [Some(&run.cur[0]), None, Some(&run.en), None, None, None], pw, grid1(n));
        rec.dispatch("box_h", [Some(&run.en), Some(&run.tmp_full), None, None, None, None], P { klen: r, ..pw }, grid2(w, h));
        rec.dispatch(
            "focus_out",
            [Some(&run.cur[0]), Some(&run.tmp_full), Some(&run.en), None, Some(&run.en2), None],
            P { klen: r, off_in: index as u32, f0: dim, f1: w0, f2: w1, f3: tex, ..pw },
            grid2(w, h),
        );
        rec.submit();
        let rgba = g.read(&run.en, (n * 4) as u64).await.map_err(|e| JsValue::from_str(&e))?;
        log(&format!("[lapstack] In focus {index} rendered ({:.0} ms)", now() - t0));
        let o = js_sys::Object::new();
        set(&o, "index", index as u32);
        set(&o, "w", w as u32);
        set(&o, "h", h as u32);
        set(&o, "rgba", js_sys::Uint8Array::from(&rgba[..]));
        Ok(o.into())
    }

    /// Retouch: apply a stroke of soft dabs `[x, y, radius, hardness, ...]`
    /// (image px) copying the loaded source into the fused image (16-bit).
    /// Returns the updated bbox as {x, y, w, h, rgba}.
    pub fn stroke(&mut self, dabs: &[f32], target: &str) -> Result<JsValue, JsValue> {
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let (w, h) = (run.w, run.h);
        let (_, src) = run.src_rgb16.as_ref().ok_or_else(|| JsValue::from_str("no source loaded"))?;
        let dmap = target == "dmap";
        let fused = if dmap { run.dmap_rgb16.as_mut() } else { run.fused_rgb16.as_mut() }
            .ok_or_else(|| JsValue::from_str(if dmap { "no depth-map rendering" } else { "not finished" }))?;
        // bbox of the stroke
        let (mut x0, mut y0, mut x1, mut y1) = (w as f32, h as f32, 0f32, 0f32);
        for d in dabs.chunks_exact(4) {
            x0 = x0.min(d[0] - d[2]);
            y0 = y0.min(d[1] - d[2]);
            x1 = x1.max(d[0] + d[2]);
            y1 = y1.max(d[1] + d[2]);
        }
        let bx = x0.floor().max(0.0) as usize;
        let by = y0.floor().max(0.0) as usize;
        let ex = (x1.ceil() as isize + 1).clamp(0, w as isize) as usize;
        let ey = (y1.ceil() as isize + 1).clamp(0, h as isize) as usize;
        if ex <= bx || ey <= by {
            return Ok(JsValue::NULL);
        }
        let (bw, bh) = (ex - bx, ey - by);
        let mut before = Vec::with_capacity(bw * bh * 3);
        for r in 0..bh {
            before.extend_from_slice(&fused[((by + r) * w + bx) * 3..((by + r) * w + bx + bw) * 3]);
        }
        // soft dabs: weight 1 inside hardness·r, falling to 0 at r (smoothstep)
        for d in dabs.chunks_exact(4) {
            let (cx, cy, r, hard) = (d[0], d[1], d[2].max(0.5), d[3].clamp(0.0, 0.999));
            let (lx, ly) = ((cx - r).floor().max(0.0) as usize, (cy - r).floor().max(0.0) as usize);
            let (hx, hy) = (((cx + r).ceil() as isize + 1).clamp(0, w as isize) as usize, ((cy + r).ceil() as isize + 1).clamp(0, h as isize) as usize);
            for y in ly..hy {
                for x in lx..hx {
                    let t = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / r;
                    if t >= 1.0 {
                        continue;
                    }
                    let m = if t <= hard { 1.0 } else { let u = (t - hard) / (1.0 - hard); 1.0 - u * u * (3.0 - 2.0 * u) };
                    let i = (y * w + x) * 3;
                    for k in 0..3 {
                        let f = fused[i + k] as f32;
                        fused[i + k] = (f + (src[i + k] as f32 - f) * m + 0.5) as u16;
                    }
                }
            }
        }
        let mut after = Vec::with_capacity(bw * bh * 3);
        for r in 0..bh {
            after.extend_from_slice(&fused[((by + r) * w + bx) * 3..((by + r) * w + bx + bw) * 3]);
        }
        run.undo_bytes += 2 * before.len() * 2;
        run.undo.push(Patch { dmap, x: bx, y: by, w: bw, h: bh, before, after });
        for p in run.redo.drain(..) {
            let _ = p;
        }
        while run.undo_bytes > UNDO_CAP && run.undo.len() > 1 {
            let p = run.undo.remove(0);
            run.undo_bytes -= 2 * p.before.len() * 2;
        }
        Ok(patch_obj(fused, w, dmap, bx, by, bw, bh))
    }

    fn restore(&mut self, redo: bool) -> Result<JsValue, JsValue> {
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let w = run.w;
        let Some(p) = (if redo { run.redo.pop() } else { run.undo.pop() }) else { return Ok(JsValue::NULL) };
        let fused = if p.dmap { run.dmap_rgb16.as_mut() } else { run.fused_rgb16.as_mut() }.ok_or_else(|| JsValue::from_str("not finished"))?;
        let pixels = if redo { &p.after } else { &p.before };
        for r in 0..p.h {
            fused[((p.y + r) * w + p.x) * 3..((p.y + r) * w + p.x + p.w) * 3].copy_from_slice(&pixels[r * p.w * 3..(r + 1) * p.w * 3]);
        }
        let out = patch_obj(fused, w, p.dmap, p.x, p.y, p.w, p.h);
        if redo { run.undo.push(p) } else { run.redo.push(p) }
        Ok(out)
    }

    pub fn undo(&mut self) -> Result<JsValue, JsValue> {
        self.restore(false)
    }
    pub fn redo(&mut self) -> Result<JsValue, JsValue> {
        self.restore(true)
    }
    /// [undo depth, redo depth]
    pub fn history(&self) -> Vec<u32> {
        self.run.as_ref().map_or(vec![0, 0], |r| vec![r.undo.len() as u32, r.redo.len() as u32])
    }

    /// Encode the fused image or the depth map. `format`: "png" (fused at the
    /// input bit depth, depth map 8-bit gray), "png8" (8-bit), "jpeg" (8-bit,
    /// `quality` 1..100). Returns the file bytes.
    pub fn encode(&self, kind: &str, format: &str, quality: u8) -> Result<js_sys::Uint8Array, JsValue> {
        use image::ImageEncoder;
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no result"))?;
        let (w, h) = (run.w as u32, run.h as u32);
        let err = |e: image::ImageError| JsValue::from_str(&format!("encode: {e}"));
        let mut out = Vec::new();
        let (pixels8, pixels16, color): (Vec<u8>, Option<&[u16]>, image::ExtendedColorType) = match kind {
            "fused" | "dmap" => {
                let v = if kind == "dmap" { run.dmap_rgb16.as_ref() } else { run.fused_rgb16.as_ref() }
                    .ok_or_else(|| JsValue::from_str(if kind == "dmap" { "no depth-map rendering" } else { "not finished" }))?;
                if format == "png" && run.bits == 16 {
                    (Vec::new(), Some(v), image::ExtendedColorType::Rgb16)
                } else {
                    (v.iter().map(|&s| (s as f32 / 65535.0 * 255.0 + 0.5) as u8).collect(), None, image::ExtendedColorType::Rgb8)
                }
            }
            "winner" => {
                let (d, dw, dh) = run.winner_small.as_ref().ok_or_else(|| JsValue::from_str("not finished"))?;
                let full = upsample_index(d, *dw, *dh, run.w, run.h, run.depth_level);
                let (lo, hi) = full.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
                let range = (hi - lo).max(1e-6);
                (full.iter().map(|&v| ((v - lo) / range * 255.0 + 0.5) as u8).collect(), None, image::ExtendedColorType::L8)
            }
            "depth" | "depth16" => {
                // full-resolution frame index, 65535 = last frame
                let full: std::borrow::Cow<'_, [u16]> = match &run.depth_full {
                    Some(f) => std::borrow::Cow::Borrowed(f),
                    None => {
                        let (d, dw, dh) = run.depth_small.as_ref().ok_or_else(|| JsValue::from_str("not finished"))?;
                        let k = 65535.0 / (run.count.max(2) - 1) as f32;
                        std::borrow::Cow::Owned(upsample_index(d, *dw, *dh, run.w, run.h, run.depth_level).iter().map(|&v| (v * k + 0.5) as u16).collect())
                    }
                };
                if kind == "depth16" {
                    let bytes: Vec<u8> = full.iter().flat_map(|v| v.to_be_bytes()).collect();
                    let enc = image::codecs::png::PngEncoder::new(&mut out);
                    enc.write_image(&bytes, w, h, image::ExtendedColorType::L16).map_err(err)?;
                    return Ok(js_sys::Uint8Array::from(&out[..]));
                }
                let (lo, hi) = full.iter().fold((u16::MAX, 0u16), |(a, b), &v| (a.min(v), b.max(v)));
                let range = (hi as f32 - lo as f32).max(1e-6);
                (full.iter().map(|&v| ((v - lo) as f32 / range * 255.0 + 0.5) as u8).collect(), None, image::ExtendedColorType::L8)
            }
            _ => return Err(JsValue::from_str("kind must be fused|dmap|depth|depth16|winner")),
        };
        match format {
            "jpeg" => {
                let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality.clamp(1, 100));
                enc.write_image(&pixels8, w, h, color).map_err(err)?;
            }
            _ => {
                let enc = image::codecs::png::PngEncoder::new(&mut out);
                match pixels16 {
                    Some(v) => enc.write_image(bytemuck::cast_slice(v), w, h, color).map_err(err)?,
                    None => enc.write_image(&pixels8, w, h, color).map_err(err)?,
                }
            }
        }
        Ok(js_sys::Uint8Array::from(&out[..]))
    }

    /// Kept for the test page: PNG at the input bit depth.
    pub fn encode_png(&self, kind: &str) -> Result<js_sys::Uint8Array, JsValue> {
        self.encode(kind, "png", 90)
    }

    /// Full-resolution depth map (u16, 65535 = last frame); for tests.
    pub fn depth_full(&self) -> Result<js_sys::Uint16Array, JsValue> {
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no result"))?;
        let d = run.depth_full.as_ref().ok_or_else(|| JsValue::from_str("no depth-from-focus result"))?;
        Ok(js_sys::Uint16Array::from(&d[..]))
    }
}

impl Engine {
    fn setup(&self, w: usize, h: usize, bits: u32, params: Params) -> Result<Run, String> {
        let g = &self.gpu;
        let n = w * h;
        let need = (3 * n * 4) as u64;
        if need > g.limits.max_buffer_size || need > g.limits.max_storage_buffer_binding_size {
            return Err(format!(
                "{w}x{h} needs {} MB storage buffers; this GPU allows {} MB per buffer / {} MB per binding. Use smaller frames.",
                need >> 20,
                g.limits.max_buffer_size >> 20,
                g.limits.max_storage_buffer_binding_size >> 20
            ));
        }
        let top_rule = TopRule::parse(&params.top).ok_or_else(|| format!("unknown top rule '{}'", params.top))?;
        let fp = FuseParams {
            levels: params.levels,
            energy_radius: params.energy_radius,
            top_rule,
            top_radius: params.top_radius,
            entropy_bins: params.entropy_bins,
            use_chroma: params.use_chroma,
            depth_level: params.depth_level,
        };
        let levels = params.levels.unwrap_or_else(|| auto_levels(w, h, 32)).max(1);
        let depth_level = params.depth_level.min(levels - 1);
        let mut dims = vec![(w, h)];
        for _ in 0..levels {
            let (cw, ch) = *dims.last().unwrap();
            dims.push((half(cw), half(ch)));
        }
        let cur: Vec<_> = dims.iter().enumerate().map(|(l, &(lw, lh))| g.buffer_f32(&format!("cur{l}"), 3 * lw * lh)).collect();
        let acc: Vec<_> = dims
            .iter()
            .enumerate()
            .map(|(l, &(lw, lh))| g.buffer_f32(&format!("acc{l}"), if l == depth_level { 4 } else { 3 } * lw * lh))
            .collect();
        let best: Vec<_> = dims[..levels]
            .iter()
            .map(|&(lw, lh)| g.buffer_init("best", bytemuck::cast_slice(&vec![-1.0f32; lw * lh])))
            .collect();
        let wtv = binomial(params.energy_radius);
        let klen = wtv.len() as u32;
        let wt = g.buffer_init("window", bytemuck::cast_slice(&wtv));
        let pf = (w.max(h)).div_ceil(params.proxy_edge.max(64)).max(1);
        let (pw, ph) = (w.div_ceil(pf), h.div_ceil(pf));
        // peaking from the first band above the finest (noise) one, when there is one
        let pl = 1usize.min(levels - 1);
        let (lw1, lh1) = dims[pl];
        let kf = (lw1.max(lh1)).div_ceil(params.proxy_edge.max(64)).max(1);
        let (kw, kh) = (lw1.div_ceil(kf), lh1.div_ceil(kf));
        let dff = match params.depth.as_str() {
            "dff" => Some(DepthGpu::new(g, w, h, params.depth_params())?),
            "winner" => None,
            other => return Err(format!("depth must be dff|winner, not '{other}'")),
        };
        let (ref_pyr, tgt_pyr, aligner) = if params.align {
            (Some(LumaPyr::new(g, w, h, "ref")), Some(LumaPyr::new(g, w, h, "tgt")), Some(Aligner::new(g, w, h)))
        } else {
            (None, None, None)
        };
        log(&format!(
            "[lapstack] run: {w}x{h} {bits}-bit, {levels} levels (residual {}x{}), window {klen}x{klen}, top {:?}, align {}, depth {}",
            dims[levels].0, dims[levels].1, top_rule, params.align,
            match &dff { Some(d) => format!("from focus on a {}x{} grid", d.dw, d.dh), None => format!("winner map of level {depth_level}") }
        ));
        Ok(Run {
            w,
            h,
            bits,
            fp,
            params,
            levels,
            depth_level,
            dims,
            cur,
            acc,
            best,
            tmp_half: g.buffer_f32("tmp_half", (half(w) * h).max(half(h) * w)),
            tmp_full: g.buffer_f32("tmp_full", n),
            en: g.buffer_f32("en", n),
            en2: g.buffer_f32("en2", n),
            wt,
            klen,
            up: g.buffer("upload u16", (3 * n).div_ceil(2) as u64 * 4),
            aff: g.buffer_init("affine", bytemuck::cast_slice(&[0f32; 4])),
            proxy: (g.buffer("proxy", (pw * ph * 4) as u64), pw, ph, pf),
            peak: (g.buffer_f32("peak", kw * kh), kw, kh, kf, pl),
            ref_pyr,
            tgt_pyr,
            aligner,
            tops: Vec::new(),
            sims: Vec::new(),
            guess: Sim::id(),
            count: 0,
            fused_rgb16: None,
            depth_small: None,
            winner_small: None,
            depth_full: None,
            dff,
            render_count: 0,
            dmap_rgb16: None,
            src_rgb16: None,
            src_gpu: None,
            undo: Vec::new(),
            redo: Vec::new(),
            undo_bytes: 0,
        })
    }
}

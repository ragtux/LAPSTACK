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
mod gif;
mod gpu;

use align::{Aligner, LumaPyr, Sim, affine_inv};
use gpu::{Gpu, P, Rec, grid1, grid2};
use depth::DepthGpu;
use lapstack_core::depth::{DepthParams, FocusMeasure, Upsample};
use lapstack_core::fuse::{FuseParams, TopRule, binomial, fuse_residuals, upsample_index};
use lapstack_core::align::{Rect, common_area};
use lapstack_core::pyramid::{Img3, auto_levels, half};
use lapstack_core::view::{self, Layout, View};
use lapstack_core::mesh::{self, MeshParams, TexFormat};
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
    /// Bring every frame to frame 0's brightness (one gain per channel, see lapstack_core::brightness).
    pub brightness: bool,
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
            brightness: true,
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
    /// Brightness normalisation: frame 0's channel means per 64x64 block
    /// (blocks across, blocks down), the per-block partial sums of a frame, and
    /// the gains found for each frame (applied again when a frame is re-warped).
    ref_blk: (wgpu::Buffer, usize, usize),
    bright: wgpu::Buffer,
    gains: Vec<[f32; 3]>,
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
    /// DFF confidence on the working grid, in [0, 1] (`None` for the winner map).
    conf_small: Option<Vec<f32>>,
    /// Pyramid winner map of level `depth_level` (values, w, h): free by-product of fusion.
    winner_small: Option<(Vec<f32>, usize, usize)>,
    /// Depth from focus: full-resolution depth, 65535 = last frame.
    depth_full: Option<Vec<u16>>,
    dff: Option<DepthGpu>,
    /// Depth-map rendering (second pass): frames folded so far, and the result.
    render_count: usize,
    dmap_rgb16: Option<Vec<u16>>,
    /// The weighted average (`render_push` in its wav mode), Helicon's method A.
    wav_rgb16: Option<Vec<u16>>,
    /// The slabbed depth-map rendering in progress (`render_slabs_begin`).
    srender: Option<SlabRender>,
    /// Retouch: the currently loaded aligned source frame (index, RGB u16).
    src_rgb16: Option<(usize, Vec<u16>)>,
    /// Retouch: the on-demand slab, the other brush source — the frames
    /// `lo..=hi` fused on their own (`slab_begin` / `slab_push` / `slab_finish`).
    /// `slab` is the fold in progress with its residuals, `slab_rgb16` the last
    /// one finished, held on the CPU as (lo, hi, RGB u16).
    slab: Option<Slab>,
    slab_rgb16: Option<(usize, usize, Vec<u16>)>,
    /// The first frame's EXIF / ICC profile / XMP, for the saved files.
    meta: lapstack_core::meta::Meta,
    /// The area every aligned frame covers with real pixels (`finish`); the
    /// saved images are cropped to it on request. `None` = the whole frame.
    crop: Option<Rect>,
    /// The source frame currently warped into `cur[0]` (`load_source`); In focus renders from it.
    src_gpu: Option<usize>,
    undo: Vec<Patch>,
    redo: Vec<Patch>,
    undo_bytes: usize,
    /// Retouch strokes, undos and redos so far: the view base is keyed on it.
    edits: u64,
    /// The image the stereo / rocking views are cut from (`view_prepare`).
    view_base: Option<ViewBase>,
    /// A refold in progress (`refold_begin` … `refold_end`).
    refold: Option<Refold>,
}

/// A depth-map rendering from slabs (LAP within slabs, the depth-map blend
/// across them): its own accumulator and weight sum — the run's `acc` / `best`
/// are busy fusing the slabs — the slab ranges, and how many are blended in.
struct SlabRender {
    acc: wgpu::Buffer,
    wt: wgpu::Buffer,
    slabs: Vec<(usize, usize)>,
    done: usize,
}

/// A slab being fused: its frame range and the residuals folded so far.
struct Slab {
    lo: usize,
    hi: usize,
    tops: Vec<Img3>,
}

/// One retouch stroke's effect on the fused image: the bbox and its pixels
/// before and after (RGB u16, row-major within the bbox).
struct Patch {
    /// Which result was painted: an index into `KINDS`.
    target: u8,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    before: Vec<u16>,
    after: Vec<u16>,
}

/// Undo memory cap (before + after copies); oldest strokes are dropped first.
const UNDO_CAP: usize = 600 << 20;

/// The image the stereo and rocking views are cut from (`Engine::view_prepare`):
/// the LAP or DFR master and the full-resolution depth map, cut to the crop
/// and shrunk to the view size. `rgb` / `z` are owned only when something
/// was cut or shrunk (or the depth had to be upsampled); otherwise the run's
/// own arrays serve.
struct ViewBase {
    /// (source, cropped, w, h, edits)
    key: (String, bool, usize, usize, u64),
    w: usize,
    h: usize,
    rgb: Option<Vec<u16>>,
    z: Option<Vec<u16>>,
}

/// Zerene-style synthetic stereo (`Engine::refold_*`): the stack folded
/// again, every frame shifted sideways in proportion to its index, into one
/// accumulator per view — at full resolution for a stereo pair, at the
/// animation's size for a rocking sequence (each warped frame is
/// block-averaged by `k` first, the shift then an integer number of source
/// pixels). Views are done `per_pass` at a time, one pass over the frames
/// each, as many as a GPU memory budget allows.
struct Refold {
    shifts: Vec<f32>,
    near_first: bool,
    k: usize,
    /// the pyramid at the view size
    dims: Vec<(usize, usize)>,
    levels: usize,
    fp: FuseParams,
    per_pass: usize,
    /// the frame pyramid, REDUCE/EXPAND scratch, energy plane and window-sum
    /// scratch at the view size; `None` = the run's own (k == 1)
    work: Option<(Vec<wgpu::Buffer>, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)>,
    /// per view of a pass: its accumulator (acc, best; `None` = the run's
    /// own) and the residuals folded so far
    accs: Vec<(Option<(Vec<wgpu::Buffer>, Vec<wgpu::Buffer>)>, Vec<Img3>)>,
    /// the first view of the pass in progress
    pass: Option<usize>,
    /// the crop, in view pixels
    crop: Rect,
    /// the finished views, cropped
    done: Vec<Option<Vec<u16>>>,
}

/// The GPU memory a refold may take for its own accumulators.
const REFOLD_BUDGET: usize = 1 << 30;

impl Run {
    /// The fold buffers of view `v` of the refold pass.
    fn refold_bufs(&self, v: usize) -> FoldBufs<'_> {
        let rf = self.refold.as_ref().unwrap();
        let (cur, tmp_half, en) = match &rf.work {
            Some((cur, th, en, _)) => (&cur[..], th, en),
            None => (&self.cur[..], &self.tmp_half, &self.en),
        };
        let (acc, best) = match &rf.accs[v].0 {
            Some((a, b)) => (&a[..], &b[..]),
            None => (&self.acc[..], &self.best[..]),
        };
        FoldBufs { dims: &rf.dims, levels: rf.levels, cur, acc, best, tmp_half, en }
    }
    fn refold_scratch(&self) -> &wgpu::Buffer {
        match &self.refold.as_ref().unwrap().work {
            Some((_, _, _, s)) => s,
            None => &self.tmp_full, // en2 holds the depth map
        }
    }
}

/// Encode an RGB u16 image as PNG (16-bit when `bits16`, else 8-bit) or JPEG.
fn encode_rgb16(v: &[u16], w: u32, h: u32, format: &str, quality: u8, bits16: bool) -> Result<Vec<u8>, String> {
    use image::ImageEncoder;
    let err = |e: image::ImageError| format!("encode: {e}");
    let to8 = || -> Vec<u8> { v.iter().map(|&s| (s as f32 / 65535.0 * 255.0 + 0.5) as u8).collect() };
    let mut out = Vec::new();
    match format {
        "jpeg" => image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality.clamp(1, 100)).write_image(&to8(), w, h, image::ExtendedColorType::Rgb8).map_err(err)?,
        "png" if bits16 => image::codecs::png::PngEncoder::new(&mut out).write_image(bytemuck::cast_slice(v), w, h, image::ExtendedColorType::Rgb16).map_err(err)?,
        _ => image::codecs::png::PngEncoder::new(&mut out).write_image(&to8(), w, h, image::ExtendedColorType::Rgb8).map_err(err)?,
    }
    Ok(out)
}

/// A stacked image kept past its run — the LAP or DFR master of an earlier run,
/// or an image file loaded as one — to view, compare, brush from and save
/// (`keep`, `keep_file`, `drop_kept`; `stroke` takes `from = "kept:ID"`, `encode`
/// `kind = "kept:ID"`). Its own crop and metadata go with it, so it saves as it
/// would have when it was the result. The page owns the ids and the memory budget.
struct Kept {
    id: u32,
    w: usize,
    h: usize,
    bits: u32,
    rgb16: Vec<u16>,
    crop: Option<Rect>,
    meta: lapstack_core::meta::Meta,
}

#[wasm_bindgen]
pub struct Engine {
    gpu: Gpu,
    run: Option<Run>,
    kept: Vec<Kept>,
}

/// Decode a frame on the CPU and area-average it to `edge` px on the long side:
/// {w, h, bits, proxy_w, proxy_h, proxy: Uint8Array (RGBA8)}. Used for the
/// filmstrip / Source view before a run (the browser cannot decode TIFF).
#[wasm_bindgen]
/// `raw`: a camera raw — its embedded JPEG preview stands in, developing the
/// frame being the run's job (seconds per frame here).
pub fn thumbnail(bytes: &[u8], edge: usize, raw: bool) -> Result<JsValue, JsValue> {
    let f = match if raw { lapstack_core::raw::preview(bytes) } else { None } {
        Some(img) => decode::frame_of(img),
        None => decode::decode_any(bytes, raw).map_err(|e| JsValue::from_str(&e))?,
    };
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
    Ok(Engine { gpu, run: None, kept: Vec::new() })
}

/// The full-resolution depth map (u16, 65535 = last frame): the DFF map, or
/// the pyramid winner map upsampled.
fn depth_full_u16(run: &Run) -> Result<std::borrow::Cow<'_, [u16]>, JsValue> {
    match &run.depth_full {
        Some(f) => Ok(std::borrow::Cow::Borrowed(f)),
        None => {
            let (d, dw, dh) = run.depth_small.as_ref().ok_or_else(|| JsValue::from_str("not finished"))?;
            let k = 65535.0 / (run.count.max(2) - 1) as f32;
            Ok(std::borrow::Cow::Owned(upsample_index(d, *dw, *dh, run.w, run.h, run.depth_level).iter().map(|&v| (v * k + 0.5) as u16).collect()))
        }
    }
}

/// The DFF confidence at full resolution, bilinear from the working grid like the
/// native pipeline (`lapstack_core::depth`), quantised to u16 with 65535 = 1.
fn conf_full_u16(run: &Run) -> Result<Vec<u16>, JsValue> {
    let c = run.conf_small.as_ref().ok_or_else(|| JsValue::from_str("no confidence map with the winner depth"))?;
    let dff = run.dff.as_ref().ok_or_else(|| JsValue::from_str("not finished"))?;
    let full = if dff.k == 1 { std::borrow::Cow::Borrowed(&c[..]) } else { std::borrow::Cow::Owned(lapstack_core::depth::upsample_bilinear(c, dff.dw, dff.dh, run.w, run.h, dff.k)) };
    Ok(full.iter().map(|&v| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).collect())
}

fn set(obj: &js_sys::Object, k: &str, v: impl Into<JsValue>) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(k), &v.into());
}

/// {target, x, y, w, h, rgba: Uint8Array} of a bbox of an RGB u16 image.
/// RGBA8 display copy of an interleaved RGB u16 image of `n` pixels.
fn rgba8_of(rgb: &[u16], n: usize) -> Vec<u8> {
    let mut rgba = vec![255u8; n * 4];
    for i in 0..n {
        for k in 0..3 {
            rgba[4 * i + k] = (rgb[3 * i + k] as f32 / 65535.0 * 255.0 + 0.5) as u8;
        }
    }
    rgba
}
/// The stacked images a run can hold, by name: the pyramid result, the depth-map
/// rendering, the weighted average — the retouch targets, the encoder's kinds, the
/// stereo / mesh sources, what `keep` takes.
const KINDS: [&str; 3] = ["fused", "dmap", "wav"];
fn kind_index(kind: &str) -> u8 {
    KINDS.iter().position(|k| *k == kind).unwrap_or(0) as u8
}
fn missing(kind: &str) -> &'static str {
    match kind {
        "dmap" => "no depth-map rendering",
        "wav" => "no weighted average",
        _ => "not finished",
    }
}
fn master_slot<'a>(run: &'a mut Run, kind: &str) -> &'a mut Option<Vec<u16>> {
    match kind {
        "dmap" => &mut run.dmap_rgb16,
        "wav" => &mut run.wav_rgb16,
        _ => &mut run.fused_rgb16,
    }
}
fn master<'a>(run: &'a Run, kind: &str) -> Result<&'a [u16], JsValue> {
    match kind {
        "dmap" => &run.dmap_rgb16,
        "wav" => &run.wav_rgb16,
        _ => &run.fused_rgb16,
    }
    .as_deref()
    .ok_or_else(|| JsValue::from_str(missing(kind)))
}
fn patch_obj(img: &[u16], iw: usize, kind: &str, x: usize, y: usize, w: usize, h: usize) -> JsValue {
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
    set(&o, "target", kind);
    set(&o, "x", x as u32);
    set(&o, "y", y as u32);
    set(&o, "w", w as u32);
    set(&o, "h", h as u32);
    set(&o, "rgba", js_sys::Uint8Array::from(&rgba[..]));
    o.into()
}

/// Record the warp of the uploaded frame (`up`) into `cur[0]` with the
/// registration the run found for frame `index`, moved `dx` pixels to the
/// right on top of it (the refold's synthetic-stereo shift; 0 otherwise),
/// then the run's brightness gain for it: how a frame is brought back after
/// the run (the depth-map render, the source frame, a slab, a refold).
fn record_rewarp(g: &Gpu, run: &Run, rec: &mut Rec<'_>, index: usize, dx: f32) -> Result<(), String> {
    let sim = *run.sims.get(index).ok_or("unknown frame index")?;
    let (w, h) = (run.w, run.h);
    let registered = run.params.align && index != 0;
    let identity = !registered && dx == 0.0;
    let mut p = P { w: w as u32, h: h as u32, flag: identity as u32, ..Default::default() };
    if !identity {
        let inv = affine_inv(if registered { sim } else { Sim::id() }.matrix(w, h));
        // the output moves dx right, so its source is taken dx to the left: t' = t − A·(dx, 0)
        let t = [inv[0][2] - inv[0][0] * dx as f64, inv[1][2] - inv[1][0] * dx as f64];
        g.queue.write_buffer(&run.aff, 0, bytemuck::cast_slice(&[t[0] as f32, t[1] as f32, 0.0, 0.0]));
        p.f0 = inv[0][0] as f32;
        p.f1 = inv[0][1] as f32;
        p.f2 = inv[1][0] as f32;
        p.f3 = inv[1][1] as f32;
    }
    rec.dispatch("warp", [None, None, Some(&run.cur[0]), None, Some(&run.aff), Some(&run.up)], p, grid2(w, h));
    if let Some(gn) = run.gains.get(index).copied().filter(|gn| !lapstack_core::brightness::is_unity(*gn)) {
        rec.dispatch("gain3", [None, None, Some(&run.cur[0]), None, None, None], P { w: (w * h) as u32, f0: gn[0], f1: gn[1], f2: gn[2], ..Default::default() }, grid1(3 * w * h));
    }
    Ok(())
}

/// The buffers a fold works in — a frame's Laplacian pyramid `cur`
/// (levels + 1 buffers), the accumulator `acc` and its best-energy planes,
/// the REDUCE/EXPAND scratch and the energy plane — at the sizes in `dims`:
/// the run's own, or a refold's at the view size.
struct FoldBufs<'a> {
    dims: &'a [(usize, usize)],
    levels: usize,
    cur: &'a [wgpu::Buffer],
    acc: &'a [wgpu::Buffer],
    best: &'a [wgpu::Buffer],
    tmp_half: &'a wgpu::Buffer,
    en: &'a wgpu::Buffer,
}

impl Run {
    fn fold_bufs(&self) -> FoldBufs<'_> {
        FoldBufs { dims: &self.dims, levels: self.levels, cur: &self.cur, acc: &self.acc, best: &self.best, tmp_half: &self.tmp_half, en: &self.en }
    }
}

/// Record the fold of the frame in `cur[0]`: its Laplacian pyramid (band-pass
/// levels in `cur[l]`, the residual left in `cur[levels]`), the region energy
/// of every band-pass level and the winner-take-all select into `acc` / `best`.
/// `scratch` is a w×h f32 buffer for the window sums. With `winner` =
/// Some(index) the level `depth_level` records the frame index (the winner
/// map); with `peak` the focus-peaking map is taken from level `peak.4`. The
/// run does both; a slab and a refold neither.
fn record_fold(run: &Run, rec: &mut Rec<'_>, scratch: &wgpu::Buffer, winner: Option<usize>, peak: bool) {
    record_fold_in(&run.fold_bufs(), run.klen, &run.wt, run.fp.use_chroma, rec, scratch, winner.map(|i| (i, run.depth_level)), peak.then_some(&run.peak));
}

fn record_fold_in(
    fb: &FoldBufs<'_>,
    klen: u32,
    wt: &wgpu::Buffer,
    use_chroma: bool,
    rec: &mut Rec<'_>,
    scratch: &wgpu::Buffer,
    winner: Option<(usize, usize)>,
    peak: Option<&(wgpu::Buffer, usize, usize, usize, usize)>,
) {
    // build: L_l = G_l - EXPAND(REDUCE(G_l)), per plane
    for l in 0..fb.levels {
        let (fw, fh) = fb.dims[l];
        let (cw, ch) = fb.dims[l + 1];
        for c in 0..3 {
            let pr = P { w: fw as u32, h: fh as u32, ow: cw as u32, oh: ch as u32, off_in: (c * fw * fh) as u32, off_out: (c * cw * ch) as u32, ..Default::default() };
            rec.dispatch("red_h", [Some(&fb.cur[l]), Some(fb.tmp_half), None, None, None, None], pr, grid2(cw, fh));
            rec.dispatch("red_v", [None, Some(fb.tmp_half), Some(&fb.cur[l + 1]), None, None, None], pr, grid2(cw, ch));
            let pe = P { w: fw as u32, h: fh as u32, ow: cw as u32, oh: ch as u32, off_in: (c * cw * ch) as u32, off_out: (c * fw * fh) as u32, flag: 1, ..Default::default() };
            rec.dispatch("exp_h", [Some(&fb.cur[l + 1]), Some(fb.tmp_half), None, None, None, None], pe, grid2(fw, ch));
            rec.dispatch("exp_v", [None, Some(fb.tmp_half), Some(&fb.cur[l]), None, None, None], pe, grid2(fw, fh));
        }
    }
    // region energy + winner-take-all per band-pass level
    for l in 0..fb.levels {
        let (lw, lh) = fb.dims[l];
        let ln = lw * lh;
        let pl = P { w: lw as u32, h: lh as u32, klen, flag: use_chroma as u32, ..Default::default() };
        rec.dispatch("energy", [Some(&fb.cur[l]), None, None, Some(fb.en), None, None], pl, grid1(ln));
        if klen > 1 {
            rec.dispatch("win_h", [None, Some(scratch), None, Some(fb.en), Some(wt), None], pl, grid2(lw, lh));
            rec.dispatch("win_v", [None, Some(scratch), None, Some(fb.en), Some(wt), None], pl, grid2(lw, lh));
        }
        if let Some(pk) = peak.filter(|pk| l == pk.4) {
            let (kw, kh, kf) = (pk.1, pk.2, pk.3);
            rec.dispatch(
                "down1",
                [Some(fb.en), None, Some(&pk.0), None, None, None],
                P { w: lw as u32, h: lh as u32, ow: kw as u32, oh: kh as u32, klen: kf as u32, ..Default::default() },
                grid2(kw, kh),
            );
        }
        let ps = P { w: lw as u32, h: lh as u32, flag: winner.is_some_and(|(_, dl)| l == dl) as u32, f0: winner.map_or(0.0, |(i, _)| i as f32), ..Default::default() };
        rec.dispatch("sel", [Some(&fb.cur[l]), Some(&fb.best[l]), Some(&fb.acc[l]), Some(fb.en), None, None], ps, grid1(ln));
    }
}

/// Record the collapse: fuse the residuals `tops` (on the CPU) into
/// `acc[levels]`, expand-and-add down the accumulator pyramid, and clamp
/// the image left in `acc[0]` to [0, 1].
fn record_collapse(g: &Gpu, run: &Run, rec: &mut Rec<'_>, tops: &[Img3]) {
    record_collapse_in(g, &run.fold_bufs(), &run.fp, rec, tops);
}

fn record_collapse_in(g: &Gpu, fb: &FoldBufs<'_>, fp: &FuseParams, rec: &mut Rec<'_>, tops: &[Img3]) {
    let (w, h) = fb.dims[0];
    let n = w * h;
    let top = fuse_residuals(tops, fp);
    let (tw, th) = fb.dims[fb.levels];
    let mut flat = Vec::with_capacity(3 * tw * th);
    for c in 0..3 {
        flat.extend_from_slice(&top.p[c]);
    }
    g.queue.write_buffer(&fb.acc[fb.levels], 0, bytemuck::cast_slice(&flat));
    for l in (0..fb.levels).rev() {
        let (fw, fh) = fb.dims[l];
        let (cw, ch) = fb.dims[l + 1];
        for c in 0..3 {
            let pe = P { w: fw as u32, h: fh as u32, ow: cw as u32, oh: ch as u32, off_in: (c * cw * ch) as u32, off_out: (c * fw * fh) as u32, flag: 2, ..Default::default() };
            rec.dispatch("exp_h", [Some(&fb.acc[l + 1]), Some(fb.tmp_half), None, None, None, None], pe, grid2(fw, ch));
            rec.dispatch("exp_v", [None, Some(fb.tmp_half), Some(&fb.acc[l]), None, None, None], pe, grid2(fw, fh));
        }
    }
    rec.dispatch("clamp01", [None, None, Some(&fb.acc[0]), None, None, None], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(3 * n));
}

/// Read a float image (3 planes in `img`, w×h) back as RGB u16.
async fn read_rgb16(g: &Gpu, img: &wgpu::Buffer, w: usize, h: usize) -> Result<Vec<u16>, String> {
    let n = w * h;
    let words = (3 * n).div_ceil(2);
    let buf = g.buffer("rgb16 out", (words * 4) as u64);
    let mut rec = g.rec();
    rec.dispatch("to_rgb16", [Some(img), None, Some(&buf), None, None, None], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(words));
    rec.submit();
    let r16 = g.read(&buf, (words * 4) as u64).await?;
    let mut v: Vec<u16> = bytemuck::cast_slice(&r16).to_vec();
    v.truncate(3 * n);
    Ok(v)
}

/// The window `r` of an interleaved RGB u16 image `stride` pixels wide.
fn cut_rgb(v: &[u16], stride: usize, r: &Rect) -> Vec<u16> {
    let mut out = Vec::with_capacity(r.w * r.h * 3);
    for y in r.y..r.y + r.h {
        out.extend_from_slice(&v[(y * stride + r.x) * 3..(y * stride + r.x + r.w) * 3]);
    }
    out
}

/// RGB u16 to RGBA8 for display.
fn rgb16_to_rgba8(v: &[u16]) -> Vec<u8> {
    let mut rgba = vec![255u8; v.len() / 3 * 4];
    for (o, p) in rgba.chunks_exact_mut(4).zip(v.chunks_exact(3)) {
        for k in 0..3 {
            o[k] = (p[k] as f32 / 65535.0 * 255.0 + 0.5) as u8;
        }
    }
    rgba
}

/// Read a float image (3 planes in `img`) back as (RGBA8 for display, RGB
/// u16 master); `tmp_full` is the RGBA8 staging.
async fn readback_image(g: &Gpu, run: &Run, img: &wgpu::Buffer) -> Result<(Vec<u8>, Vec<u16>), String> {
    let (w, h, n) = (run.w, run.h, run.w * run.h);
    let pw = P { w: w as u32, h: h as u32, ..Default::default() };
    let mut rec = g.rec();
    rec.dispatch("to_rgba8", [Some(img), None, Some(&run.tmp_full), None, None, None], pw, grid1(n));
    let rgb16 = g.buffer("rgb16 out", ((3 * n).div_ceil(2) * 4) as u64);
    rec.dispatch("to_rgb16", [Some(img), None, Some(&rgb16), None, None, None], pw, grid1((3 * n).div_ceil(2)));
    rec.submit();
    let rgba = g.read(&run.tmp_full, (n * 4) as u64).await?;
    let r16 = g.read(&rgb16, ((3 * n).div_ceil(2) * 4) as u64).await?;
    drop(rgb16);
    let mut v16: Vec<u16> = bytemuck::cast_slice(&r16).to_vec();
    v16.truncate(3 * n);
    Ok((rgba, v16))
}

/// Fuse the residuals `tops`, collapse the accumulator pyramid into `acc[0]`
/// (which keeps the float image) and read the result back: (RGBA8 for
/// display, RGB u16 master).
async fn collapse(g: &Gpu, run: &Run, tops: &[Img3]) -> Result<(Vec<u8>, Vec<u16>), String> {
    let mut rec = g.rec();
    record_collapse(g, run, &mut rec, tops);
    rec.submit();
    readback_image(g, run, &run.acc[0]).await
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

    /// Keep the run's `kind` (fused | dmap | wav) master under `id`, taking it out of the
    /// run — one about to be reset, or whose result the page has let go. Returns
    /// false when there is none to keep.
    pub fn keep(&mut self, id: u32, kind: &str) -> bool {
        let Some(run) = self.run.as_mut() else { return false };
        let Some(rgb16) = master_slot(run, kind).take() else { return false };
        self.kept.retain(|k| k.id != id);
        self.kept.push(Kept { id, w: run.w, h: run.h, bits: run.bits, rgb16, crop: run.crop, meta: run.meta.clone() });
        run.view_base = None;
        true
    }

    /// Keep an image file (PNG / JPEG / TIFF, 8- or 16-bit) as a result under `id`:
    /// a saved result of an earlier session, or another program's — to compare
    /// with and to brush from. Its EXIF / ICC / XMP go with it, and it has no crop.
    /// Returns {w, h, bits, rgba: Uint8Array (RGBA8)} for the page's display copy.
    pub fn keep_file(&mut self, id: u32, bytes: &[u8], raw: bool) -> Result<JsValue, JsValue> {
        let frame = decode::decode_any(bytes, raw).map_err(|e| JsValue::from_str(&e))?;
        let (w, h) = (frame.w, frame.h);
        let rgba = rgba8_of(&frame.rgb, w * h);
        self.kept.retain(|k| k.id != id);
        self.kept.push(Kept { id, w, h, bits: frame.bits, rgb16: frame.rgb, crop: None, meta: lapstack_core::meta::extract(bytes) });
        let o = js_sys::Object::new();
        set(&o, "w", w as u32);
        set(&o, "h", h as u32);
        set(&o, "bits", frame.bits);
        set(&o, "rgba", js_sys::Uint8Array::from(&rgba[..]));
        Ok(o.into())
    }

    pub fn drop_kept(&mut self, id: u32) {
        self.kept.retain(|k| k.id != id);
    }

    pub fn drop_all_kept(&mut self) {
        self.kept.clear();
    }

    /// Bytes held by the kept results (the page's budget counts them).
    pub fn kept_bytes(&self) -> f64 {
        self.kept.iter().map(|k| (k.rgb16.len() * 2) as f64).sum()
    }

    /// Decode, align (chained to frame 0) and fold one frame. The first call
    /// sets up the run from `params_json`. Returns {index, w, h, bits,
    /// proxy_w, proxy_h, proxy: Uint8Array (RGBA8), sim: [dx_px, dy_px, scale, rot_deg], ms}.
    /// `given`: the frame's registration in that same form, from a project file,
    /// used in place of the search (empty = search).
    /// `raw`: the bytes are a camera raw's, developed rather than decoded (so everywhere below).
    pub async fn push(&mut self, bytes: &[u8], params_json: &str, given: &[f64], raw: bool) -> Result<JsValue, JsValue> {
        let t0 = now();
        let frame = decode::decode_any(bytes, raw).map_err(|e| JsValue::from_str(&e))?;
        let t_dec = now();
        if self.run.is_none() {
            let params: Params = serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&format!("params: {e}")))?;
            let mut run = self.setup(frame.w, frame.h, frame.bits, params).map_err(|e| JsValue::from_str(&e))?;
            run.meta = lapstack_core::meta::extract(bytes);
            log(&format!("[lapstack] metadata of the first frame: {}", run.meta.describe()));
            self.run = Some(run);
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
        if run.params.align && run.count > 0 && given.len() == 4 {
            sim = Sim { xoff: given[0] / w as f64, yoff: given[1] / h as f64, scale: given[2], rot: given[3].to_radians() };
        } else if run.params.align && run.count > 0 {
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
        // ---- brightness: frame 0 leaves its block means; every other frame's channel
        // means over the pixels its warp covers are compared with frame 0's over the
        // same pixels, and the gains (one per channel) are applied before anything
        // reads the frame (see lapstack_core::brightness for why means)
        let mut gain = [1f32; 3];
        if run.params.brightness {
            let (bx, by) = (run.ref_blk.1, run.ref_blk.2);
            let pb = P { w: w as u32, h: h as u32, ow: bx as u32, oh: by as u32, ..p };
            if run.count == 0 {
                rec.dispatch("blk_mean", [Some(&run.cur[0]), None, Some(&run.ref_blk.0), None, None, None], pb, grid2(bx, by));
            } else {
                rec.dispatch("bright", [Some(&run.cur[0]), Some(&run.ref_blk.0), None, Some(&run.bright), Some(&run.aff), None], pb, (bx as u32, by as u32));
                rec.submit();
                let part = g.read_f32(&run.bright, bx * by * 8).await.map_err(|e| JsValue::from_str(&e))?;
                let mut s = [0f64; 8];
                for r in part.chunks_exact(8) {
                    for k in 0..7 {
                        s[k] += r[k] as f64;
                    }
                }
                if s[3] >= 64.0 {
                    gain = [0, 1, 2].map(|c| if s[c] > 1e-9 { (s[4 + c] / s[c]) as f32 } else { 1.0 }.clamp(lapstack_core::brightness::GAIN_MIN, lapstack_core::brightness::GAIN_MAX));
                }
                rec = g.rec();
                if !lapstack_core::brightness::is_unity(gain) {
                    rec.dispatch("gain3", [None, None, Some(&run.cur[0]), None, None, None], P { w: n as u32, f0: gain[0], f1: gain[1], f2: gain[2], ..Default::default() }, grid1(3 * n));
                }
            }
        }
        run.gains.push(gain);
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
        // Laplacian pyramid, region energy, winner-take-all (en2 is free scratch during the run)
        record_fold(run, &mut rec, &run.en2, Some(run.count), true);
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
        set(&o, "gain", js_sys::Array::from_iter(gain.iter().map(|&v| JsValue::from_f64(v as f64))));
        set(&o, "ms", t1 - t0);
        Ok(o.into())
    }

    /// Fuse the residuals, collapse, and read back the result. Returns
    /// {w, h, bits, frames, rgba: Uint8Array, depth_w, depth_h, depth: Float32Array,
    /// conf: Float32Array, winner_w, winner_h, winner: Float32Array, ms} — `depth` is
    /// the depth-from-focus map on its working grid, `conf` its confidence on the same
    /// grid ([0, 1]; empty with the winner depth), `winner` the pyramid winner index of `depth_level`.
    pub async fn finish(&mut self) -> Result<JsValue, JsValue> {
        let t0 = now();
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no frames pushed"))?;
        let (w, h, n) = (run.w, run.h, run.w * run.h);
        let (rgba, v16) = collapse(g, run, &run.tops).await.map_err(|e| JsValue::from_str(&e))?;
        run.fused_rgb16 = Some(v16);
        let pw = P { w: w as u32, h: h as u32, ..Default::default() };
        let t_fuse = now();
        // the pyramid winner map (plane 3 of the depth level's accumulator) is always there
        let (ww, wh) = run.dims[run.depth_level];
        let wn = ww * wh;
        let winner = g.read_f32(&run.acc[run.depth_level], 4 * wn).await.map_err(|e| JsValue::from_str(&e))?[3 * wn..].to_vec();
        let (depth, conf, dw, dh) = if let Some(dff) = &mut run.dff {
            // depth from focus, guided by the fused luma
            let mut rec = g.rec();
            rec.dispatch("luma_f32", [Some(&run.acc[0]), None, Some(&run.en), None, None, None], pw, grid1(n));
            rec.submit();
            let (dw, dh) = (dff.dw, dff.dh);
            let (depth_w, conf_w, full) = dff
                .finish(g, w, h, &run.en, &run.tmp_full, &run.en2, &run.up, &|s| log(s))
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            run.depth_full = Some(full);
            (depth_w, Some(conf_w), dw, dh)
        } else {
            // the renderer reads the full-res depth from en2
            let full = upsample_index(&winner, ww, wh, w, h, run.depth_level);
            g.queue.write_buffer(&run.en2, 0, bytemuck::cast_slice(&full));
            (winner.clone(), None, ww, wh)
        };
        run.depth_small = Some((depth.clone(), dw, dh));
        run.conf_small = conf.clone();
        run.winner_small = Some((winner.clone(), ww, wh));
        let t1 = now();
        log(&format!("[lapstack] finish: {:.0} ms (collapse {:.0} ms, depth {:.0} ms)", t1 - t0, t_fuse - t0, t1 - t_fuse));
        let o = js_sys::Object::new();
        set(&o, "w", w as u32);
        set(&o, "h", h as u32);
        set(&o, "bits", run.bits);
        set(&o, "frames", run.count as u32);
        // the window every frame covers without a smeared edge: [x, y, w, h], or null for the whole frame
        let area = common_area(&run.sims, w, h);
        run.crop = (!area.is_full(w, h)).then_some(area);
        match run.crop {
            Some(r) => set(&o, "crop", js_sys::Array::from_iter([r.x, r.y, r.w, r.h].iter().map(|&v| JsValue::from(v as u32)))),
            None => set(&o, "crop", JsValue::NULL),
        }
        set(&o, "rgba", js_sys::Uint8Array::from(&rgba[..]));
        set(&o, "depth_w", dw as u32);
        set(&o, "depth_h", dh as u32);
        set(&o, "depth", js_sys::Float32Array::from(&depth[..]));
        set(&o, "conf", js_sys::Float32Array::from(conf.as_deref().unwrap_or(&[])));
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
    /// `wav`: the weighted average instead (Helicon's method A, twin of core `wav.rs`):
    /// the re-warped frame is weighed by its contrast — the depth pass's focus measure
    /// on the working grid, box-smoothed by `smooth` grid pixels, raised to `power`,
    /// plus a floor of (1e-4)^power so a flat pixel averages every frame — into the
    /// same accumulator; `render_finish(true)` normalises it into `wav_rgb16`.
    pub async fn render_push(&mut self, index: usize, bytes: &[u8], raw: bool, wav: bool, power: f32, smooth: u32) -> Result<JsValue, JsValue> {
        let t0 = now();
        let frame = decode::decode_any(bytes, raw).map_err(|e| JsValue::from_str(&e))?;
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        if run.fused_rgb16.is_none() {
            return Err(JsValue::from_str("finish the run first"));
        }
        if frame.w != run.w || frame.h != run.h {
            return Err(JsValue::from_str("frame size differs from the run"));
        }
        let (w, h, n) = (run.w, run.h, run.w * run.h);
        g.queue.write_buffer(&run.up, 0, bytemuck::cast_slice(&frame.rgb));
        drop(frame);
        let mut rec = g.rec();
        if run.render_count == 0 {
            // accumulator = acc[0] (the pyramid result was read back already), weights = best[0]
            rec.clear(&run.acc[0]);
            rec.clear(&run.best[0]);
        }
        record_rewarp(g, run, &mut rec, index, 0.0).map_err(|e| JsValue::from_str(&e))?;
        run.src_gpu = None;
        if wav {
            let dff = run.dff.as_ref().ok_or_else(|| JsValue::from_str("the weighted average needs the depth-from-focus pass"))?;
            rec.dispatch("luma_f32", [Some(&run.cur[0]), None, Some(&run.en), None, None, None], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(n));
            let weight = dff.record_weight(&mut rec, &run.en, &run.tmp_full, w, h, smooth);
            rec.dispatch(
                "wav_acc",
                [Some(&run.cur[0]), Some(&run.acc[0]), Some(&run.best[0]), None, Some(weight), None],
                P { w: w as u32, h: h as u32, ow: dff.dw as u32, oh: dff.dh as u32, klen: dff.k as u32, f0: power.max(0.0), f1: 1e-4f32.powf(power.max(0.0)), ..Default::default() },
                grid1(n),
            );
        } else {
            rec.dispatch(
                "dmap_acc",
                [Some(&run.cur[0]), Some(&run.acc[0]), Some(&run.best[0]), None, Some(&run.en2), None],
                P { w: w as u32, h: h as u32, f0: index as f32, f1: index as f32, ..Default::default() },
                grid1(n),
            );
        }
        rec.submit();
        run.render_count += 1;
        let o = js_sys::Object::new();
        set(&o, "index", index as u32);
        set(&o, "ms", now() - t0);
        Ok(o.into())
    }

    /// Slabbed depth-map rendering (Zerene's slabbing, second pass): the
    /// stack is cut into slabs of `size` frames overlapping by `overlap`,
    /// each is fused on its own (`slab_begin` / `slab_push`, LAP within the
    /// slab) and `render_slab_finish` blends the collapsed slab in with
    /// weight 1 − dist(depth, its range) per pixel, so a pixel takes the
    /// slab(s) whose frames hold its depth — fine detail and crossing
    /// structures from LAP within a slab, and the far-out-of-focus frames
    /// that build noise and halos over a whole stack never blend in. The
    /// run's accumulator fuses the slabs, so the blend gets buffers of its
    /// own (4 floats per pixel, freed by `render_finish`). Returns the slab
    /// ranges as [[lo, hi], …].
    pub fn render_slabs_begin(&mut self, size: usize, overlap: usize) -> Result<JsValue, JsValue> {
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        if run.fused_rgb16.is_none() {
            return Err(JsValue::from_str("finish the run first"));
        }
        let n = run.w * run.h;
        let slabs = lapstack_core::slab_ranges(run.count, size, overlap);
        let acc = g.buffer_f32("slab render acc", 3 * n);
        let wt = g.buffer_f32("slab render weights", n);
        let mut rec = g.rec();
        rec.clear(&acc);
        rec.clear(&wt);
        rec.submit();
        log(&format!("[lapstack] depth-map rendering from {} slabs of {size} frames, overlap {overlap}: {slabs:?}", slabs.len()));
        let out = js_sys::Array::new();
        for &(lo, hi) in &slabs {
            out.push(&js_sys::Array::from_iter([JsValue::from(lo as u32), JsValue::from(hi as u32)]));
        }
        run.srender = Some(SlabRender { acc, wt, slabs, done: 0 });
        run.render_count = 0;
        Ok(out.into())
    }

    /// Collapse the slab in progress and blend it into the slabbed depth-map
    /// rendering. Returns {lo, hi, ms}.
    pub async fn render_slab_finish(&mut self) -> Result<JsValue, JsValue> {
        let t0 = now();
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let slab = run.slab.take().ok_or_else(|| JsValue::from_str("no slab begun"))?;
        let sr = run.srender.as_ref().ok_or_else(|| JsValue::from_str("no slabbed rendering begun"))?;
        if slab.tops.is_empty() {
            return Err(JsValue::from_str("no frames folded into the slab"));
        }
        if sr.slabs.get(sr.done) != Some(&(slab.lo, slab.hi)) {
            return Err(JsValue::from_str(&format!("slab {}..{} is not slab {} of the rendering", slab.lo, slab.hi, sr.done)));
        }
        let (w, h, n) = (run.w, run.h, run.w * run.h);
        let mut rec = g.rec();
        record_collapse(g, run, &mut rec, &slab.tops);
        rec.dispatch(
            "dmap_acc",
            [Some(&run.acc[0]), Some(&sr.acc), Some(&sr.wt), None, Some(&run.en2), None],
            P { w: w as u32, h: h as u32, f0: slab.lo as f32, f1: slab.hi as f32, ..Default::default() },
            grid1(n),
        );
        rec.submit();
        let ms = now() - t0;
        log(&format!("[lapstack] slab {}..{} ({} frames) blended into the depth-map rendering ({ms:.0} ms)", slab.lo, slab.hi, slab.tops.len()));
        run.srender.as_mut().unwrap().done += 1;
        let o = js_sys::Object::new();
        set(&o, "lo", slab.lo as u32);
        set(&o, "hi", slab.hi as u32);
        set(&o, "ms", ms);
        Ok(o.into())
    }

    /// Drop a depth-map rendering in progress (cancelled), slabbed or not.
    pub fn render_cancel(&mut self) {
        if let Some(run) = self.run.as_mut() {
            run.srender = None;
            run.slab = None;
            run.render_count = 0;
        }
    }

    /// Normalise the depth-map rendering (from frames, or from slabs) and read
    /// it back: {w, h, rgba, ms}.
    pub async fn render_finish(&mut self, wav: bool) -> Result<JsValue, JsValue> {
        let t0 = now();
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let (w, h, n) = (run.w, run.h, run.w * run.h);
        let pw = P { w: w as u32, h: h as u32, ..Default::default() };
        let sr = run.srender.take();
        let (acc, wt, what) = match &sr {
            Some(sr) => {
                if sr.done == 0 {
                    return Err(JsValue::from_str("no slabs rendered"));
                }
                (&sr.acc, &sr.wt, format!("{} slabs", sr.done))
            }
            None => {
                if run.render_count == 0 {
                    return Err(JsValue::from_str("no frames rendered"));
                }
                (&run.acc[0], &run.best[0], format!("{} frames", run.render_count))
            }
        };
        let mut rec = g.rec();
        rec.dispatch(if wav { "wav_norm" } else { "dmap_norm" }, [None, Some(acc), Some(wt), None, None, None], pw, grid1(n));
        rec.submit();
        let (rgba, v16) = readback_image(g, run, acc).await.map_err(|e| JsValue::from_str(&e))?;
        drop(sr);
        if wav {
            run.wav_rgb16 = Some(v16);
        } else {
            run.dmap_rgb16 = Some(v16);
        }
        run.render_count = 0;
        log(&format!("[lapstack] {} finished from {what} ({:.0} ms)", if wav { "weighted average" } else { "depth-map rendering" }, now() - t0));
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
    pub async fn load_source(&mut self, index: usize, bytes: &[u8], readback: bool, raw: bool) -> Result<JsValue, JsValue> {
        let t0 = now();
        let frame = decode::decode_any(bytes, raw).map_err(|e| JsValue::from_str(&e))?;
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        if run.fused_rgb16.is_none() {
            return Err(JsValue::from_str("finish the run first"));
        }
        if frame.w != run.w || frame.h != run.h {
            return Err(JsValue::from_str("frame size differs from the run"));
        }
        let (w, h) = (run.w, run.h);
        g.queue.write_buffer(&run.up, 0, bytemuck::cast_slice(&frame.rgb));
        drop(frame);
        let mut rec = g.rec();
        record_rewarp(g, run, &mut rec, index, 0.0).map_err(|e| JsValue::from_str(&e))?;
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

    /// Retouch: start a slab — the frames `lo..=hi` fused on their own, with
    /// the run's registration, brightness gains and fusion parameters — the
    /// brush source with a thick plane of focus (Zerene's slabs, made on
    /// demand: one at a time, around the scrubbed frame, instead of a batch
    /// of files). It reuses the run's accumulator, which is free once the
    /// result is read back: the best-energy planes are reset here,
    /// `slab_push` folds the frames one by one and `slab_finish` collapses
    /// the result. The GPU-resident source frame is lost (`cur` is the
    /// slab's pyramid); the depth map (`en2`) is kept.
    pub fn slab_begin(&mut self, lo: usize, hi: usize) -> Result<(), JsValue> {
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        if run.fused_rgb16.is_none() {
            return Err(JsValue::from_str("finish the run first"));
        }
        if lo > hi || hi >= run.count {
            return Err(JsValue::from_str(&format!("slab {lo}..={hi} is not within the run's {} frames", run.count)));
        }
        let mut rec = g.rec();
        for (l, b) in run.best.iter().enumerate() {
            let (lw, lh) = run.dims[l];
            rec.dispatch("fill", [None, None, Some(b), None, None, None], P { w: (lw * lh) as u32, f0: -1.0, ..Default::default() }, grid1(lw * lh));
        }
        rec.submit();
        run.slab = Some(Slab { lo, hi, tops: Vec::new() });
        Ok(())
    }

    /// Fold frame `index` (decoded again from `bytes`) into the slab. Returns {index, ms}.
    pub async fn slab_push(&mut self, index: usize, bytes: &[u8], raw: bool) -> Result<JsValue, JsValue> {
        let t0 = now();
        let frame = decode::decode_any(bytes, raw).map_err(|e| JsValue::from_str(&e))?;
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let (lo, hi) = run.slab.as_ref().map(|s| (s.lo, s.hi)).ok_or_else(|| JsValue::from_str("no slab begun"))?;
        if index < lo || index > hi {
            return Err(JsValue::from_str(&format!("frame {index} is outside the slab {lo}..={hi}")));
        }
        if frame.w != run.w || frame.h != run.h {
            return Err(JsValue::from_str("frame size differs from the run"));
        }
        g.upload(&run.up, bytemuck::cast_slice(&frame.rgb)).await.map_err(|e| JsValue::from_str(&e))?;
        drop(frame);
        let mut rec = g.rec();
        record_rewarp(g, run, &mut rec, index, 0.0).map_err(|e| JsValue::from_str(&e))?;
        run.src_gpu = None;
        // tmp_full stands in for en2 as the window-sum scratch: en2 holds the depth map
        record_fold(run, &mut rec, &run.tmp_full, None, false);
        rec.submit();
        let (tw, th) = run.dims[run.levels];
        let top = g.read_f32(&run.cur[run.levels], 3 * tw * th).await.map_err(|e| JsValue::from_str(&e))?;
        run.slab.as_mut().unwrap().tops.push(Img3 { w: tw, h: th, p: [top[..tw * th].to_vec(), top[tw * th..2 * tw * th].to_vec(), top[2 * tw * th..].to_vec()] });
        let ms = now() - t0;
        log(&format!("[lapstack] slab {lo}..{hi}: frame {index} folded ({ms:.0} ms)"));
        let o = js_sys::Object::new();
        set(&o, "index", index as u32);
        set(&o, "ms", ms);
        Ok(o.into())
    }

    /// Collapse the slab, hold it on the CPU as the brush source and return
    /// it for display: {lo, hi, w, h, rgba: Uint8Array}.
    pub async fn slab_finish(&mut self) -> Result<JsValue, JsValue> {
        let t0 = now();
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let slab = run.slab.take().ok_or_else(|| JsValue::from_str("no slab begun"))?;
        if slab.tops.is_empty() {
            return Err(JsValue::from_str("no frames folded into the slab"));
        }
        let (rgba, v16) = collapse(g, run, &slab.tops).await.map_err(|e| JsValue::from_str(&e))?;
        run.slab_rgb16 = Some((slab.lo, slab.hi, v16));
        log(&format!("[lapstack] slab {}..{} collapsed from {} frames ({:.0} ms)", slab.lo, slab.hi, slab.tops.len(), now() - t0));
        let o = js_sys::Object::new();
        set(&o, "lo", slab.lo as u32);
        set(&o, "hi", slab.hi as u32);
        set(&o, "w", run.w as u32);
        set(&o, "h", run.h as u32);
        set(&o, "rgba", js_sys::Uint8Array::from(&rgba[..]));
        Ok(o.into())
    }

    /// Drop a slab in progress (its request was superseded); the finished one is kept.
    pub fn slab_cancel(&mut self) {
        if let Some(run) = self.run.as_mut() {
            run.slab = None;
        }
    }

    /// The slab held on the CPU as a brush source, [lo, hi], or [-1, -1].
    pub fn slab_range(&self) -> Vec<i32> {
        self.run.as_ref().and_then(|r| r.slab_rgb16.as_ref()).map_or(vec![-1, -1], |(lo, hi, _)| vec![*lo as i32, *hi as i32])
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
    /// (image px) copying `from` into the `target` image (16-bit). `target` is
    /// "fused" (the pyramid result) or "dmap" (the depth-map rendering); `from`
    /// is "source" (the loaded aligned frame), "slab" (the on-demand slab) or
    /// the other result — the pyramid image can be painted into the depth-map
    /// rendering and back.
    /// Returns the updated bbox as {x, y, w, h, rgba}.
    pub fn stroke(&mut self, dabs: &[f32], target: &str, from: &str) -> Result<JsValue, JsValue> {
        let Engine { run, kept, .. } = self;
        let run = run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let (w, h) = (run.w, run.h);
        let kind = if KINDS.contains(&target) { target } else { "fused" };
        if from == target {
            return Err(JsValue::from_str("the brush source is the paint target"));
        }
        // a kept result as the source: it has to be the run's size (the same frames, as a rule)
        let kept_src: Option<&[u16]> = match from.strip_prefix("kept:") {
            Some(id) => {
                let id: u32 = id.parse().map_err(|_| JsValue::from_str("bad kept id"))?;
                let k = kept.iter().find(|k| k.id == id).ok_or_else(|| JsValue::from_str("no such kept result"))?;
                if (k.w, k.h) != (w, h) {
                    return Err(JsValue::from_str(&format!("the kept result is {}x{}, the stack {}x{}", k.w, k.h, w, h)));
                }
                Some(&k.rgb16)
            }
            None => None,
        };
        run.edits += 1;
        // the bbox of the stroke, before anything is taken out of the run
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
        // The target's master is taken out of the run while it is painted, so the source —
        // another master, the loaded frame, the slab, a kept result — can be borrowed
        // from the run alongside it; it goes back before the undo record is pushed.
        let err = |m: &str| JsValue::from_str(m);
        let mut fused = master_slot(run, kind).take().ok_or_else(|| err(missing(kind)))?;
        let src_res: Result<&[u16], JsValue> = match (kept_src, from) {
            (Some(k), _) => Ok(k),
            (None, "fused" | "dmap" | "wav") => master(run, from),
            (None, "slab") => run.slab_rgb16.as_ref().map(|s| &s.2[..]).ok_or_else(|| err("no slab")),
            (None, _) => run.src_rgb16.as_ref().map(|s| &s.1[..]).ok_or_else(|| err("no source loaded")),
        };
        let src = match src_res {
            Ok(s) => s,
            Err(e) => {
                *master_slot(run, kind) = Some(fused);
                return Err(e);
            }
        };
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
        let out = patch_obj(&fused, w, kind, bx, by, bw, bh);
        *master_slot(run, kind) = Some(fused);
        run.undo_bytes += 2 * before.len() * 2;
        run.undo.push(Patch { target: kind_index(kind), x: bx, y: by, w: bw, h: bh, before, after });
        for p in run.redo.drain(..) {
            let _ = p;
        }
        while run.undo_bytes > UNDO_CAP && run.undo.len() > 1 {
            let p = run.undo.remove(0);
            run.undo_bytes -= 2 * p.before.len() * 2;
        }
        Ok(out)
    }

    fn restore(&mut self, redo: bool) -> Result<JsValue, JsValue> {
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let w = run.w;
        run.edits += 1;
        let Some(p) = (if redo { run.redo.pop() } else { run.undo.pop() }) else { return Ok(JsValue::NULL) };
        let kind = KINDS[p.target as usize];
        let fused = master_slot(run, kind).as_mut().ok_or_else(|| JsValue::from_str(missing(kind)))?;
        let pixels = if redo { &p.after } else { &p.before };
        for r in 0..p.h {
            fused[((p.y + r) * w + p.x) * 3..((p.y + r) * w + p.x + p.w) * 3].copy_from_slice(&pixels[r * p.w * 3..(r + 1) * p.w * 3]);
        }
        let out = patch_obj(fused, w, kind, p.x, p.y, p.w, p.h);
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

    /// What the first frame carried, for the Save step: {exif, icc, xmp, chrm: byte
    /// counts (0 = absent; chrm 1), text: one line}.
    pub fn meta_info(&self) -> Result<JsValue, JsValue> {
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no run"))?;
        let m = &run.meta;
        let o = js_sys::Object::new();
        set(&o, "exif", m.exif.as_ref().map_or(0, |v| v.len()) as u32);
        set(&o, "icc", m.icc.as_ref().map_or(0, |v| v.len()) as u32);
        set(&o, "xmp", m.xmp.as_ref().map_or(0, |v| v.len()) as u32);
        set(&o, "chrm", m.chrm.is_some() as u32);
        set(&o, "text", m.describe());
        Ok(o.into())
    }

    /// Encode the fused image or the depth map. `format`: "png" (fused at the
    /// input bit depth, depth map 8-bit gray), "png8" (8-bit), "jpeg" (8-bit,
    /// `quality` 1..100). With `metadata` the stacked images (not the maps)
    /// carry the first frame's EXIF / ICC profile / XMP; with `crop` every
    /// image is cut to the area all frames cover (`finish`'s crop). Returns the file bytes.
    pub fn encode(&self, kind: &str, format: &str, quality: u8, metadata: bool, crop: bool) -> Result<js_sys::Uint8Array, JsValue> {
        use image::ImageEncoder;
        if let Some(id) = kind.strip_prefix("kept:") {
            let id: u32 = id.parse().map_err(|_| JsValue::from_str("bad kept id"))?;
            let k = self.kept.iter().find(|k| k.id == id).ok_or_else(|| JsValue::from_str("no such kept result"))?;
            let (v, w, h): (std::borrow::Cow<'_, [u16]>, u32, u32) = match if crop { k.crop } else { None } {
                Some(r) => {
                    let mut out = Vec::with_capacity(r.w * r.h * 3);
                    for y in r.y..r.y + r.h {
                        out.extend_from_slice(&k.rgb16[(y * k.w + r.x) * 3..(y * k.w + r.x + r.w) * 3]);
                    }
                    (std::borrow::Cow::Owned(out), r.w as u32, r.h as u32)
                }
                None => (std::borrow::Cow::Borrowed(&k.rgb16[..k.w * k.h * 3]), k.w as u32, k.h as u32),
            };
            let out = encode_rgb16(&v, w, h, format, quality, k.bits == 16).map_err(|e| JsValue::from_str(&e))?;
            return Ok(js_sys::Uint8Array::from(&if metadata { lapstack_core::meta::embed(out, &k.meta) } else { out }[..]));
        }
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no result"))?;
        let area = if crop { run.crop } else { None };
        // the `r` window of a `ch`-channel interleaved plane `run.w` wide
        let cut = |v: &[u16], ch: usize, r: &Rect| -> Vec<u16> {
            let mut out = Vec::with_capacity(r.w * r.h * ch);
            for y in r.y..r.y + r.h {
                out.extend_from_slice(&v[(y * run.w + r.x) * ch..(y * run.w + r.x + r.w) * ch]);
            }
            out
        };
        let (w, h) = area.map_or((run.w as u32, run.h as u32), |r| (r.w as u32, r.h as u32));
        let err = |e: image::ImageError| JsValue::from_str(&format!("encode: {e}"));
        let mut out = Vec::new();
        let (pixels8, pixels16, color): (Vec<u8>, Option<std::borrow::Cow<'_, [u16]>>, image::ExtendedColorType) = match kind {
            "fused" | "dmap" | "wav" => {
                let v = master(run, kind)?;
                let v: std::borrow::Cow<'_, [u16]> = match &area { Some(r) => std::borrow::Cow::Owned(cut(v, 3, r)), None => std::borrow::Cow::Borrowed(v) };
                if format == "png" && run.bits == 16 {
                    (Vec::new(), Some(v), image::ExtendedColorType::Rgb16)
                } else {
                    (v.iter().map(|&s| (s as f32 / 65535.0 * 255.0 + 0.5) as u8).collect(), None, image::ExtendedColorType::Rgb8)
                }
            }
            "winner" => {
                let (d, dw, dh) = run.winner_small.as_ref().ok_or_else(|| JsValue::from_str("not finished"))?;
                let full = upsample_index(d, *dw, *dh, run.w, run.h, run.depth_level);
                let full = match &area { Some(r) => lapstack_core::pyramid::crop_plane(&full, run.w, r), None => full };
                let (lo, hi) = full.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
                let range = (hi - lo).max(1e-6);
                (full.iter().map(|&v| ((v - lo) / range * 255.0 + 0.5) as u8).collect(), None, image::ExtendedColorType::L8)
            }
            "conf" => {
                // the confidence at full resolution, 65535 = 1: the CLI's --save-conf
                let full = conf_full_u16(run)?;
                let full = match &area { Some(r) => cut(&full, 1, r), None => full };
                let bytes: Vec<u8> = full.iter().flat_map(|v| v.to_be_bytes()).collect();
                let enc = image::codecs::png::PngEncoder::new(&mut out);
                enc.write_image(&bytes, w, h, image::ExtendedColorType::L16).map_err(err)?;
                return Ok(js_sys::Uint8Array::from(&out[..]));
            }
            "depth" | "depth16" => {
                // full-resolution frame index, 65535 = last frame
                let full = depth_full_u16(run)?;
                let full: std::borrow::Cow<'_, [u16]> = match &area { Some(r) => std::borrow::Cow::Owned(cut(&full, 1, r)), None => full };
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
            _ => return Err(JsValue::from_str("kind must be fused|dmap|wav|depth|depth16|conf|winner")),
        };
        match format {
            "jpeg" => {
                let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality.clamp(1, 100));
                enc.write_image(&pixels8, w, h, color).map_err(err)?;
            }
            _ => {
                let enc = image::codecs::png::PngEncoder::new(&mut out);
                match pixels16 {
                    Some(v) => enc.write_image(bytemuck::cast_slice(&v), w, h, color).map_err(err)?,
                    None => enc.write_image(&pixels8, w, h, color).map_err(err)?,
                }
            }
        }
        if metadata && matches!(kind, "fused" | "dmap" | "wav") {
            out = lapstack_core::meta::embed(out, &run.meta);
        }
        Ok(js_sys::Uint8Array::from(&out[..]))
    }

    /// Kept for the test page: PNG at the input bit depth.
    pub fn encode_png(&self, kind: &str) -> Result<js_sys::Uint8Array, JsValue> {
        self.encode(kind, "png", 90, false, false)
    }

    /// Prepare the image the stereo / rocking views are cut from: the
    /// `source` (fused | dmap) master and the depth map, cut to the crop
    /// (`crop`) and shrunk to `ow`×`oh` (0 = the crop's own size). Kept until
    /// the source, crop, size or retouch changes. Returns {w, h}.
    pub fn view_prepare(&mut self, source: &str, crop: bool, ow: u32, oh: u32) -> Result<JsValue, JsValue> {
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no result"))?;
        let key = (source.to_string(), crop, ow as usize, oh as usize, run.edits);
        if run.view_base.as_ref().is_none_or(|b| b.key != key) {
            let area = if crop { run.crop } else { None }.unwrap_or(Rect::full(run.w, run.h));
            let (ow, oh) = if ow == 0 || oh == 0 { (area.w, area.h) } else { (ow as usize, oh as usize) };
            let same = area.is_full(run.w, run.h) && ow == run.w && oh == run.h;
            let src = master(run, source)?;
            // full-resolution depth, 65535 = last frame: the DFF map, or the winner map upsampled
            let full = depth_full_u16(run)?;
            let rgb = (!same).then(|| view::shrink(src, run.w, &area, 3, ow, oh));
            let z = if same && run.depth_full.is_some() { None } else { Some(view::shrink(&full, run.w, &area, 1, ow, oh)) };
            run.view_base = Some(ViewBase { key, w: ow, h: oh, rgb, z });
        }
        let b = run.view_base.as_ref().unwrap();
        let o = js_sys::Object::new();
        set(&o, "w", b.w as u32);
        set(&o, "h", b.h as u32);
        Ok(o.into())
    }

    /// The prepared base: (rgb u16 interleaved, depth u16, w, h).
    fn view_base(&self) -> Result<(&[u16], &[u16], usize, usize), JsValue> {
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no result"))?;
        let b = run.view_base.as_ref().ok_or_else(|| JsValue::from_str("no view prepared"))?;
        let rgb: &[u16] = match &b.rgb {
            Some(v) => v,
            None => master(run, &b.key.0)?,
        };
        let z: &[u16] = match &b.z {
            Some(v) => v,
            None => run.depth_full.as_deref().ok_or_else(|| JsValue::from_str("no depth map"))?,
        };
        Ok((rgb, z, b.w, b.h))
    }

    /// One view of the prepared base as RGBA8 (w·h·4 bytes): `shift` is the
    /// far end's shift as a fraction of the width, positive = seen from the
    /// right; `near_first` = frame 0 is the near end (see core `view`).
    pub fn view_rgba(&self, shift: f32, near_first: bool) -> Result<js_sys::Uint8Array, JsValue> {
        let (rgb, z, w, h) = self.view_base()?;
        let v = view::render_new(rgb, w, h, 3, z, 1.0 / 65535.0, &View::new(shift, near_first));
        Ok(js_sys::Uint8Array::from(&rgb16_to_rgba8(&v)[..]))
    }

    /// The stereo pair of the prepared base — the views from the left and
    /// the right at ∓`shift` — in `layout` (sbs | cross | anaglyph), encoded
    /// like `encode` (`format` png | png8 | jpeg, the first frame's metadata
    /// embedded with `metadata`).
    pub fn view_stereo(&self, shift: f32, near_first: bool, layout: &str, format: &str, quality: u8, metadata: bool) -> Result<js_sys::Uint8Array, JsValue> {
        let layout = Layout::parse(layout).ok_or_else(|| JsValue::from_str("layout must be sbs | cross | anaglyph"))?;
        let (rgb, z, w, h) = self.view_base()?;
        let (px, pw, ph) = view::stereo(rgb, w, h, 3, z, 1.0 / 65535.0, shift, near_first, layout);
        let run = self.run.as_ref().unwrap();
        let mut out = encode_rgb16(&px, pw as u32, ph as u32, format, quality, run.bits == 16).map_err(|e| JsValue::from_str(&e))?;
        if metadata {
            out = lapstack_core::meta::embed(out, &run.meta);
        }
        Ok(js_sys::Uint8Array::from(&out[..]))
    }

    /// The 3D model (core `mesh`): the `source` (fused | dmap) master, cut to
    /// the crop with `crop`, as a relief of its depth map with the image as
    /// its texture. `format` glb | obj | stl; `grid` vertices along the long
    /// edge; `relief` the stack's depth as a fraction of the width;
    /// `near_first` as in `view_rgba`; `texture_edge` caps the texture's
    /// long edge (0 = as it is); `texture` jpeg (at `quality`) | png.
    /// Returns [{name, bytes}], the names built on `stem`: one file for glb
    /// and stl, three (obj, mtl, texture) for obj.
    pub fn mesh(&self, stem: &str, format: &str, source: &str, crop: bool, grid: u32, relief: f32, near_first: bool, texture_edge: u32, texture: &str, quality: u8) -> Result<JsValue, JsValue> {
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no result"))?;
        let area = if crop { run.crop } else { None }.unwrap_or(Rect::full(run.w, run.h));
        let src = master(run, source)?;
        let t0 = now();
        let full = depth_full_u16(run)?;
        let mp = MeshParams { grid: grid as usize, relief, near_first };
        let m = mesh::heightfield(&full, run.w, &area, 1.0 / 65535.0, &mp);
        let tf = if texture == "png" { TexFormat::Png } else { TexFormat::Jpeg(quality.clamp(1, 100)) };
        let files = js_sys::Array::new();
        let push = |name: String, bytes: &[u8]| {
            let o = js_sys::Object::new();
            set(&o, "name", name);
            set(&o, "bytes", js_sys::Uint8Array::from(bytes));
            files.push(&o);
        };
        let mut tex_note = String::new();
        match format {
            "stl" => push(format!("{stem}.stl"), &mesh::stl(&m)),
            "glb" | "obj" => {
                let (tex, tw, th) = mesh::texture(src, run.w, &area, 1.0 / 65535.0, texture_edge as usize);
                let image = mesh::encode_texture(&tex, tw, th, tf).map_err(|e| JsValue::from_str(&e))?;
                tex_note = format!(", texture {tw}x{th} {}", tf.ext());
                if format == "glb" {
                    push(format!("{stem}.glb"), &mesh::glb(&m, &image, tf.mime()));
                } else {
                    let (mtl, texn) = (format!("{stem}.mtl"), format!("{stem}_texture.{}", tf.ext()));
                    push(format!("{stem}.obj"), &mesh::obj(&m, &mtl));
                    push(mtl.clone(), &mesh::mtl(&texn));
                    push(texn, &image);
                }
            }
            _ => return Err(JsValue::from_str("format must be glb | obj | stl")),
        }
        log(&format!("[lapstack] 3D model ({format}): {}x{} vertices, {} triangles, relief {:.0} % of the width{tex_note} ({:.1}s)", m.nx, m.ny, m.triangles(), relief * 100.0, (now() - t0) / 1000.0));
        Ok(files.into())
    }

    /// Start a refold (see `Refold`): views at `shifts` (each the far end's
    /// shift as a fraction of the width, positive = seen from the right),
    /// rendered at the size that block-averages the frames to about
    /// `ow`×`oh` (0 = full resolution). Returns {w, h (the cropped view
    /// size), k, per_pass, passes}; then `refold_pass_begin(p)`,
    /// `refold_push` per frame, `refold_pass_finish()` for each pass.
    pub fn refold_begin(&mut self, shifts: &[f32], near_first: bool, ow: u32, oh: u32) -> Result<JsValue, JsValue> {
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        if run.fused_rgb16.is_none() {
            return Err(JsValue::from_str("finish the run first"));
        }
        if shifts.is_empty() {
            return Err(JsValue::from_str("no views"));
        }
        let (w, h) = (run.w, run.h);
        let k = if ow == 0 || oh == 0 { 1 } else { ((w as f32 / ow as f32).max(h as f32 / oh as f32).round() as usize).max(1) };
        let (vw, vh) = (w.div_ceil(k), h.div_ceil(k));
        let (dims, levels, fp) = if k == 1 {
            (run.dims.clone(), run.levels, run.fp.clone())
        } else {
            let levels = auto_levels(vw, vh, 32).max(1);
            let mut dims = vec![(vw, vh)];
            for _ in 0..levels {
                let (cw, ch) = *dims.last().unwrap();
                dims.push((half(cw), half(ch)));
            }
            (dims, levels, FuseParams { levels: Some(levels), ..run.fp.clone() })
        };
        let per_view = 4 * (3 * dims.iter().map(|&(a, b)| a * b).sum::<usize>() + dims[..levels].iter().map(|&(a, b)| a * b).sum::<usize>());
        let owned = (REFOLD_BUDGET / per_view).max(if k == 1 { 0 } else { 1 });
        let per_pass = (owned + (k == 1) as usize).min(shifts.len()).max(1);
        let work = (k > 1).then(|| {
            let cur: Vec<_> = dims.iter().enumerate().map(|(l, &(lw, lh))| g.buffer_f32(&format!("refold cur{l}"), 3 * lw * lh)).collect();
            (cur, g.buffer_f32("refold tmp_half", (half(vw) * vh).max(half(vh) * vw)), g.buffer_f32("refold en", vw * vh), g.buffer_f32("refold scratch", vw * vh))
        });
        let accs = (0..per_pass)
            .map(|v| {
                let own = !(k == 1 && v == 0);
                let bufs = own.then(|| {
                    let acc: Vec<_> = dims.iter().enumerate().map(|(l, &(lw, lh))| g.buffer_f32(&format!("refold acc{l}"), 3 * lw * lh)).collect();
                    let best: Vec<_> = dims[..levels].iter().map(|&(lw, lh)| g.buffer_f32("refold best", lw * lh)).collect();
                    (acc, best)
                });
                (bufs, Vec::new())
            })
            .collect();
        let crop = match run.crop {
            Some(r) => {
                let (x0, y0) = (r.x.div_ceil(k), r.y.div_ceil(k));
                Rect { x: x0, y: y0, w: ((r.x + r.w) / k).saturating_sub(x0).max(1), h: ((r.y + r.h) / k).saturating_sub(y0).max(1) }
            }
            None => Rect::full(vw, vh),
        };
        let passes = shifts.len().div_ceil(per_pass);
        log(&format!(
            "[lapstack] refold: {} views at {vw}x{vh} (1/{k}), {levels} levels, {per_pass} per pass ({} MB each), {passes} pass{}",
            shifts.len(), per_view >> 20, if passes == 1 { "" } else { "es" }
        ));
        run.refold = Some(Refold { shifts: shifts.to_vec(), near_first, k, dims, levels, fp, per_pass, work, accs, pass: None, crop, done: vec![None; shifts.len()] });
        run.src_gpu = None;
        let o = js_sys::Object::new();
        set(&o, "w", crop.w as u32);
        set(&o, "h", crop.h as u32);
        set(&o, "k", k as u32);
        set(&o, "per_pass", per_pass as u32);
        set(&o, "passes", passes as u32);
        Ok(o.into())
    }

    /// Start pass `pass` of the refold: reset the accumulators of its views.
    pub fn refold_pass_begin(&mut self, pass: u32) -> Result<(), JsValue> {
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let (first, count) = {
            let rf = run.refold.as_ref().ok_or_else(|| JsValue::from_str("no refold begun"))?;
            let first = pass as usize * rf.per_pass;
            if first >= rf.shifts.len() {
                return Err(JsValue::from_str("no such pass"));
            }
            (first, (rf.shifts.len() - first).min(rf.per_pass))
        };
        let mut rec = g.rec();
        for v in 0..count {
            let fb = run.refold_bufs(v);
            for (l, b) in fb.best.iter().enumerate() {
                let (lw, lh) = fb.dims[l];
                rec.dispatch("fill", [None, None, Some(b), None, None, None], P { w: (lw * lh) as u32, f0: -1.0, ..Default::default() }, grid1(lw * lh));
            }
        }
        rec.submit();
        let rf = run.refold.as_mut().unwrap();
        rf.pass = Some(first);
        for a in rf.accs.iter_mut() {
            a.1.clear();
        }
        Ok(())
    }

    /// Fold frame `index` (decoded again from `bytes`) into every view of the
    /// pass, shifted by its index. Returns {index, ms}.
    pub async fn refold_push(&mut self, index: usize, bytes: &[u8], raw: bool) -> Result<JsValue, JsValue> {
        let t0 = now();
        let frame = decode::decode_any(bytes, raw).map_err(|e| JsValue::from_str(&e))?;
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        if frame.w != run.w || frame.h != run.h {
            return Err(JsValue::from_str("frame size differs from the run"));
        }
        if index >= run.count {
            return Err(JsValue::from_str("unknown frame index"));
        }
        let (w, h) = (run.w, run.h);
        // the shift of this frame in each view: the far end of the stack moves shift × width, about the middle
        let (k, dxs) = {
            let rf = run.refold.as_ref().ok_or_else(|| JsValue::from_str("no refold begun"))?;
            let first = rf.pass.ok_or_else(|| JsValue::from_str("no pass begun"))?;
            let count = (rf.shifts.len() - first).min(rf.per_pass);
            let z = if run.count > 1 { index as f32 / (run.count - 1) as f32 } else { 0.5 };
            let sign = if rf.near_first { 1.0 } else { -1.0 };
            (rf.k, (0..count).map(|v| rf.shifts[first + v] * w as f32 * (z - 0.5) * sign).collect::<Vec<f32>>())
        };
        g.upload(&run.up, bytemuck::cast_slice(&frame.rgb)).await.map_err(|e| JsValue::from_str(&e))?;
        drop(frame);
        if k > 1 {
            // warped once at full resolution; each view block-averages it with its (integer) shift
            let mut rec = g.rec();
            record_rewarp(g, run, &mut rec, index, 0.0).map_err(|e| JsValue::from_str(&e))?;
            rec.submit();
        }
        for (v, &dx) in dxs.iter().enumerate() {
            let mut rec = g.rec();
            let fb = run.refold_bufs(v);
            if k == 1 {
                record_rewarp(g, run, &mut rec, index, dx).map_err(|e| JsValue::from_str(&e))?;
            } else {
                let (vw, vh) = fb.dims[0];
                for c in 0..3 {
                    rec.dispatch(
                        "down1",
                        [Some(&run.cur[0]), None, Some(&fb.cur[0]), None, None, None],
                        P { w: w as u32, h: h as u32, ow: vw as u32, oh: vh as u32, klen: k as u32, off_in: (c * w * h) as u32, off_out: (c * vw * vh) as u32, f0: dx, ..Default::default() },
                        grid2(vw, vh),
                    );
                }
            }
            record_fold_in(&fb, run.klen, &run.wt, run.fp.use_chroma, &mut rec, run.refold_scratch(), None, None);
            rec.submit();
            let (tw, th) = fb.dims[fb.levels];
            let top = g.read_f32(&fb.cur[fb.levels], 3 * tw * th).await.map_err(|e| JsValue::from_str(&e))?;
            run.refold.as_mut().unwrap().accs[v].1.push(Img3 { w: tw, h: th, p: [top[..tw * th].to_vec(), top[tw * th..2 * tw * th].to_vec(), top[2 * tw * th..].to_vec()] });
        }
        let o = js_sys::Object::new();
        set(&o, "index", index as u32);
        set(&o, "ms", now() - t0);
        Ok(o.into())
    }

    /// Collapse the pass's views and keep them (cropped) on the CPU.
    pub async fn refold_pass_finish(&mut self) -> Result<(), JsValue> {
        let t0 = now();
        let g = &self.gpu;
        let run = self.run.as_mut().ok_or_else(|| JsValue::from_str("no run"))?;
        let (first, count) = {
            let rf = run.refold.as_ref().ok_or_else(|| JsValue::from_str("no refold begun"))?;
            let first = rf.pass.ok_or_else(|| JsValue::from_str("no pass begun"))?;
            (first, (rf.shifts.len() - first).min(rf.per_pass))
        };
        for v in 0..count {
            let tops = std::mem::take(&mut run.refold.as_mut().unwrap().accs[v].1);
            if tops.is_empty() {
                return Err(JsValue::from_str("no frames folded into the refold"));
            }
            let (vw, vh, v16) = {
                let rf = run.refold.as_ref().unwrap();
                let fb = run.refold_bufs(v);
                let mut rec = g.rec();
                record_collapse_in(g, &fb, &rf.fp, &mut rec, &tops);
                rec.submit();
                let (vw, vh) = fb.dims[0];
                (vw, vh, read_rgb16(g, &fb.acc[0], vw, vh).await.map_err(|e| JsValue::from_str(&e))?)
            };
            let rf = run.refold.as_mut().unwrap();
            let cropped = if rf.crop.is_full(vw, vh) { v16 } else { cut_rgb(&v16, vw, &rf.crop) };
            rf.done[first + v] = Some(cropped);
        }
        let rf = run.refold.as_mut().unwrap();
        rf.pass = None;
        log(&format!("[lapstack] refold: views {}..{} collapsed ({:.0} ms)", first, first + count - 1, now() - t0));
        Ok(())
    }

    /// A finished view as RGBA8: {index, w, h, rgba}.
    pub fn refold_view(&self, index: usize) -> Result<JsValue, JsValue> {
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no run"))?;
        let rf = run.refold.as_ref().ok_or_else(|| JsValue::from_str("no refold begun"))?;
        let v = rf.done.get(index).and_then(|d| d.as_ref()).ok_or_else(|| JsValue::from_str("view not rendered"))?;
        let o = js_sys::Object::new();
        set(&o, "index", index as u32);
        set(&o, "w", rf.crop.w as u32);
        set(&o, "h", rf.crop.h as u32);
        set(&o, "rgba", js_sys::Uint8Array::from(&rgb16_to_rgba8(v)[..]));
        Ok(o.into())
    }

    /// Views 0 (left) and 1 (right) of the refold as a stereo pair in
    /// `layout` (sbs | cross | anaglyph), encoded like `view_stereo`.
    pub fn refold_stereo(&self, layout: &str, format: &str, quality: u8, metadata: bool) -> Result<js_sys::Uint8Array, JsValue> {
        let layout = Layout::parse(layout).ok_or_else(|| JsValue::from_str("layout must be sbs | cross | anaglyph"))?;
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no run"))?;
        let rf = run.refold.as_ref().ok_or_else(|| JsValue::from_str("no refold begun"))?;
        let get = |i: usize| rf.done.get(i).and_then(|d| d.as_deref()).ok_or_else(|| JsValue::from_str("the pair's views are not rendered"));
        let (px, pw, ph) = view::compose_pair(get(0)?, get(1)?, rf.crop.w, rf.crop.h, 3, layout);
        let mut out = encode_rgb16(&px, pw as u32, ph as u32, format, quality, run.bits == 16).map_err(|e| JsValue::from_str(&e))?;
        if metadata {
            out = lapstack_core::meta::embed(out, &run.meta);
        }
        Ok(js_sys::Uint8Array::from(&out[..]))
    }

    /// Drop the refold (its views and GPU buffers).
    pub fn refold_end(&mut self) {
        if let Some(run) = self.run.as_mut() {
            run.refold = None;
        }
    }

    /// Full-resolution depth map (u16, 65535 = last frame); for tests.
    pub fn depth_full(&self) -> Result<js_sys::Uint16Array, JsValue> {
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no result"))?;
        let d = run.depth_full.as_ref().ok_or_else(|| JsValue::from_str("no depth-from-focus result"))?;
        Ok(js_sys::Uint16Array::from(&d[..]))
    }

    /// Full-resolution confidence map (u16, 65535 = 1); for tests.
    pub fn conf_full(&self) -> Result<js_sys::Uint16Array, JsValue> {
        let run = self.run.as_ref().ok_or_else(|| JsValue::from_str("no result"))?;
        Ok(js_sys::Uint16Array::from(&conf_full_u16(run)?[..]))
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
            ref_blk: (g.buffer_f32("brightness reference", w.div_ceil(64) * h.div_ceil(64) * 3), w.div_ceil(64), h.div_ceil(64)),
            bright: g.buffer_f32("brightness partials", w.div_ceil(64) * h.div_ceil(64) * 8),
            gains: Vec::new(),
            ref_pyr,
            tgt_pyr,
            aligner,
            tops: Vec::new(),
            sims: Vec::new(),
            guess: Sim::id(),
            count: 0,
            fused_rgb16: None,
            depth_small: None,
            conf_small: None,
            winner_small: None,
            depth_full: None,
            dff,
            render_count: 0,
            dmap_rgb16: None,
            wav_rgb16: None,
            srender: None,
            src_rgb16: None,
            slab: None,
            slab_rgb16: None,
            src_gpu: None,
            undo: Vec::new(),
            redo: Vec::new(),
            undo_bytes: 0,
            edits: 0,
            view_base: None,
            refold: None,
            meta: Default::default(),
            crop: None,
        })
    }
}

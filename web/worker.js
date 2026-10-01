// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

// lapstack worker: owns the WASM engine and the WebGPU device. The page sends
// {type:'init'|'run'|'cancel'|'load_source'|'slab'|'stroke'|'save'|…}; the worker answers with progress and
// results (large buffers are transferred, not copied).
// The WASM glue and binary are imported on 'init' with a per-load query
// string so a rebuilt pkg/ is never served from the browser cache (a stale
// glue shows up as "engine.<new method> is not a function"). Not a top-level
// await: that would suspend the script before onmessage is installed and the
// page's first message would be lost.
const V = Date.now();
let init = null, create_engine = null, thumbnail = null, GifWriter = null;
async function loadWasm() {
  if (init) return;
  const mod = await import(`./pkg/lapstack_web.js?t=${V}`);
  ({ default: init, create_engine, thumbnail, GifWriter = null } = mod);
}
// content credentials live in their own module (pkg-cc, ~3 MB): loaded on the first save that asks for them
let cc = null;
async function loadCc() {
  if (cc) return cc;
  const mod = await import(`./pkg-cc/lapstack_cc.js?t=${V}`);
  await mod.default({ module_or_path: `./pkg-cc/lapstack_cc_bg.wasm?t=${V}` });
  cc = mod; return cc;
}
// camera raw decoding is a module of its own too (pkg-raw: rawler behind a small interface,
// LGPL — its source is in legal/, and a build of your own dropped in here is what the engine
// uses): loaded when the first raw comes in. The engine (lapstack-core's raw.rs, through
// lapstack-web's raw_bridge) reaches it by the three globals below; a developed image stays
// in the raw module's memory and the engine copies from a view on it, then free() gives it back.
let rawMod = null, rawMem = null;
async function loadRaw() {
  if (rawMod) return;
  const mod = await import(`./pkg-raw/lapstack_raw.js?t=${V}`);
  const wasm = await mod.default({ module_or_path: `./pkg-raw/lapstack_raw_bg.wasm?t=${V}` });
  rawMod = mod; rawMem = wasm.memory;
}
const needRaw = async (files) => { if (!rawMod && files.some((f) => f && isRaw(f.name))) await loadRaw(); };
const rawView = (d) => {
  const view = d.bits === 32 ? new Float32Array(rawMem.buffer, d.ptr(), d.len()) : d.bits === 16 ? new Uint16Array(rawMem.buffer, d.ptr(), d.len()) : new Uint8Array(rawMem.buffer, d.ptr(), d.len());
  return { w: d.w, h: d.h, channels: d.channels, bits: d.bits, turns: d.turns, flip: d.flip, color: d.color, data: view, free: () => d.free() };
};
globalThis.lapstackRawDevelop = (bytes, linear) => {
  if (!rawMod) throw new Error('the raw decoder (pkg-raw) is not loaded');
  return rawView(linear ? rawMod.develop_linear(bytes) : rawMod.develop(bytes));
};
globalThis.lapstackRawPreview = (bytes) => { const d = rawMod ? rawMod.preview(bytes) : undefined; return d ? rawView(d) : null; };
globalThis.lapstackRawMetadata = (bytes) => (rawMod && rawMod.metadata(bytes)) || null;

let engine = null;
let initError = null;     // why 'init' left no engine (no WebGPU adapter, usually): every later call reports it
let gifw = null;          // the animated GIF being written (gif_begin … gif_end), see gif.rs
let cancelled = false;
let running = false;
let thumbJob = null;      // {files, indices, edge, gen} being decoded in the background
let thumbGen = 0;
let sourceGen = -1;       // newest 'load_source' request seen; older ones still queued are skipped
let slabGen = -1;         // newest 'slab' request seen; an older one still queued, or in progress, is dropped
let refoldCancel = false; // set by 'refold_cancel': the refold in progress stops at the next frame

// Decode + downscale added frames one by one (yielding between frames so a
// 'run' or 'cancel' message can interleave); a run pauses this until it ends.
async function thumbLoop() {
  while (thumbJob && thumbJob.files.length) {
    if (running || !thumbnail) { await new Promise((r) => setTimeout(r, 200)); continue; }
    const job = thumbJob;
    const f = job.files.shift(), i = job.indices.shift(), uid = job.uids.shift();
    try {
      const bytes = new Uint8Array(await f.arrayBuffer());
      if (job.gen !== thumbGen) return;
      await needRaw([f]);
      const t = thumbnail(bytes, job.edge, isRaw(f.name), job.rotate || 0);
      const [proxy, strip] = await proxyBitmaps(t.proxy.buffer, t.proxy_w, t.proxy_h);
      post({ type: 'thumb', index: i, uid, name: f.name, w: t.w, h: t.h, bits: t.bits, proxy, strip }, [proxy, strip]);
    } catch (e) {
      post({ type: 'thumb-error', index: i, name: f.name, text: (e && e.message) ? e.message : String(e) });
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  thumbJob = null;
}

function post(msg, transfer) { self.postMessage(msg, transfer || []); }
// A proxy's bitmaps, made here rather than on the page: the full one for the view and a
// strip-sized one for its thumb. Both are transferable, and on the page each frame's
// createImageBitmap pair cost ~15 ms of main thread, a dropped frame per fused frame.
const STRIP_W = 160, STRIP_H = 100;
async function proxyBitmaps(rgba, w, h) {
  const img = new ImageData(new Uint8ClampedArray(rgba), w, h);
  const s = Math.min(STRIP_W / w, STRIP_H / h);
  return Promise.all([createImageBitmap(img), createImageBitmap(img, { resizeWidth: Math.max(1, Math.round(w * s)), resizeHeight: Math.max(1, Math.round(h * s)), resizeQuality: 'medium' })]);
}
// Start reading a frame's bytes. A File read only progresses while this thread's
// event loop is idle, which it is while a push awaits the GPU, so the next frame's
// read is started before the current push and is usually complete by the time
// it is needed (the fold spent a third of its time reading otherwise). The early
// catch keeps an abandoned read (cancel) from surfacing as an unhandled rejection;
// awaiting the promise still throws.
// a camera raw, by its name: the engine develops it instead of decoding it (lapstack-core's raw.rs)
const isRaw = (name) => /\.(ari|arw|cr2|cr3|crm|crw|dcr|dcs|dng|erf|iiq|kdc|mef|mos|mrw|nef|nrw|orf|ori|pef|raf|raw|rw2|rwl|srw|3fr|fff|x3f|qtk)$/i.test(name || '');
function readAhead(f) { const p = f.arrayBuffer(); p.catch(() => {}); return p; }

// Engine calls must not overlap (wasm-bindgen rejects re-entrant use of the
// engine while an async call is in flight), so everything that touches it
// runs through one promise chain; only cancel/clear are handled immediately.
let chain = Promise.resolve();
const enqueue = (fn) => { chain = chain.then(fn).catch((e) => post({ type: 'error', text: (e && e.message) ? e.message : String(e) })); };
// Requests carrying an `rid` are remote calls from the page (see call() there): the reply
// echoes the rid, and a failure answers that call instead of surfacing as a run error.
// Without an engine (init failed) a call that would reach it fails with that reason instead
// of a null property error; thumbnails and the content credentials don't need the engine.
const NO_ENGINE = new Set(['init', 'thumbs', 'make_cert', 'sign']);
const needEngine = (m) => { if (!engine && !NO_ENGINE.has(m.type)) throw new Error('WebGPU failed to initialise, so there is nothing to run on: ' + (initError || 'the engine is not ready')); };
const rpc = (m, fn) => enqueue(async () => {
  try { await fn(); } catch (e) { post({ type: 'rpc-error', rid: m.rid, text: (e && e.message) ? e.message : String(e) }); }
});

self.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'cancel') { cancelled = true; return; }
  if (m.type === 'refold_cancel') { refoldCancel = true; return; }
  if (m.type === 'clear') { thumbGen++; thumbJob = null; return; }
  if (m.type === 'load_source') sourceGen = Math.max(sourceGen, m.gen);
  if (m.type === 'slab') slabGen = Math.max(slabGen, m.gen);
  if (m.rid) rpc(m, () => handleCall(m)); else enqueue(() => handle(m));
};

// ---------- remote calls (Save step) ----------
// The page composes export frames itself and only needs the engine for what it cannot
// do: full-resolution aligned frames (plain or In focus), the stereo / rocking views, the
// 3D model, GIF quantisation + LZW, image encoding, and content credentials.
async function handleCall(m) {
  needEngine(m);
  if (m.type === 'save') {
    // m.overlay: the scale bar and caption (JSON, see overlay.rs) burned into the stacked images; '' = none
    const bytes = engine.encode(m.kind, m.format || 'png', m.quality || 90, !!m.meta, !!m.crop, m.overlay || '');
    post({ type: 'png', rid: m.rid, kind: m.kind, format: m.format || 'png', bytes: bytes.buffer }, [bytes.buffer]);
  } else if (m.type === 'overlay') {
    // the scale bar and caption for an m.w×m.h output (m.scale = its pixels per frame pixel) as
    // RGBA patches over the corners it occupies, for the page to draw (the viewer's preview,
    // the Save step's thumbnails, the animation frames), with the overlay's description
    const r = engine.overlay_patches(m.json || '', m.w, m.h, m.scale || 1);
    const patches = r.patches.map((q) => ({ x: q.x, y: q.y, w: q.w, h: q.h, rgba: q.rgba.buffer }));
    post({ type: 'overlay', rid: m.rid, text: r.text, patches }, patches.map((q) => q.rgba));
  } else if (m.type === 'export_source') {
    // the aligned full-res frame (or its In focus rendering, m.focus = focusParams), like load_source
    // without the scrub bookkeeping; the page passes the file bytes it has already read
    if (running) throw new Error('a run is in progress');
    await needRaw([m.file]);
    const held = engine.source_gpu_index() === m.index;
    let r = null;
    if (!held) r = await engine.load_source(m.index, new Uint8Array(m.bytes || await m.file.arrayBuffer()), !m.focus, isRaw(m.file && m.file.name));
    if (m.focus) r = await engine.source_focus(m.focus.dim, m.focus.w0, m.focus.w1, m.focus.tex);
    else if (held) r = await engine.source_readback();
    post({ type: 'export_source', rid: m.rid, index: r.index, w: r.w, h: r.h, rgba: r.rgba.buffer }, [r.rgba.buffer]);
  } else if (m.type === 'gif_begin') {
    if (!GifWriter) throw new Error('this build has no GIF writer (rebuild web/pkg)');
    if (gifw) { gifw.free(); gifw = null; }
    gifw = new GifWriter(m.w, m.h, m.loop !== false, m.dither !== false);
    post({ type: 'gif', rid: m.rid });
  } else if (m.type === 'gif_frame') {
    if (!gifw) throw new Error('no GIF in progress');
    const bytes = gifw.push(new Uint8Array(m.rgba), m.delay);
    post({ type: 'gif', rid: m.rid, bytes: bytes.buffer }, [bytes.buffer]);
  } else if (m.type === 'gif_end') {
    if (!gifw) throw new Error('no GIF in progress');
    const bytes = gifw.finish(); gifw.free(); gifw = null;
    post({ type: 'gif', rid: m.rid, bytes: bytes.buffer, done: true }, [bytes.buffer]);
  } else if (m.type === 'gif_abort') {
    if (gifw) { gifw.free(); gifw = null; }
    post({ type: 'gif', rid: m.rid });
  } else if (m.type === 'view' || m.type === 'view_stereo') {
    // a synthetic stereo / rocking view (view.rs): the base — the LAP or DFR master and the
    // depth map, cropped and shrunk to m.w×m.h (0 = the crop's own size) — is prepared once
    // per size and kept, then sheared by m.shift (a fraction of the width) per call
    const b = engine.view_prepare(m.source || 'fused', !!m.crop, m.w || 0, m.h || 0);
    if (m.type === 'view') {
      const rgba = engine.view_rgba(m.shift, m.near !== false);
      post({ type: 'view', rid: m.rid, w: b.w, h: b.h, rgba: rgba.buffer }, [rgba.buffer]);
    } else {
      const bytes = engine.view_stereo(m.shift, m.near !== false, m.layout || 'sbs', m.format || 'png', m.quality || 90, !!m.meta, m.overlay || '');
      post({ type: 'png', rid: m.rid, kind: 'stereo', format: m.format || 'png', bytes: bytes.buffer }, [bytes.buffer]);
    }
  } else if (m.type === 'mesh') {
    // the 3D model (mesh.rs): the LAP or DFR master as a relief of the depth map, textured;
    // one file (glb, stl) or three (obj + mtl + texture), each {name, bytes}
    const files = engine.mesh(m.stem, m.format || 'glb', m.source || 'fused', !!m.crop, m.grid || 1000, m.relief || 0.25, m.near !== false, m.texture_edge || 0, m.texture || 'jpeg', m.quality || 92)
      .map((f) => ({ name: f.name, bytes: f.bytes.buffer }));
    post({ type: 'mesh', rid: m.rid, files }, files.map((f) => f.bytes));
  } else if (m.type === 'refold') {
    // Refold synthetic stereo: the stack folded again (m.files, in frame order), each
    // frame shifted by its index, into one accumulator per view in m.shifts, at about m.w×m.h
    // (0 = full resolution). As many views per pass over the frames as the GPU budget allows;
    // progress goes to the page as 'refold-progress'. The views stay in the engine until
    // 'refold_end': 'refold_view' returns one as RGBA8, 'refold_stereo' encodes views 0 and 1.
    if (running) throw new Error('a run is in progress');
    await needRaw(m.files);
    refoldCancel = false;
    const plan = engine.refold_begin(new Float32Array(m.shifts), m.near !== false, m.w || 0, m.h || 0);
    const total = plan.passes * m.files.length; let done = 0;
    try {
      for (let p = 0; p < plan.passes; p++) {
        engine.refold_pass_begin(p);
        let next = readAhead(m.files[0]);
        for (let i = 0; i < m.files.length; i++) {
          if (refoldCancel) throw new Error('cancelled');
          post({ type: 'refold-progress', text: `refold${plan.passes > 1 ? ` ${p + 1}/${plan.passes}` : ''}: ${m.files[i].name}`, done, total });
          const bytes = new Uint8Array(await next);
          next = i + 1 < m.files.length ? readAhead(m.files[i + 1]) : null;
          await engine.refold_push(i, bytes, isRaw(m.files[i].name)); done++;
        }
        await engine.refold_pass_finish();
      }
    } catch (e) { engine.refold_end(); throw e; }
    post({ type: 'refold', rid: m.rid, w: plan.w, h: plan.h, k: plan.k, passes: plan.passes });
  } else if (m.type === 'refold_view') {
    const r = engine.refold_view(m.index);
    post({ type: 'view', rid: m.rid, w: r.w, h: r.h, rgba: r.rgba.buffer }, [r.rgba.buffer]);
  } else if (m.type === 'refold_stereo') {
    const bytes = engine.refold_stereo(m.layout || 'sbs', m.format || 'png', m.quality || 90, !!m.meta, m.overlay || '');
    post({ type: 'png', rid: m.rid, kind: 'stereo', format: m.format || 'png', bytes: bytes.buffer }, [bytes.buffer]);
  } else if (m.type === 'refold_end') {
    engine.refold_end();
    post({ type: 'refold', rid: m.rid });
  } else if (m.type === 'crop_set') {   // the Save step's own window, on top of the automatic crop: the engine returns the window in force
    const c = engine.crop_set(m.x, m.y, m.w, m.h);
    post({ type: 'crop_set', rid: m.rid, crop: Array.from(c) });
  } else if (m.type === 'crop_clear') {
    engine.crop_clear();
    post({ type: 'crop_clear', rid: m.rid });
  } else if (m.type === 'dust_set' || m.type === 'dust_update' || m.type === 'dust_clear') {
    // the dust map (lapstack-core's dust.rs): a frame of an evenly lit blank surface, whose spots
    // the engine takes out of every frame it decodes from now on; 'dust_update' finds the spots
    // again with other settings, 'dust_clear' lets it go. The reply is the engine's dust_info.
    if (running) throw new Error('a run is in progress');
    if (m.type === 'dust_set') await needRaw([m.file]);
    let r = null;
    if (m.type === 'dust_set') r = engine.dust_set(m.file.name, new Uint8Array(await m.file.arrayBuffer()), isRaw(m.file.name), m.edge || 1400, m.threshold, m.margin, m.mode);
    else if (m.type === 'dust_update') r = engine.dust_update(m.threshold, m.margin, m.mode);
    else engine.dust_clear();
    const proxy = r && r.proxy ? r.proxy.buffer : null;
    post({ type: 'dust', rid: m.rid, info: r ? { ...r, rects: r.rects, proxy } : null }, proxy ? [proxy, r.rects.buffer] : []);
  } else if (m.type === 'keep_file') {
    // an image file kept as a result (see 'keep' below): decoded here, its RGBA8 back for the page's copy
    await needRaw([m.file]);
    const r = engine.keep_file(m.id, new Uint8Array(await m.file.arrayBuffer()), isRaw(m.file.name));
    post({ type: 'kept_file', rid: m.rid, id: m.id, w: r.w, h: r.h, bits: r.bits, rgba: r.rgba.buffer }, [r.rgba.buffer]);
  } else if (m.type === 'make_cert') {
    const [cert, key] = (await loadCc()).make_cert(m.name, Date.now() / 1000);
    post({ type: 'cert', rid: m.rid, cert, key });
  } else if (m.type === 'sign') {
    const bytes = (await loadCc()).sign_image(new Uint8Array(m.bytes), m.mime, m.manifest, m.cert, m.key);
    post({ type: 'signed', rid: m.rid, bytes: bytes.buffer }, [bytes.buffer]);
  } else throw new Error('unknown call ' + m.type);
}

async function handle(m) {
  try {
    needEngine(m);
    if (m.type === 'init') {
      if (m.debug && self.navigator.gpu) {
        const dbg = (t) => post({ type: 'debug', text: t });
        dbg('worker global: window=' + typeof self.window + ' navigator.gpu=' + typeof self.navigator.gpu);
        // surface WebGPU validation / shader errors, which are otherwise only on the worker console
        const rd = GPUAdapter.prototype.requestDevice;
        GPUAdapter.prototype.requestDevice = async function (d) {
          const dev = await rd.call(this, d);
          dev.addEventListener('uncapturederror', (e) => dbg('WebGPU error: ' + e.error.message.slice(0, 600)));
          return dev;
        };
        for (const k of ['error', 'warn']) { const o = console[k].bind(console); console[k] = (...a) => { dbg(k + ': ' + a.join(' ').slice(0, 600)); o(...a); }; }
        const orig = self.navigator.gpu.requestAdapter.bind(self.navigator.gpu);
        self.navigator.gpu.requestAdapter = async (opts) => {
          dbg('requestAdapter opts=' + JSON.stringify(opts));
          try { const a = await orig(opts); dbg('requestAdapter -> ' + (a ? a.info.vendor : 'null')); return a; }
          catch (e) { dbg('requestAdapter threw ' + e); throw e; }
        };
      }
      await loadWasm();
      await init({ module_or_path: `./pkg/lapstack_web_bg.wasm?t=${V}` });
      // adapter description straight from the browser (wgpu's copy is sparse on the web)
      let adapter = null;
      try { adapter = await self.navigator.gpu?.requestAdapter(); if (!adapter) adapter = await self.navigator.gpu?.requestAdapter(); } catch {}
      const ainfo = adapter ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description, fallback: !!adapter.isFallbackAdapter } : null;
      try { engine = await create_engine(); } catch (e) { initError = (e && e.message) ? e.message : String(e); throw e; }
      post({ type: 'ready', info: { ...JSON.parse(engine.info()), ...(ainfo || {}) } });
    } else if (m.type === 'thumbs') {
      if (thumbJob) { thumbJob.files.push(...m.files); thumbJob.indices.push(...m.indices); thumbJob.uids.push(...m.uids); }
      else { thumbJob = { files: [...m.files], indices: [...m.indices], uids: [...m.uids], edge: m.edge, rotate: m.rotate || 0, gen: thumbGen }; thumbLoop(); }
    } else if (m.type === 'run') {
      if (running) return;
      running = true; cancelled = false;
      await needRaw(m.files);
      engine.reset();
      const params = JSON.stringify(m.params);
      const t0 = performance.now();
      let next = m.files.length ? readAhead(m.files[0]) : null;
      for (let i = 0; i < m.files.length; i++) {
        if (cancelled) break;
        const f = m.files[i];
        post({ type: 'stage', text: `decoding ${f.name}`, done: i, total: m.files.length });
        const bytes = new Uint8Array(await next);
        next = i + 1 < m.files.length ? readAhead(m.files[i + 1]) : null;
        const r = await engine.push(bytes, params, m.sims && m.sims[i] ? new Float64Array(m.sims[i]) : new Float64Array(0), isRaw(f.name));   // a project's registration, or the search
        const peak = r.peak;
        const [proxy, strip] = await proxyBitmaps(r.proxy.buffer, r.proxy_w, r.proxy_h);
        post({ type: 'frame', index: r.index, name: f.name, w: r.w, h: r.h, bits: r.bits, proxy, strip,
               peak_w: r.peak_w, peak_h: r.peak_h, peak: peak.buffer, sim: r.sim, gain: r.gain, ms: r.ms,
               done: i + 1, total: m.files.length }, [proxy, strip, peak.buffer]);
      }
      if (cancelled) { engine.reset(); post({ type: 'cancelled' }); running = false; return; }
      post({ type: 'stage', text: 'collapsing', done: m.files.length, total: m.files.length });
      const res = await engine.finish();
      res.meta = engine.meta_info();   // what the first frame carried, for the Save step
      post({ type: 'done', w: res.w, h: res.h, bits: res.bits, frames: res.frames, meta: res.meta, crop: res.crop,
             rgba: res.rgba.buffer, depth_w: res.depth_w, depth_h: res.depth_h, depth: res.depth.buffer, conf: res.conf.buffer,
             winner_w: res.winner_w, winner_h: res.winner_h, winner: res.winner.buffer,
             ms: performance.now() - t0 }, [res.rgba.buffer, res.depth.buffer, res.conf.buffer, res.winner.buffer]);
      // optional second pass: render a second image from the depth map (frames are decoded
      // again) — blending the frames nearest each pixel's depth, or, slabbed, the LAP fusions
      // of overlapping slabs of the stack (each slab fused like the retouch slab, then
      // blended in by its frame range; frames in the overlaps are decoded once per slab)
      if (m.params.render_dmap) {
        const t1 = performance.now();
        if (m.params.render_slabs) {
          const slabs = engine.render_slabs_begin(m.params.slab_size || 10, m.params.slab_overlap ?? 2);   // [[lo, hi], …]
          const total = slabs.reduce((s, [lo, hi]) => s + hi - lo + 1, 0);
          let done = 0;
          for (let k = 0; k < slabs.length && !cancelled; k++) {
            const [lo, hi] = slabs[k];
            engine.slab_begin(lo, hi);
            let next = readAhead(m.files[lo]);
            for (let i = lo; i <= hi && !cancelled; i++) {
              post({ type: 'stage', text: `slab ${k + 1}/${slabs.length}: ${m.files[i].name}`, done, total });
              const bytes = new Uint8Array(await next);
              next = i < hi ? readAhead(m.files[i + 1]) : null;
              await engine.slab_push(i, bytes, isRaw(m.files[i].name)); done++;
            }
            if (!cancelled) await engine.render_slab_finish();
          }
        } else {
          let next = readAhead(m.files[0]);
          for (let i = 0; i < m.files.length; i++) {
            if (cancelled) break;
            post({ type: 'stage', text: `rendering from depth map: ${m.files[i].name}`, done: i, total: m.files.length });
            const bytes = new Uint8Array(await next);
            next = i + 1 < m.files.length ? readAhead(m.files[i + 1]) : null;
            await engine.render_push(i, bytes, isRaw(m.files[i].name), false, 0, 0, 0);
          }
        }
        if (cancelled) { engine.render_cancel(); post({ type: 'render-cancelled' }); running = false; return; }
        const r = await engine.render_finish(false);
        post({ type: 'done2', w: r.w, h: r.h, rgba: r.rgba.buffer, ms: performance.now() - t1 }, [r.rgba.buffer]);
      }
      // optional third image: the weighted average — the frames decoded
      // once more, each weighed by its contrast (the depth pass's focus measure) into one average
      if (m.params.render_wav) {
        const t2 = performance.now();
        let next = readAhead(m.files[0]);
        for (let i = 0; i < m.files.length; i++) {
          if (cancelled) break;
          post({ type: 'stage', text: `weighted average: ${m.files[i].name}`, done: i, total: m.files.length });
          const bytes = new Uint8Array(await next);
          next = i + 1 < m.files.length ? readAhead(m.files[i + 1]) : null;
          await engine.render_push(i, bytes, isRaw(m.files[i].name), true, m.params.wav_power ?? 2, m.params.wav_smooth ?? 3, m.params.wav_gate ?? 0.5);
        }
        if (cancelled) { engine.render_cancel(); post({ type: 'render-cancelled' }); running = false; return; }
        const r = await engine.render_finish(true);
        post({ type: 'done3', w: r.w, h: r.h, rgba: r.rgba.buffer, ms: performance.now() - t2 }, [r.rgba.buffer]);
      }
      running = false;
    } else if (m.type === 'replay') {
      // A project's retouch strokes applied again, in order, to the run just made: each
      // stroke's source is brought back first — the frame decoded and warped (load_source),
      // the slab fused (slab_begin … slab_finish) — then the stroke goes through the engine
      // like a live one, and its patch reaches the page the same way. m.strokes: [{target,
      // from, index?, lo?, hi?, dabs}], m.files: the frames in order.
      if (running) return;
      await needRaw(m.files);
      let done = 0, skipped = 0;
      for (const s of m.strokes) {
        post({ type: 'stage', text: `retouch ${done + skipped + 1}/${m.strokes.length}`, done: done + skipped, total: m.strokes.length });
        try {
          if (s.from === 'source') {
            if (engine.source_index() !== s.index) await engine.load_source(s.index, new Uint8Array(await m.files[s.index].arrayBuffer()), true, isRaw(m.files[s.index].name));
          } else if (s.from === 'slab') {
            const r = engine.slab_range();
            if (!(r[0] === s.lo && r[1] === s.hi)) {
              engine.slab_begin(s.lo, s.hi);
              for (let i = s.lo; i <= s.hi; i++) await engine.slab_push(i, new Uint8Array(await m.files[i].arrayBuffer()), isRaw(m.files[i].name));
              await engine.slab_finish();
            }
          }
          const r = engine.stroke(new Float32Array(s.dabs), s.target, s.from);
          const hist = engine.history();
          if (r) { const rgba = r.rgba; post({ type: 'patch', target: r.target, x: r.x, y: r.y, w: r.w, h: r.h, rgba: rgba.buffer, undo: hist[0], redo: hist[1], replay: true }, [rgba.buffer]); }
          else post({ type: 'patch', x: 0, y: 0, w: 0, h: 0, rgba: null, undo: hist[0], redo: hist[1], replay: true });
          done++;
        } catch (e) {   // its source is not to be had: the stroke is skipped, with the empty patch every stroke answers with (the page's history keeps step)
          skipped++; const hist = engine.history();
          post({ type: 'patch', x: 0, y: 0, w: 0, h: 0, rgba: null, undo: hist[0], redo: hist[1], replay: true });
          post({ type: 'debug', text: `replay: stroke ${done + skipped} skipped: ${(e && e.message) || e}` });
        }
      }
      post({ type: 'replayed', done, skipped });
    } else if (m.type === 'keep') {
      // the run's LAP or DFR master becomes a kept result (the page sends this before the
      // next run, or when it lets a result go): the engine takes it out of the run
      if (!engine.keep(m.id, m.kind)) post({ type: 'debug', text: `keep ${m.id} ${m.kind}: nothing to keep` });
    } else if (m.type === 'drop_kept') {
      engine.drop_kept(m.id);
    } else if (m.type === 'drop_all_kept') {
      engine.drop_all_kept();
    } else if (m.type === 'load_source') {
      // m.focus = {dim, w0, w1, tex}: the In focus rendering of the frame instead of the
      // plain frame. A request the page has since superseded (the user scrubbed on
      // while this one waited behind another) is skipped: each costs a full decode.
      // The decode + warp is skipped when the engine already holds that frame on the
      // GPU; the plain frame also needs the CPU readback (the retouch brush source).
      // m.bytes: the file, read by the page so the read overlapped our previous decode
      // (a read issued here would wait for it); without them the File is read here.
      if (running) return;
      if (m.gen < sourceGen) { post({ type: 'source-skipped', index: m.index, gen: m.gen, focus: !!m.focus }); return; }
      await needRaw([m.file]);
      const held = engine.source_gpu_index() === m.index;
      let r = null;
      if (!held) {
        const bytes = new Uint8Array(m.bytes || await m.file.arrayBuffer());
        r = await engine.load_source(m.index, bytes, !m.focus, isRaw(m.file && m.file.name));
      }
      if (m.focus) r = await engine.source_focus(m.focus.dim, m.focus.w0, m.focus.w1, m.focus.tex);
      else if (held) r = await engine.source_readback();
      const rgba = r.rgba;
      post({ type: 'source', index: r.index, w: r.w, h: r.h, rgba: rgba.buffer, gen: m.gen, focus: !!m.focus, prefetch: !!m.prefetch }, [rgba.buffer]);
    } else if (m.type === 'slab') {
      // The on-demand slab: frames m.lo..m.hi (m.files, in that order) fused on their own
      // as the retouch brush source. Each frame is decoded again, so a request the page
      // has since superseded (the user scrubbed on) is dropped, also between frames.
      if (running) return;
      const total = m.hi - m.lo + 1;
      const skip = () => post({ type: 'slab-skipped', lo: m.lo, hi: m.hi, gen: m.gen });
      if (m.gen < slabGen) { skip(); return; }
      await needRaw(m.files);
      engine.slab_begin(m.lo, m.hi);
      let next = readAhead(m.files[0]);
      for (let i = 0; i < total; i++) {
        post({ type: 'slab-progress', lo: m.lo, hi: m.hi, done: i, total, gen: m.gen });
        const bytes = new Uint8Array(await next);
        next = i + 1 < total ? readAhead(m.files[i + 1]) : null;
        await engine.slab_push(m.lo + i, bytes, isRaw(m.files[i].name));
        if (m.gen < slabGen) { engine.slab_cancel(); skip(); return; }
      }
      const r = await engine.slab_finish();
      post({ type: 'slab', lo: r.lo, hi: r.hi, w: r.w, h: r.h, rgba: r.rgba.buffer, gen: m.gen }, [r.rgba.buffer]);
    } else if (m.type === 'stroke' || m.type === 'undo' || m.type === 'redo') {
      const r = m.type === 'stroke' ? engine.stroke(m.dabs, m.target || 'fused', m.from || 'source') : m.type === 'undo' ? engine.undo() : engine.redo();
      const hist = engine.history();
      if (r) { const rgba = r.rgba; post({ type: 'patch', target: r.target, x: r.x, y: r.y, w: r.w, h: r.h, rgba: rgba.buffer, undo: hist[0], redo: hist[1] }, [rgba.buffer]); }
      else post({ type: 'patch', x: 0, y: 0, w: 0, h: 0, rgba: null, undo: hist[0], redo: hist[1] });
    } else if (m.type === 'depth_full') {
      const d = engine.depth_full();
      post({ type: 'depth_full', w: 0, data: d.buffer }, [d.buffer]);
    } else if (m.type === 'conf_full') {
      const d = engine.conf_full();
      post({ type: 'conf_full', w: 0, data: d.buffer }, [d.buffer]);
    } else if (m.type === 'save') {
      const bytes = engine.encode(m.kind, m.format || 'png', m.quality || 90, !!m.meta, !!m.crop, m.overlay || '');
      post({ type: 'png', kind: m.kind, format: m.format || 'png', bytes: bytes.buffer }, [bytes.buffer]);
    }
  } catch (e) {
    running = false;
    post({ type: 'error', text: (e && e.message) ? e.message : String(e) });
  }
}

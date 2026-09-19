// lapstack worker: owns the WASM engine and the WebGPU device. The page sends
// {type:'init'|'run'|'cancel'|'save'}; the worker answers with progress and
// results (large buffers are transferred, not copied).
// The WASM glue and binary are imported on 'init' with a per-load query
// string so a rebuilt pkg/ is never served from the browser cache (a stale
// glue shows up as "engine.<new method> is not a function"). Not a top-level
// await: that would suspend the script before onmessage is installed and the
// page's first message would be lost.
const V = Date.now();
let init = null, create_engine = null, thumbnail = null;
async function loadWasm() {
  if (init) return;
  const mod = await import(`./pkg/lapstack_web.js?t=${V}`);
  ({ default: init, create_engine, thumbnail } = mod);
}

let engine = null;
let cancelled = false;
let running = false;
let thumbJob = null;      // {files, indices, edge, gen} being decoded in the background
let thumbGen = 0;

// Decode + downscale added frames one by one (yielding between frames so a
// 'run' or 'cancel' message can interleave); a run pauses this until it ends.
async function thumbLoop() {
  while (thumbJob && thumbJob.files.length) {
    if (running || !thumbnail) { await new Promise((r) => setTimeout(r, 200)); continue; }
    const job = thumbJob;
    const f = job.files.shift(), i = job.indices.shift();
    try {
      const bytes = new Uint8Array(await f.arrayBuffer());
      if (job.gen !== thumbGen) return;
      const t = thumbnail(bytes, job.edge);
      post({ type: 'thumb', index: i, name: f.name, w: t.w, h: t.h, bits: t.bits, proxy_w: t.proxy_w, proxy_h: t.proxy_h, proxy: t.proxy.buffer }, [t.proxy.buffer]);
    } catch (e) {
      post({ type: 'thumb-error', index: i, name: f.name, text: (e && e.message) ? e.message : String(e) });
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  thumbJob = null;
}

function post(msg, transfer) { self.postMessage(msg, transfer || []); }

// Engine calls must not overlap (wasm-bindgen rejects re-entrant use of the
// engine while an async call is in flight), so everything that touches it
// runs through one promise chain; only cancel/clear are handled immediately.
let chain = Promise.resolve();
const enqueue = (fn) => { chain = chain.then(fn).catch((e) => post({ type: 'error', text: (e && e.message) ? e.message : String(e) })); };

self.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'cancel') { cancelled = true; return; }
  if (m.type === 'clear') { thumbGen++; thumbJob = null; return; }
  enqueue(() => handle(m));
};

async function handle(m) {
  try {
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
      engine = await create_engine();
      post({ type: 'ready', info: { ...JSON.parse(engine.info()), ...(ainfo || {}) } });
    } else if (m.type === 'thumbs') {
      if (thumbJob) { thumbJob.files.push(...m.files); thumbJob.indices.push(...m.indices); }
      else { thumbJob = { files: [...m.files], indices: [...m.indices], edge: m.edge, gen: thumbGen }; thumbLoop(); }
    } else if (m.type === 'run') {
      if (running) return;
      running = true; cancelled = false;
      engine.reset();
      const params = JSON.stringify(m.params);
      const t0 = performance.now();
      for (let i = 0; i < m.files.length; i++) {
        if (cancelled) break;
        const f = m.files[i];
        post({ type: 'stage', text: `decoding ${f.name}`, done: i, total: m.files.length });
        const bytes = new Uint8Array(await f.arrayBuffer());
        const r = await engine.push(bytes, params);
        const proxy = r.proxy, peak = r.peak;
        post({ type: 'frame', index: r.index, name: f.name, w: r.w, h: r.h, bits: r.bits,
               proxy_w: r.proxy_w, proxy_h: r.proxy_h, proxy: proxy.buffer,
               peak_w: r.peak_w, peak_h: r.peak_h, peak: peak.buffer, sim: r.sim, ms: r.ms,
               done: i + 1, total: m.files.length }, [proxy.buffer, peak.buffer]);
      }
      if (cancelled) { engine.reset(); post({ type: 'cancelled' }); running = false; return; }
      post({ type: 'stage', text: 'collapsing', done: m.files.length, total: m.files.length });
      const res = await engine.finish();
      post({ type: 'done', w: res.w, h: res.h, bits: res.bits, frames: res.frames,
             rgba: res.rgba.buffer, depth_w: res.depth_w, depth_h: res.depth_h, depth: res.depth.buffer,
             winner_w: res.winner_w, winner_h: res.winner_h, winner: res.winner.buffer,
             ms: performance.now() - t0 }, [res.rgba.buffer, res.depth.buffer, res.winner.buffer]);
      // optional second pass: render a second image from the depth map (frames are decoded again)
      if (m.params.render_dmap) {
        const t1 = performance.now();
        for (let i = 0; i < m.files.length; i++) {
          if (cancelled) break;
          post({ type: 'stage', text: `rendering from depth map: ${m.files[i].name}`, done: i, total: m.files.length });
          const bytes = new Uint8Array(await m.files[i].arrayBuffer());
          await engine.render_push(i, bytes);
        }
        if (cancelled) { post({ type: 'render-cancelled' }); running = false; return; }
        const r = await engine.render_finish();
        post({ type: 'done2', w: r.w, h: r.h, rgba: r.rgba.buffer, ms: performance.now() - t1 }, [r.rgba.buffer]);
      }
      running = false;
    } else if (m.type === 'load_source') {
      // m.focus = {dim, w0, w1, tex}: the In focus rendering of the frame instead of the
      // plain frame; the decode is skipped when the engine already holds that frame.
      if (running) return;
      let r = null;
      if (!(m.focus && engine.source_index() === m.index)) {
        const bytes = new Uint8Array(await m.file.arrayBuffer());
        r = await engine.load_source(m.index, bytes);
      }
      if (m.focus) r = engine.source_focus(m.focus.dim, m.focus.w0, m.focus.w1, m.focus.tex);
      const rgba = r.rgba;
      post({ type: 'source', index: r.index, w: r.w, h: r.h, rgba: rgba.buffer, gen: m.gen, focus: !!m.focus }, [rgba.buffer]);
    } else if (m.type === 'stroke' || m.type === 'undo' || m.type === 'redo') {
      const r = m.type === 'stroke' ? engine.stroke(m.dabs, m.target || 'fused') : m.type === 'undo' ? engine.undo() : engine.redo();
      const hist = engine.history();
      if (r) { const rgba = r.rgba; post({ type: 'patch', target: r.target, x: r.x, y: r.y, w: r.w, h: r.h, rgba: rgba.buffer, undo: hist[0], redo: hist[1] }, [rgba.buffer]); }
      else post({ type: 'patch', x: 0, y: 0, w: 0, h: 0, rgba: null, undo: hist[0], redo: hist[1] });
    } else if (m.type === 'depth_full') {
      const d = engine.depth_full();
      post({ type: 'depth_full', w: 0, data: d.buffer }, [d.buffer]);
    } else if (m.type === 'save') {
      const bytes = engine.encode(m.kind, m.format || 'png', m.quality || 90);
      post({ type: 'png', kind: m.kind, format: m.format || 'png', bytes: bytes.buffer }, [bytes.buffer]);
    }
  } catch (e) {
    running = false;
    post({ type: 'error', text: (e && e.message) ? e.message : String(e) });
  }
}

// lapstack worker: owns the WASM engine and the WebGPU device. The page sends
// {type:'init'|'run'|'cancel'|'save'}; the worker answers with progress and
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

let engine = null;
let gifw = null;          // the animated GIF being written (gif_begin … gif_end), see gif.rs
let cancelled = false;
let running = false;
let thumbJob = null;      // {files, indices, edge, gen} being decoded in the background
let thumbGen = 0;
let sourceGen = -1;       // newest 'load_source' request seen; older ones still queued are skipped

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
      const [proxy, strip] = await proxyBitmaps(t.proxy.buffer, t.proxy_w, t.proxy_h);
      post({ type: 'thumb', index: i, name: f.name, w: t.w, h: t.h, bits: t.bits, proxy, strip }, [proxy, strip]);
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
function readAhead(f) { const p = f.arrayBuffer(); p.catch(() => {}); return p; }

// Engine calls must not overlap (wasm-bindgen rejects re-entrant use of the
// engine while an async call is in flight), so everything that touches it
// runs through one promise chain; only cancel/clear are handled immediately.
let chain = Promise.resolve();
const enqueue = (fn) => { chain = chain.then(fn).catch((e) => post({ type: 'error', text: (e && e.message) ? e.message : String(e) })); };
// Requests carrying an `rid` are remote calls from the page (see call() there): the reply
// echoes the rid, and a failure answers that call instead of surfacing as a run error.
const rpc = (m, fn) => enqueue(async () => {
  try { await fn(); } catch (e) { post({ type: 'rpc-error', rid: m.rid, text: (e && e.message) ? e.message : String(e) }); }
});

self.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'cancel') { cancelled = true; return; }
  if (m.type === 'clear') { thumbGen++; thumbJob = null; return; }
  if (m.type === 'load_source') sourceGen = Math.max(sourceGen, m.gen);
  if (m.rid) rpc(m, () => handleCall(m)); else enqueue(() => handle(m));
};

// ---------- remote calls (Save step) ----------
// The page composes export frames itself and only needs the engine for what it cannot
// do: full-resolution aligned frames (plain or In focus), GIF quantisation + LZW, image
// encoding, and content credentials.
async function handleCall(m) {
  if (m.type === 'save') {
    const bytes = engine.encode(m.kind, m.format || 'png', m.quality || 90);
    post({ type: 'png', rid: m.rid, kind: m.kind, format: m.format || 'png', bytes: bytes.buffer }, [bytes.buffer]);
  } else if (m.type === 'export_source') {
    // the aligned full-res frame (or its In focus rendering, m.focus = focusParams), like load_source
    // without the scrub bookkeeping; the page passes the file bytes it has already read
    if (running) throw new Error('a run is in progress');
    const held = engine.source_gpu_index() === m.index;
    let r = null;
    if (!held) r = await engine.load_source(m.index, new Uint8Array(m.bytes || await m.file.arrayBuffer()), !m.focus);
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
      let next = m.files.length ? readAhead(m.files[0]) : null;
      for (let i = 0; i < m.files.length; i++) {
        if (cancelled) break;
        const f = m.files[i];
        post({ type: 'stage', text: `decoding ${f.name}`, done: i, total: m.files.length });
        const bytes = new Uint8Array(await next);
        next = i + 1 < m.files.length ? readAhead(m.files[i + 1]) : null;
        const r = await engine.push(bytes, params);
        const peak = r.peak;
        const [proxy, strip] = await proxyBitmaps(r.proxy.buffer, r.proxy_w, r.proxy_h);
        post({ type: 'frame', index: r.index, name: f.name, w: r.w, h: r.h, bits: r.bits, proxy, strip,
               peak_w: r.peak_w, peak_h: r.peak_h, peak: peak.buffer, sim: r.sim, ms: r.ms,
               done: i + 1, total: m.files.length }, [proxy, strip, peak.buffer]);
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
        let next = readAhead(m.files[0]);
        for (let i = 0; i < m.files.length; i++) {
          if (cancelled) break;
          post({ type: 'stage', text: `rendering from depth map: ${m.files[i].name}`, done: i, total: m.files.length });
          const bytes = new Uint8Array(await next);
          next = i + 1 < m.files.length ? readAhead(m.files[i + 1]) : null;
          await engine.render_push(i, bytes);
        }
        if (cancelled) { post({ type: 'render-cancelled' }); running = false; return; }
        const r = await engine.render_finish();
        post({ type: 'done2', w: r.w, h: r.h, rgba: r.rgba.buffer, ms: performance.now() - t1 }, [r.rgba.buffer]);
      }
      running = false;
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
      const held = engine.source_gpu_index() === m.index;
      let r = null;
      if (!held) {
        const bytes = new Uint8Array(m.bytes || await m.file.arrayBuffer());
        r = await engine.load_source(m.index, bytes, !m.focus);
      }
      if (m.focus) r = await engine.source_focus(m.focus.dim, m.focus.w0, m.focus.w1, m.focus.tex);
      else if (held) r = await engine.source_readback();
      const rgba = r.rgba;
      post({ type: 'source', index: r.index, w: r.w, h: r.h, rgba: rgba.buffer, gen: m.gen, focus: !!m.focus, prefetch: !!m.prefetch }, [rgba.buffer]);
    } else if (m.type === 'stroke' || m.type === 'undo' || m.type === 'redo') {
      const r = m.type === 'stroke' ? engine.stroke(m.dabs, m.target || 'fused', m.from || 'source') : m.type === 'undo' ? engine.undo() : engine.redo();
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

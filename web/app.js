// lapstack browser UI. A focus-stacking workbench: filmstrip, parameter
// panel, run/cancel with progress + log, viewer tabs (Source / Fused / Depth),
// tiled-free zoom/pan on a canvas, A/B compare with a draggable divider and
// hold-to-flip, depth map gray/Turbo, frame scrubbing on the Source view.
// All heavy work happens in worker.js (WASM + WebGPU).

const $ = (id) => document.getElementById(id);
const logEl = $('log');
function log(s) {
  logEl.textContent += s + '\n';
  if (!document.body.classList.contains('log-collapsed')) logEl.scrollTop = logEl.scrollHeight;
}

// log strip toolbar: collapse to just the toolbar, and copy the whole log
const setLogCollapsed = (on) => {
  document.body.classList.toggle('log-collapsed', on);
  const b = $('log-toggle');
  b.textContent = on ? '▴' : '▾';
  b.title = on ? 'expand log' : 'collapse log';
  b.setAttribute('aria-expanded', String(!on));
  if (!on) logEl.scrollTop = logEl.scrollHeight;
};
$('log-toggle').onclick = () => setLogCollapsed(!document.body.classList.contains('log-collapsed'));
$('log-copy').onclick = async () => {
  const b = $('log-copy');
  try {
    await navigator.clipboard.writeText(logEl.textContent);
  } catch {
    // clipboard API needs a secure context; fall back to a scratch selection
    const ta = document.createElement('textarea');
    ta.value = logEl.textContent; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch {}
    ta.remove();
  }
  b.textContent = '✓'; b.disabled = true;
  setTimeout(() => { b.textContent = '⧉'; b.disabled = false; }, 900);
};

// ---------- state ----------
const st = {
  files: [],            // File objects
  frames: [],           // per processed frame: {name, w, h, proxy: ImageBitmap, sim}
  result: null,         // {w, h, bits, fused: OffscreenCanvas, dmap: OffscreenCanvas|null, depth: Float32Array, dw, dh, winner: Float32Array, ww, wh}
  depthBmp: new Map(),  // 'layer:lut' -> ImageBitmap (layer = depth | winner)
  step: 'stack', view: 'source', selected: 0,
  compare: false, cmp: 'depth', cmpMode: 'swipe', divider: 0.5, flipped: false,
  turbo: false, slice: true,
  sliceBmps: new Map(),   // 'layer:frame index' -> ImageBitmap (magenta band over pixels assigned to that frame)
  peak: { on: false, thr: 0.5, max: 0, pixmax: null, floor: 0 },   // focus peaking, see peakMask()
  zoom: 1, ox: 0, oy: 0, fitted: true,
  running: false,
  retouch: { size: 100, hard: 0.5, painting: false, dabs: [], last: null, cursor: null, target: 'fused',
             srcIndex: -1, srcCanvas: null, loading: -1, gen: 0, undo: 0, redo: 0 },
};
window.__st = st; window.__draw = () => draw();
const dpr = () => window.devicePixelRatio || 1;

// ---------- settings (persisted) ----------
const PK = 'lapstack.settings';
const stepDefaults = { 'p-coarsen': 2, 'p-levels': 0, 'p-energy': 1, 'p-topr': 2, 'p-depthscale': 2, 'p-depthlevel': 2, 'p-proxy': 1400 };
function readParams() {
  const n = (id) => Number($(id).textContent === 'auto' ? 0 : $(id).textContent);
  return {
    align: $('p-align').checked, shift: $('p-shift').checked, scale: $('p-scale').checked, rotation: $('p-rotation').checked,
    coarsen: n('p-coarsen'), levels: n('p-levels') || null, energy_radius: n('p-energy'), top: $('p-top').value,
    top_radius: n('p-topr'), use_chroma: $('p-chroma').checked, proxy_edge: n('p-proxy'),
    depth_scale: n('p-depthscale'), depth_level: n('p-depthlevel'), render_dmap: $('p-dmap').checked,
    turbo: st.turbo, slice: st.slice, peak_on: st.peak.on, peak_thr: st.peak.thr, cmp_mode: st.cmpMode,
    brush_size: st.retouch.size, brush_hard: st.retouch.hard,
  };
}
function setStep(id, v) {
  const el = $(id); const lo = Number(el.dataset.min), hi = Number(el.dataset.max);
  v = Math.min(hi, Math.max(lo, v));
  el.textContent = (v === 0 && el.dataset.zero) ? el.dataset.zero : String(v);
}
function applyParams(p) {
  if (!p) return;
  $('p-align').checked = p.align ?? true; $('p-shift').checked = p.shift ?? true; $('p-scale').checked = p.scale ?? true; $('p-rotation').checked = p.rotation ?? true;
  setStep('p-coarsen', p.coarsen ?? 2); setStep('p-levels', p.levels ?? 0); setStep('p-energy', p.energy_radius ?? 1);
  $('p-top').value = p.top ?? 'de'; setStep('p-topr', p.top_radius ?? 2); $('p-chroma').checked = p.use_chroma ?? false;
  setStep('p-proxy', p.proxy_edge ?? 1400); st.turbo = p.turbo ?? false;
  setStep('p-depthscale', p.depth_scale ?? 2); setStep('p-depthlevel', p.depth_level ?? 2); $('p-dmap').checked = p.render_dmap ?? false; st.cmpMode = p.cmp_mode ?? 'swipe';
  st.peak.on = p.peak_on ?? false; st.peak.thr = p.peak_thr ?? 0.5; st.slice = p.slice ?? true;
  st.retouch.size = p.brush_size ?? 100; st.retouch.hard = p.brush_hard ?? 0.5;
}
function saveParams() { try { localStorage.setItem(PK, JSON.stringify(readParams())); } catch {} }
try { applyParams(JSON.parse(localStorage.getItem(PK))); } catch {}
for (const id of Object.keys(stepDefaults)) setStep(id, Number($(id).textContent === 'auto' ? 0 : $(id).textContent));
document.querySelectorAll('#params [data-step]').forEach((b) => b.addEventListener('click', () => {
  const id = b.dataset.step; const cur = Number($(id).textContent === 'auto' ? 0 : $(id).textContent);
  setStep(id, cur + Number(b.dataset.d)); saveParams();
}));
document.querySelectorAll('#params input, #params select').forEach((el) => el.addEventListener('change', saveParams));
// Run menu (DFR lives here, not in the parameter panel): the Run label shows the state
const runLabel = () => { $('run').textContent = $('p-dmap').checked ? 'Run LAP + DFR' : 'Run LAP'; };
$('p-dmap').addEventListener('change', () => { saveParams(); runLabel(); });
$('run-more').addEventListener('click', (e) => { e.stopPropagation(); $('runmenu').hidden = !$('runmenu').hidden; });
$('runmenu').addEventListener('click', (e) => e.stopPropagation());
document.addEventListener('click', () => { $('runmenu').hidden = true; });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('runmenu').hidden = true; });
runLabel();

// ---------- worker ----------
// (query string: never run a stale cached worker after a rebuild; serve.sh also sends no-store)
const worker = new Worker('./worker.js?t=' + Date.now(), { type: 'module' });
worker.onmessage = (ev) => {
  const m = ev.data;
  switch (m.type) {
    case 'ready': {
      $('status').textContent = 'ready'; $('progress').className = '';
      const name = [m.info.vendor, m.info.architecture, m.info.description].filter(Boolean).join(' ') || 'WebGPU adapter';
      $('gpuinfo').textContent = `${name} · buffers ≤ ${m.info.max_buffer_mb} MB, bindings ≤ ${m.info.max_storage_mb} MB`;
      log(`[lapstack] WebGPU ready: ${JSON.stringify(m.info)}`);
      if (/swiftshader|llvmpipe|software/i.test(name) || m.info.fallback) {
        toast('WebGPU is running on a software adapter (' + name + '): it will work but slowly, and large frames may exceed its buffer limits. ' + gpuHint(), 0);
      }
      break;
    }
    case 'stage': setProgress(m.text, m.done, m.total); break;
    case 'frame': onFrame(m); break;
    case 'thumb': onThumb(m); break;
    case 'source': onSource(m); break;
    case 'patch': onPatch(m); break;
    case 'thumb-error': log(`[lapstack] cannot decode ${m.name}: ${m.text}`); break;
    case 'done': onDone(m); break;
    case 'done2': onDone2(m); break;
    case 'render-cancelled': endRun('cancelled (depth-map render)'); log('[lapstack] depth-map render cancelled; the LAP result is kept'); finishRun(); break;
    case 'cancelled': endRun('cancelled'); log('[lapstack] cancelled'); break;
    case 'error': endRun('error'); log('[lapstack] error: ' + m.text); toast(/no WebGPU adapter/i.test(m.text) ? 'No WebGPU adapter. ' + gpuHint() : m.text, 0); break;
    case 'png': download(m); break;
    case 'debug': log('[worker] ' + m.text); break;
  }
};
worker.postMessage({ type: 'init' });

// ---------- toasts ----------
function toast(text, ms = 8000) {
  const t = document.createElement('div'); t.className = 'toast'; t.textContent = text;
  const x = document.createElement('button'); x.textContent = '×'; x.addEventListener('click', () => t.remove()); t.appendChild(x);
  $('toasts').appendChild(t); if (ms) setTimeout(() => t.remove(), ms);
}
function gpuHint() {
  const ua = navigator.userAgent;
  if (/Firefox/.test(ua)) return 'Firefox: set dom.webgpu.enabled = true in about:config (and dom.webgpu.workers.enabled), or use Chrome/Edge.';
  if (/Linux/.test(ua)) return 'Chrome on Linux ships WebGPU behind command-line switches, and the chrome://flags pair only gives the SwiftShader software adapter (leave those flags at Default). Start a separate WebGPU profile with ./web/chrome.sh (google-chrome --user-data-dir=~/.config/lapstack-chrome --enable-unsafe-webgpu --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan). Avoid the flags page\'s ANGLE=Vulkan option: it blanks the 2D canvas on NVIDIA/Wayland. probe.html shows what the browser sees.';
  return 'This browser has no WebGPU adapter. Chrome/Edge 113+ (Windows/macOS) and Safari 26 support it; check chrome://gpu, and that hardware acceleration is on.';
}

// ---------- files / filmstrip ----------
function addFiles(list) {
  const files = [...list].filter((f) => /\.(png|jpe?g|tiff?)$/i.test(f.name)).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  if (!files.length) return;
  st.files.push(...files);
  renderFilmstrip();
  $('run').disabled = st.running || !st.files.length;
  $('tab-source').disabled = false;
  if (st.view === 'source') draw();
  log(`[lapstack] ${files.length} frame(s) added (${st.files.length} total)`);
  for (const f of files) makeThumb(f);
  const indices = files.map((f) => st.files.indexOf(f));
  worker.postMessage({ type: 'thumbs', files, indices, edge: readParams().proxy_edge });
}
async function onThumb(m) {
  if (!st.files[m.index] || st.files[m.index].name !== m.name) return; // stale (cleared)
  const cur = st.frames[m.index];
  if (cur && cur.proxy && cur.sim) return; // the run already supplied an aligned proxy
  const bmp = await createImageBitmap(new ImageData(new Uint8ClampedArray(m.proxy), m.proxy_w, m.proxy_h));
  st.frames[m.index] = { ...(cur || {}), name: m.name, w: m.w, h: m.h, bits: m.bits, proxy: bmp };
  renderFilmstrip();
  if (st.view === 'source' && st.selected === m.index) draw();
}
async function makeThumb(f) {
  if (!/\.(png|jpe?g)$/i.test(f.name)) return; // the browser cannot decode TIFF; the run supplies a proxy
  try {
    const bmp = await createImageBitmap(f, { resizeWidth: 320, resizeQuality: 'medium' });
    const i = st.files.indexOf(f); if (i < 0) return;
    st.frames[i] = st.frames[i] || { name: f.name };
    if (!st.frames[i].proxy) { st.frames[i].thumb = bmp; renderFilmstrip(); if (st.view === 'source' && st.selected === i) draw(); }
  } catch {}
}
function renderFilmstrip() {
  const fs = $('filmstrip'); fs.innerHTML = '';
  if (!st.files.length) { fs.innerHTML = '<div class="empty dim">Add frames, or drop them here.</div>'; return; }
  st.files.forEach((f, i) => {
    const d = document.createElement('div'); d.className = 'thumb' + (i === st.selected && scrubbable() ? ' sel' : '');
    const fr = st.frames[i];
    const bmp = fr && (fr.proxy || fr.thumb);
    if (bmp) {
      const c = document.createElement('canvas'); c.width = 160; c.height = 100;
      const s = Math.min(160 / bmp.width, 100 / bmp.height);
      c.getContext('2d').drawImage(bmp, (160 - bmp.width * s) / 2, (100 - bmp.height * s) / 2, bmp.width * s, bmp.height * s);
      d.appendChild(c);
    } else { const ph = document.createElement('div'); ph.className = 'ph'; ph.textContent = fr ? '…' : String(i); d.appendChild(ph); }
    const n = document.createElement('div'); n.className = 'name'; n.textContent = f.name; d.appendChild(n);
    if (fr && fr.sim) { const s = document.createElement('div'); s.className = 'sim'; s.textContent = `${fr.sim[0].toFixed(1)}, ${fr.sim[1].toFixed(1)} px · ×${fr.sim[2].toFixed(4)} · ${fr.sim[3].toFixed(2)}°`; d.appendChild(s); }
    if (st.peak.on && fr && fr.peak) { const s = document.createElement('div'); s.className = 'sim pct'; s.textContent = `${peakPercent(fr).toFixed(1)} % in focus`; d.appendChild(s); }
    d.addEventListener('click', () => { st.selected = i; if (!scrubbable()) st.view = 'source'; updateTabs(); renderFilmstrip(); draw(); });
    fs.appendChild(d);
  });
}
$('add').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
$('clear').addEventListener('click', () => { if (st.running) return; worker.postMessage({ type: 'clear' }); st.files = []; st.frames = []; st.result = null; st.depthBmp.clear(); st.sliceBmps.clear(); st.peak.pixmax = null; resetRetouch(); st.step = 'stack'; if (st.view === 'retouch') st.view = 'source'; gotoStep('stack'); renderFilmstrip(); updateTabs(); setView('source'); });
document.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('drop'); });
document.addEventListener('dragleave', () => document.body.classList.remove('drop'));
document.addEventListener('drop', (e) => { e.preventDefault(); document.body.classList.remove('drop'); if (!st.running) addFiles(e.dataTransfer.files); });

// ---------- run ----------
function setProgress(text, done, total) {
  $('progress').className = 'running'; $('status').textContent = `${text} ${total ? `${done}/${total}` : ''}`;
  $('fill').style.width = total ? `${(100 * done / total).toFixed(1)}%` : '100%';   // no total = indeterminate: full pole
}
$('run').addEventListener('click', () => {
  if (st.running || !st.files.length) return;
  st.running = true; st.frames = st.frames.map((f) => (f ? { name: f.name, thumb: f.thumb, proxy: f.proxy, w: f.w, h: f.h, bits: f.bits } : f)); st.result = null; st.depthBmp.clear(); st.sliceBmps.clear(); st.peak.pixmax = null; resetRetouch(); if (st.step !== 'stack') gotoStep('stack');
  runLabel(); $('runwrap').hidden = true; $('runmenu').hidden = true; $('cancel').hidden = false; $('clear').disabled = true;
  setProgress('starting', 0, st.files.length);
  const params = readParams(); delete params.turbo;
  log(`[lapstack] run: ${st.files.length} frames, ${JSON.stringify(params)}`);
  st.t0 = performance.now(); st.rendering = !!params.render_dmap; window.__app_done = null;
  worker.postMessage({ type: 'run', files: st.files, params });
});
$('cancel').addEventListener('click', () => worker.postMessage({ type: 'cancel' }));
function endRun(status) {
  st.running = false; $('runwrap').hidden = false; $('cancel').hidden = true; $('clear').disabled = false;
  $('progress').className = status.startsWith('done') ? 'done' : 'error'; $('fill').style.width = '0';
  $('run').disabled = !st.files.length; $('status').textContent = status; updateTabs();
}
async function onFrame(m) {
  const img = new ImageData(new Uint8ClampedArray(m.proxy), m.proxy_w, m.proxy_h);
  const bmp = await createImageBitmap(img);
  const peak = { w: m.peak_w, h: m.peak_h, data: new Float32Array(m.peak), bmp: null, bmpThr: -1, pct: null, pctThr: -1 };
  st.peak.pixmax = null;
  st.frames[m.index] = { name: m.name, w: m.w, h: m.h, bits: m.bits, proxy: bmp, sim: m.sim, peak };
  setProgress('fusing', m.done, m.total);
  log(`[lapstack]   frame ${String(m.index).padStart(3)}: dx=${m.sim[0].toFixed(2)}px dy=${m.sim[1].toFixed(2)}px scale=${m.sim[2].toFixed(5)} rot=${m.sim[3].toFixed(3)}°  (${m.ms.toFixed(0)} ms)`);
  renderFilmstrip();
  if (st.view === 'source' && st.selected === m.index) draw();
}
async function onDone(m) {
  const img = new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h);
  const fused = new OffscreenCanvas(m.w, m.h);
  fused.getContext('2d').putImageData(img, 0, 0);
  st.result = { w: m.w, h: m.h, bits: m.bits, fused, dmap: null, depth: new Float32Array(m.depth), dw: m.depth_w, dh: m.depth_h,
                winner: new Float32Array(m.winner), ww: m.winner_w, wh: m.winner_h };
  resetRetouch();
  const secs = ((performance.now() - st.t0) / 1000).toFixed(1);
  log(`[lapstack] fused ${m.frames} frames -> ${m.w}x${m.h} ${m.bits}-bit  (${secs}s)`);
  st.frameCount = m.frames;
  if (st.rendering) { setProgress('rendering from depth map', 0, st.files.length); setView('fused'); return; }
  endRun(`done in ${secs}s`);
  finishRun();
}
// second stacked image, rendered from the depth map (optional second pass)
async function onDone2(m) {
  const img = new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h);
  const cv = new OffscreenCanvas(m.w, m.h);
  cv.getContext('2d').putImageData(img, 0, 0);
  if (st.result) st.result.dmap = cv;
  const secs = ((performance.now() - st.t0) / 1000).toFixed(1);
  log(`[lapstack] depth-map render done (${(m.ms / 1000).toFixed(1)}s)  (${secs}s total)`);
  endRun(`done in ${secs}s`);
  // both stacked images side by side
  st.compare = true; st.cmpMode = 'split'; st.cmp = 'dmap'; st.flipped = false;
  finishRun();
}
function finishRun() {
  st.rendering = false;
  setView('fused');
  window.__app_done = JSON.stringify({ ok: true, w: st.result?.w, h: st.result?.h, frames: st.frameCount, dmap: !!(st.result && st.result.dmap), secs: ((performance.now() - st.t0) / 1000).toFixed(1) });
}
// ---------- save step ----------
$('sv-format').addEventListener('change', () => { const j = $('sv-format').value === 'jpeg'; $('sv-qrow').hidden = !j; $('sv-quality').hidden = !j; });
$('sv-quality').addEventListener('input', (e) => { $('sv-qval').textContent = e.target.value; });
$('sv-image').addEventListener('click', () => worker.postMessage({ type: 'save', kind: $('sv-which').value, format: $('sv-format').value, quality: Number($('sv-quality').value) }));
$('sv-depth').addEventListener('click', () => worker.postMessage({ type: 'save', kind: 'depth', format: 'png' }));
$('sv-depth16').addEventListener('click', () => worker.postMessage({ type: 'save', kind: 'depth16', format: 'png' }));
$('sv-winner').addEventListener('click', () => worker.postMessage({ type: 'save', kind: 'winner', format: 'png' }));
function download(m) {
  const jpeg = m.format === 'jpeg';
  const blob = new Blob([m.bytes], { type: jpeg ? 'image/jpeg' : 'image/png' });
  const base = ($('sv-name').value || 'stacked').replace(/[^\w.-]+/g, '_');
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${base}${m.kind === 'fused' ? '' : '_' + (m.kind === 'dmap' ? 'dfr' : m.kind)}.${jpeg ? 'jpg' : 'png'}`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  log(`[lapstack] saved ${a.download} (${(blob.size / 1e6).toFixed(1)} MB)`);
}

// ---------- depth LUT ----------
function turbo(t) { // Google Turbo colormap, polynomial fit
  const r = 34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05))));
  const g = 23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (1073.77 + t * 707.56))));
  const b = 27.2 + t * (3211.1 + t * (-15327.97 + t * (27814 + t * (-22569.18 + t * 6838.66))));
  return [r, g, b].map((v) => Math.max(0, Math.min(255, v)));
}
// the two depth layers: 'depth' = depth from focus (DFF, working grid), 'winner' = LAP winner index
const isDepthLayer = (t) => t === 'depth' || t === 'winner';
function depthData(layer) {
  const r = st.result; if (!r) return null;
  return layer === 'winner' ? { data: r.winner, w: r.ww, h: r.wh } : { data: r.depth, w: r.dw, h: r.dh };
}
async function depthBitmap(layer, useTurbo) {
  const D = depthData(layer); if (!D) return null;
  const key = `${layer}:${useTurbo ? 'turbo' : 'gray'}`;
  if (st.depthBmp.has(key)) return st.depthBmp.get(key);
  let lo = Infinity, hi = -Infinity; for (const v of D.data) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const range = Math.max(hi - lo, 1e-6);
  const px = new Uint8ClampedArray(D.w * D.h * 4);
  for (let i = 0; i < D.w * D.h; i++) {
    const t = (D.data[i] - lo) / range;
    const [cr, cg, cb] = useTurbo ? turbo(t) : [t * 255, t * 255, t * 255];
    px[4 * i] = cr; px[4 * i + 1] = cg; px[4 * i + 2] = cb; px[4 * i + 3] = 255;
  }
  const bmp = await createImageBitmap(new ImageData(px, D.w, D.h));
  st.depthBmp.set(key, bmp); return bmp;
}

// ---------- focus peaking ----------
// A pixel counts as "in focus" in frame i when that frame's level-1 contrast
// sqrt(E_i(x)) is at least `thr` × the largest contrast any frame has at x, and
// that maximum is above a noise floor (2 % of the stack's 99.5th percentile),
// so flat areas do not flicker. Scrubbing then shows the in-focus band sweep.
function peakStats() {
  const frs = st.frames.filter((f) => f && f.peak);
  if (!frs.length) return null;
  if (st.peak.pixmax && st.peak.pixmax.n === frs.length) return st.peak;
  const n = frs[0].peak.data.length; const pm = new Float32Array(n);
  for (const f of frs) { const d = f.peak.data; for (let i = 0; i < n; i++) if (d[i] > pm[i]) pm[i] = d[i]; }
  const sorted = Float32Array.from(pm).sort();
  st.peak.floor = 0.02 * sorted[Math.floor(0.995 * (n - 1))];
  pm.n = frs.length; st.peak.pixmax = pm;
  for (const f of frs) { f.peak.pct = null; f.peak.bmp = null; }
  return st.peak;
}
function peakMask(fr) { const ps = peakStats(); return { p: fr.peak, pm: ps.pixmax, thr: st.peak.thr, floor: ps.floor }; }
function peakPercent(fr) {
  const { p, pm, thr, floor } = peakMask(fr);
  if (p.pctThr === thr && p.pct !== null) return p.pct;
  const t2 = thr * thr;
  let n = 0; for (let i = 0; i < p.data.length; i++) if (pm[i] > floor && p.data[i] >= t2 * pm[i]) n++;
  p.pct = 100 * n / p.data.length; p.pctThr = thr; return p.pct;
}
async function peakBitmap(fr) {
  const { p, pm, thr, floor } = peakMask(fr);
  if (p.bmp && p.bmpThr === thr) return p.bmp;
  const px = new Uint8ClampedArray(p.w * p.h * 4);
  const t2 = thr * thr;
  for (let i = 0; i < p.data.length; i++) if (pm[i] > floor && p.data[i] >= t2 * pm[i]) { px[4 * i] = 255; px[4 * i + 1] = 0; px[4 * i + 2] = 255; px[4 * i + 3] = 150; }
  p.bmp = await createImageBitmap(new ImageData(px, p.w, p.h)); p.bmpThr = thr; return p.bmp;
}
function setPeakThr(v) { st.peak.thr = Math.min(1, Math.max(0.05, v)); $('peakthr').textContent = st.peak.thr.toFixed(2); saveParams(); renderFilmstrip(); draw(); }
$('peak').addEventListener('change', (e) => { st.peak.on = e.target.checked; saveParams(); updateTabs(); renderFilmstrip(); draw(); });

// ---------- depth slice ----------
// Magenta band over the pixels whose depth index is the scrubbed frame (the depth "slice").
async function sliceBitmap(layer, ix) {
  const D = depthData(layer); if (!D) return null;
  const key = `${layer}:${ix}`;
  if (st.sliceBmps.has(key)) return st.sliceBmps.get(key);
  const px = new Uint8ClampedArray(D.w * D.h * 4);
  for (let i = 0; i < D.data.length; i++) if (Math.round(D.data[i]) === ix) { px[4 * i] = 255; px[4 * i + 1] = 0; px[4 * i + 2] = 255; px[4 * i + 3] = 150; }
  const bmp = await createImageBitmap(new ImageData(px, D.w, D.h));
  if (st.sliceBmps.size > 32) st.sliceBmps.clear();
  st.sliceBmps.set(key, bmp); return bmp;
}
$('slice').addEventListener('change', (e) => { st.slice = e.target.checked; saveParams(); updateTabs(); draw(); });
$('peak-minus').addEventListener('click', () => setPeakThr(st.peak.thr - 0.05));
$('peak-plus').addEventListener('click', () => setPeakThr(st.peak.thr + 0.05));

// ---------- retouch ----------
// Two panes (source | fused) with one transform. The brush copies the aligned
// source frame into the fused image: a live preview is composited on the
// display canvas while dragging, then the worker applies the stroke to the
// 16-bit master and sends back the exact bbox, which replaces the preview.
const R = st.retouch;
function resetRetouch() { R.srcIndex = -1; R.srcCanvas = null; R.loading = -1; R.gen++; R.undo = 0; R.redo = 0; R.painting = false; R.dabs = []; }
let srcTimer = null;
function ensureSource() {
  if (st.view !== 'retouch' || !st.result || !st.files[st.selected]) return;
  if (R.srcIndex === st.selected || R.loading === st.selected) return;
  clearTimeout(srcTimer);
  srcTimer = setTimeout(() => {
    if (R.srcIndex === st.selected || R.loading === st.selected || st.running) return;
    R.loading = st.selected; R.gen++;
    worker.postMessage({ type: 'load_source', index: st.selected, file: st.files[st.selected], gen: R.gen });
    updateTabs();
  }, 250);
}
async function onSource(m) {
  if (R.loading === m.index) R.loading = -1;
  if (m.index !== st.selected || !st.result) { ensureSource(); return; } // stale: the user scrubbed on
  const cv = new OffscreenCanvas(m.w, m.h);
  cv.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h), 0, 0);
  R.srcCanvas = cv; R.srcIndex = m.index;
  updateTabs(); draw();
}
const targetCanvas = () => (R.target === 'dmap' && st.result && st.result.dmap) ? st.result.dmap : st.result && st.result.fused;
function onPatch(m) {
  R.undo = m.undo; R.redo = m.redo; updateTabs();
  if (!m.rgba || !st.result) return;
  const cv = m.target === 'dmap' ? st.result.dmap : st.result.fused; if (!cv) return;
  cv.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h), m.x, m.y);
  draw();
}
const dabCv = new OffscreenCanvas(16, 16);
function previewDab(x, y) {
  if (!R.srcCanvas || !st.result) return;
  const r = R.size, d = Math.ceil(2 * r) + 2;
  if (dabCv.width !== d) { dabCv.width = d; dabCv.height = d; }
  const c = dabCv.getContext('2d');
  c.globalCompositeOperation = 'source-over'; c.clearRect(0, 0, d, d);
  c.drawImage(R.srcCanvas, x - r, y - r, d, d, 0, 0, d, d);
  const g = c.createRadialGradient(r + 1, r + 1, r * R.hard, r + 1, r + 1, r);
  g.addColorStop(0, 'rgba(0,0,0,1)'); g.addColorStop(1, 'rgba(0,0,0,0)');
  c.globalCompositeOperation = 'destination-in'; c.fillStyle = g; c.fillRect(0, 0, d, d);
  targetCanvas().getContext('2d').drawImage(dabCv, x - r, y - r);
}
function addDab(x, y) {
  const step = Math.max(1, R.size / 3);
  if (R.last) {
    const dx = x - R.last[0], dy = y - R.last[1], dist = Math.hypot(dx, dy);
    if (dist < step) return;
    const n = Math.floor(dist / step);
    for (let i = 1; i <= n; i++) { const px = R.last[0] + dx * i / n, py = R.last[1] + dy * i / n; R.dabs.push(px, py, R.size, R.hard); previewDab(px, py); }
    R.last = [R.last[0] + dx * 1, R.last[1] + dy * 1];
  } else { R.dabs.push(x, y, R.size, R.hard); previewDab(x, y); R.last = [x, y]; }
}
function endStroke() {
  if (!R.painting) return;
  R.painting = false;
  if (R.dabs.length) worker.postMessage({ type: 'stroke', dabs: new Float32Array(R.dabs), target: R.target });
  R.dabs = []; R.last = null;
}
window.__retouchStroke = (pts) => { R.painting = true; R.dabs = []; R.last = null; for (const [x, y] of pts) addDab(x, y); endStroke(); };
// brush size slider is logarithmic: 0..100 -> 2..2000 px
const sizeFromSlider = (v) => Math.round(2 * Math.pow(1000, v / 100));
const sliderFromSize = (px) => Math.round(100 * Math.log(px / 2) / Math.log(1000));
function setBrush(size, hard) {
  R.size = Math.min(2000, Math.max(2, Math.round(size))); R.hard = Math.min(0.95, Math.max(0, Math.round(hard * 100) / 100));
  $('br-size').textContent = `${R.size} px`; $('br-hard').textContent = R.hard.toFixed(2);
  $('br-size-in').value = String(sliderFromSize(R.size)); $('br-hard-in').value = String(Math.round(R.hard * 100));
  saveParams(); draw();
}
$('br-size-in').addEventListener('input', (e) => setBrush(sizeFromSlider(Number(e.target.value)), R.hard));
$('br-hard-in').addEventListener('input', (e) => setBrush(R.size, Number(e.target.value) / 100));
$('undo').addEventListener('click', () => worker.postMessage({ type: 'undo' })); $('redo').addEventListener('click', () => worker.postMessage({ type: 'redo' }));
$('br-target').addEventListener('change', (e) => { R.target = e.target.value; updateTabs(); draw(); });

// ---------- viewer ----------
const canvas = $('view'), ctx = canvas.getContext('2d');
const canvas2 = $('view2'), ctx2 = canvas2.getContext('2d');
function imageDims() {
  if (st.result) return [st.result.w, st.result.h];
  const f = st.frames[st.selected]; if (f && f.w) return [f.w, f.h];
  const b = f && f.thumb; if (b) return [b.width, b.height];
  return [0, 0];
}
function fit() {
  const [w, h] = imageDims(); if (!w) return;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  st.zoom = Math.min(cw / w, ch / h); st.ox = (cw - w * st.zoom) / 2; st.oy = (ch - h * st.zoom) / 2; st.fitted = true; draw();
}
function zoom100() {
  const [w, h] = imageDims(); if (!w) return;
  const cw = canvas.clientWidth, ch = canvas.clientHeight; const z = 1 / dpr();
  const cx = (cw / 2 - st.ox) / st.zoom, cy = (ch / 2 - st.oy) / st.zoom; // keep the centre
  st.zoom = z; st.ox = cw / 2 - cx * z; st.oy = ch / 2 - cy * z; st.fitted = false; draw();
}
function layerFor(tab) {
  if (tab === 'fused') return st.result ? { bmp: st.result.fused, w: st.result.w, h: st.result.h } : null;
  if (tab === 'dmap') return st.result && st.result.dmap ? { bmp: st.result.dmap, w: st.result.w, h: st.result.h } : null;
  if (isDepthLayer(tab)) {
    if (!st.result) return null;
    const bmp = st.depthBmp.get(`${tab}:${st.turbo ? 'turbo' : 'gray'}`); if (!bmp) { depthBitmap(tab, st.turbo).then(draw); return null; }
    let overlay = null;
    if (st.slice && st.files.length) { overlay = st.sliceBmps.get(`${tab}:${st.selected}`) || null; if (!overlay) sliceBitmap(tab, st.selected).then(draw); }
    return { bmp, w: st.result.w, h: st.result.h, pixelated: true, overlay, overlayPixelated: true };
  }
  const f = st.frames[st.selected]; const bmp = f && (f.proxy || f.thumb); if (!bmp) return null;
  const [w, h] = f.w ? [f.w, f.h] : [bmp.width, bmp.height];
  let overlay = null;
  if (st.peak.on && f.peak) {
    if (f.peak.bmp && f.peak.bmpThr === st.peak.thr) overlay = f.peak.bmp; else peakBitmap(f).then(draw);
  }
  return { bmp, w, h, overlay };
}
function drawLayer(L, c = ctx) {
  if (!L) return;
  c.imageSmoothingEnabled = !L.pixelated && st.zoom * L.bmp.width / L.w < 1.0 ? true : !L.pixelated;
  if (L.pixelated) c.imageSmoothingEnabled = false;
  c.drawImage(L.bmp, 0, 0, L.w, L.h);
  if (L.overlay) { c.imageSmoothingEnabled = !L.overlayPixelated; c.drawImage(L.overlay, 0, 0, L.w, L.h); }
}
function sizeCanvas(cv, d) {
  const cw = cv.clientWidth, ch = cv.clientHeight;
  if (cv.width !== Math.round(cw * d) || cv.height !== Math.round(ch * d)) { cv.width = Math.round(cw * d); cv.height = Math.round(ch * d); }
}
function drawCursor(c, d) {
  if (!R.cursor) return;
  c.setTransform(st.zoom * d, 0, 0, st.zoom * d, st.ox * d, st.oy * d);
  c.lineWidth = 1.5 / (st.zoom * d); c.strokeStyle = 'rgba(255,255,255,.9)'; c.beginPath(); c.arc(R.cursor[0], R.cursor[1], R.size, 0, 2 * Math.PI); c.stroke();
  c.strokeStyle = 'rgba(0,0,0,.6)'; c.beginPath(); c.arc(R.cursor[0], R.cursor[1], R.size * R.hard, 0, 2 * Math.PI); c.stroke();
}
function draw() {
  const d = dpr();
  const retouch = st.view === 'retouch' && !!st.result;
  const split = retouch || (st.compare && st.cmpMode === 'split' && !!st.result);
  $('vwrap').classList.toggle('split', split); $('vwrap').classList.toggle('paint', retouch); canvas2.hidden = !split; $('panelabels').hidden = !split;
  sizeCanvas(canvas, d); if (split) sizeCanvas(canvas2, d);
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#141416'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const [w, h] = imageDims(); if (!w) { $('zoom').textContent = ''; return; }
  if (st.fitted) { st.zoom = Math.min(cw / w, ch / h); st.ox = (cw - w * st.zoom) / 2; st.oy = (ch - h * st.zoom) / 2; }
  ctx.setTransform(st.zoom * d, 0, 0, st.zoom * d, st.ox * d, st.oy * d);
  if (split) {
    // two panes, one transform: retouch = source | target result; compare = view | partner
    let L, Rt, labels;
    if (retouch) {
      const f = st.frames[st.selected]; const pbmp = f && (f.proxy || f.thumb);
      L = R.srcCanvas && R.srcIndex === st.selected ? { bmp: R.srcCanvas, w, h } : (pbmp ? { bmp: pbmp, w, h } : null);
      Rt = { bmp: targetCanvas(), w, h };
      labels = ['source', `${layerName(R.target)} — drag to paint, shift+drag pans`];
    } else {
      const A = layerFor(st.view), B = layerFor(st.cmp);
      [L, Rt] = st.flipped ? [B, A] : [A, B];
      labels = st.flipped ? [layerName(st.cmp), layerName(st.view)] : [layerName(st.view), layerName(st.cmp)];
    }
    drawLayer(L); if (retouch) drawCursor(ctx, d);
    ctx2.setTransform(1, 0, 0, 1, 0, 0); ctx2.fillStyle = '#141416'; ctx2.fillRect(0, 0, canvas2.width, canvas2.height);
    ctx2.setTransform(st.zoom * d, 0, 0, st.zoom * d, st.ox * d, st.oy * d);
    drawLayer(Rt, ctx2); if (retouch) drawCursor(ctx2, d);
    const [l1, l2] = $('panelabels').children; l1.textContent = labels[0]; l2.textContent = labels[1];
    $('divider').hidden = true; $('zoom').textContent = `${(st.zoom * d * 100).toFixed(0)}%`;
    return;
  }
  const A = layerFor(st.view);
  if (st.compare && st.result) {
    const B = layerFor(st.cmp);
    const [left, right] = st.flipped ? [B, A] : [A, B];
    const xs = (st.divider * cw - st.ox) / st.zoom; // divider in image px
    ctx.save(); ctx.beginPath(); ctx.rect(-1e6, -1e6, 1e6 + xs, 2e6); ctx.clip(); drawLayer(left); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(xs, -1e6, 1e6, 2e6); ctx.clip(); drawLayer(right); ctx.restore();
    $('divider').hidden = false; $('divider').style.left = `${st.divider * cw - 1}px`;
  } else { drawLayer(A); $('divider').hidden = true; }
  $('zoom').textContent = `${(st.zoom * d * 100).toFixed(0)}%`;
}
new ResizeObserver(() => draw()).observe($('vwrap'));
// One wheel rule for both panes: plain wheel scrubs whenever any visible layer
// depends on a frame (shift = 10 frames); ctrl/cmd+wheel always zooms at the
// cursor; plain wheel zooms only when nothing on screen is scrubbable.
function onWheel(cv, e) {
  e.preventDefault();
  if (st.view !== 'retouch' && scrubbable() && !(e.ctrlKey || e.metaKey) && st.files.length > 1) { scrub((e.deltaY > 0 ? 1 : -1) * (e.shiftKey ? 10 : 1)); return; }
  const f = Math.pow(1.0015, -e.deltaY); const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  st.ox = mx - (mx - st.ox) * f; st.oy = my - (my - st.oy) * f; st.zoom *= f; st.fitted = false; draw();
}
let drag = null;
function imgXY(cv, e) { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left - st.ox) / st.zoom, (e.clientY - r.top - st.oy) / st.zoom]; }
for (const cv of [canvas, canvas2]) {
  cv.addEventListener('pointerdown', (e) => {
    const paint = st.view === 'retouch' && st.result && e.button === 0 && !e.shiftKey && !(e.buttons & 4);
    try { cv.setPointerCapture(e.pointerId); } catch {}
    if (paint) {
      if (!R.srcCanvas || R.srcIndex !== st.selected) { toast('Source frame still loading — wait for "loaded" before painting.', 3000); return; }
      R.painting = true; R.dabs = []; R.last = null; const [x, y] = imgXY(cv, e); addDab(x, y); draw(); return;
    }
    drag = { x: e.clientX, y: e.clientY, ox: st.ox, oy: st.oy }; cv.classList.add('drag');
  });
  cv.addEventListener('pointermove', (e) => {
    if (st.view === 'retouch') { R.cursor = imgXY(cv, e); }
    if (R.painting) { const [x, y] = imgXY(cv, e); addDab(x, y); draw(); return; }
    if (drag) { st.ox = drag.ox + e.clientX - drag.x; st.oy = drag.oy + e.clientY - drag.y; st.fitted = false; }
    if (drag || st.view === 'retouch') draw();
  });
  cv.addEventListener('pointerup', () => { endStroke(); drag = null; cv.classList.remove('drag'); });
  cv.addEventListener('pointerleave', () => { if (st.view === 'retouch') { R.cursor = null; draw(); } });
  cv.addEventListener('dblclick', () => (st.fitted ? zoom100() : fit()));
  cv.addEventListener('wheel', (e) => onWheel(cv, e), { passive: false });
}
let ddrag = false;
$('divider').addEventListener('pointerdown', (e) => { ddrag = true; $('divider').setPointerCapture(e.pointerId); e.stopPropagation(); });
$('divider').addEventListener('pointermove', (e) => { if (!ddrag) return; const r = canvas.getBoundingClientRect(); st.divider = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); draw(); });
$('divider').addEventListener('pointerup', () => { ddrag = false; });
$('fit').addEventListener('click', fit); $('z100').addEventListener('click', zoom100);

// ---------- header: view / context / compare / scrub ----------
const LAYERS = [['fused', 'LAP'], ['dmap', 'DFR'], ['depth', 'Focus depth'], ['winner', 'Winner'], ['source', 'Source']];
// Header groups: Source | Stack (LAP, DFR) | Depth (Focus depth, Winner). The sub-control
// lists the group's layers and is hidden when the group has only one.
const GROUPS = { source: ['source'], stack: ['fused', 'dmap'], depth: ['depth', 'winner'] };
const groupOf = (v) => Object.keys(GROUPS).find((g) => GROUPS[g].includes(v)) || null;
const lastIn = { stack: 'fused', depth: 'depth' };   // last layer picked in each group
const layerName = (id) => (LAYERS.find((l) => l[0] === id) || [id, id])[1];
const haveDmap = () => !!(st.result && st.result.dmap);
function scrubbable() {
  const usesFrame = (t) => t === 'source' || t === 'retouch' || (isDepthLayer(t) && st.slice);
  return usesFrame(st.view) || (st.compare && usesFrame(st.cmp));
}
function updateTabs() {
  const have = !!st.result;
  $('tab-source').disabled = !st.files.length; $('tab-stack').disabled = !have; $('tab-depth').disabled = !have; $('ab').disabled = !have || st.step !== 'stack';
  if (!haveDmap() && st.view === 'dmap') st.view = 'fused';
  if (!haveDmap() && lastIn.stack === 'dmap') lastIn.stack = 'fused';
  if (!haveDmap()) R.target = 'fused'; $('br-target').value = R.target; $('br-target-sec').hidden = $('br-target-row').hidden = !haveDmap();
  $('sv-which-row').hidden = !haveDmap(); if (!haveDmap()) $('sv-which').value = 'fused';
  $('cm-swipe').classList.toggle('on', st.cmpMode !== 'split'); $('cm-split').classList.toggle('on', st.cmpMode === 'split');
  document.querySelectorAll('#steps button').forEach((b) => { if (b.dataset.step !== 'stack') b.disabled = !have; });
  const retouch = st.view === 'retouch' && have;
  $('viewseg').hidden = st.step !== 'stack';
  $('ctx-retouch').hidden = !retouch;
  $('undo').disabled = !R.undo; $('redo').disabled = !R.redo; $('hist').textContent = R.undo || R.redo ? `${R.undo} undo · ${R.redo} redo` : '';
  $('src-status').textContent = retouch ? (R.loading === st.selected ? `loading ${st.files[st.selected]?.name}…` : (R.srcIndex === st.selected ? `source: ${st.files[st.selected]?.name}` : '')) : '';
  $('ab').parentElement.hidden = st.step !== 'stack';
  if (retouch) { st.compare = false; ensureSource(); }
  const group = groupOf(st.view);
  document.querySelectorAll('#viewseg button').forEach((b) => b.classList.toggle('on', b.dataset.group === group));
  const subs = (GROUPS[group] || []).filter((t) => t !== 'dmap' || haveDmap());
  $('subseg').hidden = st.step !== 'stack' || subs.length < 2;
  document.querySelectorAll('#subseg button').forEach((b) => { b.hidden = !subs.includes(b.dataset.tab); b.classList.toggle('on', b.dataset.tab === st.view); });
  // compare partner: any layer but the current view
  const sel = $('cmp-sel'); sel.innerHTML = '';
  for (const [id, name] of LAYERS) { if (id === st.view) continue; if (id === 'source' && !st.files.length) continue; if (id === 'dmap' && !haveDmap()) continue; const o = document.createElement('option'); o.value = id; o.textContent = name; sel.appendChild(o); }
  if (![...sel.options].some((o) => o.value === st.cmp)) st.cmp = sel.options[0] ? sel.options[0].value : 'depth';
  sel.value = st.cmp;
  $('ab').checked = st.compare; $('ctx-compare').hidden = !(st.compare && have) || retouch; $('cmpbar').hidden = $('ctx-compare').hidden;
  const depthShown = isDepthLayer(st.view) || (st.compare && isDepthLayer(st.cmp));
  // put each context group next to the layer it acts on: the shown layer (left) or the compare partner (after "vs")
  const place = (el, onView, onPartner) => { const slot = (!onView && onPartner) ? $('cmp-ctx') : $('view-ctx'); if (el.parentElement !== slot) slot.appendChild(el); };
  place($('ctx-depth'), isDepthLayer(st.view), st.compare && isDepthLayer(st.cmp));
  place($('ctx-source'), st.view === 'source', st.compare && st.cmp === 'source');
  const usesFrame = (t) => t === 'source' || t === 'retouch' || (isDepthLayer(t) && st.slice);
  place($('scrub'), usesFrame(st.view), st.compare && usesFrame(st.cmp));
  $('ctx-depth').hidden = !depthShown; $('lut-gray').classList.toggle('on', !st.turbo); $('lut-turbo').classList.toggle('on', st.turbo); $('slice').checked = st.slice;
  const havePeaks = st.frames.some((f) => f && f.peak);
  const sourceShown = st.view === 'source' || (st.compare && st.cmp === 'source');
  if (retouch) $('ctx-depth').hidden = true;
  $('ctx-source').hidden = !(havePeaks && sourceShown);
  $('peak').checked = st.peak.on; $('peakthr').textContent = st.peak.thr.toFixed(2); $('peakstep').hidden = !st.peak.on;
  const scrubbing = st.files.length > 1 && scrubbable();
  $('scrub').hidden = !scrubbing; $('scrubber').max = String(Math.max(0, st.files.length - 1)); $('scrubber').value = String(st.selected);
  $('scrubname').textContent = st.files[st.selected] ? `${st.selected + 1}/${st.files.length}` : '';
  $('scrub').title = st.files[st.selected] ? st.files[st.selected].name : '';
}
function setView(v) { st.view = v; const g = groupOf(v); if (g && g !== 'source') lastIn[g] = v; updateTabs(); renderFilmstrip(); draw(); }
// ---------- workflow steps ----------
function gotoStep(step) {
  if (step !== 'stack' && !st.result) return;
  st.step = step;
  if (step === 'retouch') st.view = 'retouch';
  else if (step === 'save') { st.view = 'fused'; st.compare = false; }
  else if (st.view === 'retouch') st.view = 'fused';
  document.querySelectorAll('#steps button').forEach((b) => b.classList.toggle('on', b.dataset.step === step));
  $('params').hidden = step !== 'stack'; $('brushpanel').hidden = step !== 'retouch'; $('savepanel').hidden = step !== 'save';
  if (step === 'save' && st.result) {
    const strokes = R.undo;
    $('sv-info').textContent = `${st.result.w}×${st.result.h}, ${st.result.bits}-bit input, ${st.files.length} frames` + (strokes ? `, ${strokes} retouch stroke${strokes > 1 ? 's' : ''}` : '');
  }
  if (step === 'retouch') setBrush(R.size, R.hard);
  updateTabs(); renderFilmstrip(); draw();
}
document.querySelectorAll('#steps button').forEach((b) => b.addEventListener('click', () => gotoStep(b.dataset.step)));
document.querySelectorAll('#viewseg button').forEach((b) => b.addEventListener('click', () => { const g = b.dataset.group; setView(g === 'source' ? 'source' : lastIn[g]); }));
document.querySelectorAll('#subseg button').forEach((b) => b.addEventListener('click', () => setView(b.dataset.tab)));
$('cmp-sel').addEventListener('change', (e) => { st.cmp = e.target.value; updateTabs(); draw(); });
$('cm-swipe').addEventListener('click', () => { st.cmpMode = 'swipe'; saveParams(); updateTabs(); draw(); });
$('cm-split').addEventListener('click', () => { st.cmpMode = 'split'; saveParams(); updateTabs(); draw(); });
$('ab').addEventListener('change', (e) => { st.compare = e.target.checked; if (st.compare && st.view === 'source') st.view = 'fused'; updateTabs(); draw(); });
$('lut-gray').addEventListener('click', () => { st.turbo = false; saveParams(); updateTabs(); draw(); });
$('lut-turbo').addEventListener('click', () => { st.turbo = true; saveParams(); updateTabs(); draw(); });
const flip = (on) => { st.flipped = on; draw(); };
$('flip').addEventListener('pointerdown', () => flip(true)); $('flip').addEventListener('pointerup', () => flip(false)); $('flip').addEventListener('pointerleave', () => flip(false));
// Keep the selected thumb near the middle of the filmstrip while scrubbing
// (keys, wheel, slider), so frames above and below stay in view and the strip
// scrolls under the selection instead of the selection running off-screen.
// Clicking a thumb does not recentre: it is already on screen.
function revealSelected() {
  const el = $('filmstrip').children[st.selected];
  if (el && el.classList.contains('sel')) el.scrollIntoView({ block: 'center', behavior: 'instant' });
}
function scrub(delta) { if (!st.files.length) return; st.selected = Math.min(st.files.length - 1, Math.max(0, st.selected + delta)); updateTabs(); renderFilmstrip(); revealSelected(); draw(); }
// ResizeObserver: two panes change size when the split toggles
new ResizeObserver(() => draw()).observe(canvas2);
$('scrubber').addEventListener('input', (e) => { st.selected = Number(e.target.value); updateTabs(); renderFilmstrip(); revealSelected(); draw(); });
for (const id of ['scrub', 'filmstrip']) $(id).addEventListener('wheel', (e) => { if (id === 'filmstrip' && !e.shiftKey && !scrubbable()) return; e.preventDefault(); scrub((e.deltaY > 0 ? 1 : -1) * (e.shiftKey ? 10 : 1)); }, { passive: false });
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'range') return;
  if (e.target.tagName === 'SELECT') return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); worker.postMessage({ type: e.shiftKey ? 'redo' : 'undo' }); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); worker.postMessage({ type: 'redo' }); return; }
  if (e.key === '[') setBrush(R.size / 1.25, R.hard); else if (e.key === ']') setBrush(R.size * 1.25, R.hard);
  else if (e.key === '1') gotoStep('stack'); else if (e.key === '2') gotoStep('retouch'); else if (e.key === '3') gotoStep('save');
  else if (e.key === 'ArrowLeft') scrub(e.shiftKey ? -10 : -1); else if (e.key === 'ArrowRight') scrub(e.shiftKey ? 10 : 1);
  else if (e.key === 'f') fit(); else if (e.key === 'z' && !e.ctrlKey) zoom100();
  else if (e.key === ' ' && st.compare) { e.preventDefault(); flip(true); }
});
document.addEventListener('keyup', (e) => { if (e.key === ' ') flip(false); });

// ---------- autorun (smoke tests): ?autorun=test/frames[&align=0] ----------
const q = new URLSearchParams(location.search);
if (q.get('autorun')) {
  (async () => {
    const dir = q.get('autorun');
    const names = await (await fetch(`./${dir}/list.json`)).json();
    const files = [];
    for (const n of names) files.push(new File([await (await fetch(`./${dir}/` + n)).blob()], n));
    addFiles(files);
    if (q.has('align')) $('p-align').checked = q.get('align') === '1';
    if (q.has('render')) { $('p-dmap').checked = q.get('render') === '1'; runLabel(); }
    if (q.get('norun')) return;
    const tick = () => { if ($('status').textContent === 'ready') $('run').click(); else setTimeout(tick, 100); };
    tick();
  })();
}
updateTabs(); draw();

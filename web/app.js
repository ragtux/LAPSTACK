// lapstack browser UI. A focus-stacking workbench: filmstrip, parameter
// panel, run/cancel with progress + log, viewer layers (Source / Stack / Depth),
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
// keyboard shortcut card (bottom right of the canvas): collapses to its title row; the
// choice is remembered, open by default
const setKeysCollapsed = (on) => {
  $('keys').classList.toggle('collapsed', on);
  $('keys-arrow').textContent = on ? '▴' : '▾';
  $('keys-toggle').title = on ? 'show shortcuts (?)' : 'collapse shortcuts (?)';
  $('keys-toggle').setAttribute('aria-expanded', String(!on));
  try { localStorage.setItem('lapstack.keys', on ? '0' : '1'); } catch {}
};
const toggleKeys = () => setKeysCollapsed(!$('keys').classList.contains('collapsed'));
$('keys-toggle').onclick = toggleKeys;
try { if (localStorage.getItem('lapstack.keys') === '0') setKeysCollapsed(true); } catch {}
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
  depthBmp: new Map(),  // 'lut' -> ImageBitmap of the depth map (gray | turbo)
  step: 'stack', view: 'source', selected: 0,
  compare: false, cmp: 'depth', cmpMode: 'swipe', divider: 0.5, flipped: false,
  turbo: false, slice: true,
  sliceBmps: new Map(),   // 'slice:i' -> magenta band over pixels whose depth is frame i; 'focus:i' -> the In focus preview of frame i (proxy res)
  focusPending: new Set(), // frame indices whose In focus mask is being built
  peak: { on: false, strip: false, thr: 0.5, max: 0, pixmax: null, floor: 0 },   // focus peaking, see peakMask(); `on` is the canvas overlay, `strip` the filmstrip thumbs
  zoom: 1, ox: 0, oy: 0, fitted: true,
  pick: false,          // ctrl+G: the next canvas click jumps to the frame that won that pixel
  running: false,
  retouch: { size: 100, hard: 0.5, painting: false, dabs: [], last: null, cursor: null, target: 'fused',
             wasmIndex: -1, loading: -1, gen: 0, genMin: 0, undo: 0, redo: 0,
             gpuIndex: -1, prefetch: -1, ahead: null, dir: 1, lastSel: -1 },   // see ensureSource(): the frame the worker holds on the GPU, the one being prefetched, the read-ahead slot, the scrub direction
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
    turbo: st.turbo, slice: st.slice, peak_on: st.peak.on, peak_strip: st.peak.strip, peak_thr: st.peak.thr, cmp_mode: st.cmpMode,
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
  st.peak.on = p.peak_on ?? false; st.peak.strip = p.peak_strip ?? false; st.peak.thr = p.peak_thr ?? 0.5; st.slice = p.slice ?? true;
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
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { $('runmenu').hidden = true; setPick(false); } });
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
    case 'source-skipped': onSourceSkipped(m); break;
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
      const g = c.getContext('2d'); const r = [(160 - bmp.width * s) / 2, (100 - bmp.height * s) / 2, bmp.width * s, bmp.height * s];
      const peaking = st.peak.strip && fr.peak;   // with preview peaking on, the thumb is the dimmed frame under its magenta in-focus band
      if (peaking) g.filter = 'grayscale(1) brightness(0.6)';
      g.drawImage(bmp, ...r);
      if (peaking) { g.filter = 'none'; g.drawImage(peakThumb(fr, Math.round(r[2]), Math.round(r[3])), r[0], r[1]); }
      d.appendChild(c);
    } else { const ph = document.createElement('div'); ph.className = 'ph'; ph.textContent = fr ? '…' : String(i); d.appendChild(ph); }
    const n = document.createElement('div'); n.className = 'name'; n.textContent = f.name; d.appendChild(n);
    if (fr && fr.sim) { const s = document.createElement('div'); s.className = 'sim'; s.textContent = `${fr.sim[0].toFixed(1)}, ${fr.sim[1].toFixed(1)} px · ×${fr.sim[2].toFixed(4)} · ${fr.sim[3].toFixed(2)}°`; d.appendChild(s); }
    if (st.peak.strip && fr && fr.peak) { const s = document.createElement('div'); s.className = 'sim pct'; s.textContent = `${peakPercent(fr).toFixed(1)} % in focus`; d.appendChild(s); }
    d.addEventListener('click', () => { st.selected = i; if (!scrubbable()) st.view = 'source'; updateTabs(); renderFilmstrip(); draw(); });
    fs.appendChild(d);
  });
}
$('add').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
$('clear').addEventListener('click', () => { if (st.running) return; setPick(false); worker.postMessage({ type: 'clear' }); st.files = []; st.frames = []; st.result = null; st.depthBmp.clear(); st.sliceBmps.clear(); st.peak.pixmax = null; resetRetouch(); st.step = 'stack'; if (st.view === 'retouch') st.view = 'source'; gotoStep('stack'); renderFilmstrip(); updateTabs(); setView('source'); });
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
  st.running = true; setPick(false); st.frames = st.frames.map((f) => (f ? { name: f.name, thumb: f.thumb, proxy: f.proxy, w: f.w, h: f.h, bits: f.bits } : f)); st.result = null; st.depthBmp.clear(); st.sliceBmps.clear(); st.peak.pixmax = null; resetRetouch(); if (st.step !== 'stack') gotoStep('stack');
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
// the depth layer: depth from focus (DFF) on its working grid, a fractional frame index
const isDepthLayer = (t) => t === 'depth';
function depthData() {
  const r = st.result; if (!r) return null;
  return { data: r.depth, w: r.dw, h: r.dh };
}
async function depthBitmap(useTurbo) {
  const D = depthData(); if (!D) return null;
  const key = useTurbo ? 'turbo' : 'gray';
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
  for (const f of frs) { f.peak.pct = null; f.peak.bmp = null; f.peak.tc = null; }
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
// The filmstrip's peaking band: the mask box-averaged to thumb size, so the
// alpha is the in-focus fraction of each thumb pixel (no 5 MB bitmap per frame).
function peakThumb(fr, tw, th) {
  const { p, pm, thr, floor } = peakMask(fr);
  if (p.tc && p.tcThr === thr && p.tc.width === tw && p.tc.height === th) return p.tc;
  const t2 = thr * thr; const cnt = new Uint16Array(tw * th), tot = new Uint16Array(tw * th);
  for (let y = 0, i = 0; y < p.h; y++) {
    const row = Math.floor(y * th / p.h) * tw;
    for (let x = 0; x < p.w; x++, i++) { const j = row + Math.floor(x * tw / p.w); tot[j]++; if (pm[i] > floor && p.data[i] >= t2 * pm[i]) cnt[j]++; }
  }
  const px = new Uint8ClampedArray(tw * th * 4);
  for (let j = 0; j < tw * th; j++) { px[4 * j] = 255; px[4 * j + 2] = 255; px[4 * j + 3] = tot[j] ? Math.round(230 * cnt[j] / tot[j]) : 0; }
  const c = new OffscreenCanvas(tw, th); c.getContext('2d').putImageData(new ImageData(px, tw, th), 0, 0);
  p.tc = c; p.tcThr = thr; return c;
}
function setPeakThr(v) { st.peak.thr = Math.min(1, Math.max(0.05, v)); $('peakthr').textContent = st.peak.thr.toFixed(2); saveParams(); renderFilmstrip(); draw(); }
$('peak').addEventListener('change', (e) => { st.peak.on = e.target.checked; saveParams(); updateTabs(); draw(); });
$('peak-strip').addEventListener('change', (e) => { st.peak.strip = e.target.checked; saveParams(); renderFilmstrip(); });

// ---------- depth slice ----------
// Magenta band over the pixels whose depth index is the scrubbed frame (the depth "slice").
async function sliceBitmap(ix) {
  const D = depthData(); if (!D) return null;
  const key = `slice:${ix}`;
  if (st.sliceBmps.has(key)) return st.sliceBmps.get(key);
  const px = new Uint8ClampedArray(D.w * D.h * 4);
  for (let i = 0; i < D.data.length; i++) if (Math.round(D.data[i]) === ix) { px[4 * i] = 255; px[4 * i + 1] = 0; px[4 * i + 2] = 255; px[4 * i + 3] = 150; }
  const bmp = await createImageBitmap(new ImageData(px, D.w, D.h));
  if (st.sliceBmps.size > 32) st.sliceBmps.clear();
  st.sliceBmps.set(key, bmp); return bmp;
}
$('slice').addEventListener('change', (e) => { st.slice = e.target.checked; saveParams(); updateTabs(); draw(); });

// ---------- in focus ----------
// The scrubbed frame's real pixels, showing only the parts of it the result
// uses (Helicon's "source map"): each pixel is weighted by how far, in frames,
// the depth map puts it from the frame — w = 1 within ±FOCUS_W0 frames, 0
// beyond ±FOCUS_W1, linear between — so the plane of focus stands out with a
// crisp edge and scrubbing sweeps it through the scene. The out-of-focus part
// is dimmed to FOCUS_DIM but keeps its outlines: the local contrast of its
// luminance (|luma − box blur|, radius ≈ width/1000) is added back in grey,
// scaled by FOCUS_TEX and soft-clipped at FOCUS_CAP so hard edges stay light
// grey, so blurred edges and fibres read as light lines.
//   out = rgb·(w + (1 − w)·dim) + (1 − w)·soft(tex·|luma − box(luma)|),  soft(t) = cap·(1 − e^(−t/cap))
// Two renderings share the formula: a preview built here from the proxy and
// the depth map's working grid, and the full-resolution frame rendered by the
// engine from the guided-upsampled depth map (source_focus), which replaces
// the preview once it has loaded (srcCache, key 'focus:i').
const FOCUS_DIM = 0.08;   // brightness left to a pixel the frame does not contribute
const FOCUS_W0 = 0.5;     // frames: fully lit within this distance of the plane
const FOCUS_W1 = 0.75;    // frames: fully dimmed beyond this distance (a quarter-frame feather)
const FOCUS_TEX = 3;      // gain of the out-of-focus local contrast
const FOCUS_CAP = 0.4;    // brightness the local contrast term saturates towards
const focusParams = () => ({ dim: FOCUS_DIM, w0: FOCUS_W0, w1: FOCUS_W1, tex: FOCUS_TEX });
const focusWeight = (d, ix) => Math.min(1, Math.max(0, (FOCUS_W1 - Math.abs(d - ix)) / (FOCUS_W1 - FOCUS_W0)));
// |luma − box blur| of an RGBA image, radius r, as a Float32Array in [0, 1]
function localContrast(px, w, h, r) {
  const luma = new Float32Array(w * h), hb = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) luma[i] = (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
  for (let y = 0; y < h; y++) {
    let sum = 0; for (let x = 0; x <= Math.min(r, w - 1); x++) sum += luma[y * w + x];
    for (let x = 0; x < w; x++) {
      const lo = Math.max(0, x - r), hi = Math.min(w - 1, x + r);
      hb[y * w + x] = sum / (hi - lo + 1);
      if (x + r + 1 < w) sum += luma[y * w + x + r + 1];
      if (x >= r) sum -= luma[y * w + x - r];
    }
  }
  const col = new Float32Array(w);
  for (let y = 0; y <= Math.min(r, h - 1); y++) for (let x = 0; x < w; x++) col[x] += hb[y * w + x];
  for (let y = 0; y < h; y++) {
    const lo = Math.max(0, y - r), hi = Math.min(h - 1, y + r), cnt = hi - lo + 1;
    for (let x = 0; x < w; x++) out[y * w + x] = Math.abs(luma[y * w + x] - col[x] / cnt);
    if (y + r + 1 < h) for (let x = 0; x < w; x++) col[x] += hb[(y + r + 1) * w + x];
    if (y >= r) for (let x = 0; x < w; x++) col[x] -= hb[(y - r) * w + x];
  }
  return out;
}
function sampleBilinear(D, x, y) {
  const x0 = Math.max(0, Math.min(D.w - 1, Math.floor(x))), y0 = Math.max(0, Math.min(D.h - 1, Math.floor(y)));
  const x1 = Math.min(D.w - 1, x0 + 1), y1 = Math.min(D.h - 1, y0 + 1);
  const fx = Math.max(0, Math.min(1, x - x0)), fy = Math.max(0, Math.min(1, y - y0));
  const a = D.data[y0 * D.w + x0], b = D.data[y0 * D.w + x1], c = D.data[y1 * D.w + x0], d = D.data[y1 * D.w + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}
// the In focus preview of frame ix at proxy resolution (an image, not an overlay)
async function focusBitmap(ix) {
  const D = depthData(); const f = st.frames[ix]; const src = f && (f.proxy || f.thumb);
  if (!D || !src) return null;
  const key = `focus:${ix}`;
  if (st.sliceBmps.has(key)) return st.sliceBmps.get(key);
  if (st.focusPending.has(ix)) return null;
  st.focusPending.add(ix);
  let bmp;
  try {
    const w = src.width, h = src.height;
    const oc = new OffscreenCanvas(w, h); const c = oc.getContext('2d', { willReadFrequently: true });
    c.drawImage(src, 0, 0);
    const img = c.getImageData(0, 0, w, h), px = img.data;
    const hp = localContrast(px, w, h, Math.max(1, Math.round(w / 1000)));
    for (let y = 0; y < h; y++) {
      const dy = (y + 0.5) * D.h / h - 0.5;
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const wgt = focusWeight(sampleBilinear(D, (x + 0.5) * D.w / w - 0.5, dy), ix);
        const gain = wgt + (1 - wgt) * FOCUS_DIM, add = (1 - wgt) * FOCUS_CAP * (1 - Math.exp(-FOCUS_TEX * hp[i] / FOCUS_CAP)) * 255;
        px[4 * i] = px[4 * i] * gain + add; px[4 * i + 1] = px[4 * i + 1] * gain + add; px[4 * i + 2] = px[4 * i + 2] * gain + add;
      }
    }
    bmp = await createImageBitmap(img);
  } finally { st.focusPending.delete(ix); }
  if (st.sliceBmps.size > 32) st.sliceBmps.clear();
  st.sliceBmps.set(key, bmp); return bmp;
}
$('peak-minus').addEventListener('click', () => setPeakThr(st.peak.thr - 0.05));
$('peak-plus').addEventListener('click', () => setPeakThr(st.peak.thr + 0.05));

// ---------- retouch ----------
// Two panes (source | fused) with one transform. The brush copies the aligned
// source frame into the fused image: a live preview is composited on the
// display canvas while dragging, then the worker applies the stroke to the
// 16-bit master and sends back the exact bbox, which replaces the preview.
const R = st.retouch;
// Full-res sources: an LRU of decoded, run-aligned frames. Each costs 4 bytes/px
// (a 45 MP frame is 179 MB), so the cache is sized in bytes, not frames: ~10
// frames at 12 MP, 2 at 45 MP, 1 above that — never fewer than the one on screen.
let SRC_BUDGET = 512 * 1024 * 1024;
const srcCache = new Map(); // frame index (plain frame) | 'focus:i' (In focus rendering) -> OffscreenCanvas, least recently used first
const srcLimit = () => { const [w, h] = imageDims(); return w ? Math.max(1, Math.floor(SRC_BUDGET / (w * h * 4))) : 1; };
function srcGet(i) { const cv = srcCache.get(i); if (!cv) return null; srcCache.delete(i); srcCache.set(i, cv); return cv; } // touch
function srcPut(i, cv) {
  srcCache.delete(i); srcCache.set(i, cv);
  for (const k of [...srcCache.keys()]) {
    if (srcCache.size <= srcLimit()) break;
    if (k === i || k === st.selected || k === `focus:${st.selected}`) continue;   // never drop what is on screen
    srcCache.get(k).width = 1; srcCache.delete(k);   // 1x1 releases the backing store now, not at the next GC
  }
}
function srcClear() { for (const cv of srcCache.values()) cv.width = 1; srcCache.clear(); }
window.__srcCache = srcCache; window.__srcBudget = (b) => { SRC_BUDGET = b; };
function resetRetouch() { srcClear(); R.wasmIndex = -1; R.gpuIndex = -1; R.loading = -1; R.prefetch = -1; R.ahead = null; R.gen++; R.genMin = R.gen; R.undo = 0; R.redo = 0; R.painting = false; R.dabs = []; }
let srcTimer = null;
// The Source and In focus layers are drawn from the proxy (proxy_edge px long
// side) until the full-res aligned frame arrives, so a run's result is never
// compared against a soft image. Retouch needs the same frame to paint from.
// layers drawn from the scrubbed frame's pixels: Source, and In focus (the frame under its dimming mask)
const usesSource = (t) => t === 'source' || t === 'focus';
const shown = (t) => st.view === t || (st.compare && st.cmp === t);
const sourceShown = () => shown('source') || shown('focus');
// Frame bytes are read here, on the main thread, not in the worker: a File read
// only progresses while its thread's event loop is free, so a read issued in the
// worker waits behind the decode running there, while one issued here overlaps
// it. One read runs ahead, for the frame the scrub direction predicts next; the
// worker needs no bytes for the frame it still holds warped on the GPU.
function readBytes(i) {
  const f = st.files[i];
  if (R.ahead && R.ahead.file === f) { const p = R.ahead.p; R.ahead = null; return p; }
  return f.arrayBuffer();
}
function readAhead(i) {
  const f = st.files[i];
  if (!f || (R.ahead && R.ahead.file === f)) return;
  R.ahead = { file: f, p: f.arrayBuffer() };
}
// ask the worker for frame `index`: `focus` = the In focus rendering (focusParams()) or null for the plain frame
function requestSource(index, focus, prefetch) {
  const gen = ++R.gen, file = st.files[index];
  const post = (bytes) => {
    if (!st.result || gen < R.genMin || st.files[index] !== file) return;   // cleared or re-run while the bytes were read
    worker.postMessage({ type: 'load_source', index, file, bytes, gen, focus, prefetch }, bytes ? [bytes] : []);
  };
  if (R.gpuIndex === index) post(null); else readBytes(index).then(post, () => post(null));   // on a read error the worker reads the File itself
  if (srcLimit() >= 2) readAhead(index + R.dir);
}
function ensureSource() {
  if (!st.result || !st.files[st.selected]) return;
  if (st.selected !== R.lastSel) { R.dir = st.selected < R.lastSel ? -1 : 1; R.lastSel = st.selected; }
  const forPaint = st.view === 'retouch';                      // strokes copy from the worker's own 16-bit copy
  // the plain frame first (Source, retouch), then the In focus rendering; one request at a time
  const needPlain = () => (forPaint && R.wasmIndex !== st.selected) || (shown('source') && !srcCache.has(st.selected));
  const needFocus = () => shown('focus') && !srcCache.has(`focus:${st.selected}`);
  if (needPlain() || needFocus()) {
    if (R.loading === st.selected) return;
    clearTimeout(srcTimer);
    srcTimer = setTimeout(() => {
      if (!(needPlain() || needFocus()) || R.loading === st.selected || st.running) return;
      R.loading = st.selected;
      requestSource(st.selected, needPlain() ? null : focusParams(), false);
      updateTabs();
    }, 250);
    return;
  }
  // This frame is on screen at full res and the worker is idle: warm the cache with
  // the frame the scrub direction predicts next (one ahead; the cache keeps at least
  // two frames for this to help). A real request supersedes a queued prefetch.
  if (forPaint || R.loading >= 0 || R.prefetch >= 0 || st.running || srcLimit() < 2 || !sourceShown()) return;
  const n = st.selected + R.dir; if (!st.files[n]) return;
  const focus = !shown('source');   // Source shown (alone or beside In focus): the plain frame, as needPlain() will ask first
  if (srcCache.has(focus ? `focus:${n}` : n)) return;
  R.prefetch = n; requestSource(n, focus ? focusParams() : null, true);
}
async function onSource(m) {
  if (R.loading === m.index) R.loading = -1;
  if (R.prefetch === m.index) R.prefetch = -1;
  if (!st.result || m.gen < R.genMin) return;   // aligned against a run that is gone
  R.gpuIndex = m.index;         // the worker has this frame warped on the GPU now
  const cv = new OffscreenCanvas(m.w, m.h);
  cv.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h), 0, 0);
  srcPut(m.focus ? `focus:${m.index}` : m.index, cv);   // keep it even if the user has scrubbed on: that is what the cache is for
  if (srcCache.size === 1) log(`[lapstack] full-res source cache: up to ${srcLimit()} frame(s), ${(m.w * m.h * 4 / 1e6).toFixed(0)} MB each`);
  if (!m.focus) R.wasmIndex = m.index;   // the worker's 16-bit copy is this frame now (In focus is rendered on the GPU without one)
  ensureSource();               // the other rendering of this frame, or the frame scrubbed to meanwhile
  updateTabs(); draw();
}
// the worker dropped a request that a newer one had superseded; ask again for what is on screen now
function onSourceSkipped(m) { if (R.loading === m.index) R.loading = -1; if (R.prefetch === m.index) R.prefetch = -1; ensureSource(); updateTabs(); }
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
  const src = srcGet(st.selected); if (!src || !st.result) return;
  const r = R.size, d = Math.ceil(2 * r) + 2;
  if (dabCv.width !== d) { dabCv.width = d; dabCv.height = d; }
  const c = dabCv.getContext('2d');
  c.globalCompositeOperation = 'source-over'; c.clearRect(0, 0, d, d);
  c.drawImage(src, x - r, y - r, d, d, 0, 0, d, d);
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
    const bmp = st.depthBmp.get(st.turbo ? 'turbo' : 'gray'); if (!bmp) { depthBitmap(st.turbo).then(draw); return null; }
    let overlay = null;
    if (st.slice && st.files.length) { overlay = st.sliceBmps.get(`slice:${st.selected}`) || null; if (!overlay) sliceBitmap(st.selected).then(draw); }
    return { bmp, w: st.result.w, h: st.result.h, pixelated: true, overlay, overlayPixelated: true };
  }
  const f = st.frames[st.selected]; const bmp = f && (f.proxy || f.thumb); if (!bmp) return null;
  const [w, h] = f.w ? [f.w, f.h] : [bmp.width, bmp.height];
  const full = srcGet(st.selected); // full res, aligned like the run
  let overlay = null;
  if (tab === 'focus') {
    if (!st.result) return null;
    const fullFocus = srcGet(`focus:${st.selected}`);   // full res, rendered by the engine
    if (fullFocus) return { bmp: fullFocus, w, h };
    const preview = st.sliceBmps.get(`focus:${st.selected}`); if (!preview) { focusBitmap(st.selected).then(draw); return { bmp, w, h }; }
    return { bmp: preview, w, h };
  }
  if (st.peak.on && f.peak) {
    if (f.peak.bmp && f.peak.bmpThr === st.peak.thr) overlay = f.peak.bmp; else peakBitmap(f).then(draw);
  }
  return { bmp: full || bmp, w, h, overlay };
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
// Pane labels: a chip over every visible image. 'a' is the view layer, 'b' the
// compare partner, plain = neutral (the retouch source). Positions are inline so
// one element serves the pane centres, the swipe divider and the centred single view.
function showLabel(el, text, kind, pos) {
  el.hidden = !text;
  if (!text) return;
  el.textContent = text;
  el.className = 'plab' + (kind ? ' ' + kind : '');
  el.style.left = pos.left || 'auto';
  el.style.right = pos.right || 'auto';
  el.style.transform = pos.transform || 'none';
}
function draw() {
  const d = dpr();
  const retouch = st.view === 'retouch' && !!st.result;
  const split = retouch || (st.compare && st.cmpMode === 'split' && !!st.result);
  $('vwrap').classList.toggle('split', split); $('vwrap').classList.toggle('paint', retouch); canvas2.hidden = !split;
  sizeCanvas(canvas, d); if (split) sizeCanvas(canvas2, d);
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#141416'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const [w, h] = imageDims(); if (!w) { $('zoom').textContent = ''; $('panelabels').hidden = true; return; }
  if (st.fitted) { st.zoom = Math.min(cw / w, ch / h); st.ox = (cw - w * st.zoom) / 2; st.oy = (ch - h * st.zoom) / 2; }
  ctx.setTransform(st.zoom * d, 0, 0, st.zoom * d, st.ox * d, st.oy * d);
  // labels ride the top of the image itself, so they stay on it when it is letterboxed,
  // but never under the vs toggle / compare row floating at the top of the canvas
  const vs = $('ab').parentElement, vr = $('vh-right');
  const hdr = $('vhead').offsetTop + Math.max(vs.offsetTop + vs.offsetHeight, vr.offsetTop + vr.offsetHeight);
  $('panelabels').style.top = `${Math.max(hdr, Math.min(st.oy, ch - 40))}px`;
  if (split) {
    // two panes, one transform: retouch = source | target result; compare = view | partner
    let L, Rt, labels;
    if (retouch) {
      const f = st.frames[st.selected]; const pbmp = f && (f.proxy || f.thumb);
      const fullSrc = srcGet(st.selected);
      L = fullSrc ? { bmp: fullSrc, w, h } : (pbmp ? { bmp: pbmp, w, h } : null);
      Rt = { bmp: targetCanvas(), w, h };
      labels = ['Source', `${layerName(R.target)} — drag to paint, shift+drag pans`];
    } else {
      const A = layerFor(st.view), B = layerFor(st.cmp);
      [L, Rt] = st.flipped ? [B, A] : [A, B];
      labels = st.flipped ? [layerLabel(st.cmp), layerLabel(st.view)] : [layerLabel(st.view), layerLabel(st.cmp)];
    }
    drawLayer(L); if (retouch) drawCursor(ctx, d);
    ctx2.setTransform(1, 0, 0, 1, 0, 0); ctx2.fillStyle = '#141416'; ctx2.fillRect(0, 0, canvas2.width, canvas2.height);
    ctx2.setTransform(st.zoom * d, 0, 0, st.zoom * d, st.ox * d, st.oy * d);
    drawLayer(Rt, ctx2); if (retouch) drawCursor(ctx2, d);
    const [k1, k2] = retouch ? ['', 'a'] : (st.flipped ? ['b', 'a'] : ['a', 'b']);
    const [l1, l2] = $('panelabels').children; $('panelabels').hidden = false;
    showLabel(l1, labels[0], k1, { left: '25%', transform: 'translateX(-50%)' });
    showLabel(l2, labels[1], k2, { left: '75%', transform: 'translateX(-50%)' });
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
    // the two chips ride the divider, one on each side; each drops out as its side closes
    const x = st.divider * cw;
    const [n1, n2] = st.flipped ? [layerLabel(st.cmp), layerLabel(st.view)] : [layerLabel(st.view), layerLabel(st.cmp)];
    const [k1, k2] = st.flipped ? ['b', 'a'] : ['a', 'b'];
    const [l1, l2] = $('panelabels').children; $('panelabels').hidden = false;
    showLabel(l1, x > 40 ? n1 : '', k1, { right: `${Math.round(cw - x + 8)}px` });
    showLabel(l2, cw - x > 40 ? n2 : '', k2, { left: `${Math.round(x + 8)}px` });
  } else {
    drawLayer(A); $('divider').hidden = true;
    const [l1, l2] = $('panelabels').children; $('panelabels').hidden = false;
    showLabel(l1, layerLabel(st.view), 'a', { left: '50%', transform: 'translateX(-50%)' });
    showLabel(l2, '', '', {});
  }
  $('zoom').textContent = `${(st.zoom * d * 100).toFixed(0)}%`;
}
new ResizeObserver(() => { layoutScrub(); draw(); }).observe($('vwrap'));
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
// ---------- ctrl+G: jump to the frame that won a pixel ----------
// The LAP winner map holds, per cell of the depth level's grid, the frame that
// won there (the map "Save winner map" writes). A single cell is noisy, so a
// click takes the most common index in the 3x3 around it, with the clicked cell
// breaking ties.
function setPick(on) {
  st.pick = !!on && !!st.result;
  $('vwrap').classList.toggle('pick', st.pick);
  $('pickhint').hidden = !st.pick;
}
function frameAt(x, y) {
  const r = st.result; if (!r || !r.winner) return -1;
  const cx = Math.round((x + 0.5) * r.ww / r.w - 0.5), cy = Math.round((y + 0.5) * r.wh / r.h - 0.5);
  if (cx < 0 || cy < 0 || cx >= r.ww || cy >= r.wh) return -1;
  const votes = new Map();
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const px = cx + dx, py = cy + dy; if (px < 0 || py < 0 || px >= r.ww || py >= r.wh) continue;
    const v = Math.round(r.winner[py * r.ww + px]);
    votes.set(v, (votes.get(v) || 0) + (dx || dy ? 1 : 1.5));
  }
  let best = -1, n = 0;
  for (const [v, c] of votes) if (c > n) { best = v; n = c; }
  return best;
}
function pickAt(cv, e) {
  const [x, y] = imgXY(cv, e), [w, h] = imageDims();
  if (x < 0 || y < 0 || x >= w || y >= h) return;   // outside the image: wait for a click on it
  const i = frameAt(x, y);
  setPick(false);
  if (i < 0 || !st.files[i]) { toast('No winner recorded for that pixel.', 3000); return; }
  st.selected = i;
  if (!scrubbable()) st.view = 'source';            // otherwise the jump would be invisible
  updateTabs(); renderFilmstrip(); revealSelected(); draw();
  log(`[lapstack] (${Math.round(x)}, ${Math.round(y)}) is sharpest in frame ${i + 1}/${st.files.length}: ${st.files[i].name}`);
}
for (const cv of [canvas, canvas2]) {
  cv.addEventListener('pointerdown', (e) => {
    if (st.pick && e.button === 0) { e.preventDefault(); pickAt(cv, e); return; }
    const paint = st.view === 'retouch' && st.result && e.button === 0 && !e.shiftKey && !(e.buttons & 4);
    try { cv.setPointerCapture(e.pointerId); } catch {}
    if (paint) {
      if (R.wasmIndex !== st.selected || !srcGet(st.selected)) { toast('Source frame still loading — wait for "loaded" before painting.', 3000); return; }
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
const LAYERS = [['fused', 'LAP'], ['dmap', 'DFR'], ['depth', 'Focus depth'], ['focus', 'In focus'], ['source', 'Source']];
// Header groups: Source | Stack (LAP, DFR) | Depth (Focus depth, In focus). The sub-control
// lists the group's layers and is hidden when the group has only one.
const GROUPS = { source: ['source'], stack: ['fused', 'dmap'], depth: ['depth', 'focus'] };
const groupOf = (v) => Object.keys(GROUPS).find((g) => GROUPS[g].includes(v)) || null;
const lastIn = { stack: 'fused', depth: 'depth' };   // last layer picked in each group
const layerName = (id) => (LAYERS.find((l) => l[0] === id) || [id, id])[1];
// while the full-res frame decodes, say so: the pane is showing the proxy
const layerLabel = (id) => (usesSource(id) && st.result && !srcCache.has(id === 'focus' ? `focus:${st.selected}` : st.selected))
  ? `${layerName(id)} — loading full res…` : layerName(id);
const haveDmap = () => !!(st.result && st.result.dmap);
// layers that depend on the scrubbed frame
const usesFrame = (t) => usesSource(t) || t === 'retouch' || (isDepthLayer(t) && st.slice);
function scrubbable() { return usesFrame(st.view) || (st.compare && usesFrame(st.cmp)); }
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
  $('src-status').textContent = retouch ? (R.loading === st.selected ? `loading ${st.files[st.selected]?.name}…` : (R.wasmIndex === st.selected ? `source: ${st.files[st.selected]?.name}` : '')) : '';
  $('ab').parentElement.hidden = st.step !== 'stack';
  if (retouch) st.compare = false;
  ensureSource();
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
  $('ctx-depth').hidden = !depthShown; $('lut-gray').classList.toggle('on', !st.turbo); $('lut-turbo').classList.toggle('on', st.turbo); $('slice').checked = st.slice;
  const havePeaks = st.frames.some((f) => f && f.peak);
  const peakShown = st.view === 'source' || (st.compare && st.cmp === 'source');   // peaking is a Source overlay
  if (retouch) $('ctx-depth').hidden = true;
  $('ctx-source').hidden = !(havePeaks && peakShown);
  $('peak').checked = st.peak.on; $('peakthr').textContent = st.peak.thr.toFixed(2); $('peakstep').hidden = !(st.peak.on || st.peak.strip);
  $('peak-strip').checked = st.peak.strip; $('peak-strip').disabled = !havePeaks;
  // shortcut card: rows for a result / retouch / compare appear once they apply
  const when = { result: have, retouch, compare: st.compare && have && !retouch };
  document.querySelectorAll('#keys-list [data-when]').forEach((r) => { r.hidden = !when[r.dataset.when]; });
  const scrubbing = st.files.length > 1 && scrubbable();
  $('scrub').hidden = !scrubbing; $('scrubber').max = String(Math.max(0, st.files.length - 1)); $('scrubber').value = String(st.selected);
  $('scrubname').textContent = st.files[st.selected] ? `${st.selected + 1}/${st.files.length}` : '';
  $('scrub').title = st.files[st.selected] ? st.files[st.selected].name : '';
  layoutScrub();
}
// The scrubber wants to sit centred on the left edge at min(50%, 420px) tall. The chip
// column floats above it on the same edge, so when the centred track would run into the
// chips, shorten it (down to 160px) to keep the centre; if even that overlaps, centre it
// in the free band between the chips and the zoom bar instead.
function layoutScrub() {
  const sc = $('scrub'); if (sc.hidden) return;
  const ch = $('vwrap').clientHeight, gap = 10;
  const colBottom = $('vh-left').offsetHeight ? $('vhead').offsetTop + $('vh-left').offsetHeight + gap : 0;
  const free = ch - 60;                                          // above the zoom bar
  const full = Math.min(ch * .5, 420);
  let top, h = Math.min(full, 2 * (ch / 2 - colBottom), 2 * (free - ch / 2));
  if (h >= Math.min(full, 160)) { top = ch / 2 - h / 2; }
  else { h = Math.max(120, Math.min(full, free - colBottom)); top = colBottom + (free - colBottom - h) / 2; }
  sc.style.top = `${Math.round(top)}px`; sc.style.height = `${Math.round(h)}px`; sc.style.transform = 'none';
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
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'g') {
    e.preventDefault();
    if (!st.result) { toast('Run first: the sharpest frame comes from the run\'s winner map.', 3000); return; }
    setPick(!st.pick);
    return;
  }
  if (e.key === '[') setBrush(R.size / 1.25, R.hard); else if (e.key === ']') setBrush(R.size * 1.25, R.hard);
  else if (e.key === '1') gotoStep('stack'); else if (e.key === '2') gotoStep('retouch'); else if (e.key === '3') gotoStep('save');
  else if (e.key === 'ArrowLeft') scrub(e.shiftKey ? -10 : -1); else if (e.key === 'ArrowRight') scrub(e.shiftKey ? 10 : 1);
  else if (e.key === 'f') fit(); else if (e.key === 'z' && !e.ctrlKey) zoom100();
  else if (e.key === ' ' && st.compare) { e.preventDefault(); flip(true); }
  else if (e.key === '?') toggleKeys();
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

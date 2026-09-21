// lapstack browser UI. A focus-stacking workbench: filmstrip, parameter
// panel, run/cancel with progress + log, viewer layers (Source / Stack / Depth),
// tiled-free zoom/pan on a canvas, A/B compare with a draggable divider and
// hold-to-flip, depth map gray/Turbo, frame scrubbing on the Source view.
// All heavy work happens in worker.js (WASM + WebGPU).

const $ = (id) => document.getElementById(id);
const logEl = $('log');
function log(s) {
  // one block per line: text appended to a single block re-lays out every line of the log
  const d = document.createElement('div'); d.textContent = s || '\n'; logEl.append(d);
  if (!document.body.classList.contains('log-collapsed')) logEl.scrollTop = logEl.scrollHeight;
}

// status bar (the foot of the log strip's toolbar): with the log open it is a square showing
// only the barber pole / check / cross, so the stage text lives in its tooltip; collapsed it
// stretches to the full bar and shows the text as well
const setStatus = (s) => { $('status').textContent = s; $('progress').title = s; };

// log strip toolbar: collapse to just the toolbar row + the status bar, and copy the whole log
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

// parameter panel: each section heading collapses the rows under it (the .sec-body that follows),
// so a short window can be cut down to the groups in use; the collapsed set is remembered
const collapsedSecs = new Set((() => { try { return JSON.parse(localStorage.getItem('lapstack.sections') || '[]'); } catch { return []; } })());
for (const h of document.querySelectorAll('#params button.section')) {
  const set = (on) => {
    h.classList.toggle('collapsed', on); h.setAttribute('aria-expanded', String(!on));
    h.title = on ? `show ${h.textContent.toLowerCase()}` : `hide ${h.textContent.toLowerCase()}`;
    collapsedSecs[on ? 'add' : 'delete'](h.dataset.sec);
  };
  set(collapsedSecs.has(h.dataset.sec));
  h.onclick = () => {
    set(!h.classList.contains('collapsed'));
    try { localStorage.setItem('lapstack.sections', JSON.stringify([...collapsedSecs])); } catch {}
  };
}
$('log-copy').onclick = async () => {
  const b = $('log-copy');
  try {
    await navigator.clipboard.writeText(logEl.innerText);
  } catch {
    // clipboard API needs a secure context; fall back to a scratch selection
    const ta = document.createElement('textarea');
    ta.value = logEl.innerText; ta.style.position = 'fixed'; ta.style.opacity = '0';
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
  retouch: { on: false, prev: null, size: 100, hard: 0.5, from: 'source', painting: false, dabs: [], last: null, cursor: null, hold: false,   // on: retouch mode (a compare split, stack layer | brush source); from: 'source' = the scrubbed frame, 'stack' = the other stacked result, 'slab' = the on-demand slab (see brushFrom); prev: the compare state to restore on exit; hold: the hover preview waits for the next pointer move (see onPatch)
             wasmIndex: -1, loading: -1, gen: 0, genMin: 0, undo: 0, redo: 0,
             gpuIndex: -1, prefetch: -1, ahead: null, dir: 1, lastSel: -1,     // see ensureSource(): the frame the worker holds on the GPU, the one being prefetched, the read-ahead slot, the scrub direction
             slab: null, slabLoading: null, slabProgress: null, slabGen: 0, slabGenMin: 0 },   // see ensureSlab(): the slab held ({lo, hi, canvas}), the one being built ({lo, hi, gen}), its progress
};
window.__st = st; window.__draw = () => draw();
// The batch: the frame list cut into stacks (stacksOf), run in turn. While one is in hand
// `all` holds every file and st.files the stack being run; `stacks` keeps each stack's
// status for the filmstrip's headers (lo/hi index into `all`, or st.files once shown again).
const B = { all: null, frames: null, stacks: [], k: -1, cancelled: false, done: false, t0: 0 };
const dpr = () => window.devicePixelRatio || 1;

// ---------- settings (persisted) ----------
const PK = 'lapstack.settings';
const stepDefaults = { 'p-coarsen': 2, 'p-levels': 0, 'p-energy': 1, 'p-topr': 2, 'p-depthscale': 2, 'p-depthlevel': 2, 'p-proxy': 1400, 'p-slab': 5, 'p-dslab-size': 10, 'p-dslab-ov': 2, 'p-split-n': 30, 'p-split-gap': 10 };
function readParams() {
  const n = (id) => Number($(id).textContent === 'auto' ? 0 : $(id).textContent);
  return {
    align: $('p-align').checked, shift: $('p-shift').checked, scale: $('p-scale').checked, rotation: $('p-rotation').checked, brightness: $('p-bright').checked,
    coarsen: n('p-coarsen'), levels: n('p-levels') || null, energy_radius: n('p-energy'), top: $('p-top').value,
    top_radius: n('p-topr'), use_chroma: $('p-chroma').checked, proxy_edge: n('p-proxy'),
    depth_scale: n('p-depthscale'), depth_level: n('p-depthlevel'), render_dmap: $('p-dmap').checked,
    render_slabs: $('p-dslabs').checked, slab_size: n('p-dslab-size'), slab_overlap: n('p-dslab-ov'),
    turbo: st.turbo, slice: st.slice, peak_on: st.peak.on, peak_strip: st.peak.strip, peak_thr: st.peak.thr, cmp_mode: st.retouch.on && st.retouch.prev ? st.retouch.prev.cmpMode : st.cmpMode,
    brush_size: st.retouch.size, brush_hard: st.retouch.hard, brush_from: st.retouch.from, brush_slab: n('p-slab'),
    split: $('p-split').value, split_n: n('p-split-n'), split_gap: n('p-split-gap'),
  };
}
function setStep(id, v) {
  const el = $(id); const lo = Number(el.dataset.min), hi = Number(el.dataset.max);
  v = Math.min(hi, Math.max(lo, v));
  el.textContent = (v === 0 && el.dataset.zero) ? el.dataset.zero : String(v);
}
function applyParams(p) {
  if (!p) return;
  $('p-align').checked = p.align ?? true; $('p-shift').checked = p.shift ?? true; $('p-scale').checked = p.scale ?? true; $('p-rotation').checked = p.rotation ?? true; $('p-bright').checked = p.brightness ?? true;
  setStep('p-coarsen', p.coarsen ?? 2); setStep('p-levels', p.levels ?? 0); setStep('p-energy', p.energy_radius ?? 1);
  $('p-top').value = p.top ?? 'de'; setStep('p-topr', p.top_radius ?? 2); $('p-chroma').checked = p.use_chroma ?? false;
  setStep('p-proxy', p.proxy_edge ?? 1400); st.turbo = p.turbo ?? false;
  setStep('p-depthscale', p.depth_scale ?? 2); setStep('p-depthlevel', p.depth_level ?? 2); $('p-dmap').checked = p.render_dmap ?? false; st.cmpMode = p.cmp_mode ?? 'swipe';
  $('p-dslabs').checked = p.render_slabs ?? false; setStep('p-dslab-size', p.slab_size ?? 10); setStep('p-dslab-ov', p.slab_overlap ?? 2);
  st.peak.on = p.peak_on ?? false; st.peak.strip = p.peak_strip ?? false; st.peak.thr = p.peak_thr ?? 0.5; st.slice = p.slice ?? true;
  st.retouch.size = p.brush_size ?? 100; st.retouch.hard = p.brush_hard ?? 0.5; st.retouch.from = ['stack', 'slab'].includes(p.brush_from) ? p.brush_from : 'source';
  setStep('p-slab', p.brush_slab ?? 5);
  $('p-split').value = ['count', 'gap', 'dir'].includes(p.split) ? p.split : 'none'; setStep('p-split-n', p.split_n ?? 30); setStep('p-split-gap', p.split_gap ?? 10);
}
function saveParams() { try { localStorage.setItem(PK, JSON.stringify(readParams())); } catch {} }
try { applyParams(JSON.parse(localStorage.getItem(PK))); } catch {}
for (const id of Object.keys(stepDefaults)) setStep(id, Number($(id).textContent === 'auto' ? 0 : $(id).textContent));
document.querySelectorAll('#params [data-step], #runmenu [data-step]').forEach((b) => b.addEventListener('click', () => {
  const id = b.dataset.step; const cur = Number($(id).textContent === 'auto' ? 0 : $(id).textContent);
  setStep(id, cur + Number(b.dataset.d));
  // a slab's overlap stays short of its size (the engine clamps it too), so consecutive slabs advance
  if (id === 'p-dslab-size' || id === 'p-dslab-ov') setStep('p-dslab-ov', Math.min(Number($('p-dslab-ov').textContent), Number($('p-dslab-size').textContent) - 1));
  saveParams();
}));
document.querySelectorAll('#params input, #params select').forEach((el) => el.addEventListener('change', saveParams));
// the slab half-width is a brush setting: a change moves the slab's range (ensureSlab, via updateTabs)
document.querySelectorAll('[data-step="p-slab"]').forEach((b) => b.addEventListener('click', () => { updateTabs(); renderFilmstrip(); draw(); }));
// Run menu (DFR and the batch split live here, not in the parameter panel): the Run label
// shows the state, with the number of stacks a split makes
const runLabel = () => {
  const base = $('p-dmap').checked ? 'Run LAP + DFR' : 'Run LAP';
  const stacks = B.all ? null : stacksOf(st.files), n = stacks ? stacks.length : 0;
  $('run').textContent = n > 1 ? `${base} ×${n}` : base;
  $('run').title = n > 1 ? `batch: ${n} stacks, run in turn and saved as they finish` : '';
  $('dslab-ctl').hidden = !$('p-dslabs').checked;
  const rule = $('p-split').value;
  $('split-ctl').hidden = rule === 'none' || rule === 'dir'; $('split-n-row').hidden = rule !== 'count'; $('split-gap-row').hidden = rule !== 'gap';
  $('split-info').textContent = splitInfo(stacks);
};
// what the split makes of the frames in hand, for the Run menu
function splitInfo(stacks) {
  const files = B.all || st.files, rule = $('p-split').value;
  if (!files.length) return rule === 'none' ? '' : 'no frames yet';
  if (rule === 'none') return `one stack of ${files.length} frames`;
  if (stacks === null) return 'reading the capture times…';
  const sizes = stacks.map((s) => s.hi - s.lo + 1), lo = Math.min(...sizes), hi = Math.max(...sizes);
  let text = `${stacks.length} stack${stacks.length > 1 ? 's' : ''} of ${lo === hi ? lo : `${lo}–${hi}`} frames`;
  if (rule === 'gap') {
    const t = files.map((f) => f.ctime ?? f.lastModified / 1000), gaps = t.slice(1).map((v, i) => Math.abs(v - t[i]));
    const none = files.filter((f) => f.ctime == null).length;
    if (gaps.length) text += ` · frames ${median(gaps).toFixed(1)} s apart, the longest pause ${Math.max(...gaps).toFixed(1)} s`;
    if (none) text += ` · ${none} frame${none > 1 ? 's' : ''} without a capture time (their file date is used)`;
  }
  return text;
}
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : 0; };
$('p-dmap').addEventListener('change', () => { saveParams(); runLabel(); });
$('p-dslabs').addEventListener('change', () => { if ($('p-dslabs').checked) $('p-dmap').checked = true; saveParams(); runLabel(); });   // slabs are a way of rendering DFR
// the split rule: the filmstrip regroups, and the statuses of an earlier batch no longer apply
const splitChanged = () => { B.stacks = []; saveParams(); ensureTimes(); runLabel(); renderFilmstrip(); };
$('p-split').addEventListener('change', splitChanged);
document.querySelectorAll('[data-step="p-split-n"], [data-step="p-split-gap"]').forEach((b) => b.addEventListener('click', splitChanged));
$('run-more').addEventListener('click', (e) => { e.stopPropagation(); $('runmenu').hidden = !$('runmenu').hidden; });
$('runmenu').addEventListener('click', (e) => e.stopPropagation());
document.addEventListener('click', () => { $('runmenu').hidden = true; $('cmp-menu').hidden = true; });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { $('runmenu').hidden = true; $('cmp-menu').hidden = true; if (st.pick) setPick(false); else exitRetouch(); } });
runLabel();

// ---------- worker ----------
// (query string: never run a stale cached worker after a rebuild; serve.sh also sends no-store)
const worker = new Worker('./worker.js?t=' + Date.now(), { type: 'module' });
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.rid && onReply(m)) return;
  switch (m.type) {
    case 'ready': {
      setStatus('ready'); $('progress').className = '';
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
    case 'slab': onSlab(m); break;
    case 'slab-skipped': onSlabSkipped(m); break;
    case 'slab-progress': onSlabProgress(m); break;
    case 'refold-progress': setProgress(m.text, m.done, m.total); $('sv-progress').textContent = m.text; break;
    case 'patch': onPatch(m); break;
    case 'thumb-error': log(`[lapstack] cannot decode ${m.name}: ${m.text}`); break;
    case 'done': onDone(m); break;
    case 'done2': onDone2(m); break;
    case 'render-cancelled': endRun('cancelled (depth-map render)'); log('[lapstack] depth-map render cancelled; the LAP result is kept'); if (inBatch()) B.cancelled = true; finishRun(); break;
    case 'cancelled': endRun('cancelled'); log('[lapstack] cancelled'); if (inBatch()) stackFailed('cancelled'); break;
    case 'error': endRun('error'); log('[lapstack] error: ' + m.text); toast(/no WebGPU adapter/i.test(m.text) ? 'No WebGPU adapter. ' + gpuHint() : m.text, 0); if (inBatch()) stackFailed('error'); break;
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
// A file's path as it was added: folder/name from a folder pick or a dropped folder, else the
// name; files sort by it (natural order, f2 before f10), so a folder's frames stay together.
let fileUid = 0;
const fileKey = (f) => f.relPath || f.webkitRelativePath || f.name;
const folderOf = (f) => { const k = fileKey(f), i = k.lastIndexOf('/'); return i < 0 ? '' : k.slice(0, i); };
const stemOf = (name) => name.replace(/\.[^.]+$/, '');
function addFiles(list) {
  const files = [...list].filter((f) => /\.(png|jpe?g|tiff?)$/i.test(f.name)).sort((a, b) => fileKey(a).localeCompare(fileKey(b), undefined, { numeric: true }));
  if (!files.length) return;
  if (st.running || SV.exporting) { toast('Frames can be added once the run or save in progress is done.'); return; }
  for (const f of files) f.uid = ++fileUid;
  if (B.all && !st.running) showAll();   // a batch's stack is on screen: the new frames join the whole list
  B.stacks = [];
  st.files.push(...files);
  ensureTimes();
  renderFilmstrip(); runLabel();
  $('run').disabled = st.running || !st.files.length;
  $('tab-source').disabled = false;
  if (st.view === 'source') draw();
  log(`[lapstack] ${files.length} frame(s) added (${st.files.length} total)`);
  for (const f of files) makeThumb(f);
  const indices = files.map((f) => st.files.indexOf(f));
  worker.postMessage({ type: 'thumbs', files, indices, uids: files.map((f) => f.uid), edge: readParams().proxy_edge });
}
// Every frame's capture time (captureTime: EXIF, else XMP), read once per file, a few small
// reads each: the split by pause needs them all, the name tokens the first. undefined =
// still being read, null = none (the file's date stands in).
let timesPending = 0;
function ensureTimes() {
  for (const f of (B.all || st.files)) {
    if (f.ctime !== undefined || f.ctimeReading) continue;
    f.ctimeReading = true; timesPending++;
    captureTime(f).then((t) => { f.ctime = t; }, () => { f.ctime = null; }).then(() => { if (--timesPending === 0) { runLabel(); renderFilmstrip(); } });
  }
}
// The frame list cut into stacks by the Run menu's rule: [{lo, hi, t0?, t1?}] over `files`,
// null while the capture times a split by pause needs are still being read.
function stacksOf(files) {
  const n = files.length; if (!n) return [];
  const rule = $('p-split').value, out = [];
  const num = (id) => Number($(id).textContent);
  if (rule === 'count') { const per = Math.max(1, num('p-split-n')); for (let lo = 0; lo < n; lo += per) out.push({ lo, hi: Math.min(lo + per, n) - 1 }); }
  else if (rule === 'dir') { let lo = 0; for (let i = 1; i <= n; i++) if (i === n || folderOf(files[i]) !== folderOf(files[lo])) { out.push({ lo, hi: i - 1 }); lo = i; } }
  else if (rule === 'gap') {
    if (files.some((f) => f.ctime === undefined)) return null;
    const gap = num('p-split-gap'), t = files.map((f) => f.ctime ?? f.lastModified / 1000);
    let lo = 0; for (let i = 1; i <= n; i++) if (i === n || Math.abs(t[i] - t[i - 1]) > gap) { out.push({ lo, hi: i - 1, t0: t[lo], t1: t[i - 1] }); lo = i; }
  } else out.push({ lo: 0, hi: n - 1 });
  return out;
}
// a dropped folder: its files, walked through the entries API (the entries must be taken
// before the event returns; the walk itself can wait)
function droppedFiles(dt) {
  const entries = [...(dt.items || [])].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (!entries.some((e) => e.isDirectory)) return Promise.resolve([...dt.files]);
  const out = [];
  const walk = async (e, prefix) => {
    if (e.isFile) { const f = await new Promise((res, rej) => e.file(res, rej)); f.relPath = prefix + f.name; out.push(f); }
    else if (e.isDirectory) {
      const rd = e.createReader();
      for (;;) { const batch = await new Promise((res, rej) => rd.readEntries(res, rej)); if (!batch.length) break; for (const c of batch) await walk(c, prefix + e.name + '/'); }
    }
  };
  return (async () => { for (const e of entries) await walk(e, ''); return out; })();
}
// A proxy arrives as two bitmaps made by the worker (see proxyBitmaps there): full size for
// the view, and a strip-sized one for its thumb. Drawing the full proxy into the 160x100
// thumb canvas would cost ~30 ms of main thread per frame when the canvas is flushed.
const STRIP_W = 160, STRIP_H = 100;
function onThumb(m) {
  // the file by its id: the list may have been cleared, or a batch may be showing one stack of it
  let i = st.files.findIndex((f) => f.uid === m.uid), frames = st.frames;
  if (i < 0 && B.all) { i = B.all.findIndex((f) => f.uid === m.uid); frames = B.frames; }
  if (i < 0) return; // stale (cleared)
  const cur = frames[i];
  if (cur && cur.proxy && cur.sim) return; // the run already supplied an aligned proxy
  frames[i] = { ...(cur || {}), name: m.name, w: m.w, h: m.h, bits: m.bits, proxy: m.proxy, strip: m.strip };
  if (frames !== st.frames) return;
  renderThumb(i);
  if (st.view === 'source' && st.selected === i) draw();
}
async function makeThumb(f) {
  if (!/\.(png|jpe?g)$/i.test(f.name)) return; // the browser cannot decode TIFF; the run supplies a proxy
  try {
    const bmp = await createImageBitmap(f, { resizeWidth: 320, resizeQuality: 'medium' });
    const i = st.files.indexOf(f); if (i < 0) return;
    st.frames[i] = st.frames[i] || { name: f.name };
    if (!st.frames[i].proxy) { st.frames[i].thumb = bmp; renderThumb(i); if (st.view === 'source' && st.selected === i) draw(); }
  } catch {}
}
function renderFilmstrip() {
  const fs = $('filmstrip'); fs.innerHTML = '';
  if (B.all) fs.appendChild(batchBanner());
  if (!st.files.length) { fs.insertAdjacentHTML('beforeend', '<div class="empty dim">Add frames, or drop them here.</div>'); return; }
  // the split's stacks head their frames (one stack, or a batch's stack in hand, has no header)
  const stacks = B.all ? [] : stacksOf(st.files) || [];
  const heads = new Map();
  if (stacks.length > 1) stacks.forEach((s, k) => heads.set(s.lo, groupHead(s, k, stacks)));
  st.files.forEach((f, i) => { if (heads.has(i)) fs.appendChild(heads.get(i)); fs.appendChild(thumbEl(i)); });
}
const clock = (t) => { const s = Math.floor(((t % 86400) + 86400) % 86400); return `${pad2(Math.floor(s / 3600))}:${pad2(Math.floor(s / 60) % 60)}:${pad2(s % 60)}`; };
// a stack's header in the filmstrip: its number and size, the status a batch gave it, and in
// the tooltip its frames, capture times and the pause before it
function groupHead(s, k, stacks) {
  const d = document.createElement('div'); d.className = 'fs-group';
  const n = s.hi - s.lo + 1, a = document.createElement('span'); a.textContent = `stack ${k + 1}/${stacks.length} · ${n} frame${n > 1 ? 's' : ''}`;
  let tip = `${fileKey(st.files[s.lo])} .. ${fileKey(st.files[s.hi])}`;
  if (s.t0 !== undefined) { tip += `\n${clock(s.t0)} .. ${clock(s.t1)}`; if (k > 0) tip += `, ${(s.t0 - stacks[k - 1].t1).toFixed(0)} s after the last`; }
  d.title = tip;
  const st_ = document.createElement('span'); st_.className = 'st';
  const b = B.stacks.find((x) => x.lo === s.lo && x.hi === s.hi);
  const label = { queued: 'queued', running: 'running…', saving: 'saving…', saved: '✓ saved', done: '✓ done', failed: '✗ failed', 'save-failed': '✗ save failed', cancelled: 'cancelled' };
  if (b) { st_.textContent = label[b.status] || b.status; st_.className += /saved|done/.test(b.status) ? ' ok' : /fail/.test(b.status) ? ' bad' : b.status === 'running' ? ' run' : ''; }
  d.append(a, st_);
  return d;
}
// the banner over a batch's stack in hand: where the batch is, and the way back to every frame
function batchBanner() {
  const d = document.createElement('div'); d.className = 'fs-batch';
  const n = B.stacks.length, saved = B.stacks.filter((x) => x.status === 'saved' || x.status === 'done').length, failed = B.stacks.filter((x) => /fail/.test(x.status)).length;
  const s = B.stacks[B.k], size = s ? s.hi - s.lo + 1 : st.files.length;
  const line = document.createElement('div');
  line.innerHTML = B.done
    ? `<b>Batch done</b> · ${n} stacks, ${saved} saved${failed ? `, ${failed} failed` : ''}${B.cancelled ? ', cancelled' : ''} · showing stack ${Math.min(B.k, n - 1) + 1}`
    : `<b>Batch ${B.k + 1}/${n}</b> · stack of ${size} frames${saved ? ` · ${saved} saved` : ''}${failed ? ` · ${failed} failed` : ''}`;
  d.appendChild(line);
  if (B.done) { const b = document.createElement('button'); b.textContent = 'all frames'; b.title = 'show every frame again, grouped into its stacks (the result on screen is dropped)'; b.addEventListener('click', showAll); d.appendChild(b); }
  return d;
}
// redraw one frame's thumb in place (a run delivers one frame at a time; rebuilding the whole strip
// each time costs a drawImage per frame on the main thread, which stutters with a long stack)
function renderThumb(i) {
  const fs = $('filmstrip'), cur = fs.querySelector(`.thumb[data-i="${i}"]`);
  if (!cur || fs.querySelectorAll('.thumb').length !== st.files.length) return renderFilmstrip();
  fs.replaceChild(thumbEl(i), cur);
}
function thumbEl(i) {
  const f = st.files[i];
  const d = document.createElement('div'); d.className = 'thumb' + (i === st.selected && scrubbable() ? ' sel' : ''); d.dataset.i = i;
  if (R.on && brushFrom() === 'slab') { const [lo, hi] = slabWanted(); if (i >= lo && i <= hi) d.classList.add('slab'); }   // the frames the brush source is fused from
  const fr = st.frames[i];
  const bmp = fr && (fr.strip || fr.thumb);
  if (bmp) {
    const c = document.createElement('canvas'); c.width = STRIP_W; c.height = STRIP_H;
    const s = Math.min(STRIP_W / bmp.width, STRIP_H / bmp.height);
    const g = c.getContext('2d'); const r = [(STRIP_W - bmp.width * s) / 2, (STRIP_H - bmp.height * s) / 2, bmp.width * s, bmp.height * s];
    const peaking = st.peak.strip && fr.peak;   // with preview peaking on, the thumb is the dimmed frame under its magenta in-focus band
    if (peaking) g.filter = 'grayscale(1) brightness(0.6)';
    g.drawImage(bmp, ...r);
    if (peaking) { g.filter = 'none'; g.drawImage(peakThumb(fr, Math.round(r[2]), Math.round(r[3])), r[0], r[1]); }
    d.appendChild(c);
  } else { const ph = document.createElement('div'); ph.className = 'ph'; ph.textContent = fr ? '…' : String(i); d.appendChild(ph); }
  const n = document.createElement('div'); n.className = 'name'; n.textContent = f.name; n.title = fileKey(f); d.appendChild(n);
  if (fr && fr.sim) { const s = document.createElement('div'); s.className = 'sim'; s.textContent = `${fr.sim[0].toFixed(1)}, ${fr.sim[1].toFixed(1)} px · ×${fr.sim[2].toFixed(4)} · ${fr.sim[3].toFixed(2)}°`; d.appendChild(s); }
  // the brightness gain on its own line (the registration line fills the column), only when there is one
  if (fr && gainText(fr.gain)) { const s = document.createElement('div'); s.className = 'sim'; s.textContent = `brightness ${gainText(fr.gain)}`; s.title = 'gain that brings this frame to frame 0\'s brightness'; d.appendChild(s); }
  if (st.peak.strip && fr && fr.peak) { const s = document.createElement('div'); s.className = 'sim pct'; s.textContent = `${peakPercent(fr).toFixed(1)} % in focus`; d.appendChild(s); }
  d.addEventListener('click', () => { st.selected = i; if (!scrubbable()) st.view = 'source'; updateTabs(); renderFilmstrip(); draw(); });
  return d;
}
$('add').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
$('addf').addEventListener('click', () => $('dir').click());
$('dir').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
$('clear').addEventListener('click', () => { if (st.running || SV.exporting) return; setPick(false); worker.postMessage({ type: 'clear' }); st.files = []; st.frames = []; st.result = null; st.depthBmp.clear(); st.sliceBmps.clear(); st.peak.pixmax = null; resetRetouch(); B.all = null; B.frames = null; B.stacks = []; B.done = false; st.step = 'stack'; gotoStep('stack'); renderFilmstrip(); runLabel(); updateTabs(); setView('source'); });
document.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('drop'); });
document.addEventListener('dragleave', () => document.body.classList.remove('drop'));
document.addEventListener('drop', (e) => { e.preventDefault(); document.body.classList.remove('drop'); if (!st.running) droppedFiles(e.dataTransfer).then(addFiles); });

// ---------- run ----------
function setProgress(text, done, total) {
  $('progress').className = 'running'; setStatus(`${text} ${total ? `${done}/${total}` : ''}`);
  $('fill').style.width = total ? `${(100 * done / total).toFixed(1)}%` : '100%';   // no total = indeterminate: full pole
}
$('run').addEventListener('click', () => {
  if (st.running || SV.exporting || !st.files.length) return;
  const stacks = B.all ? [] : stacksOf(st.files);
  if (stacks === null) { toast('The capture times are still being read; try again in a moment.'); return; }
  if (stacks.length > 1) runBatch(stacks); else startRun();
});
function startRun() {
  st.running = true; setPick(false); st.frames = st.frames.map((f) => (f ? { name: f.name, thumb: f.thumb, proxy: f.proxy, strip: f.strip, w: f.w, h: f.h, bits: f.bits } : f)); st.result = null; st.depthBmp.clear(); st.sliceBmps.clear(); st.peak.pixmax = null; resetRetouch(); if (st.step !== 'stack') gotoStep('stack');
  runLabel(); $('runwrap').hidden = true; $('runmenu').hidden = true; $('cancel').hidden = false; $('clear').disabled = true;
  setProgress('starting', 0, st.files.length);
  const params = readParams(); delete params.turbo;
  log(`[lapstack] run: ${st.files.length} frames, ${JSON.stringify(params)}`);
  st.t0 = performance.now(); st.rendering = !!params.render_dmap; window.__app_done = null;
  worker.postMessage({ type: 'run', files: st.files, params });
}
$('cancel').addEventListener('click', () => { worker.postMessage({ type: 'cancel' }); if (inBatch() && SV.exporting) { SV.cancel = true; worker.postMessage({ type: 'refold_cancel' }); } });
// ---------- batch ----------
// The stacks run in turn through the ordinary run: each becomes the file list in hand, is run,
// and its ticked outputs are saved (saveSelected, with the Save step's settings) before the
// next starts. The last stack stays on screen; showAll brings every frame back.
const inBatch = () => !!B.all && !B.done;
function runBatch(stacks) {
  B.all = st.files; B.frames = st.frames; B.stacks = stacks.map((s) => ({ ...s, status: 'queued', files: [] })); B.k = -1; B.cancelled = false; B.done = false; B.t0 = performance.now();
  window.__batch_done = null;
  // the file names must differ between stacks: the stack number goes in unless a per-stack token is on
  if (!['fn-exif', 'fn-fname', 'fn-first', 'fn-stack'].some((id) => $(id).checked)) { $('fn-stack').checked = true; saveSaveSettings(); log('[lapstack] batch: the stack number is added to the file names (no other name token tells the stacks apart)'); }
  const sel = OUTPUTS.filter((o) => SV.sel.has(o.id)).map((o) => o.token);
  log(`[lapstack] batch: ${stacks.length} stacks (${splitInfo(stacks)}); saving ${sel.length ? sel.join(', ') : 'nothing — tick files in the Save step'} ${SV.dir ? `to the folder "${SV.dir.name}"` : 'as downloads'}`);
  nextStack();
}
function loadStack(s) {
  st.files = B.all.slice(s.lo, s.hi + 1); st.frames = B.frames.slice(s.lo, s.hi + 1);
  st.result = null; st.depthBmp.clear(); st.sliceBmps.clear(); st.peak.pixmax = null; resetRetouch(); st.selected = 0;
  if (st.step !== 'stack') gotoStep('stack');
  renderFilmstrip(); updateTabs();
}
function nextStack() {
  B.k++;
  if (B.cancelled || B.k >= B.stacks.length) return finishBatch();
  const s = B.stacks[B.k]; s.status = 'running';
  loadStack(s);
  log(`[lapstack] batch ${B.k + 1}/${B.stacks.length}: ${st.files.length} frames, ${fileKey(st.files[0])} .. ${fileKey(st.files[st.files.length - 1])}`);
  startRun();
}
// the stack in hand is stacked: save it, then the next (called from finishRun)
async function stackDone() {
  const s = B.stacks[B.k];
  for (let i = 0; i < st.frames.length; i++) if (st.frames[i]) B.frames[s.lo + i] = st.frames[i];   // the aligned proxies, for the grouped list later
  if (B.cancelled) { s.status = 'cancelled'; return finishBatch(); }
  if (OUTPUTS.some((o) => o.avail() && SV.sel.has(o.id))) {
    s.status = 'saving'; renderFilmstrip();
    const f0 = st.files[0]; SV.exifFor = f0; SV.exif = await exifDate(f0);   // the name tokens read the stack's first frame
    $('cancel').hidden = false;
    const status = await saveSelected();
    $('cancel').hidden = true;
    s.status = status === 'saved' ? 'saved' : status === 'cancelled' ? 'cancelled' : 'save-failed'; s.files = SV.lastSaved || [];
    if (status === 'cancelled') B.cancelled = true;
  } else s.status = 'done';
  nextStack();
}
function stackFailed(why) {
  const s = B.stacks[B.k]; if (!s || s.status !== 'running') return;
  s.status = why === 'cancelled' ? 'cancelled' : 'failed';
  if (why === 'cancelled') B.cancelled = true;
  nextStack();
}
function finishBatch() {
  const n = B.stacks.length, saved = B.stacks.filter((s) => s.status === 'saved' || s.status === 'done').length, failed = B.stacks.filter((s) => /fail/.test(s.status)).length;
  const secs = ((performance.now() - B.t0) / 1000).toFixed(1);
  B.done = true;
  const files = B.stacks.reduce((a, s) => a + (s.files ? s.files.length : 0), 0);
  const text = `batch: ${n} stacks, ${saved} done, ${failed} failed${B.cancelled ? ', cancelled' : ''}, ${files} file${files === 1 ? '' : 's'} saved ${SV.dir ? `to "${SV.dir.name}"` : 'as downloads'}  (${secs}s)`;
  log(`[lapstack] ${text}`); toast(text, failed || B.cancelled ? 0 : 12000);
  for (const s of B.stacks) if (/fail/.test(s.status)) log(`[lapstack]   stack ${B.stacks.indexOf(s) + 1} (${fileKey(B.all[s.lo])} ..): ${s.status}`);
  window.__batch_done = JSON.stringify({ ok: !failed && !B.cancelled, stacks: B.stacks.map((s) => ({ lo: s.lo, hi: s.hi, status: s.status, files: s.files })), secs });
  renderFilmstrip(); runLabel(); updateTabs();
}
// every frame again, grouped into its stacks with what the batch made of them; the result goes
function showAll() {
  if (!B.all || st.running || SV.exporting) return;
  st.files = B.all; st.frames = B.frames; B.all = null; B.frames = null; B.done = false;
  st.result = null; st.depthBmp.clear(); st.sliceBmps.clear(); st.peak.pixmax = null; resetRetouch(); st.selected = 0;
  st.step = 'stack'; gotoStep('stack'); setView('source'); renderFilmstrip(); runLabel(); updateTabs();
}
function endRun(status) {
  st.running = false; $('runwrap').hidden = false; $('cancel').hidden = true; $('clear').disabled = false;
  $('progress').className = status.startsWith('done') ? 'done' : 'error'; $('fill').style.width = '0';
  $('run').disabled = !st.files.length; setStatus(status); updateTabs();
}
function onFrame(m) {
  const peak = { w: m.peak_w, h: m.peak_h, data: new Float32Array(m.peak), bmp: null, bmpThr: -1, pct: null, pctThr: -1 };
  st.peak.pixmax = null;
  st.frames[m.index] = { name: m.name, w: m.w, h: m.h, bits: m.bits, proxy: m.proxy, strip: m.strip, sim: m.sim, gain: m.gain || null, peak };
  setProgress('fusing', m.done, m.total);
  log(`[lapstack]   frame ${String(m.index).padStart(3)}: dx=${m.sim[0].toFixed(2)}px dy=${m.sim[1].toFixed(2)}px scale=${m.sim[2].toFixed(5)} rot=${m.sim[3].toFixed(3)}°` + (m.gain ? ` gain=${m.gain.map((v) => v.toFixed(3)).join('/')}` : '') + `  (${m.ms.toFixed(0)} ms)`);
  renderThumb(m.index);
  if (st.view === 'source' && st.selected === m.index) draw();
}
async function onDone(m) {
  const img = new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h);
  const fused = new OffscreenCanvas(m.w, m.h);
  fused.getContext('2d').putImageData(img, 0, 0);
  st.result = { w: m.w, h: m.h, bits: m.bits, fused, dmap: null, depth: new Float32Array(m.depth), dw: m.depth_w, dh: m.depth_h,
                winner: new Float32Array(m.winner), ww: m.winner_w, wh: m.winner_h, meta: m.meta || null,   // meta: what the first frame carried (EXIF / ICC / XMP sizes)
                crop: m.crop ? { x: m.crop[0], y: m.crop[1], w: m.crop[2], h: m.crop[3] } : null };            // crop: the window every aligned frame covers, null = all of it
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
  if (inBatch()) stackDone();
}
// the brightness gain of a frame, for the filmstrip: "×0.983", or the three channels when they differ; '' at unity
function gainText(g) {
  if (!g || g.every((v) => Math.abs(v - 1) < 0.0005)) return '';
  const spread = Math.max(...g) - Math.min(...g);
  return spread > 0.005 ? `×${g.map((v) => v.toFixed(2)).join('/')}` : `×${((g[0] + g[1] + g[2]) / 3).toFixed(3)}`;
}
// ---------- worker calls ----------
// Requests with a reply: the worker echoes `rid` on the answer (or rpc-error), see handleCall there.
const calls = { n: 0, pending: new Map() };
function call(msg, transfer) {
  return new Promise((res, rej) => { const rid = ++calls.n; calls.pending.set(rid, { res, rej }); worker.postMessage({ ...msg, rid }, transfer || []); });
}
function onReply(m) {
  const p = calls.pending.get(m.rid); if (!p) return false;
  calls.pending.delete(m.rid);
  if (m.type === 'rpc-error') p.rej(new Error(m.text)); else p.res(m);
  return true;
}

// ---------- save step ----------
// One row per output the run can produce. Stills come from the engine's encoder; the
// animations are composed here frame by frame (the layer as the viewer draws it, at the
// chosen size) and quantised + LZW-encoded by the worker (gif.rs), the bytes streaming
// back so the file is assembled as a Blob. `token` is the layer part of the file name.
const OUTPUTS = [
  { id: 'lap', token: 'lap', kind: 'fused', name: 'LAP stack', desc: 'the fused image', avail: () => !!st.result },
  { id: 'dfr', token: 'dfr', kind: 'dmap', name: 'DFR stack', desc: 'rendered from the depth map', avail: () => haveDmap() },
  { id: 'stereo', token: 'stereo', kind: 'stereo', name: 'Stereo pair', desc: () => `synthetic stereo: the stacked image seen from the left and from the right, ${refolding() ? 'each view folded from the shifted frames' : 'sheared by its depth map'}`, avail: () => !!st.result },
  { id: 'mesh', token: '3d', kind: 'mesh', name: '3D model', desc: () => MESH_DESC[m3().format], ext: () => m3().format, avail: () => !!st.result },
  { id: 'depth', token: 'depth', kind: 'depth', name: 'Depth map', desc: '8-bit gray PNG, min–max scaled', ext: 'png', avail: () => !!st.result },
  { id: 'depth16', token: 'depth16', kind: 'depth16', name: 'Depth map, 16-bit', desc: '16-bit gray PNG, 65535 = last frame', ext: 'png', avail: () => !!st.result },
  { id: 'winner', token: 'winner', kind: 'winner', name: 'Winner map', desc: '8-bit gray PNG, LAP winner index', ext: 'png', avail: () => !!st.result },
  { id: 'anim-depth', token: 'depth-slice', anim: true, name: 'Focus depth, Turbo, slice sweeping', desc: 'animated GIF: the depth map with the magenta slice moving through the frames', ext: 'gif', avail: () => !!st.result && st.files.length > 1 },
  { id: 'anim-focus', token: 'infocus', anim: true, name: 'In focus sweep', desc: 'animated GIF: each frame\'s in-focus plane lit, the rest dimmed to outlines', ext: 'gif', avail: () => !!st.result && st.files.length > 1 },
  { id: 'anim-peak', token: 'peaking', anim: true, name: 'Source with focus peaking', desc: 'animated GIF: the aligned frames under their magenta peaking band', ext: 'gif', avail: () => !!st.result && st.files.length > 1 && st.frames.some((f) => f && f.peak) },
  { id: 'anim-rock', token: 'rocking', anim: true, rock: true, name: 'Rocking', desc: () => `animated GIF: the stacked image rocking from side to side, ${refolding() ? 'each view folded from the shifted frames' : 'sheared by its depth map'}`, ext: 'gif', avail: () => !!st.result },
];
const MESH_DESC = { glb: 'glTF binary: the stacked image as a textured relief of its depth map, one file', obj: 'Wavefront OBJ + MTL + texture image: the textured relief as Helicon writes it, three files', stl: 'binary STL: the relief alone, no texture, for printing' };
const MESH_MIME = { glb: 'model/gltf-binary', obj: 'model/obj', mtl: 'model/mtl', stl: 'model/stl', jpg: 'image/jpeg', png: 'image/png' };
const SV = { sel: new Set(['lap']), exif: null, exifFor: null, exporting: false, cancel: false, dir: null, lastSaved: [] };   // dir: the folder saved files go to (File System Access), null = downloads
const SK = 'lapstack.save';
const svIds = ['fn-app', 'fn-exif', 'fn-fname', 'fn-now', 'fn-first', 'fn-stack', 'fn-layer', 'sv-name', 'sv-format', 'sv-quality', 'sv-meta', 'sv-crop', 'an-edge', 'an-fps', 'an-loop', 'v3-method', 'v3-src', 'v3-shift', 'v3-layout', 'v3-rock', 'v3-near', 'm3-format', 'm3-relief', 'm3-grid', 'm3-tex', 'cc-on', 'cc-name'];
const svSteps = ['an-step', 'v3-views'];   // the card's steppers (a number in a span between − and +)
const svLive = ['sv-quality', 'sv-name', 'cc-name', 'v3-shift', 'v3-rock', 'm3-relief'];   // re-render on every input, not on change
// The crop: the run reports the window every aligned frame covers with real pixels
// (outside it some frame only has its smeared edge). With the Save card's switch on,
// every saved file is cut to it and the viewer shows it as the bright window.
const cropArea = () => (st.result && st.result.crop && $('sv-crop').checked) ? st.result.crop : null;
const outDims = () => { const r = cropArea(); return r ? [r.w, r.h] : imageDims(); };
// the crop as a source rectangle in a bitmap's own pixels (bitmaps come at frame, proxy or grid resolution)
const srcRect = (bw, bh) => { const r = cropArea(); if (!r) return [0, 0, bw, bh]; const k = bw / st.result.w, l = bh / st.result.h; return [r.x * k, r.y * l, r.w * k, r.h * l]; };
function saveSaveSettings() {
  const o = { sel: [...SV.sel] };
  for (const id of svSteps) o[id] = Number($(id).textContent);
  for (const id of svIds) { const el = $(id); o[id] = el.type === 'checkbox' ? el.checked : el.value; }
  try { localStorage.setItem(SK, JSON.stringify(o)); } catch {}
}
try {
  const o = JSON.parse(localStorage.getItem(SK));
  if (o) {
    for (const id of svIds) if (id in o) { const el = $(id); if (el.type === 'checkbox') el.checked = !!o[id]; else el.value = o[id]; }
    for (const id of svSteps) if (o[id]) setStep(id, Number(o[id]));
    if (Array.isArray(o.sel)) SV.sel = new Set(o.sel);
  }
} catch {}
// file-name tokens: joined with "_", lower case, anything else becomes "_"
const pad2 = (n) => String(n).padStart(2, '0');
const stamp = (d) => `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}-${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}`;
const clean = (t) => t.toLowerCase().replace(/[^a-z0-9.-]+/g, '_').replace(/^_+|_+$/g, '');
// "20260911_123456", "2026-09-11 12.34.56", "IMG_20260911T123456" … in a file name; time is optional
function nameDate(name) {
  const m = /(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?:[-_.T ]?([01]\d|2[0-3])[-_.:]?([0-5]\d)[-_.:]?([0-5]\d)?)?/.exec(name);
  if (!m) return null;
  return `${m[1]}${m[2]}${m[3]}` + (m[4] ? `-${m[4]}${m[5]}${m[6] || '00'}` : '');
}
// Bytes of a File read on demand in 64 KB chunks, so the capture date of a 270 MB TIFF whose
// IFD trails the pixels costs a few small reads and not the pixels.
function chunked(file) {
  const CH = 1 << 16, cache = new Map();
  const chunk = async (k) => { let c = cache.get(k); if (!c) { c = new Uint8Array(await file.slice(k * CH, (k + 1) * CH).arrayBuffer()); cache.set(k, c); } return c; };
  return async (off, len) => {   // [off, off + len) as a Uint8Array, short at the end of the file
    if (off < 0 || off >= file.size) return new Uint8Array(0);
    len = Math.min(len, file.size - off);
    const out = new Uint8Array(len);
    for (let p = off; p < off + len;) { const k = Math.floor(p / CH), c = await chunk(k), q = p - k * CH, n = Math.min(off + len - p, c.length - q); if (n <= 0) break; out.set(c.subarray(q, q + n), p - off); p += n; }
    return out;
  };
}
// The capture date of a file as EXIF writes it, "YYYY:MM:DD HH:MM:SS", or null: DateTimeOriginal,
// else DateTimeDigitized, else DateTime — the TIFF structure inside a JPEG APP1, a TIFF, or a
// PNG eXIf chunk — or, without one, the XMP packet's CreateDate (a JPEG APP1, TIFF tag 700, a
// PNG iTXt): raw converters write TIFFs with XMP and no EXIF at all.
async function captureDate(file) {
  try {
    const get = chunked(file);
    const u16 = (b, o, le) => le ? b[o] | b[o + 1] << 8 : b[o] << 8 | b[o + 1];
    const u32 = (b, o, le) => le ? (b[o] | b[o + 1] << 8 | b[o + 2] << 16) + b[o + 3] * 16777216 : b[o] * 16777216 + (b[o + 1] << 16 | b[o + 2] << 8 | b[o + 3]);
    const ascii = (b, o, n) => String.fromCharCode(...b.subarray(o, o + n));
    const head = await get(0, 8);
    let t = -1, xmp = null;   // t: the TIFF structure's offset in the file; xmp: [offset, length] of the packet
    if (head[0] === 0xFF && head[1] === 0xD8) {
      for (let p = 2; ;) {
        const h = await get(p, 34); if (h.length < 4 || h[0] !== 0xFF) break;
        const mk = h[1], len = u16(h, 2, false);
        if (mk === 0xE1 && ascii(h, 4, 6) === 'Exif\0\0') { t = p + 10; if (xmp) break; }
        else if (mk === 0xE1 && ascii(h, 4, 29) === 'http://ns.adobe.com/xap/1.0/\0') { xmp = [p + 33, len - 31]; if (t >= 0) break; }
        if (mk === 0xDA || mk === 0xD9) break;
        p += 2 + len;
      }
    } else if ((head[0] === 0x49 && head[1] === 0x49 && head[2] === 42) || (head[0] === 0x4D && head[1] === 0x4D && head[3] === 42)) t = 0;
    else if (head[0] === 0x89 && head[1] === 0x50) {
      for (let p = 8; p + 8 <= file.size;) {
        const h = await get(p, 30); if (h.length < 8) break;
        const len = u32(h, 0, false), type = ascii(h, 4, 4);
        if (type === 'eXIf') { t = p + 8; if (xmp) break; }
        else if (type === 'iTXt' && h.length >= 27 && ascii(h, 8, 17) === 'XML:com.adobe.xmp' && h[26] === 0) { xmp = [p + 30, len - 22]; if (t >= 0) break; }   // uncompressed, empty language / translation
        if (type === 'IEND') break;
        p += 12 + len;
      }
    }
    const xmpDate = async () => {
      if (!xmp) return null;
      const m = /(?:CreateDate|DateTimeOriginal)(?:="|>)(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/.exec(new TextDecoder().decode(await get(xmp[0], Math.min(xmp[1], 1 << 20))));
      return m ? `${m[1]}:${m[2]}:${m[3]} ${m[4]}:${m[5]}:${m[6]}` : null;
    };
    if (t < 0) return xmpDate();
    const th = await get(t, 8); const le = th[0] === 0x49;
    // the wanted tags of the IFD at off (TIFF-relative): ASCII as a string, SHORT / LONG as a number, BYTE / UNDEFINED as [offset, count]
    const ifd = async (off, want) => {
      const out = {}; const nb = await get(t + off, 2); if (nb.length < 2) return out;
      const n = u16(nb, 0, le), es = await get(t + off + 2, 12 * n);
      for (let i = 0; i + 12 <= es.length; i += 12) {
        const tag = u16(es, i, le), type = u16(es, i + 2, le), cnt = u32(es, i + 4, le);
        if (!want.includes(tag)) continue;
        const p = cnt <= 4 ? t + off + 2 + i + 8 : t + u32(es, i + 8, le);
        if (type === 2) { const b = await get(p, Math.min(cnt, 40)); out[tag] = ascii(b, 0, b.length).replace(/\0[\s\S]*$/, ''); }
        else if (type === 1 || type === 7) out[tag] = [p, cnt];
        else out[tag] = type === 3 ? u16(es, i + 8, le) : u32(es, i + 8, le);
      }
      return out;
    };
    const ifd0 = await ifd(u32(th, 4, le), [0x0132, 0x8769, 700]);
    let dt = null;
    if (ifd0[0x8769]) { const ex = await ifd(ifd0[0x8769], [0x9003, 0x9004]); dt = ex[0x9003] || ex[0x9004]; }
    dt = dt || ifd0[0x0132];
    const m = dt && /^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})/.exec(dt);
    if (m && m[1] !== '0000') return `${m[1]}:${m[2]}:${m[3]} ${m[4]}:${m[5]}:${m[6]}`;
    if (t === 0 && !xmp && Array.isArray(ifd0[700])) xmp = ifd0[700];   // a TIFF's XMP is tag 700 of IFD0 (BYTE or UNDEFINED, at the entry's offset)
    return xmpDate();
  } catch { return null; }
}
// the date as a file-name token, "YYYYMMDD-HHMMSS"
async function exifDate(file) { const d = await captureDate(file); return d ? d.replace(/^(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})$/, '$1$2$3-$4$5$6') : null; }
// the date as seconds by the camera's clock (no time zone: only differences between frames are used)
async function captureTime(file) { const m = /^(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})$/.exec(await captureDate(file) || ''); return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]) / 1000 : null; }
window.__svTest = { exifDate, nameDate, captureTime, stacksOf, SV, B, save: (kind, format, quality, meta, crop) => call({ type: 'save', kind, format, quality, meta, crop }),   // tests
                    mesh: (stem, format, extra) => { const m = m3(), v = v3(); return call({ type: 'mesh', stem, format, source: v.source, crop: v.crop, grid: m.grid, relief: m.relief, near: v.near, texture_edge: m.tex, texture: 'jpeg', quality: 90, ...(extra || {}) }); },
                    view: viewFrame, stereo: (format) => call({ type: 'view_stereo', ...v3(), format: format || 'png', quality: 90, meta: false }),
                    refold, refoldView: (index) => call({ type: 'refold_view', index }), refoldEnd: () => call({ type: 'refold_end' }),
                    sign: (bytes, mime, name) => signBlob(new Blob([bytes], { type: mime }), OUTPUTS[0], name, mime).then((b) => b.arrayBuffer()) };
function svExt(o) { const e = typeof o.ext === 'function' ? o.ext() : o.ext; return e || ($('sv-format').value === 'jpeg' ? 'jpg' : 'png'); }
function svName(o, now = new Date()) {
  const parts = [];
  if ($('fn-app').checked) parts.push('lapstack');
  if ($('fn-exif').checked && SV.exif) parts.push(SV.exif);
  if ($('fn-fname').checked && st.files[0]) { const d = nameDate(st.files[0].name); if (d) parts.push(d); }
  if ($('fn-now').checked) parts.push(stamp(now));
  if ($('fn-first').checked && st.files[0]) { const c = clean(stemOf(st.files[0].name)); if (c) parts.push(c); }
  if ($('fn-stack').checked && B.all) parts.push('s' + pad2(B.k + 1));
  const custom = clean($('sv-name').value); if (custom) parts.push(custom);
  if ($('fn-layer').checked) parts.push(o.token);
  return (parts.join('_') || 'stacked') + '.' + svExt(o);
}
// the stereo / rocking settings: the shifts as fractions of the width (the card shows percent)
const v3 = () => ({ method: $('v3-method').value, source: haveDmap() ? $('v3-src').value : 'fused', crop: !!cropArea(), near: $('v3-near').checked, shift: Number($('v3-shift').value) / 100, rock: Number($('v3-rock').value) / 100, views: Math.max(4, Number($('v3-views').textContent)), layout: $('v3-layout').value });
const refolding = () => $('v3-method').value === 'refold';
// the 3D model's settings (mesh.rs): the relief as a fraction of the width (the card shows percent); the
// image and the near end are the stereo section's
const m3 = () => ({ format: $('m3-format').value, grid: Number($('m3-grid').value), relief: Number($('m3-relief').value) / 100, tex: Number($('m3-tex').value) });
// the mesh's vertex grid for the output size (core mesh::grid_dims) and a rough file size
function meshPlan() {
  const [W, H] = outDims(), m = m3(), long = Math.max(W, H, 2), g = Math.min(Math.max(m.grid, 2), long);
  const axis = (n) => Math.min(Math.max(Math.round((Math.max(n, 2) - 1) * (g - 1) / (long - 1)) + 1, 2), Math.max(n, 2));
  const nx = axis(W), ny = axis(H), nv = nx * ny, nt = 2 * (nx - 1) * (ny - 1);
  const ts = m.tex && long > m.tex ? m.tex / long : 1, tw = Math.max(1, Math.round(W * ts)), th = Math.max(1, Math.round(H * ts));
  const tex = tw * th * ($('sv-format').value === 'jpeg' ? 0.4 : 1.5);
  const digits = String(nv).length;   // an OBJ face line carries nine indices
  const size = m.format === 'stl' ? 84 + nt * 50 : m.format === 'obj' ? nv * 78 + nt * (9 * digits + 9) + tex : nv * 32 + nt * 12 + tex;
  return { nx, ny, nv, nt, tw, th, size };
}
// Zerene's way: the stack folded again, each frame shifted by its index, one accumulator per view
// (see Refold in lib.rs); the frames are read again. The views wait in the engine until refold_end.
function refold(shifts, w, h) { const v = v3(); return call({ type: 'refold', files: st.files, shifts, near: v.near, w, h }); }
// one view of the stacked image, sheared by the engine at w×h (view.rs): {w, h, rgba}. `shift` is the
// far end's shift as a fraction of the width, positive = seen from the right.
function viewFrame(shift, w, h) { const v = v3(); return call({ type: 'view', source: v.source, crop: v.crop, w, h, shift, near: v.near }); }
// the shift of frame i of a rocking cycle of n: a sine sweep of ±a, easing at the ends, closing on itself
const rockShift = (a, i, n) => a * Math.sin(2 * Math.PI * i / n);
// the animation's frame order (every Nth frame, last frame always in; back and forth or forward), size and delay;
// the rocking animation's frames are one cycle of views instead
function animPlan(o) {
  const [W, H] = outDims(); const edge = Number($('an-edge').value), step = Math.max(1, Number($('an-step').textContent));
  const s = edge ? Math.min(1, edge / Math.max(W, H)) : 1;
  const ow = Math.max(1, Math.round(W * s)), oh = Math.max(1, Math.round(H * s));
  const delay = Math.max(2, Math.round(100 / Number($('an-fps').value)));
  if (o && o.rock) return { ow, oh, seq: [...Array(v3().views).keys()], delay };
  const n = st.files.length, fwd = [];
  for (let i = 0; i < n; i += step) fwd.push(i);
  if (n && fwd[fwd.length - 1] !== n - 1) fwd.push(n - 1);
  const seq = $('an-loop').value === 'pingpong' && fwd.length > 2 ? fwd.concat(fwd.slice(1, -1).reverse()) : fwd;
  return { ow, oh, seq, delay };
}
const fmtMB = (b) => b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB` : `${Math.round(b / 1e3)} KB`;
async function renderSave() {
  if (st.step !== 'save' || !st.result) return;
  const strokes = R.undo;
  const cr = st.result.crop;
  $('sv-crop').disabled = !cr;
  $('sv-crop-info').textContent = cr ? `${cr.w}×${cr.h} of ${st.result.w}×${st.result.h}, from (${cr.x}, ${cr.y}) — the bright window in the viewer` : 'the aligned frames cover the whole image: nothing to cut';
  const [ow, oh] = outDims();
  $('sv-info').textContent = `${st.result.w}×${st.result.h}, ${st.result.bits}-bit input, ${st.files.length} frames` + (cropArea() ? `, saved as ${ow}×${oh}` : '') + (strokes ? `, ${strokes} retouch stroke${strokes > 1 ? 's' : ''}` : '');
  // tokens: EXIF is read once per first frame (async: the name preview refreshes when it lands)
  const f0 = st.files[0];
  if (f0 && SV.exifFor !== f0) { SV.exifFor = f0; SV.exif = null; exifDate(f0).then((d) => { if (SV.exifFor === f0) { SV.exif = d; renderSave(); } }); }
  $('fn-exif-val').textContent = SV.exif || (f0 ? 'none found' : '—'); $('fn-exif').disabled = !SV.exif;
  const nd = f0 ? nameDate(f0.name) : null; $('fn-fname-val').textContent = nd || 'none found'; $('fn-fname').disabled = !nd;
  $('fn-now-val').textContent = stamp(new Date());
  $('fn-first-val').textContent = f0 ? clean(stemOf(f0.name)) || 'none' : '—'; $('fn-first').disabled = !f0;
  $('fn-stack-val').textContent = B.all ? 's' + pad2(B.k + 1) : 'in a batch';
  const j = $('sv-format').value === 'jpeg'; $('sv-qrow').hidden = !j; $('sv-quality').hidden = !j;
  // metadata: what the engine found in the first frame (the run reads it); nothing found disables the box
  const mt = st.result.meta, have = !!(mt && (mt.exif || mt.icc || mt.xmp || mt.chrm));
  $('sv-meta').disabled = !have;
  $('sv-meta-info').textContent = !mt ? '—' : have ? `${f0 ? f0.name : 'first frame'}: ${mt.text}` + (!mt.icc && mt.chrm ? ' (no ICC profile: PNG gets a cHRM chunk)' : '') : `nothing found in ${f0 ? f0.name : 'the first frame'}`;
  $('fn-preview').textContent = svName(OUTPUTS[0]);
  // stereo / rocking: the DFR image is on offer only when it was rendered
  $('v3-src').querySelector('[value="dmap"]').disabled = !haveDmap(); if (!haveDmap()) $('v3-src').value = 'fused';
  $('v3-shift-val').textContent = `±${$('v3-shift').value} %`; $('v3-rock-val').textContent = `±${$('v3-rock').value} %`;
  $('v3-src').disabled = refolding();   // the refold fuses the frames itself
  // the 3D model: its vertex grid and a size estimate for the chosen format
  { const p = meshPlan(), m = m3(); $('m3-relief-val').textContent = `${$('m3-relief').value} %`;
    $('m3-info').textContent = `${p.nx}×${p.ny} vertices, ${p.nt >= 1e6 ? (p.nt / 1e6).toFixed(1) + ' M' : Math.round(p.nt / 1e3) + ' k'} triangles` + (m.format === 'stl' ? '' : `, texture ${p.tw}×${p.th}`) + ` · roughly ${fmtMB(p.size)} as ${m.format.toUpperCase()}`; }
  // the file list
  const list = $('sv-files'); list.innerHTML = '';
  const avail = OUTPUTS.filter((o) => o.avail());
  for (const o of OUTPUTS) {
    const ok = o.avail(), on = ok && SV.sel.has(o.id);
    const row = document.createElement('div'); row.className = 'svf' + (on ? ' on' : '') + (ok ? '' : ' off'); row.dataset.id = o.id;
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = on; cb.disabled = !ok;
    const th = document.createElement('canvas'); th.width = 192; th.height = 128;
    const meta = document.createElement('div');
    const nm = document.createElement('div'); nm.className = 'fname'; nm.textContent = ok ? svName(o) : `${o.name.toLowerCase()} — not available`; nm.title = nm.textContent;
    const ds = document.createElement('div'); ds.className = 'desc'; ds.textContent = `${o.name} · ${typeof o.desc === 'function' ? o.desc() : o.desc}`;
    meta.append(nm, ds);
    const state = document.createElement('span'); state.className = 'state'; state.textContent = ok ? (o.anim ? 'GIF' : svExt(o).toUpperCase()) : '';
    row.append(cb, th, meta, state);
    if (ok) row.addEventListener('click', (e) => { if (e.target !== cb) cb.checked = !cb.checked; if (cb.checked) SV.sel.add(o.id); else SV.sel.delete(o.id); row.classList.toggle('on', cb.checked); saveSaveSettings(); updateAnimInfo(); updateSaveButtons(); });
    list.appendChild(row);
    if (ok) thumbInto(o, th).catch(() => {});
    else if (o.id === 'dfr') nm.textContent = 'dfr — run with DFR (Run ▾) to render it';
    else if (o.id === 'anim-peak') nm.textContent = 'peaking — no peaking data for these frames';
  }
  $('sv-all').checked = avail.length > 0 && avail.every((o) => SV.sel.has(o.id));
  updateAnimInfo();
  updateSaveButtons();
}
// the animations' frame count, size and a rough GIF size for the selected ones
function updateAnimInfo() {
  const plan = animPlan();
  const anims = OUTPUTS.filter((o) => o.anim && o.avail() && SV.sel.has(o.id));
  const est = anims.reduce((b, o) => { const p = animPlan(o); return b + p.seq.length * p.ow * p.oh * (o.id === 'anim-depth' ? 0.15 : 0.7); }, 0);
  $('an-info').textContent = `${plan.seq.length} frames of ${plan.ow}×${plan.oh} (rocking: ${v3().views})` + (anims.length ? ` · roughly ${fmtMB(est)} for the ${anims.length} selected animation${anims.length > 1 ? 's' : ''}` : '') +
    (est > 1e9 ? ' — a GIF that large may exhaust the browser: use a smaller long edge or a larger frame step.' : '');
  $('an-info').classList.toggle('warn', est > 1e9);
}
function updateSaveButtons() {
  const n = OUTPUTS.filter((o) => o.avail() && SV.sel.has(o.id)).length;
  $('sv-go').textContent = n ? `Save ${n} file${n > 1 ? 's' : ''}` : 'Save'; $('sv-go').disabled = !n || SV.exporting;
  $('sv-cancel').hidden = !SV.exporting;
}
// a small preview of an output: the layer as the viewer shows it, the animations at their middle frame
async function thumbInto(o, cv) {
  const c = cv.getContext('2d'); c.fillStyle = '#111'; c.fillRect(0, 0, cv.width, cv.height);
  const r = st.result; if (!r) return;
  const mid = Math.floor((st.files.length - 1) / 2);
  const [ow, oh] = outDims();
  const fit = (bmp, pixelated = false) => {
    const s = Math.min(cv.width / ow, cv.height / oh), w = ow * s, h = oh * s;
    c.imageSmoothingEnabled = !pixelated; c.drawImage(bmp, ...srcRect(bmp.width, bmp.height), (cv.width - w) / 2, (cv.height - h) / 2, w, h);
  };
  if (o.kind === 'fused' || o.kind === 'dmap') fit(r[o.kind]);
  else if (o.kind === 'stereo' || o.rock) {
    // the views at thumbnail size: the pair side by side (the anaglyph mixed here), the rocking at one extreme
    const v = v3(), pair = o.kind === 'stereo' && v.layout !== 'anaglyph';
    const s = Math.min(cv.width / (pair ? 2 * ow : ow), cv.height / oh), tw = Math.max(1, Math.round(ow * s)), th = Math.max(1, Math.round(oh * s));
    const x0 = (cv.width - (pair ? 2 : 1) * tw) / 2, y0 = (cv.height - th) / 2;
    if (o.rock) { const f = await viewFrame(-v.rock, tw, th); c.putImageData(new ImageData(new Uint8ClampedArray(f.rgba), f.w, f.h), x0, y0); return; }
    const l = await viewFrame(-v.shift, tw, th), rr = await viewFrame(v.shift, tw, th);
    const L = new ImageData(new Uint8ClampedArray(l.rgba), l.w, l.h), R = new ImageData(new Uint8ClampedArray(rr.rgba), rr.w, rr.h);
    if (pair) { const [a, b] = v.layout === 'cross' ? [R, L] : [L, R]; c.putImageData(a, x0, y0); c.putImageData(b, x0 + tw, y0); }
    else { for (let i = 0; i < L.data.length; i += 4) { L.data[i + 1] = R.data[i + 1]; L.data[i + 2] = R.data[i + 2]; } c.putImageData(L, x0, y0); }
  }
  else if (o.kind === 'depth' || o.kind === 'depth16') fit(await depthBitmap(false), true);
  else if (o.kind === 'mesh') await meshThumb(cv, c);
  else if (o.kind === 'winner') fit(await winnerBitmap(), true);
  else if (o.id === 'anim-depth') { fit(await depthBitmap(true), true); const ov = await sliceBitmap(mid); if (ov) fit(ov, true); }
  else if (o.id === 'anim-focus') { const b = await focusBitmap(mid); if (b) fit(b); else { const f = st.frames[mid]; if (f && (f.proxy || f.thumb)) fit(f.proxy || f.thumb); } }
  else if (o.id === 'anim-peak') { const f = st.frames[mid]; if (f && (f.proxy || f.thumb)) fit(f.proxy || f.thumb); if (f && f.peak) fit(await peakBitmap(f)); }
}
// the 3D model's thumbnail: the stacked image lit as the relief would be — the depth map's slopes
// shade it, a light from the top-left; the relief and the near end are the card's
async function meshThumb(cv, c) {
  const r = st.result, [ow, oh] = outDims();
  const s = Math.min(cv.width / ow, cv.height / oh), w = Math.max(2, Math.round(ow * s)), h = Math.max(2, Math.round(oh * s));
  const x0 = Math.round((cv.width - w) / 2), y0 = Math.round((cv.height - h) / 2);
  const src = haveDmap() && v3().source === 'dmap' ? r.dmap : r.fused;
  c.imageSmoothingEnabled = true; c.drawImage(src, ...srcRect(src.width, src.height), x0, y0, w, h);
  const img = c.getImageData(x0, y0, w, h);
  const oc = new OffscreenCanvas(w, h).getContext('2d', { willReadFrequently: true });
  const d = await depthBitmap(false); oc.imageSmoothingEnabled = true; oc.drawImage(d, ...srcRect(d.width, d.height), 0, 0, w, h);
  const z = oc.getImageData(0, 0, w, h).data;
  const k = (v3().near ? -1 : 1) * m3().relief * w / 255;   // gray → height in thumbnail pixels
  const L = [-1, -1, 1.5], ln = Math.hypot(...L); L[0] /= ln; L[1] /= ln; L[2] /= ln;
  const px = img.data;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, xl = Math.max(x - 1, 0), xr = Math.min(x + 1, w - 1), yu = Math.max(y - 1, 0), yd = Math.min(y + 1, h - 1);
    const dzx = k * (z[4 * (y * w + xr)] - z[4 * (y * w + xl)]) / (xr - xl || 1), dzy = k * (z[4 * (yd * w + x)] - z[4 * (yu * w + x)]) / (yd - yu || 1);
    const nn = Math.hypot(dzx, dzy, 1), dot = (-dzx * L[0] - dzy * L[1] + L[2]) / nn;
    const shade = Math.min(1.35, 0.3 + 0.7 * Math.max(0, dot) / L[2]);
    px[4 * i] *= shade; px[4 * i + 1] *= shade; px[4 * i + 2] *= shade;
  }
  c.putImageData(img, x0, y0);
}
async function winnerBitmap() {
  if (st.depthBmp.has('winner')) return st.depthBmp.get('winner');
  const { winner: d, ww: w, wh: h } = st.result;
  let lo = Infinity, hi = -Infinity; for (const v of d) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const k = 255 / Math.max(hi - lo, 1e-6), px = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { const g = (d[i] - lo) * k; px[4 * i] = g; px[4 * i + 1] = g; px[4 * i + 2] = g; px[4 * i + 3] = 255; }
  const bmp = await createImageBitmap(new ImageData(px, w, h)); st.depthBmp.set('winner', bmp); return bmp;
}
for (const id of svIds) $(id).addEventListener(svLive.includes(id) ? 'input' : 'change', () => { $('sv-qval').textContent = $('sv-quality').value; saveSaveSettings(); renderSave(); if (id === 'sv-crop') draw(); });
document.querySelectorAll('#savecard [data-step]').forEach((b) => b.addEventListener('click', () => { const id = b.dataset.step; setStep(id, Number($(id).textContent) + Number(b.dataset.d)); saveSaveSettings(); renderSave(); }));
$('sv-all').addEventListener('change', (e) => { for (const o of OUTPUTS) if (o.avail()) { if (e.target.checked) SV.sel.add(o.id); else SV.sel.delete(o.id); } saveSaveSettings(); renderSave(); });
// a finished file: into the chosen folder when there is one, else a download
async function downloadBlob(blob, name) {
  if (window.__saveHook) { window.__saveHook(blob, name); return; }   // tests collect the files instead of downloading
  if (SV.dir) {
    try {
      const fh = await SV.dir.getFileHandle(name, { create: true }), w = await fh.createWritable();
      await w.write(blob); await w.close();
      log(`[lapstack] saved ${SV.dir.name}/${name} (${fmtMB(blob.size)})`);
      return;
    } catch (e) { log(`[lapstack] cannot write ${name} to the folder "${SV.dir.name}" (${e.message}); downloading it instead`); }
  }
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
  log(`[lapstack] saved ${name} (${fmtMB(blob.size)})`);
}
// the destination: a folder picked with the File System Access API (Chrome, Edge), for the
// batch and the Save step alike, until the page is reloaded
async function pickDir() {
  try { SV.dir = await window.showDirectoryPicker({ mode: 'readwrite' }); log(`[lapstack] saved files go to the folder "${SV.dir.name}"`); }
  catch (e) { if (e.name !== 'AbortError') toast('Cannot open that folder: ' + e.message); }
  renderDest();
}
function renderDest() {
  const name = SV.dir ? `the folder "${SV.dir.name}"` : 'downloads';
  $('p-dir-name').textContent = name; $('sv-dest').textContent = '→ ' + name;
  $('p-dir-x').hidden = !SV.dir; $('sv-dir-x').hidden = !SV.dir;
}
if (window.showDirectoryPicker) {
  for (const id of ['p-dir', 'sv-dir']) $(id).addEventListener('click', pickDir);
  for (const id of ['p-dir-x', 'sv-dir-x']) $(id).addEventListener('click', () => { SV.dir = null; renderDest(); });
} else { $('p-dir').hidden = true; $('sv-dir').hidden = true; $('p-dir-note').hidden = false; }
renderDest();
// one animation: frames composed here at the output size, encoded by the worker, streamed back
async function renderAnim(o, progress) {
  const { ow, oh, seq, delay } = animPlan(o);
  const oc = new OffscreenCanvas(ow, oh), c = oc.getContext('2d', { willReadFrequently: true });
  let refolded = false;
  if (o.rock && refolding()) {   // one pass over the frames (or a few) before any GIF frame
    const v = v3();
    await refold(seq.map((i) => rockShift(v.rock, i, v.views)), ow, oh);
    refolded = true;
  }
  await call({ type: 'gif_begin', w: ow, h: oh, loop: true, dither: true });
  const chunks = [];
  try {
    for (let k = 0; k < seq.length; k++) {
      if (SV.cancel) throw new Error('cancelled');
      progress(k, seq.length);
      await drawAnimFrame(o, seq[k], c, ow, oh);
      const img = c.getImageData(0, 0, ow, oh);
      const r = await call({ type: 'gif_frame', rgba: img.data.buffer, delay }, [img.data.buffer]);
      if (r.bytes.byteLength) chunks.push(r.bytes);
    }
    const r = await call({ type: 'gif_end' }); chunks.push(r.bytes);
  } catch (e) { await call({ type: 'gif_abort' }).catch(() => {}); throw e; }
  finally {
    if (refolded) await call({ type: 'refold_end' }).catch(() => {});
    if (o.id !== 'anim-depth' && !o.rock) { R.gpuIndex = -1; R.wasmIndex = -1; }   // the worker's source frame is whatever we exported last
    else if (refolded) R.gpuIndex = -1;   // the refold's warps went through cur[0]
  }
  return new Blob(chunks, { type: 'image/gif' });
}
async function drawAnimFrame(o, i, c, ow, oh) {
  const id = o.id;
  if (o.rock) {
    // view i of the rocking cycle: refolded (at about the output size: scaled here), or sheared by the engine at the output size
    const v = v3(); const f = refolding() ? await call({ type: 'refold_view', index: i }) : await viewFrame(rockShift(v.rock, i, v.views), ow, oh);
    const img = new ImageData(new Uint8ClampedArray(f.rgba), f.w, f.h);
    if (f.w === ow && f.h === oh) c.putImageData(img, 0, 0);
    else { const bmp = await createImageBitmap(img); c.imageSmoothingEnabled = true; c.drawImage(bmp, 0, 0, ow, oh); bmp.close(); }
    return;
  }
  if (id === 'anim-depth') {
    c.imageSmoothingEnabled = false;
    const d = await depthBitmap(true); c.drawImage(d, ...srcRect(d.width, d.height), 0, 0, ow, oh);
    const ov = await sliceBitmap(i); if (ov) c.drawImage(ov, ...srcRect(ov.width, ov.height), 0, 0, ow, oh);
    return;
  }
  // the aligned full-res frame, plain or In focus, from the engine (a decode + warp per frame)
  const bytes = await st.files[i].arrayBuffer();
  const r = await call({ type: 'export_source', index: i, bytes, focus: id === 'anim-focus' ? focusParams() : null }, [bytes]);
  const img = new ImageData(new Uint8ClampedArray(r.rgba), r.w, r.h);
  c.imageSmoothingEnabled = true;
  if (r.w === ow && r.h === oh && !cropArea()) c.putImageData(img, 0, 0);
  else { const bmp = await createImageBitmap(img); c.drawImage(bmp, ...srcRect(r.w, r.h), 0, 0, ow, oh); bmp.close(); }
  if (id === 'anim-peak') { const f = st.frames[i]; if (f && f.peak) { const pk = await peakBitmap(f); c.drawImage(pk, ...srcRect(pk.width, pk.height), 0, 0, ow, oh); } }
}
// content credentials: a self-signed certificate per signer name, kept in this browser
const CC_KEY = 'lapstack.cc';
async function credentials() {
  const name = ($('cc-name').value || '').trim() || 'lapstack user';
  let cc = null; try { cc = JSON.parse(localStorage.getItem(CC_KEY)); } catch {}
  if (!cc || cc.name !== name || !cc.cert || !cc.key) {
    const r = await call({ type: 'make_cert', name });
    cc = { name, cert: r.cert, key: r.key };
    try { localStorage.setItem(CC_KEY, JSON.stringify(cc)); } catch {}
    log(`[lapstack] content credentials: new self-signed certificate for "${name}"`);
  }
  return cc;
}
const CC_SIGN_MAX = 512 * 1024 * 1024;   // the whole file passes through wasm memory to be signed
async function signBlob(blob, o, name, mime) {
  if (blob.size > CC_SIGN_MAX) { log(`[lapstack] ${name}: too large to sign in the browser (${fmtMB(blob.size)}), saved without content credentials`); return blob; }
  const cc = await credentials();
  const p = readParams();
  const manifest = {
    claim_generator_info: [{ name: 'lapstack', version: '0.1.0' }],
    title: name,
    assertions: [
      { label: 'c2pa.actions', data: { actions: [{ action: 'c2pa.created', digitalSourceType: 'http://cv.iptc.org/newscodes/digitalsourcetype/compositeCapture', softwareAgent: { name: 'lapstack', version: '0.1.0' } }] } },
      { label: 'org.lapstack.stack', data: { output: o.token, frames: st.files.map((f) => f.name), align: p.align, levels: p.levels, energy_radius: p.energy_radius, top: p.top, top_radius: p.top_radius, use_chroma: p.use_chroma, brightness: p.brightness, retouch_strokes: R.undo, crop: cropArea() ? [cropArea().x, cropArea().y, cropArea().w, cropArea().h] : null } },
    ],
  };
  const bytes = await blob.arrayBuffer();
  const r = await call({ type: 'sign', bytes, mime, manifest: JSON.stringify(manifest), cert: cc.cert, key: cc.key }, [bytes]);
  return new Blob([r.bytes], { type: mime });
}
async function saveSelected() {
  if (SV.exporting || !st.result || st.running) return 'skipped';
  const items = OUTPUTS.filter((o) => o.avail() && SV.sel.has(o.id));
  if (!items.length) return 'skipped';
  const saved = []; SV.lastSaved = saved;
  SV.exporting = true; SV.cancel = false; updateSaveButtons(); $('run').disabled = true; $('clear').disabled = true;
  const now = new Date(), fmt = $('sv-format').value, q = Number($('sv-quality').value), sign = $('cc-on').checked;
  const setState = (o, t) => { const row = $('sv-files').querySelector(`[data-id="${o.id}"] .state`); if (row) row.textContent = t; };
  const prog = (t, done, total) => { $('sv-progress').textContent = t; setProgress(t, done, total); };
  let status = 'saved';
  try {
    for (const o of items) {
      const name = svName(o, now);
      let blob, mime;
      if (o.kind === 'mesh') {
        // the 3D model: one file, or three for OBJ, from the engine; nothing to sign
        prog(`building ${name}`, 0, 0); setState(o, 'building…');
        const m = m3(), v = v3();
        const r = await call({ type: 'mesh', stem: name.slice(0, name.lastIndexOf('.')), format: m.format, source: v.source, crop: v.crop, grid: m.grid, relief: m.relief, near: v.near, texture_edge: m.tex, texture: fmt === 'jpeg' ? 'jpeg' : 'png', quality: q });
        let total = 0;
        for (const f of r.files) { const b = new Blob([f.bytes], { type: MESH_MIME[f.name.slice(f.name.lastIndexOf('.') + 1)] || 'application/octet-stream' }); await downloadBlob(b, f.name); saved.push(f.name); total += b.size; }
        setState(o, `saved · ${fmtMB(total)}${r.files.length > 1 ? ` in ${r.files.length} files` : ''}`);
        continue;
      }
      if (o.anim) {
        setState(o, 'rendering…');
        blob = await renderAnim(o, (k, n) => { prog(`${o.name}: frame ${k + 1}/${n}`, k, n); setState(o, `${k + 1}/${n}`); });
        mime = 'image/gif';
      } else {
        prog(`encoding ${name}`, 0, 0); setState(o, 'encoding…');
        const f = o.ext ? 'png' : fmt, meta = $('sv-meta').checked && !$('sv-meta').disabled;
        let r;
        if (o.kind === 'stereo' && refolding()) {   // the two views folded from the frames at full resolution, then composed
          const v = v3(); setState(o, 'refolding…');
          await refold([-v.shift, v.shift], 0, 0);
          try { r = await call({ type: 'refold_stereo', layout: v.layout, format: f, quality: q, meta }); }
          finally { await call({ type: 'refold_end' }).catch(() => {}); R.gpuIndex = -1; }
          prog(`encoding ${name}`, 0, 0); setState(o, 'encoding…');
        } else if (o.kind === 'stereo') r = await call({ type: 'view_stereo', ...v3(), format: f, quality: q, meta });   // the pair at the crop's full size
        else r = await call({ type: 'save', kind: o.kind, format: f, quality: q, meta, crop: !!cropArea() });
        mime = f === 'jpeg' ? 'image/jpeg' : 'image/png'; blob = new Blob([r.bytes], { type: mime });
      }
      if (sign) { prog(`signing ${name}`, 0, 0); setState(o, 'signing…'); blob = await signBlob(blob, o, name, mime); }
      await downloadBlob(blob, name); saved.push(name); setState(o, `saved · ${fmtMB(blob.size)}`);
    }
  } catch (e) {
    status = SV.cancel ? 'cancelled' : 'error';
    if (!SV.cancel) { log('[lapstack] save failed: ' + e.message); toast('Save failed: ' + e.message, 0); }
    else log('[lapstack] save cancelled');
  }
  SV.exporting = false; $('run').disabled = !st.files.length; $('clear').disabled = false;
  $('progress').className = status === 'saved' ? 'done' : 'error'; $('fill').style.width = '0'; setStatus(status);
  $('sv-progress').textContent = ''; updateSaveButtons();
  return status;
}
$('sv-go').addEventListener('click', saveSelected);
$('sv-cancel').addEventListener('click', () => { SV.cancel = true; worker.postMessage({ type: 'refold_cancel' }); });

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
// A mode over the Stack layers, not a step: while it is on, the view is a side-by-side
// compare of the shown stacked image (LAP or DFR, the paint target) and the scrubbed
// Source frame, and the brush copies the aligned source frame into the target: a live
// preview is composited on the display canvas while dragging, then the worker applies
// the stroke to the 16-bit master and sends back the exact bbox, which replaces the preview.
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
function resetRetouch() { srcClear(); slabDrop(); R.slabLoading = null; R.slabProgress = null; R.slabGen++; R.slabGenMin = R.slabGen; R.on = false; R.prev = null; R.cursor = null; R.hold = false; R.wasmIndex = -1; R.gpuIndex = -1; R.loading = -1; R.prefetch = -1; R.ahead = null; R.gen++; R.genMin = R.gen; R.undo = 0; R.redo = 0; R.painting = false; R.dabs = []; }
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
  const forPaint = R.on && brushFrom() === 'source';           // strokes copy from the worker's own 16-bit copy
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
// ---------- the slab brush source ----------
// Zerene's slabs, made on demand: the frames within ±(slab ± frames) of the scrubbed
// one, fused on their own by the worker with the run's registration, gains and fusion
// settings, so the brush copies a thick plane of focus rather than one frame's sliver.
// One slab is held at a time (a full-res canvas here, the 16-bit master in the worker);
// scrubbing moves the range, and the rebuild waits until the scrub settles, since each
// costs a decode per frame. A build the range has moved away from is dropped between
// frames (the worker compares generations).
const slabHalf = () => Number($('p-slab').textContent);
const slabWanted = () => { const h = slabHalf(); return [Math.max(0, st.selected - h), Math.min(st.files.length - 1, st.selected + h)]; };
const slabIs = (s, lo, hi) => !!s && s.lo === lo && s.hi === hi;
const slabReady = () => { const [lo, hi] = slabWanted(); return slabIs(R.slab, lo, hi); };
function slabDrop() { if (R.slab) R.slab.canvas.width = 1; R.slab = null; }
let slabTimer = null;
const slabNeeded = () => R.on && brushFrom() === 'slab' && !!st.result && st.files.length > 0 && !st.running;
function ensureSlab() {
  if (!slabNeeded()) return;
  const [lo, hi] = slabWanted();
  if (slabIs(R.slab, lo, hi) || slabIs(R.slabLoading, lo, hi)) return;
  clearTimeout(slabTimer);
  slabTimer = setTimeout(() => {
    if (!slabNeeded()) return;
    const [lo, hi] = slabWanted();
    if (slabIs(R.slab, lo, hi) || slabIs(R.slabLoading, lo, hi)) return;
    const gen = ++R.slabGen;
    R.slabLoading = { lo, hi, gen }; R.slabProgress = null;
    R.gpuIndex = -1;   // the fold takes the worker's GPU-resident source frame with it
    worker.postMessage({ type: 'slab', lo, hi, files: st.files.slice(lo, hi + 1), gen });
    log(`[lapstack] slab ${lo}..${hi}: fusing ${hi - lo + 1} frames`);
    updateTabs(); draw();
  }, 300);
}
function onSlab(m) {
  if (R.slabLoading && R.slabLoading.gen === m.gen) R.slabLoading = null;
  R.slabProgress = null;
  if (!st.result || m.gen < R.slabGenMin) return;   // from a run that is gone
  slabDrop();
  const cv = new OffscreenCanvas(m.w, m.h);
  cv.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h), 0, 0);
  R.slab = { lo: m.lo, hi: m.hi, canvas: cv };
  log(`[lapstack] slab ${m.lo}..${m.hi} ready`);
  ensureSlab();   // the range scrubbed to meanwhile
  updateTabs(); renderFilmstrip(); draw();
}
function onSlabSkipped(m) { if (R.slabLoading && R.slabLoading.gen === m.gen) R.slabLoading = null; R.slabProgress = null; ensureSlab(); updateTabs(); draw(); }
function onSlabProgress(m) { if (R.slabLoading && R.slabLoading.gen === m.gen) { R.slabProgress = { done: m.done, total: m.total }; draw(); } }
// the right pane's label for the slab: the range wanted, and how far its build is
function slabLabel() {
  if (!R.on) return R.slab ? `Slab ${R.slab.lo + 1}–${R.slab.hi + 1}` : 'Slab';
  const [lo, hi] = slabWanted(), name = `Slab ${lo + 1}–${hi + 1}`;
  if (slabIs(R.slab, lo, hi)) return name;
  const p = R.slabProgress;
  return `${name} — ${slabIs(R.slabLoading, lo, hi) ? (p ? `fusing ${p.done}/${p.total}…` : 'fusing…') : 'waiting…'}`;
}
// the paint target is the stacked image on screen: DFR when it is the view, else LAP
const target = () => (st.view === 'dmap' && haveDmap()) ? 'dmap' : 'fused';
const targetCanvas = () => st.result && st.result[target()];
// The brush source: the scrubbed frame ('source'), the slab around it ('slab'), or the
// other stacked result — DFR while LAP is painted, LAP while DFR is — the way Zerene
// brushes PMax detail into a DMap. That one needs both results, so without a depth-map
// rendering the choice falls back to the frame. The source is what the right pane shows
// (updateTabs pins st.cmp to it).
const otherResult = () => (target() === 'dmap' ? 'fused' : 'dmap');
const brushFrom = () => (R.from === 'slab' ? 'slab' : R.from === 'stack' && haveDmap() ? otherResult() : 'source');
const brushCanvas = () => (brushFrom() === 'source' ? srcGet(st.selected) : brushFrom() === 'slab' ? (slabReady() ? R.slab.canvas : null) : st.result && st.result[brushFrom()]);
const brushReady = () => (brushFrom() === 'source' ? R.wasmIndex === st.selected && !!srcGet(st.selected) : !!brushCanvas());
function setBrushFrom(from) {
  R.from = ['stack', 'slab'].includes(from) ? from : 'source';
  saveParams(); updateTabs(); renderFilmstrip(); draw();
}
$('bs-frame').addEventListener('click', () => setBrushFrom('source'));
$('bs-stack').addEventListener('click', () => setBrushFrom('stack'));
$('bs-slab').addEventListener('click', () => setBrushFrom('slab'));
function onPatch(m) {
  R.undo = m.undo; R.redo = m.redo; updateTabs();
  if (!m.rgba || !st.result) return;
  R.hold = true;   // an undo under the cursor has to be visible: the hover preview, which would paint the same pixels straight back over it, waits for the next pointer move
  const cv = m.target === 'dmap' ? st.result.dmap : st.result.fused; if (!cv) return;
  cv.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h), m.x, m.y);
  draw();
}
const dabCv = new OffscreenCanvas(16, 16);
// One dab, built into dabCv: the brush source (the aligned frame, or the other stacked
// result) under the brush, masked by the
// brush falloff — the engine's smoothstep from hardness*r to r, so what a preview shows
// and what the worker later paints have the same edge. Two knobs keep a preview's cost
// off the brush's size: `scale` is dab canvas px per image px (1 for the dabs of a
// stroke, which land in the master's display copy at its own resolution; the on-screen
// scale for the hover preview), and `clip` is an image-space [x0, y0, x1, y1] the dab is
// cut to (the visible part of the pane). Returns the dab's box in image px, or null when
// the source frame is not on hand or nothing of the dab is left.
function makeDab(x, y, scale = 1, clip = null) {
  const src = brushCanvas(); if (!src || !st.result) return null;
  const r = R.size;
  let bx = x - r - 1, by = y - r - 1, bw = 2 * r + 2, bh = 2 * r + 2;
  if (clip) {
    const x1 = Math.min(bx + bw, clip[2]), y1 = Math.min(by + bh, clip[3]);
    bx = Math.max(bx, clip[0]); by = Math.max(by, clip[1]); bw = x1 - bx; bh = y1 - by;
    if (bw <= 0 || bh <= 0) return null;
  }
  const dw = Math.max(2, Math.ceil(bw * scale)), dh = Math.max(2, Math.ceil(bh * scale));
  if (dabCv.width !== dw || dabCv.height !== dh) { dabCv.width = dw; dabCv.height = dh; }
  const c = dabCv.getContext('2d');
  c.setTransform(dw / bw, 0, 0, dh / bh, -bx * dw / bw, -by * dh / bh);   // image px, whatever the dab's own resolution
  c.globalCompositeOperation = 'source-over'; c.clearRect(bx, by, bw, bh);
  c.drawImage(src, bx, by, bw, bh, bx, by, bw, bh);
  const g = c.createRadialGradient(x, y, r * R.hard, x, y, r);
  for (let i = 0; i <= 8; i++) { const u = i / 8; g.addColorStop(u, `rgba(0,0,0,${1 - u * u * (3 - 2 * u)})`); }
  c.globalCompositeOperation = 'destination-in'; c.fillStyle = g; c.fillRect(bx, by, bw, bh);
  return [bx, by, bw, bh];
}
// a dab of the stroke under the pointer: onto the display copy of the master, where it
// stands in for the stroke until the worker's exact patch replaces it
function previewDab(x, y) {
  const b = makeDab(x, y);
  if (b) targetCanvas().getContext('2d').drawImage(dabCv, b[0], b[1], b[2], b[3]);
}
// The hover preview: the dab a click here would lay down, composited onto the paint
// pane's canvas alone. Neither the master nor its display copy is touched, so it follows
// the cursor, and leaves nothing behind when the cursor moves on or off the pane. It is
// built at the pane's own scale and cut to the pane, so a brush wider than the window
// costs what the window costs, not what the brush does.
function hoverDab(c, d) {
  if (!R.cursor || R.hold || shiftHeld) return;
  const [x, y] = R.cursor, [w, h] = imageDims();
  const clip = [Math.max(0, -st.ox / st.zoom), Math.max(0, -st.oy / st.zoom),
                Math.min(w, (canvas.clientWidth - st.ox) / st.zoom), Math.min(h, (canvas.clientHeight - st.oy) / st.zoom)];
  const b = makeDab(x, y, Math.min(1, st.zoom * d), clip);
  if (b) c.drawImage(dabCv, b[0], b[1], b[2], b[3]);
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
  if (R.dabs.length) worker.postMessage({ type: 'stroke', dabs: new Float32Array(R.dabs), target: target(), from: brushFrom() });
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
  saveParams(); drawSoon();   // the wheel and the sliders both run ahead of the frame rate: one redraw a frame
}
$('br-size-in').addEventListener('input', (e) => setBrush(sizeFromSlider(Number(e.target.value)), R.hard));
$('br-hard-in').addEventListener('input', (e) => setBrush(R.size, Number(e.target.value) / 100));
$('undo').addEventListener('click', () => worker.postMessage({ type: 'undo' })); $('redo').addEventListener('click', () => worker.postMessage({ type: 'redo' }));
// Enter: make sure a stack layer is the view and put Source beside it, side by side;
// the compare state it replaces comes back on exit. Leaving the Stack group, the stack
// step, or losing the result all end the mode (updateTabs).
function enterRetouch() {
  if (R.on || !st.result || st.step !== 'stack') return;
  if (groupOf(st.view) !== 'stack') { setView(st.compare && groupOf(st.cmp) === 'stack' ? st.cmp : lastIn.stack); }
  R.prev = { compare: st.compare, cmpMode: st.cmpMode, cmp: st.cmp };
  R.on = true; st.compare = true; st.cmpMode = 'split'; st.cmp = brushFrom(); st.flipped = false;
  setBrush(R.size, R.hard);
  updateTabs(); renderFilmstrip(); draw();
}
// state only (no redraw): updateTabs and gotoStep call this mid-refresh
function leaveRetouch(restore = true) {
  if (!R.on) return;
  endStroke(); R.on = false; R.cursor = null; R.hold = false;
  if (R.prev) { st.cmpMode = R.prev.cmpMode; if (restore && st.step === 'stack') { st.compare = R.prev.compare; st.cmp = R.prev.cmp; } }
  R.prev = null;
}
function exitRetouch() { if (!R.on) return; leaveRetouch(); updateTabs(); renderFilmstrip(); draw(); }
const toggleRetouch = () => (R.on ? exitRetouch() : enterRetouch());
$('retouch').addEventListener('click', toggleRetouch);

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
  if (tab === 'slab') {
    // the slab held (its range may lag the scrub: the label says so), else the scrubbed frame's proxy
    if (!st.result) return null;
    if (R.slab) return { bmp: R.slab.canvas, w: st.result.w, h: st.result.h };
    const f = st.frames[st.selected]; const bmp = f && (f.proxy || f.thumb);
    return bmp ? { bmp, w: st.result.w, h: st.result.h } : null;
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
// The crop window over a drawn layer: the border outside it dimmed, a hairline on its
// edge — in image space, so it sits on the same pixels in every pane and at every zoom.
function drawCrop(c = ctx) {
  const r = cropArea(); if (!r) return;
  const [W, H] = imageDims();
  c.fillStyle = 'rgba(0,0,0,.55)';
  c.fillRect(0, 0, W, r.y); c.fillRect(0, r.y + r.h, W, H - r.y - r.h);
  c.fillRect(0, r.y, r.x, r.h); c.fillRect(r.x + r.w, r.y, W - r.x - r.w, r.h);
  const lw = 1 / (st.zoom * dpr());
  c.lineWidth = lw; c.strokeStyle = 'rgba(255,255,255,.75)'; c.strokeRect(r.x - lw / 2, r.y - lw / 2, r.w + lw, r.h + lw);
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
  if (!R.cursor || shiftHeld) return;
  c.setTransform(st.zoom * d, 0, 0, st.zoom * d, st.ox * d, st.oy * d);
  c.lineWidth = 1.5 / (st.zoom * d); c.strokeStyle = 'rgba(255,255,255,.9)'; c.beginPath(); c.arc(R.cursor[0], R.cursor[1], R.size, 0, 2 * Math.PI); c.stroke();
  c.strokeStyle = 'rgba(0,0,0,.6)'; c.beginPath(); c.arc(R.cursor[0], R.cursor[1], R.size * R.hard, 0, 2 * Math.PI); c.stroke();
}
// The marks below stand in for the pointer's own cursor, which only ever lands on the pane
// under the hand. Retouch's two panes hold the same image under the same transform, so a
// mark drawn at the pointer in image space appears in both, over the same pixel: what is
// about to happen is shown where it will happen, in the Source pane as much as the target.
// ctrl+G: the cross marks the pixel whose sharpest frame the next click jumps to — the
// frame the Source pane is about to show.
function drawPick(c, d) {
  if (!R.cursor) return;
  const k = st.zoom * d, [x, y] = R.cursor, arm = 14 / k, gap = 4 / k;
  c.setTransform(k, 0, 0, k, st.ox * d, st.oy * d);
  c.beginPath();
  c.moveTo(x - arm, y); c.lineTo(x - gap, y); c.moveTo(x + gap, y); c.lineTo(x + arm, y);
  c.moveTo(x, y - arm); c.lineTo(x, y - gap); c.moveTo(x, y + gap); c.lineTo(x, y + arm);
  strokeMark(c, k);
}
// shift: the drag pans both panes at once (it suspends the brush), so the four arrows that
// say so belong in both — the move cursor they stand in for is in one.
function drawPan(c, d) {
  if (!R.cursor) return;
  const k = st.zoom * d, [x, y] = R.cursor, arm = 15 / k, gap = 4 / k, head = 4.5 / k;
  c.setTransform(k, 0, 0, k, st.ox * d, st.oy * d);
  c.beginPath();
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    const px = dx * arm, py = dy * arm;
    c.moveTo(x + dx * gap, y + dy * gap); c.lineTo(x + px, y + py);   // the arm, clear of the pixel itself
    c.moveTo(x + px - (dx - dy) * head, y + py - (dy + dx) * head);   // its head: one barb, the tip, the other
    c.lineTo(x + px, y + py);
    c.lineTo(x + px - (dx + dy) * head, y + py - (dy - dx) * head);
  }
  strokeMark(c, k);
}
// ctrl: the wheel stops scrubbing and zooms about the pointer, so a lens marks the point
// the zoom will hold still. It joins the brush ring rather than replacing it: ctrl changes
// what the wheel does, not what a click does — a click still paints.
function drawZoom(c, d) {
  if (!R.cursor) return;
  const k = st.zoom * d, [x, y] = R.cursor, r = 8 / k, t = 3.5 / k, h = 5 / k, q = Math.SQRT1_2;
  c.setTransform(k, 0, 0, k, st.ox * d, st.oy * d);
  c.beginPath();
  c.arc(x, y, r, 0, 2 * Math.PI);
  c.moveTo(x - t, y); c.lineTo(x + t, y); c.moveTo(x, y - t); c.lineTo(x, y + t);   // the + of a zoom-in lens
  c.moveTo(x + r * q, y + r * q); c.lineTo(x + (r + h) * q, y + (r + h) * q);       // its handle
  strokeMark(c, k);
}
// every mark twice: a dark liner under a light line, so it reads on any image
function strokeMark(c, k) {
  c.lineJoin = 'round'; c.lineCap = 'round';
  c.lineWidth = 3.5 / k; c.strokeStyle = 'rgba(0,0,0,.55)'; c.stroke();
  c.lineWidth = 1.5 / k; c.strokeStyle = 'rgba(255,255,255,.95)'; c.stroke();
  c.lineJoin = 'miter'; c.lineCap = 'butt';
}
// What the pointer is about to do, drawn into one pane: a pick jumps, shift pans, ctrl
// zooms, and otherwise the brush paints — the hover preview only on the pane it would
// land on, the rest in both.
function paintMarks(c, d, paintPane) {
  if (st.pick) { drawPick(c, d); return; }
  if (shiftHeld) { drawPan(c, d); return; }
  if (paintPane) hoverDab(c, d);
  drawCursor(c, d);
  if (ctrlHeld) drawZoom(c, d);
}
// Pane labels: block letters over every visible image. 'a' is the view layer, 'b' the
// compare partner, plain = neutral (the retouch source). Positions are inline so one
// element serves the pane centres, the swipe divider and the centred single view; a
// " — hint" suffix in the text becomes a smaller line under the name.
function showLabel(el, text, kind, pos) {
  el.hidden = !text;
  if (!text) return;
  const [name, hint] = text.split(' — ');
  el.textContent = name;
  if (hint) { const h = document.createElement('span'); h.className = 'hint'; h.textContent = hint; el.append(h); }
  // letters on the divider line up against it; everything else centres on its anchor
  el.className = 'plab' + (kind ? ' ' + kind : '') + (pos.right ? ' r' : pos.left && !pos.transform ? ' l' : '');
  // the box and letters take their layer's group colour (kind 'a' = the view, 'b' = the compare partner)
  const g = kind === 'a' ? groupOf(st.view) : kind === 'b' ? groupOf(st.cmp) : null;
  if (g) el.dataset.group = g; else delete el.dataset.group;
  el.style.left = pos.left || 'auto';
  el.style.right = pos.right || 'auto';
  el.style.transform = pos.transform || 'none';
}
function draw() {
  const d = dpr();
  const retouch = R.on && !!st.result;
  const split = st.compare && st.cmpMode === 'split' && !!st.result;
  $('vwrap').classList.toggle('split', split); $('vwrap').classList.toggle('paint', retouch); canvas2.hidden = !split;
  // retouch draws its own marks into both panes, so the native cursor stands down for them
  // and the two panes read the same — but only once a drawn mark has a place to be, so the
  // pointer is never left with no cursor at all
  $('vwrap').classList.toggle('marks', retouch && !!R.cursor);
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
  $('panelabels').style.top = `${Math.max(hdr, Math.min(st.oy, ch - 64))}px`;
  if (split) {
    // two panes, one transform: view | partner (retouch: paint target | Source, never flipped)
    const A = layerFor(st.view), B = layerFor(st.cmp);
    const [L, Rt] = st.flipped ? [B, A] : [A, B];
    const labels = st.flipped ? [layerLabel(st.cmp), layerLabel(st.view)] : [layerLabel(st.view), layerLabel(st.cmp)];
    if (retouch) { labels[0] += ' — drag to paint, shift+drag pans'; if (!labels[1].includes(' — ')) labels[1] += usesFrame(st.cmp) ? ' — brush source, wheel scrubs' : ' — brush source'; }   // a loading hint keeps its line
    drawLayer(L); drawCrop(ctx); if (retouch) paintMarks(ctx, d, true);   // the paint pane: the hover preview lands here
    ctx2.setTransform(1, 0, 0, 1, 0, 0); ctx2.fillStyle = '#141416'; ctx2.fillRect(0, 0, canvas2.width, canvas2.height);
    ctx2.setTransform(st.zoom * d, 0, 0, st.zoom * d, st.ox * d, st.oy * d);
    drawLayer(Rt, ctx2); drawCrop(ctx2); if (retouch) paintMarks(ctx2, d, false);
    const [k1, k2] = st.flipped ? ['b', 'a'] : ['a', 'b'];
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
    drawCrop();
    $('divider').hidden = false; $('divider').style.left = `${st.divider * cw - 1}px`;
    // the two chips ride the divider, one on each side; each drops out as its side closes
    const x = st.divider * cw;
    const [n1, n2] = st.flipped ? [layerLabel(st.cmp), layerLabel(st.view)] : [layerLabel(st.view), layerLabel(st.cmp)];
    const [k1, k2] = st.flipped ? ['b', 'a'] : ['a', 'b'];
    const [l1, l2] = $('panelabels').children; $('panelabels').hidden = false;
    showLabel(l1, x > 40 ? n1 : '', k1, { right: `${Math.round(cw - x + 8)}px` });
    showLabel(l2, cw - x > 40 ? n2 : '', k2, { left: `${Math.round(x + 8)}px` });
  } else {
    drawLayer(A); drawCrop(); $('divider').hidden = true;
    const [l1, l2] = $('panelabels').children; $('panelabels').hidden = false;
    showLabel(l1, layerLabel(st.view), 'a', { left: '50%', transform: 'translateX(-50%)' });
    showLabel(l2, '', '', {});
  }
  $('zoom').textContent = `${(st.zoom * d * 100).toFixed(0)}%`;
}
// Pointer moves arrive faster than frames, and each one now repaints a brush preview:
// coalesce them, one redraw per frame.
let drawReq = 0;
function drawSoon() { if (drawReq) return; drawReq = requestAnimationFrame(() => { drawReq = 0; draw(); }); }
new ResizeObserver(() => { layoutScrub(); draw(); }).observe($('vwrap'));
// One wheel rule for both panes: plain wheel scrubs whenever any visible layer
// depends on a frame (shift = 10 frames); ctrl/cmd+wheel always zooms at the
// cursor; plain wheel zooms only when nothing on screen is scrubbable. In
// retouch mode alt takes the wheel first, for the brush.
function onWheel(cv, e) {
  e.preventDefault();
  // alt+wheel resizes the brush under the cursor (alt+shift: its hardness), so the size is set
  // where it is about to be used, against the image, without the hand leaving the mouse. Alt is
  // the one modifier the wheel had left, and the ring redraws with every notch. Size scales
  // geometrically, like the zoom and like [ / ], but never by less than a pixel: 20 % of a small
  // brush rounds back to the size it started at, and the wheel would do nothing.
  if (R.on && e.altKey) {
    const f = Math.pow(1.0015, -e.deltaY);
    if (e.shiftKey) setBrush(R.size, R.hard - e.deltaY * 0.0004);
    else setBrush(f > 1 ? Math.max(R.size + 1, R.size * f) : Math.min(R.size - 1, R.size * f), R.hard);
    return;
  }
  // retouch is no exception — the Source pane is scrubbable, so the wheel picks the
  // frame to paint from — except mid-stroke, where changing the source under the brush
  // would be nobody's intent: there the wheel keeps zooming. With the other result as
  // the brush source nothing on screen scrubs, and the wheel zooms.
  if (scrubbable() && !R.painting && !(e.ctrlKey || e.metaKey) && st.files.length > 1) { scrub((e.deltaY > 0 ? 1 : -1) * (e.shiftKey ? 10 : 1)); return; }
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
  if (R.on) draw();   // retouch draws its own cross in both panes: arming and cancelling both change it
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
// Shift means pan: it suspends the brush (pointerdown below) and pans instead, so while it is
// held the pane shows the move cursor in place of grab — and in retouch mode in place of the
// brush's cursor: none, with the brush ring and its hover preview giving way to the four
// arrows of drawPan, in both panes, since the pan moves both. Pointer events carry the state
// too: the key may have gone down while another window had focus, and it may come up there
// (blur).
let shiftHeld = false;
function setShift(on) {
  if (on === shiftHeld || (on && R.painting)) return;   // mid-stroke shift pans nothing: the brush keeps its ring
  shiftHeld = on;
  $('vwrap').classList.toggle('shift', on);
  if (R.on) draw();
}
// Ctrl (cmd on a Mac) means zoom: it takes the wheel off the scrub and zooms about the
// pointer. Nothing about the pointer itself changes, so the pane only says so — the zoom-in
// cursor everywhere, and in retouch the lens of drawZoom beside the brush ring in both panes.
let ctrlHeld = false;
function setCtrl(on) {
  if (on === ctrlHeld) return;
  ctrlHeld = on;
  $('vwrap').classList.toggle('zoom', on);
  if (R.on) draw();
}
const setMods = (e) => { setShift(e.shiftKey); setCtrl(e.ctrlKey || e.metaKey); };
addEventListener('keydown', setMods);
addEventListener('keyup', setMods);
addEventListener('blur', () => { setShift(false); setCtrl(false); });

for (const cv of [canvas, canvas2]) {
  cv.addEventListener('pointerdown', (e) => {
    setMods(e);
    if (st.pick && e.button === 0) { e.preventDefault(); pickAt(cv, e); return; }
    const paint = R.on && st.result && e.button === 0 && !e.shiftKey && !(e.buttons & 4);
    try { cv.setPointerCapture(e.pointerId); } catch {}
    if (paint) {
      if (!brushReady()) { toast(brushFrom() === 'slab' ? 'The slab is still being fused — wait for it before painting.' : 'Source frame still loading — wait for "loaded" before painting.', 3000); return; }
      R.painting = true; R.dabs = []; R.last = null; const [x, y] = imgXY(cv, e); addDab(x, y); draw(); return;
    }
    drag = { x: e.clientX, y: e.clientY, ox: st.ox, oy: st.oy }; cv.classList.add('drag');
  });
  cv.addEventListener('pointermove', (e) => {
    setMods(e);
    if (R.on) { R.cursor = imgXY(cv, e); R.hold = false; }
    if (R.painting) { const [x, y] = imgXY(cv, e); addDab(x, y); drawSoon(); return; }
    if (drag) { st.ox = drag.ox + e.clientX - drag.x; st.oy = drag.oy + e.clientY - drag.y; st.fitted = false; }
    if (drag || R.on) drawSoon();
  });
  cv.addEventListener('pointerup', () => { endStroke(); drag = null; cv.classList.remove('drag'); });
  cv.addEventListener('pointerleave', () => { if (R.on) { R.cursor = null; draw(); } });
  cv.addEventListener('dblclick', () => (st.fitted ? zoom100() : fit()));
  cv.addEventListener('wheel', (e) => onWheel(cv, e), { passive: false });
}
let ddrag = false;
$('divider').addEventListener('pointerdown', (e) => { ddrag = true; $('divider').setPointerCapture(e.pointerId); e.stopPropagation(); });
$('divider').addEventListener('pointermove', (e) => { if (!ddrag) return; const r = canvas.getBoundingClientRect(); st.divider = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); draw(); });
$('divider').addEventListener('pointerup', () => { ddrag = false; });
$('fit').addEventListener('click', fit); $('z100').addEventListener('click', zoom100);

// ---------- header: view / context / compare / scrub ----------
const LAYERS = [['fused', 'LAP'], ['dmap', 'DFR'], ['depth', 'Focus depth'], ['focus', 'In focus'], ['source', 'Source'], ['slab', 'Slab']];   // slab: the retouch brush source, once one has been fused
// Header groups: Source | Stack (LAP, DFR) | Depth (Focus depth, In focus). The sub-control
// lists the group's layers and is hidden when the group has only one.
const GROUPS = { source: ['source'], stack: ['fused', 'dmap'], depth: ['depth', 'focus'] };
const groupOf = (v) => Object.keys(GROUPS).find((g) => GROUPS[g].includes(v)) || null;
const lastIn = { stack: 'fused', depth: 'depth' };   // last layer picked in each group
const layerName = (id) => (LAYERS.find((l) => l[0] === id) || [id, id])[1];
// while the full-res frame decodes, say so: the pane is showing the proxy
const layerLabel = (id) => id === 'slab' ? slabLabel() : (usesSource(id) && st.result && (!srcCache.has(id === 'focus' ? `focus:${st.selected}` : st.selected) || (R.on && id === 'source' && R.wasmIndex !== st.selected)))
  ? `${layerName(id)} — loading full res…` : layerName(id);
const haveDmap = () => !!(st.result && st.result.dmap);
// layers that depend on the scrubbed frame
const usesFrame = (t) => usesSource(t) || (t === 'slab' && R.on) || (isDepthLayer(t) && st.slice);   // the slab follows the scrub while it is the brush source
function scrubbable() { return usesFrame(st.view) || (st.compare && usesFrame(st.cmp)); }
function updateTabs() {
  const have = !!st.result;
  $('tab-source').disabled = !st.files.length; $('tab-stack').disabled = !have; $('tab-depth').disabled = !have; $('ab').disabled = !have || st.step !== 'stack';
  if (!haveDmap() && st.view === 'dmap') st.view = 'fused';
  if (!haveDmap() && lastIn.stack === 'dmap') lastIn.stack = 'fused';
  $('cm-swipe').classList.toggle('on', st.cmpMode !== 'split'); $('cm-split').classList.toggle('on', st.cmpMode === 'split');
  document.querySelectorAll('#steps button').forEach((b) => { if (b.dataset.step !== 'stack') b.disabled = !have; });
  const group = groupOf(st.view);
  // retouch mode ends when its target leaves the screen; while it is on, the split is pinned to target | brush source
  if (R.on && (!have || st.step !== 'stack' || group !== 'stack')) leaveRetouch();
  const retouch = R.on;
  if (retouch) { st.compare = true; st.cmpMode = 'split'; st.cmp = brushFrom(); st.flipped = false; }
  $('viewseg').hidden = st.step !== 'stack';
  $('brush').hidden = !retouch;
  // the brush source picker: the frame, or the other result (only once there are two)
  $('bs-stack').hidden = !haveDmap(); $('bs-stack').textContent = layerName(otherResult());
  $('bs-stack').title = `paint the ${layerName(otherResult())} image into ${layerName(target())} (S)`;
  const from = brushFrom(), fromStack = from === 'fused' || from === 'dmap';
  $('bs-frame').classList.toggle('on', from === 'source'); $('bs-stack').classList.toggle('on', fromStack); $('bs-slab').classList.toggle('on', from === 'slab');
  $('bs-frame-hint').hidden = from !== 'source'; $('bs-stack-hint').hidden = !fromStack; $('bs-slab-hint').hidden = from !== 'slab'; $('bs-slab-ctl').hidden = from !== 'slab';
  $('undo').disabled = !R.undo; $('redo').disabled = !R.redo; $('hist').textContent = R.undo || R.redo ? `${R.undo} undo · ${R.redo} redo` : '';
  $('ab').parentElement.hidden = st.step !== 'stack';
  // the Retouch button: whenever a stacked image is on screen (as the view or the compare partner)
  const canRetouch = have && st.step === 'stack' && (group === 'stack' || (st.compare && groupOf(st.cmp) === 'stack'));
  $('rtseg').hidden = !canRetouch; $('retouch').classList.toggle('on', retouch);
  ensureSource(); ensureSlab();
  document.querySelectorAll('#viewseg button').forEach((b) => b.classList.toggle('on', b.dataset.group === group));
  const subs = (GROUPS[group] || []).filter((t) => t !== 'dmap' || haveDmap());
  $('subseg').hidden = st.step !== 'stack' || subs.length < 2;
  document.querySelectorAll('#subseg button').forEach((b) => { b.hidden = !subs.includes(b.dataset.tab); b.classList.toggle('on', b.dataset.tab === st.view); });
  if (group) $('subseg').dataset.group = group; else delete $('subseg').dataset.group;
  // compare partner: any layer but the current view
  const choices = LAYERS.filter(([id]) => id !== st.view && (id !== 'source' || st.files.length) && (id !== 'dmap' || haveDmap()) && (id !== 'slab' || !!R.slab || (retouch && brushFrom() === 'slab')));   // the slab: once one exists, and while it is being made for the brush
  if (!choices.some(([id]) => id === st.cmp)) st.cmp = choices[0] ? choices[0][0] : 'depth';
  const menu = $('cmp-menu'); menu.innerHTML = '';
  for (const [id, name] of choices) { const b = document.createElement('button'); b.textContent = name; b.dataset.value = id; b.dataset.group = groupOf(id); b.classList.toggle('on', id === st.cmp); menu.appendChild(b); }
  $('cmp-name').textContent = layerName(st.cmp); $('cmp-sel').dataset.group = groupOf(st.cmp);
  $('ab').checked = st.compare; $('ctx-compare').hidden = !(st.compare && have); $('cmp-btn').disabled = retouch; $('cmpbar').hidden = $('ctx-compare').hidden || retouch;
  const depthShown = isDepthLayer(st.view) || (st.compare && isDepthLayer(st.cmp));
  // put each context group next to the layer it acts on: the shown layer (left) or the compare partner (after "vs")
  const place = (el, onView, onPartner) => { const slot = (!onView && onPartner) ? $('cmp-ctx') : $('view-ctx'); if (el.parentElement !== slot) slot.appendChild(el); };
  place($('ctx-depth'), isDepthLayer(st.view), st.compare && isDepthLayer(st.cmp));
  place($('ctx-source'), st.view === 'source', st.compare && st.cmp === 'source');
  $('ctx-depth').hidden = !depthShown; $('lut-gray').classList.toggle('on', !st.turbo); $('lut-turbo').classList.toggle('on', st.turbo); $('slice').checked = st.slice;
  const havePeaks = st.frames.some((f) => f && f.peak);
  const peakShown = st.view === 'source' || (st.compare && st.cmp === 'source');   // peaking is a Source overlay
  $('ctx-source').hidden = !(havePeaks && peakShown);
  $('peak').checked = st.peak.on; $('peakthr').textContent = st.peak.thr.toFixed(2); $('peakstep').hidden = !(st.peak.on || st.peak.strip);
  $('peak-strip').checked = st.peak.strip; $('peak-strip').disabled = !havePeaks;
  // shortcut card: rows for a result / retouch / compare appear once they apply
  const when = { result: have, retouch, canretouch: canRetouch, compare: st.compare && have && !retouch };
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
  if (step === 'save') { leaveRetouch(false); st.view = 'fused'; st.compare = false; }
  document.querySelectorAll('#steps button').forEach((b) => b.classList.toggle('on', b.dataset.step === step));
  $('params').hidden = step !== 'stack'; $('savepage').hidden = step !== 'save'; document.body.classList.toggle('step-save', step === 'save');
  if (step === 'save') renderSave();
  updateTabs(); renderFilmstrip(); draw();
}
document.querySelectorAll('#steps button').forEach((b) => b.addEventListener('click', () => gotoStep(b.dataset.step)));
document.querySelectorAll('#viewseg button').forEach((b) => b.addEventListener('click', () => { const g = b.dataset.group; setView(g === 'source' ? 'source' : lastIn[g]); }));
document.querySelectorAll('#subseg button').forEach((b) => b.addEventListener('click', () => setView(b.dataset.tab)));
// compare partner picker: the chip opens its list; a row picks; any click elsewhere or Esc closes it
$('cmp-btn').addEventListener('click', (e) => { e.stopPropagation(); $('cmp-menu').hidden = !$('cmp-menu').hidden; });
$('cmp-menu').addEventListener('click', (e) => { e.stopPropagation(); const b = e.target.closest('button'); if (!b) return; $('cmp-menu').hidden = true; st.cmp = b.dataset.value; updateTabs(); draw(); });
$('cm-swipe').addEventListener('click', () => { st.cmpMode = 'swipe'; saveParams(); updateTabs(); draw(); });
$('cm-split').addEventListener('click', () => { st.cmpMode = 'split'; saveParams(); updateTabs(); draw(); });
$('ab').addEventListener('change', (e) => { if (R.on) leaveRetouch(false); st.compare = e.target.checked; if (st.compare && st.view === 'source') st.view = 'fused'; updateTabs(); draw(); });
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
  if (e.key === '[' && R.on) setBrush(R.size / 1.25, R.hard); else if (e.key === ']' && R.on) setBrush(R.size * 1.25, R.hard);
  else if (e.key === 's' && R.on && !e.ctrlKey && !e.metaKey) { const order = haveDmap() ? ['source', 'stack', 'slab'] : ['source', 'slab']; setBrushFrom(order[(order.indexOf(R.from) + 1) % order.length]); }   // frame → other result (once there is one) → slab
  else if (e.key === '1') gotoStep('stack'); else if (e.key === '2') gotoStep('save');
  else if (e.key === 'r' && !e.ctrlKey && !e.metaKey) toggleRetouch();
  else if (e.key === 'ArrowLeft') scrub(e.shiftKey ? -10 : -1); else if (e.key === 'ArrowRight') scrub(e.shiftKey ? 10 : 1);
  else if (e.key === 'f') fit(); else if (e.key === 'z' && !e.ctrlKey) zoom100();
  else if (e.key === ' ' && st.compare && !R.on) { e.preventDefault(); flip(true); }
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
    if (q.has('slabs')) { $('p-dslabs').checked = q.get('slabs') === '1'; if ($('p-dslabs').checked) $('p-dmap').checked = true; runLabel(); }
    if (q.get('norun')) return;
    const tick = () => { if ($('status').textContent === 'ready') $('run').click(); else setTimeout(tick, 100); };
    tick();
  })();
}
updateTabs(); draw();

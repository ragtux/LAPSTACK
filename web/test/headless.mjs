// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

// Headless Chrome runner for web/test.html: serves ./web, launches Chrome with
// WebGPU, attaches over CDP, streams console output, and exits with the JSON
// result from window.__result.  Usage: node web/test/headless.mjs [--align]
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, '..');
const port = 8766, dbg = 9333;
const align = process.argv.includes('--align');
const url = process.env.PAGE ? `http://127.0.0.1:${port}/${process.env.PAGE}` : `http://127.0.0.1:${port}/test.html?align=${align ? 1 : 0}`;
// serve.sh sends Cache-Control: no-store, so a persistent headless profile never runs a stale app.js / style.css / pkg.
const server = spawn('bash', [path.join(webDir, 'serve.sh'), String(port)], { cwd: webDir, stdio: 'ignore' });
const profile = path.join(process.env.TMPDIR || '/tmp', 'lapstack-headless-profile');
const chrome = spawn(process.env.CHROME || 'google-chrome-stable', [
  ...(process.env.HEADFUL ? [] : ['--headless=new', '--no-sandbox', '--disable-gpu-sandbox']),
  ...(process.env.NO_GPU_FLAGS ? [] : ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--ignore-gpu-blocklist']),
  `--remote-debugging-port=${dbg}`, `--user-data-dir=${profile}`, '--no-first-run', '--window-size=1600,1000',
  ...((process.env.CHROME_FLAGS || '').split(' ').filter(Boolean)), url,
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill(); } catch {} try { server.kill(); } catch {} };
process.on('exit', cleanup);
async function pageTarget() {
  for (let i = 0; i < 100; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${dbg}/json/list`)).json();
      const t = list.find((x) => x.type === 'page' && x.url.startsWith(url.split('?')[0]));
      if (t) return t;
    } catch {}
    await sleep(200);
  }
  throw new Error('page target not found');
}
const dbgLog = (...a) => { if (process.env.HARNESS_DEBUG) console.log('[harness]', ...a); };
dbgLog('chrome pid', chrome.pid, 'server pid', server.pid, 'url', url);
const target = await pageTarget();
dbgLog('page target', target.id);
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
dbgLog('ws open');
let id = 0; const pending = new Map();
const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
  else if (m.method === 'Runtime.consoleAPICalled') console.log('[page]', m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
  else if (m.method === 'Runtime.exceptionThrown') console.log('[page exception]', JSON.stringify(m.params.exceptionDetails).slice(0, 500));
});
await send('Runtime.enable');
// FILES="glob or space-separated paths": feed real files to the page's <input id=file>
// through CDP (no in-memory copies, like a user picking them), then run PRE_EXPR.
if (process.env.FILES) {
  const fs = await import('node:fs');
  const files = process.env.FILES.split(' ').filter(Boolean).flatMap((g) => g.includes('*')
    ? fs.readdirSync(path.dirname(g)).filter((n) => new RegExp('^' + path.basename(g).replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*') + '$').test(n)).sort().map((n) => path.resolve(path.dirname(g), n))
    : [path.resolve(g)]);
  for (let i = 0; i < 200; i++) {
    const r = await send('Runtime.evaluate', { expression: "document.getElementById('status') && document.getElementById('status').textContent", returnByValue: true });
    if (r && r.result && r.result.value === 'ready') break;
    await sleep(250);
  }
  const doc = await send('DOM.getDocument', { depth: 1 });
  const node = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: '#file' });
  await send('DOM.setFileInputFiles', { nodeId: node.nodeId, files });
  console.log(`fed ${files.length} files to #file`);
  if (process.env.PRE_EXPR) await send('Runtime.evaluate', { expression: process.env.PRE_EXPR, awaitPromise: true });
}
const deadline = Date.now() + (Number(process.env.TIMEOUT_S || 300) * 1000);
let result = null;
const expr = process.env.WAIT_EXPR || 'window.__result || null';
while (Date.now() < deadline) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true });
  if (r && r.result && r.result.value) { result = JSON.parse(r.result.value); break; }
  await sleep(500);
}
if (process.env.SHOT) {
  for (const step of (process.env.SHOT_STEPS || '').split(';').filter(Boolean)) {
    await send('Runtime.evaluate', { expression: step, awaitPromise: true });
    await sleep(300);
  }
  await sleep(500);
  if (process.env.PRINT_EXPR) {
  const r = await send('Runtime.evaluate', { expression: process.env.PRINT_EXPR, returnByValue: true });
  console.log('PRINT', JSON.stringify(r && r.result && r.result.value));
}
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  const fs = await import('node:fs');
  fs.writeFileSync(process.env.SHOT, Buffer.from(shot.data, 'base64'));
  console.log('screenshot ->', process.env.SHOT);
}
ws.close();
cleanup();
if (!result) { console.log('TIMEOUT'); process.exit(2); }
console.log('RESULT', JSON.stringify(result, null, 1));
process.exit(result.ok ? 0 : 1);

// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

// lapstack as a desktop application: the browser app in web/ inside an
// Electron window. Electron carries its own Chromium, so WebGPU is there on
// Linux, Windows and macOS with the switches web/chrome.sh passes to Chrome,
// and nobody has to keep a second browser profile for it.
//
// The page is not opened from disk. WebGPU needs a secure context and a
// file:// page cannot fetch its wasm, so the main process runs a small static
// server over the web directory, bound to 127.0.0.1 on a port the system
// picks (localhost is a secure context, the same as web/serve.sh), and the
// window loads http://127.0.0.1:<port>/. Nothing listens for anyone else.
//
// In development the web directory is ../web (the build of web/build.sh must
// be there: web/pkg and web/pkg-cc). In a packaged build electron-builder
// copies it into the application's resources as web/ (package.json, extraResources).
//
//   npm start            run
//   npm run smoke        load the page in a hidden window, report the WebGPU adapter, exit
//   npm run dist         package for this platform (dist/)

'use strict';

const { app, BrowserWindow, session, dialog, shell } = require('electron');
const http = require('http');
const fs = require('fs');
const path = require('path');

// ---- Chromium switches: before the app is ready, or they are ignored ----
//
// Linux Chromium ships WebGPU behind switches, and the Vulkan trio is what
// selects the hardware adapter instead of SwiftShader (the software one, slow
// and capped at 1 GB buffers). Chromium's Wayland backend does not present
// with Vulkan on (the window stays blank), so the window runs on X11
// (XWayland) whatever the session. Windows and macOS have WebGPU on by default.
//
// The ozone platform is the one switch that has to be on the real command
// line: appended from here it reaches the GPU process but not the browser
// process, which has already chosen its platform (Wayland, in a Wayland
// session) — the GPU process then makes an X11 Vulkan surface for a Wayland
// window ("GetGeometry failed for window 1"), compositing falls back to
// software, and the window never appears. So on Linux the app relaunches
// itself once with --ozone-platform=x11 in front of its arguments.
//
// LAPSTACK_SWITCHES replaces the Linux switches with its own (space-separated
// `name=value` or `name`, without the dashes; empty = none, no relaunch): for
// telling a driver problem from an app problem, or trying another combination.
app.commandLine.appendSwitch('enable-unsafe-webgpu');
const OZONE = '--ozone-platform=x11';
let relaunching = false;
if (process.platform === 'linux') {
  const own = process.env.LAPSTACK_SWITCHES;
  if (own !== undefined) {
    for (const sw of own.split(/\s+/).filter(Boolean)) {
      const k = sw.indexOf('=');
      if (k < 0) app.commandLine.appendSwitch(sw);
      else app.commandLine.appendSwitch(sw.slice(0, k), sw.slice(k + 1));
    }
  } else if (!process.argv.some((a) => a.startsWith('--ozone-platform'))) {
    if (process.argv.includes('--smoke')) {
      // the diagnostic keeps its terminal: run the relaunched instance in place and exit with its status
      const r = require('child_process').spawnSync(process.execPath, [OZONE, ...process.argv.slice(1)], { stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 64 << 20 });
      process.stdout.write(r.stdout || '');
      process.stderr.write(r.stderr || '');
      process.exit(r.status === null ? 1 : r.status);
    }
    app.relaunch({ args: [OZONE, ...process.argv.slice(1)] });
    relaunching = true;
  } else {
    app.commandLine.appendSwitch('enable-features', 'Vulkan,VulkanFromANGLE,DefaultANGLEVulkan');
    app.commandLine.appendSwitch('ignore-gpu-blocklist');
  }
}

const SMOKE = process.argv.includes('--smoke');

// ---- the web directory ----
const webDir = app.isPackaged ? path.join(process.resourcesPath, 'web') : path.join(__dirname, '..', 'web');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.ttf': 'font/ttf',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.map': 'application/json',
};

// What the page may load: itself, its wasm ('wasm-unsafe-eval' is WebAssembly
// compilation, not eval), blobs and data URLs for the images and downloads it
// makes, nothing from the network. Electron warns at start without one.
const CSP = "default-src 'self' blob: data:; script-src 'self' 'wasm-unsafe-eval' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; worker-src 'self' blob:; connect-src 'self' blob: data:; font-src 'self' data:";

// A static server over `webDir`: GET/HEAD only, no directory listings, paths
// kept under the directory, Cache-Control: no-store (like web/serve.sh) so a
// rebuilt app.js / worker.js / wasm is picked up by a plain reload.
function serve(dir) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        res.writeHead(405).end();
        return;
      }
      let p;
      try {
        p = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
      } catch {
        res.writeHead(400).end();
        return;
      }
      if (p.endsWith('/')) p += 'index.html';
      const file = path.normalize(path.join(dir, p));
      if (!file.startsWith(dir + path.sep) && file !== dir) {
        res.writeHead(403).end();
        return;
      }
      fs.stat(file, (err, st) => {
        if (err || !st.isFile()) {
          res.writeHead(404, { 'Cache-Control': 'no-store' }).end();
          return;
        }
        res.writeHead(200, {
          'Content-Type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream',
          'Content-Length': st.size,
          'Cache-Control': 'no-store',
          'Content-Security-Policy': CSP,
        });
        if (req.method === 'HEAD') {
          res.end();
          return;
        }
        fs.createReadStream(file).on('error', () => res.destroy()).pipe(res);
      });
    });
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

// ---- permissions ----
//
// The app opens folders and saves files through the File System Access API
// (showDirectoryPicker / showSaveFilePicker) and copies the log to the
// clipboard; Electron asks the main process before granting any of that, and
// a handler that says nothing denies. Everything else (camera, location,
// notifications …) stays denied: the page never asks for it.
const ALLOWED = new Set(['fileSystem', 'clipboard-read', 'clipboard-sanitized-write']);
function permissions(ses) {
  ses.setPermissionRequestHandler((wc, permission, callback) => callback(ALLOWED.has(permission)));
  ses.setPermissionCheckHandler((wc, permission) => ALLOWED.has(permission));
  // Chromium blocks the File System Access API on some paths (the Downloads
  // folder, system directories); a stack in Downloads is a normal thing here
  ses.on('file-system-access-restricted', (e, details, callback) => callback('allow'));
  // the app's fallback when no picker is on hand is a download: Save As, in
  // the Downloads folder by default (must be set synchronously in the handler)
  ses.on('will-download', (e, item) => {
    item.setSaveDialogOptions({ defaultPath: path.join(app.getPath('downloads'), item.getFilename()) });
  });
}

// ---- the window ----
async function main() {
  if (!fs.existsSync(path.join(webDir, 'index.html'))) {
    dialog.showErrorBox('lapstack', `The web app is not at ${webDir}\n(run web/build.sh first, or package the app)`);
    app.exit(2);
    return;
  }
  const server = await serve(webDir);
  const url = `http://127.0.0.1:${server.address().port}/`;
  permissions(session.defaultSession);

  const win = new BrowserWindow({
    width: 1500,
    height: 950,
    minWidth: 900,
    minHeight: 600,
    show: false,   // shown once the page has painted (below); never in --smoke
    title: 'lapstack',
    backgroundColor: '#1a1a1a',
    autoHideMenuBar: true,
    icon: path.join(__dirname, 'build', 'icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });
  // No windows out of the app (the dust map preview's "larger view" opens a tab: not here).
  // The one exception is the toolbar's support link, which goes to the system browser.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https:\/\/(donate|buy)\.stripe\.com\//.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  win.on('closed', () => server.close());
  if (!SMOKE) win.once('ready-to-show', () => win.show());

  if (SMOKE) {
    smoke(win, url);
    return;
  }
  await win.loadURL(url);
}

// --smoke: load the page in a hidden window, ask the page for its WebGPU
// adapter, wait for the worker to report in the page's log — and, in a
// checkout with web/test/frames, let the page's autorun stack the bundled
// 8-frame test set (wasm decode, alignment and fusion on WebGPU, the depth
// pass) and wait for its result. Prints everything; exits 0 when a hardware
// adapter answered and nothing errored (1 otherwise).
function smoke(win, url) {
  const autorun = fs.existsSync(path.join(webDir, 'test', 'frames', 'list.json'));
  if (autorun) url += 'index.html?autorun=test/frames&align=1';
  const messages = [];
  win.webContents.on('console-message', (e) => messages.push(`[console ${e.level}] ${e.message}`));
  win.webContents.on('did-fail-load', (e, code, desc) => {
    console.log(`smoke: page failed to load: ${code} ${desc}`);
    app.exit(1);
  });
  win.webContents.on('did-finish-load', async () => {
    console.log(`smoke: loaded ${url}`);
    console.log(`smoke: web directory ${webDir}`);
    try {
      const info = await win.webContents.executeJavaScript(`(async () => {
        if (!navigator.gpu) return { gpu: false };
        const a = await navigator.gpu.requestAdapter();
        if (!a) return { gpu: true, adapter: null };
        const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {});
        return { gpu: true, adapter: { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, fallback: a.isFallbackAdapter, maxBufferSize: a.limits.maxBufferSize } };
      })()`);
      console.log('smoke: ' + JSON.stringify(info));
      console.log('smoke: gpu features ' + JSON.stringify(app.getGPUFeatureStatus()));
      // the worker's first lines land in the page's log within a few seconds
      await new Promise((r) => setTimeout(r, 6000));
      const log = await win.webContents.executeJavaScript(`(document.getElementById('log') || {}).innerText || ''`);
      console.log('smoke: page log:\n' + log.split('\n').slice(0, 12).map((l) => '  ' + l).join('\n'));
      let ran = null;
      if (autorun) {
        // window.__app_done is set (as JSON) when the run finishes
        const deadline = Date.now() + 120000;
        while (Date.now() < deadline && !ran) {
          ran = await win.webContents.executeJavaScript('window.__app_done || null');
          if (!ran) await new Promise((r) => setTimeout(r, 500));
        }
        console.log(ran ? `smoke: the test stack ran: ${ran}` : 'smoke: the test stack did not finish within 120 s');
      }
      for (const m of messages) console.log('smoke: ' + m);
      const soft = !info.adapter || /swiftshader|llvmpipe|software/i.test(`${info.adapter.description} ${info.adapter.vendor} ${info.adapter.architecture}`) || info.adapter.fallback;
      const errors = messages.filter((m) => /^\[console (error|3)\]/.test(m));
      console.log(`smoke: ${info.gpu ? (soft ? 'software or no adapter' : 'hardware adapter') : 'no WebGPU'}, ${errors.length} console error(s)`);
      app.exit(info.gpu && !soft && errors.length === 0 && (!autorun || ran) ? 0 : 1);
    } catch (err) {
      console.log('smoke: ' + err);
      app.exit(1);
    }
  });
  win.loadURL(url);
}

if (relaunching) {
  app.exit(0);
} else if (!app.requestSingleInstanceLock()) {
  console.error('lapstack is already running: that window was brought to the front');
  app.quit();
} else {
  app.on('second-instance', () => {
    const w = BrowserWindow.getAllWindows()[0];
    if (w) {
      if (w.isMinimized()) w.restore();
      w.focus();
    }
  });
  app.whenReady().then(main);
  app.on('window-all-closed', () => app.quit());
}

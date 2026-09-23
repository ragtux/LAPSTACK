# lapstack desktop — the browser app as an application

`desktop/` wraps the browser app in `web/` in an [Electron](https://www.electronjs.org/)
window, so lapstack opens like a desktop application: one
application, no browser profile to set up and no launcher script. Electron
carries its own Chromium, and WebGPU runs in it on Linux, Windows and macOS
with the switches `web/chrome.sh` passes to Chrome. Nothing of the app
changes: the same `index.html`, `app.js`, `worker.js` and wasm run inside
the window, with the same File System Access API for folders and saving.

```
cd desktop
npm install          # Electron and electron-builder (downloads the Electron binary)
npm start            # run: the app in a window
npm run smoke        # diagnostic: a hidden window, the WebGPU adapter, the 8-frame test stack, exit 0/1
npm run dist         # package for this platform into dist/ (dist:linux | dist:mac | dist:win)
```

The wasm must be built first (`./web/build.sh`, or `just build-web`): the app
serves `web/pkg` and `web/pkg-cc` as they are.

## How it runs

`main.js` is the whole of it, a couple of hundred lines of plain CommonJS.

**The page is served, not opened from disk.** WebGPU needs a secure context,
and a `file://` page cannot fetch its wasm, so the main process runs a small
static HTTP server over the web directory, bound to 127.0.0.1 on a port the
system picks (localhost is a secure context, the same as `web/serve.sh`), and
the window loads `http://127.0.0.1:<port>/`. It serves GET and HEAD, keeps
paths under the directory, sets the MIME types the app needs (`.wasm` as
`application/wasm`, `.js` as `text/javascript`, fonts, images), sends
`Cache-Control: no-store` like `serve.sh` so a rebuilt `app.js` or wasm is
picked up by a plain reload, and a Content-Security-Policy that allows the
page itself, its wasm (`'wasm-unsafe-eval'` is WebAssembly compilation, not
`eval`), blobs and data URLs, and nothing from the network. Nothing listens
for anyone but the window.

**Where the web files come from.** In development the directory is `../web`.
In a packaged build electron-builder copies `web/` into the application's
resources (`extraResources` in `package.json`, without `web/test/`, the shell
scripts and `probe.html`), and `main.js` resolves it with
`app.isPackaged ? process.resourcesPath/web : ../web`.

**The window.** One `BrowserWindow`, shown once the page has painted, with
`nodeIntegration` off, `contextIsolation` and the sandbox on, no preload:
the page has no reason to reach Node. Links that would open a new tab (the
dust map preview's larger view) are refused; the menu bar is hidden (Alt
shows it, with the usual reload and developer tools). A second launch brings
the running window to the front instead of opening another.

**Permissions.** The app opens folders and saves files through the File
System Access API (`showDirectoryPicker`, `showSaveFilePicker`) and copies
the log to the clipboard; Electron asks the main process before granting any
of that, and a handler that says nothing denies. `main.js` grants
`fileSystem`, `clipboard-read` and `clipboard-sanitized-write` and nothing
else, answers `'allow'` to the `file-system-access-restricted` event (Chromium
otherwise refuses the Downloads folder and a few system paths, and a stack in
Downloads is a normal thing), and sends the app's download fallback (an
`<a download>` when no picker is on hand) through a Save As dialog that
starts in the Downloads folder.

## The switches, and why

`--enable-unsafe-webgpu` everywhere. On Linux also
`--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan`,
`--ignore-gpu-blocklist` and `--ozone-platform=x11`: Linux Chromium ships
WebGPU behind switches, the Vulkan trio is what selects the hardware adapter
instead of SwiftShader (the software one: slow, and capped at 1 GB buffers,
which a 45 MP frame does not fit), and Chromium's Wayland backend does not
present with Vulkan on (the window maps and stays blank), so the window runs
on X11 — XWayland in a Wayland session — whatever the desktop, exactly as
`web/chrome.sh` does for Chrome. Windows and macOS have WebGPU on by
default; the two Linux switches are not passed there.

One of them has to be on the real command line. `app.commandLine.appendSwitch`
from `main.js` reaches the GPU process but not the browser process, which
has already chosen its platform (Wayland, in a Wayland session); the GPU
process then makes an X11 Vulkan surface for a Wayland window
(`GetGeometry failed for window 1`, `Failed to create vulkan surface`),
compositing falls back to software, and the window never appears at all. So
on Linux the app relaunches itself once with `--ozone-platform=x11` in front
of its arguments (`app.relaunch`; `--smoke` runs the relaunched instance in
place so its output stays in the terminal) — a few hundred milliseconds at
start, and the only way a packaged launcher gets the switch.

`LAPSTACK_SWITCHES` replaces the Linux switches with its own, space-separated
`name=value` or `name` without the dashes, and skips the relaunch — for
telling a driver problem from an app problem, or trying another
combination: `LAPSTACK_SWITCHES= npm start` runs with no switches at all
(WebGPU on SwiftShader, the window on the session's own backend).

## What was checked

On the development machine (NixOS, KDE on Wayland, NVIDIA RTX 3060), with
nixpkgs' Electron 43 and the Electron 44 binary npm installs (run in an FHS
environment, since NixOS cannot run it as downloaded):

- `npm run smoke`: the page loads, the worker reports `WebGPU ready` on the
  `nvidia` / `ampere` adapter (`maxBufferSize` 4 GB, not SwiftShader's 1 GB),
  the 8-frame test stack in `web/test/frames` aligns and fuses in about a
  second with the depth pass, no console errors, exit 0.
- `npm start`: the window maps as an X11 window and presents (checked with a
  compositor screenshot: the app with its panels, the log showing the NVIDIA
  adapter). Without the relaunch — the ozone switch appended from `main.js`
  — the window never appeared with the Vulkan switches on, on either
  Electron; with `--ozone-platform=wayland` it appeared blank; without the
  Vulkan switches it appeared and worked on SwiftShader. Those three were
  the whole search.
- `npm run dist:linux` built `dist/lapstack-0.1.0.AppImage` (130 MB) and
  `dist/lapstack-desktop_0.1.0_amd64.deb` (104 MB); the unpacked build and
  the AppImage (`--appimage-extract-and-run --smoke`) both start, find
  their web files in the resources and reach the NVIDIA adapter.

On NixOS the npm-installed Electron and electron-builder's AppImage and deb
tools cannot run as downloaded (dynamic executables for generic Linux):
`nix-fhs.nix` builds an FHS environment that runs them —
`nix-build nix-fhs.nix -o electron-fhs`, then
`./electron-fhs/bin/electron-fhs -c 'npm run dist:linux'` — and nixpkgs' own
Electron runs the app in development (`nix shell nixpkgs#electron -c electron .`).
Elsewhere `npm start` and `npm run dist` are the whole story.

Not checked here: Windows and macOS builds and their pickers (no machine),
and the `file-system-access-restricted` and `will-download` paths, which
were written from the Electron 44 documentation.

## Troubleshooting

- *"The web app is not at …"* at start: build the wasm (`./web/build.sh`).
- The app opens but the log says the adapter is `swiftshader` / the app warns
  about a software adapter: the GPU's Vulkan driver is not there for the
  Chromium inside Electron. `vulkaninfo` should list the card; on NVIDIA the
  driver's ICD must be installed for the user session.
- A blank window on Linux: the ozone relaunch did not happen (a wrapper that
  strips arguments, or `LAPSTACK_SWITCHES` set). `npm run smoke` says what
  the page sees.
- *"lapstack is already running"*: the single-instance lock; the running
  window was brought to the front.

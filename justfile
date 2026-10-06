# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

# lapstack developer commands — `just` lists them.
set shell := ["bash", "-euo", "pipefail", "-c"]

port := "8765"

default:
    @just --list --unsorted

# ---- native ----

# Release build of the CLI (target/release/lapstack)
build:
    cargo build --release

# Release build with the GPU paths (--gpu / --gpu-align at run time): CUDA, and wgpu for any other GPU
build-gpu:
    cargo build --release --features gpu,wgpu -p lapstack-cli

# Type-check everything native, with and without the gpu feature
check:
    cargo check --release
    cargo check --release --features gpu -p lapstack-cli

# Unit tests (lapstack-core)
test:
    cargo test --release -p lapstack-core

# clippy (advisory: the tree is not warning-free)
lint:
    cargo clippy --release

# rustfmt the tree (note: reformats many lines; the code is not rustfmt-clean)
fmt:
    cargo fmt

# Stack the bundled 8-frame sample with the native CLI -> /tmp/lapstack-smoke.png
smoke: build
    target/release/lapstack --align-coarsen 1 --save-depth -o /tmp/lapstack-smoke.png web/test/frames/f0*.png
    @echo "-> /tmp/lapstack-smoke.png (+ _depth.png)"

# ---- browser app ----

# wasm32 build + wasm-bindgen glue into web/pkg, pkg-cc and pkg-raw (pulls in lld via nix-shell when no wasm linker is on PATH)
build-web:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v wasm-ld >/dev/null || ls "$(rustc --print sysroot)"/lib/rustlib/*/bin/rust-lld >/dev/null 2>&1; then
        ./web/build.sh
    else
        nix-shell -p lld --run ./web/build.sh
    fi

# Show what is listening on the app port and which directory it serves
status port=port:
    #!/usr/bin/env bash
    set -uo pipefail
    pids=$(ss -ltnpH "sport = :{{port}}" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
    if [ -z "$pids" ]; then echo "nothing listening on :{{port}}"; exit 0; fi
    for p in $pids; do
        echo "pid $p  $(tr '\0' ' ' </proc/$p/cmdline | cut -c1-60)"
        echo "     serving $(readlink /proc/$p/cwd)"
    done

# Kill whatever is listening on the app port (a stale server from another checkout, say)
stop port=port:
    #!/usr/bin/env bash
    set -uo pipefail
    pids=$(ss -ltnpH "sport = :{{port}}" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
    if [ -z "$pids" ]; then echo "nothing listening on :{{port}}"; exit 0; fi
    for p in $pids; do
        echo "killing pid $p (was serving $(readlink /proc/$p/cwd))"
        kill "$p" || true
    done
    for _ in $(seq 20); do ss -ltnH "sport = :{{port}}" | grep -q . || exit 0; sleep 0.1; done
    echo "still listening; sending SIGKILL"; kill -9 $pids || true

# Serve web/ on the app port, replacing any server already there (background)
serve port=port: (stop port)
    #!/usr/bin/env bash
    set -euo pipefail
    setsid nohup ./web/serve.sh {{port}} >/dev/null 2>&1 &
    for _ in $(seq 30); do curl -fs "http://127.0.0.1:{{port}}/index.html" >/dev/null && break; sleep 0.1; done
    echo "serving $(pwd)/web on http://localhost:{{port}}/"

# Open the app in the WebGPU-enabled Chrome profile (starts the server if needed)
chrome port=port:
    #!/usr/bin/env bash
    set -euo pipefail
    curl -fs "http://127.0.0.1:{{port}}/index.html" >/dev/null || just serve {{port}}
    ./web/chrome.sh {{port}}

# Rebuild the wasm, restart the server, open Chrome
dev port=port: build-web (serve port) (chrome port)

# Headless Chrome end-to-end test on the GPU; e.g. `just test-web --align`
test-web *args:
    node web/test/headless.mjs {{args}}

# Everything: native tests, wasm build, browser test with alignment
test-all: test build-web (test-web "--align")

# Regenerate the browser test frames + references from a directory of aligned frames
test-frames dir:
    ./web/test/make-frames.sh {{dir}}

# The deployable app: web/ minus its tests and scripts, cache-busting pinned to the build, into web/dist
dist-web:
    ./web/dist.sh

# Regenerate THIRD-PARTY.md (every crate a build can contain, its license and notices) from the Cargo metadata
third-party:
    python3 tools/third-party.py

# ---- desktop app (Electron wrapper of web/, see desktop/README.md) ----

# Install the desktop app's dependencies (once; downloads Electron)
desktop-install:
    cd desktop && npm install

# Run the desktop app in development (needs `just build-web` first; on NixOS see desktop/README.md for the FHS env)
desktop:
    cd desktop && npm start

# Hidden-window smoke test of the desktop app: adapter, the 8-frame test stack, console errors; exit 0/1
desktop-smoke:
    cd desktop && npm run smoke

# Package the desktop app for this platform into desktop/dist (AppImage + deb, dmg, or nsis)
desktop-dist:
    cd desktop && npm run dist

# Remove build outputs
clean:
    cargo clean
    rm -rf web/pkg web/pkg-cc web/pkg-raw web/dist desktop/dist

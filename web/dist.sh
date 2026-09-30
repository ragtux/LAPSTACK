#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

# The deployable browser app, into web/dist: web/ without its tests, probes and
# scripts (the filter the desktop app's extraResources uses), the licence and
# the third-party notices under /legal, and worker.js's cache-busting query
# pinned to a hash of the wasm modules instead of Date.now() — in development a
# per-load stamp makes a rebuilt pkg/ show up on a plain reload; deployed, it
# would make every visit download the 6 MB engine again. With the stamp fixed
# per build, the glue and the wasm cache until the next deploy changes it, and
# the entry files (index.html, app.js, worker.js, style.css) are what the
# server must serve with a short or no-cache lifetime.
#   ./web/build.sh && ./web/dist.sh     # then rsync web/dist/ to the host
set -euo pipefail
cd "$(dirname "$0")"
[ -f pkg/lapstack_web_bg.wasm ] && [ -f pkg-cc/lapstack_cc_bg.wasm ] || { echo "web/pkg or web/pkg-cc missing: run web/build.sh first" >&2; exit 1; }
rm -rf dist && mkdir -p dist/legal
rsync -a --exclude test --exclude test.html --exclude probe.html --exclude '*.sh' --exclude '*.d.ts' --exclude dist ./ dist/
cp ../LICENSE dist/legal/LICENSE.txt
cp ../THIRD-PARTY.md dist/legal/THIRD-PARTY.md
V=$(cat pkg/lapstack_web_bg.wasm pkg/lapstack_web.js pkg-cc/lapstack_cc_bg.wasm pkg-cc/lapstack_cc.js | sha256sum | cut -c1-12)
grep -q "^const V = Date.now();$" dist/worker.js || { echo "worker.js: the cache-busting stamp is not where dist.sh expects it" >&2; exit 1; }
sed -i "s/^const V = Date.now();$/const V = '$V';  \/\/ pinned by web\/dist.sh: the build's wasm hash/" dist/worker.js
du -sh dist | cut -f1 | xargs -I{} echo "web/dist: {} (build $V)"
find dist -type f | sort | sed 's|^dist/|  |'

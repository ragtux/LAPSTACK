#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

# The deployable browser app, into web/dist: web/ without its tests, probes and
# scripts (the filter the desktop app's extraResources uses); under /legal the
# license, the third-party notices and the source of the raw decoder module
# (pkg-raw is lapstack-raw + rawler, LGPL: the source has to travel with the
# build, and a reader of it can rebuild the module and drop it in); and
# worker.js's cache-busting query
# pinned to a hash of the wasm modules instead of Date.now() — in development a
# per-load stamp makes a rebuilt pkg/ show up on a plain reload; deployed, it
# would make every visit download the 6 MB engine again. With the stamp fixed
# per build, the glue and the wasm cache until the next deploy changes it, and
# the entry files (index.html, app.js, worker.js, style.css) are what the
# server must serve with a short or no-cache lifetime.
#   ./web/build.sh && ./web/dist.sh     # then rsync web/dist/ to the host
set -euo pipefail
cd "$(dirname "$0")"
for m in pkg/lapstack_web_bg.wasm pkg-cc/lapstack_cc_bg.wasm pkg-raw/lapstack_raw_bg.wasm; do
    [ -f "$m" ] || { echo "web/$m missing: run web/build.sh first" >&2; exit 1; }
done
rm -rf dist && mkdir -p dist/legal
rsync -a --exclude test --exclude test.html --exclude probe.html --exclude '*.sh' --exclude '*.d.ts' --exclude dist ./ dist/
cp ../LICENSE dist/legal/LICENSE.txt
cp ../THIRD-PARTY.md dist/legal/THIRD-PARTY.md
# the raw module's source: the crate, the patched rawler, and a workspace file so it builds as it is
SRC=$(mktemp -d); mkdir -p "$SRC/lapstack-raw-src"
rsync -a --exclude target ../crates/lapstack-raw "$SRC/lapstack-raw-src/crates/"
rsync -a ../vendor/rawler "$SRC/lapstack-raw-src/vendor/"
cp ../crates/lapstack-raw/LICENSE "$SRC/lapstack-raw-src/LICENSE"
cp ../crates/lapstack-raw/README.md "$SRC/lapstack-raw-src/README.md"
cat > "$SRC/lapstack-raw-src/Cargo.toml" <<'TOML'
# The source of lapstack's raw decoder module (LGPL-2.1, see LICENSE and README.md).
# `cargo build --release -p lapstack-raw` here builds the shared library the
# command-line tool loads; README.md has the wasm module's two commands.
[workspace]
resolver = "3"
members = ["crates/lapstack-raw"]
exclude = ["vendor/rawler"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
TOML
tar czf dist/legal/lapstack-raw-src.tar.gz -C "$SRC" lapstack-raw-src
rm -rf "$SRC"
V=$(cat pkg/lapstack_web_bg.wasm pkg/lapstack_web.js pkg-cc/lapstack_cc_bg.wasm pkg-cc/lapstack_cc.js pkg-raw/lapstack_raw_bg.wasm pkg-raw/lapstack_raw.js | sha256sum | cut -c1-12)
grep -q "^const V = Date.now();$" dist/worker.js || { echo "worker.js: the cache-busting stamp is not where dist.sh expects it" >&2; exit 1; }
sed -i "s/^const V = Date.now();$/const V = '$V';  \/\/ pinned by web\/dist.sh: the build's wasm hash/" dist/worker.js
du -sh dist | cut -f1 | xargs -I{} echo "web/dist: {} (build $V)"
find dist -type f | sort | sed 's|^dist/|  |'

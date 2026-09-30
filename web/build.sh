#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

# Build the browser app: wasm32 + wasm-bindgen glue into web/pkg.
#   ./web/build.sh            # release build
#   ./web/serve.sh            # then open http://localhost:8765/
# Needs: rustup target wasm32-unknown-unknown (or a rustc with that std),
# wasm-bindgen-cli matching the wasm-bindgen crate version (cargo install
# wasm-bindgen-cli --version 0.2.128), and a wasm linker: rustup's rust-lld,
# or `wasm-ld` from LLVM/lld (NixOS: nix-shell -p lld).
set -euo pipefail
cd "$(dirname "$0")/.."
FLAGS=""
if ! rustc --print sysroot >/dev/null || ! ls "$(rustc --print sysroot)"/lib/rustlib/*/bin/rust-lld >/dev/null 2>&1; then
    if command -v wasm-ld >/dev/null; then FLAGS="-C linker=wasm-ld"; fi
fi
# Three modules: the engine (web/pkg); content credentials (web/pkg-cc, c2pa is
# large) that worker.js imports only when a save asks for them; and the raw
# decoder (web/pkg-raw: lapstack-raw, LGPL, never linked into the engine) that
# it imports when the first raw comes in.
RUSTFLAGS="${RUSTFLAGS:-} $FLAGS" cargo build --release --target wasm32-unknown-unknown -p lapstack-web -p lapstack-cc -p lapstack-raw
WB=$(command -v wasm-bindgen || echo "$HOME/.cargo/bin/wasm-bindgen")
"$WB" --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/lapstack_web.wasm
"$WB" --target web --out-dir web/pkg-cc target/wasm32-unknown-unknown/release/lapstack_cc.wasm
"$WB" --target web --out-dir web/pkg-raw target/wasm32-unknown-unknown/release/lapstack_raw.wasm
if command -v wasm-opt >/dev/null; then
    wasm-opt -O2 -o web/pkg/lapstack_web_bg.wasm web/pkg/lapstack_web_bg.wasm
    wasm-opt -O2 -o web/pkg-cc/lapstack_cc_bg.wasm web/pkg-cc/lapstack_cc_bg.wasm
    wasm-opt -O2 -o web/pkg-raw/lapstack_raw_bg.wasm web/pkg-raw/lapstack_raw_bg.wasm
fi
ls -la web/pkg/lapstack_web_bg.wasm web/pkg-cc/lapstack_cc_bg.wasm web/pkg-raw/lapstack_raw_bg.wasm

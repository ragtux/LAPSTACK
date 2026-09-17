#!/usr/bin/env bash
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
RUSTFLAGS="${RUSTFLAGS:-} $FLAGS" cargo build --release --target wasm32-unknown-unknown -p lapstack-web
WB=$(command -v wasm-bindgen || echo "$HOME/.cargo/bin/wasm-bindgen")
"$WB" --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/lapstack_web.wasm
if command -v wasm-opt >/dev/null; then wasm-opt -O2 -o web/pkg/lapstack_web_bg.wasm web/pkg/lapstack_web_bg.wasm; fi
ls -la web/pkg/lapstack_web_bg.wasm

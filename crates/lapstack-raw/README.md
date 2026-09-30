# lapstack-raw

Camera raw decoding for [lapstack](https://lapstack.ragtux.com), as a component
of its own: [rawler](https://github.com/dnglab/dnglab) (dnglab's library) behind
a small interface, built as a shared library that the lapstack command-line
tool loads at run time and as a wasm module that the browser app loads beside
its engine.

This crate is free software under the **GNU Lesser General Public License,
version 2.1** (`LICENSE`), because it is a work based on rawler, which is
LGPL-2.1. lapstack itself is proprietary and does not link this crate or
rawler; it loads the built module at run time through the interface described
in `src/lib.rs`. That is the arrangement the LGPL's section 6 asks for: you may
modify this library, or rawler inside it, rebuild it, and lapstack will use
your build.

## Building it

This directory and `vendor/rawler` (rawler 0.8.0 with a two-file change so it
runs on wasm32, described in `vendor/rawler/LAPSTACK-PATCH.md`) are all it
needs, and the source tarball that ships with lapstack (`legal/lapstack-raw-src.tar.gz`
in the browser app, next to the command-line tool's downloads) contains both
with a workspace `Cargo.toml` at its root.

Natively — the shared library the command-line tool loads:

```
cargo build --release -p lapstack-raw
# target/release/liblapstack_raw.so (Linux), liblapstack_raw.dylib (macOS), lapstack_raw.dll (Windows)
```

Put it next to the `lapstack` binary, or point `LAPSTACK_RAW_LIB` at it. The
tool looks in that variable, then beside itself, then in `../lib`, then on the
system's library path, and says where it looked when it finds none.

For the browser — the wasm module the worker imports:

```
cargo build --release --target wasm32-unknown-unknown -p lapstack-raw
wasm-bindgen --target web --out-dir pkg-raw target/wasm32-unknown-unknown/release/lapstack_raw.wasm
```

and replace the app's `pkg-raw/` directory with the result (`wasm-bindgen-cli`
0.2.128, the version of the `wasm-bindgen` crate). Needs a wasm linker:
rustup's `rust-lld`, or `wasm-ld` from LLVM (`RUSTFLAGS="-C linker=wasm-ld"`).

## The interface

`src/lib.rs` documents what crosses the boundary — the pixel layouts of
`develop`, `develop_linear` and `preview`, and the JSON of `Color` and
`Metadata` — and `ABI` numbers its revisions. `src/ffi.rs` is the C ABI of the
shared library; `src/wasm.rs` the wasm module's exports, which the worker
wires to three functions on the global object (`lapstackRawDevelop`,
`lapstackRawPreview`, `lapstackRawMetadata`) that the engine calls.

# rawler 0.8.0, vendored

This is [rawler](https://github.com/dnglab/dnglab) 0.8.0 from crates.io (LGPL-2.1, see
`LICENSE` and **Licensing** below), the raw decoder and developer behind lapstack's camera raw
input (`crates/lapstack-core/src/raw.rs`), with two changes, so that it builds and
runs in the browser (wasm32-unknown-unknown) where `std::time::Instant::now()` panics:

- `src/wasm_time.rs`: an `Instant` that is `std::time::Instant` everywhere but on
  wasm32, where it reads zero. The demosaic (`imgop/sensor/bayer/ppg.rs`,
  `imgop/sensor/xtrans/{bilinear,markesteijn}.rs`) and the CR3 decoder
  (`decompressors/crx/decoder.rs`) import it in place of `std::time::Instant`; they
  only time themselves for a debug log line.
- `data/testdata` (39 MB of the crate's test fixtures), `tests`, `benches` and `Cargo.lock` are
  left out.

The workspace's `[patch.crates-io]` points `rawler` here. To move to a newer rawler:
copy the new crate over this directory, drop `data/testdata`, `tests`, `benches` and
`Cargo.lock`, and apply the `wasm_time` change again (`grep -rn 'time::Instant' src`).

## Licensing

rawler's `Cargo.toml` declares `LGPL-2.1`, and its sources carry per-file SPDX
headers: 50 files `MIT`, 44 files `LGPL-2.1`, copyright Daniel Vogelbacher. No
file says "or any later version", so the LGPL files are LGPL-2.1 **only**.

lapstack is proprietary (`/LICENSE`). The LGPL allows that combination — a
"work that uses the Library" may be under any terms — on conditions, two of
which bind here, and this is how lapstack meets them:

- **The library's own changes stay LGPL.** `wasm_time.rs` and the five import
  lines are a modification of rawler and are under the LGPL-2.1 like the files
  they change; so is `crates/lapstack-raw`, the shim around rawler that is
  built with it. This directory, the shim and a workspace file to build them
  are the source of that modified library, and they ship with every lapstack
  build (`legal/lapstack-raw-src.tar.gz` in the browser app, and beside the
  command-line tool's downloads), so anyone who receives a build has it.
- **The user must be able to replace the library** (LGPL-2.1 §6). lapstack
  does not link rawler or `lapstack-raw`; it loads the built module at run
  time through a small interface (`crates/lapstack-raw/src/lib.rs` describes
  it; `crates/lapstack-core/src/raw.rs` is the client): natively a shared
  library found next to the binary (§6b — "a suitable shared library
  mechanism"), in the browser a wasm module of its own that the worker imports
  and the engine reaches through the global object. Rebuild the module from
  its source, put it where the old one was, and lapstack uses it;
  `crates/lapstack-raw/README.md` says how. lapstack's own parts stay
  proprietary and are not part of the LGPL work.

Until 2026-09-30 rawler was linked into the engine itself, which the LGPL does
not allow a proprietary program to distribute without relinkable object code;
no build with it was distributed.

While lapstack was AGPL-3.0-only (2026-09-24 to 2026-09-30) an earlier revision
of this note elected, under LGPL-2.1 §3, to take the LGPL files as GPL-3.0.
That election is irreversible for the copy it was made on, so on 2026-09-30
this directory was replaced by a fresh copy of the crates.io tarball with the
`wasm_time` change applied again. No election is made for this copy: it is
LGPL-2.1, as published.

`lapstack-core`'s `raw` feature is what pulls rawler in, and building the
library without it (`cargo build -p lapstack-core --no-default-features`) drops
rawler and this whole question with it. Note that this is a library-only escape
hatch today: `lapstack-cli` depends on `lapstack-core` with default features and
has no `raw` passthrough, so `-p lapstack-cli --no-default-features` still
builds rawler in.

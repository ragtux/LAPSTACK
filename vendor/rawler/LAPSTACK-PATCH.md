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
- `data/testdata` (39 MB of the crate's test fixtures) and `Cargo.lock` are left out.

The workspace's `[patch.crates-io]` points `rawler` here. To move to a newer rawler:
copy the new crate over this directory, drop `data/testdata`, and apply the
`wasm_time` change again (`grep -rn 'time::Instant' src`).

## Licensing

rawler is **not** Apache-2.0, as an earlier version of this note said. Its
`Cargo.toml` declares `LGPL-2.1`, and its sources carry per-file SPDX headers:
50 files `MIT`, 44 files `LGPL-2.1`, copyright Daniel Vogelbacher. No file says
"or any later version", so the LGPL files are LGPL-2.1 **only**.

lapstack is AGPL-3.0-only (`/LICENSE`), and that combination needs one step to
be coherent:

- The MIT files are compatible with the AGPL as they stand.
- LGPL-2.1-only is not directly compatible with AGPL-3.0. Section 3 of the
  LGPL, however, lets any recipient of a copy elect to use "the GNU General
  Public License, version 2, or any later version" for that copy instead of
  the LGPL. **lapstack elects GPL-3.0 for the LGPL-2.1 files in this
  directory**, under that section.
- GPL-3.0 section 13 then explicitly permits combining GPLv3 code with
  AGPLv3 code; the combined work's lapstack parts remain AGPL-3.0-only, and
  these vendored parts remain under GPL-3.0 as elected.

The election applies to this vendored copy only. It does not relicense
upstream rawler, and it is not a claim about any other user's copy — anyone
else may make the same election, or not, for their own.

`lapstack-core`'s `raw` feature is what pulls rawler in, and building the
library without it (`cargo build -p lapstack-core --no-default-features`) drops
rawler and this whole question with it. Note that this is a library-only escape
hatch today: `lapstack-cli` depends on `lapstack-core` with default features and
has no `raw` passthrough, so `-p lapstack-cli --no-default-features` still
builds rawler in.

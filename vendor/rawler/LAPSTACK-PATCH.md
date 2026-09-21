# rawler 0.8.0, vendored

This is [rawler](https://github.com/dnglab/dnglab) 0.8.0 from crates.io (Apache-2.0
/ LGPL-2.1, see LICENSE), the raw decoder and developer behind lapstack's camera raw
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

//! lapstack's one change to rawler: `std::time::Instant` has no implementation on
//! wasm32-unknown-unknown (`Instant::now()` panics there), and the demosaic and
//! CR3 decoder only time themselves for their debug log. On wasm the clock reads
//! zero; everywhere else this is `std::time::Instant`.
#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
pub struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
  pub fn now() -> Self {
    Instant
  }
  pub fn elapsed(&self) -> std::time::Duration {
    std::time::Duration::ZERO
  }
}

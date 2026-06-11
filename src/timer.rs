//! Monotonic clock portable to browser WASM (`std::time::Instant` panics on wasm32).

#[cfg(target_arch = "wasm32")]
pub use web_time::Instant; // web-time crate
#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

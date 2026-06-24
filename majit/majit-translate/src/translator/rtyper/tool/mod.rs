//! `translator/rtyper/tool/` — RPython-orthodox
//! `rpython/rtyper/tool/` counterparts.

pub mod gcstat;
pub mod mkrffi;
// `rffi_platform` is intentionally absent and must NEVER be added. RPython's
// `rffi_platform.py` probes the C toolchain at build time to discover struct
// layouts and constants; pyre uses Rust's real types and Charon-extracted
// layouts, so build-time C probing is permanently unused by design.
pub mod rfficache;

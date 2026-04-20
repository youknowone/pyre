//! `tool::algo` — Rust port of `rpython/tool/algo/`.
//!
//! Upstream package lists `bitstring`, `color`, `graphlib`, `regalloc`,
//! `sparsemat`, `unionfind`. Only modules that are reached by already-
//! ported downstream code land here; the rest are pulled in as their
//! consumers arrive.
//!
//! | majit         | rpython/tool/algo/         |
//! |---------------|----------------------------|
//! | `unionfind`   | `unionfind.py`             |

pub mod unionfind;

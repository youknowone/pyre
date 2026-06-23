//! `tool::algo` — Rust port of `rpython/tool/algo/`.
//!
//! Upstream package lists `bitstring`, `color`, `graphlib`, `regalloc`,
//! `sparsemat`, `unionfind`. Only modules that are reached by already-
//! ported downstream code land here; the rest are pulled in as their
//! consumers arrive.
//!
//! | majit         | rpython/tool/algo/         |
//! |---------------|----------------------------|
//! | `bitstring`   | `bitstring.py`             |
//! | `graphlib`    | `graphlib.py`              |
//! | `sparsemat`   | `sparsemat.py`             |
//! | `unionfind`   | `unionfind.py`             |

pub mod bitstring;
pub mod graphlib;
pub mod sparsemat;
pub mod unionfind;

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
//! | `color`       | `color.py`                 |
//! | `graphlib`    | `graphlib.py`              |
//! | `regalloc`    | `regalloc.py`              |
//! | `sparsemat`   | `sparsemat.py`             |
//! | `unionfind`   | `unionfind.py`             |

pub mod bitstring;
pub mod color;
pub mod graphlib;
pub mod regalloc;
pub mod sparsemat;
pub mod unionfind;

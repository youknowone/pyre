//! `rlib` — Rust port of `rpython/rlib/` helpers.
//!
//! Only the subset required by downstream annotator / rtyper ports
//! lands; new submodules appear as their dependencies are pulled in.
//!
//! | majit                 | rpython/rlib/                 |
//! |-----------------------|-------------------------------|
//! | `rarithmetic`         | `rarithmetic.py`              |

pub mod rarithmetic;

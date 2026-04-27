//! `rlib` — Rust port of `rpython/rlib/` helpers.
//!
//! Only the subset required by downstream annotator / rtyper ports
//! lands; new submodules appear as their dependencies are pulled in.
//!
//! | majit                 | rpython/rlib/                 |
//! |-----------------------|-------------------------------|
//! | `entrypoint`          | `entrypoint.py`               |
//! | `jit_marker`          | `jit.py:875-1024` (markers)   |
//! | `rarithmetic`         | `rarithmetic.py`              |

pub mod entrypoint;
pub mod jit_marker;
pub mod rarithmetic;

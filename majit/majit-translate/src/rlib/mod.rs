//! `rlib` — Rust port of `rpython/rlib/` helpers.
//!
//! Only the subset required by downstream annotator / rtyper ports
//! lands; new submodules appear as their dependencies are pulled in.
//!
//! | majit                 | rpython/rlib/                 |
//! |-----------------------|-------------------------------|
//! | `entrypoint`          | `entrypoint.py`               |
//! | `jit`                 | `jit.py:875-1024` (marker/extregistry half; the user hint API lives at `majit_metainterp::jit`) |
//! | `rarithmetic`         | `rarithmetic.py`              |
//! | `rvmprof`             | `rvmprof/` disabled-runtime `cintf` adapter |

pub mod entrypoint;
pub mod jit;
pub mod rarithmetic;
pub mod rvmprof;

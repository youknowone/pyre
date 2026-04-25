//! `translator/backendopt/` — mirror of upstream
//! `rpython/translator/backendopt/`. Houses the post-annotator graph
//! optimisations (SSA conversion, escape analysis, etc.).
//!
//! Each file name matches upstream (e.g. `ssa.py` → `ssa.rs`).

pub mod all;
pub mod ssa;

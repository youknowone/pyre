//! `tool` — Rust port of `rpython/tool/`.
//!
//! Upstream `tool/` collects small, reusable helpers used across
//! `rpython/annotator/`, `rpython/rtyper/`, and `rpython/translator/`.
//! Only the subset required by the downstream ports lands; new
//! submodules appear as their dependencies are pulled in.
//!
//! | majit                 | rpython/tool/                 |
//! |-----------------------|-------------------------------|
//! | `algo::unionfind`     | `algo/unionfind.py`           |

pub mod algo;

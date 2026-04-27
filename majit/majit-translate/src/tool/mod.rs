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
//! | `error`               | `error.py`                    |
//! | `flattenrec`          | `flattenrec.py`               |
//! | `pairtype`            | `pairtype.py`                 |
//! | `sourcetools`         | `sourcetools.py`              |
//! | `udir`                | `udir.py`                     |

pub mod algo;
pub mod error;
pub mod flattenrec;
pub mod pairtype;
pub mod sourcetools;
pub mod udir;

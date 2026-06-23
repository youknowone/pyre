//! `config/` — Rust port of `rpython/config/`.
//!
//! | majit                 | rpython/config/              |
//! |-----------------------|------------------------------|
//! | `config`              | `config.py`                  |
//! | `parse`               | `parse.py`                   |
//! | `support`             | `support.py` (partial)       |
//! | `translationoption`   | `translationoption.py`       |
//!
//! Deferred: full `support.py` (`detect_pax`, C-backend concern) lands
//! alongside first consumers.

pub mod config;
pub mod parse;
pub mod support;
pub mod translationoption;

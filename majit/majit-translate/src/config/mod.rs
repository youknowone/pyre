//! `config/` — Rust port of `rpython/config/`.
//!
//! | majit                 | rpython/config/              |
//! |-----------------------|------------------------------|
//! | `config`              | `config.py`                  |
//! | `support`             | `support.py` (partial)       |
//! | `translationoption`   | `translationoption.py`       |
//!
//! Deferred: `parse.py` (optparse glue) + full `support.py`
//! (`detect_pax`, C-backend concern) land alongside first consumers.

pub mod config;
pub mod support;
pub mod translationoption;

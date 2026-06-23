//! `translator/` — RPython-orthodox port files that mirror upstream
//! `rpython/rtyper/` and `rpython/annotator/`.
//!
//! Files in this tree must use RPython-orthodox structure (file
//! names, function names, control flow). The legacy `translate_legacy/`
//! subtree it once paired with was deleted; the residual
//! adapter walker that still drives the dual-gate Skip arm now lives
//! in [`rtyper::legacy_annotator`] / [`rtyper::legacy_resolve`]
//! inside this same tree.  Both `majit-annotator` and `majit-rtyper`
//! standalone crates remain on the roadmap.

pub mod backendopt;
pub mod c;
pub mod driver;
pub mod gensupp;
pub mod goal;
pub mod interactive;
pub mod platform;
pub mod rtyper;
pub mod simplify;
pub mod targetspec;
pub mod timing;
pub mod tool;
pub mod transform;
pub mod translator;
pub mod unsimplify;

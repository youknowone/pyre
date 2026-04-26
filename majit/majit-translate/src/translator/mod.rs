//! `translator/` — non-LEGACY home for RPython-orthodox port files
//! that mirror upstream `rpython/rtyper/` and `rpython/annotator/`.
//!
//! The matching majit-local infra still lives under
//! `translator_legacy/`; that subtree is marked LEGACY and slated for
//! deletion at roadmap commit P8.11, replaced by the standalone
//! `majit-annotator` (Phase 5 exit) and `majit-rtyper` (Phase 6 exit)
//! crates. Until those crates land, files in this `translator/` tree
//! must use clearly RPython-orthodox structure (file names, function
//! names, control flow) — anything that would be a NEW-DEVIATION
//! belongs in `translator_legacy/` instead.

pub mod backendopt;
pub mod c;
pub mod driver;
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

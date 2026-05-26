//! `translator/rtyper/` — RPython-orthodox `rpython/rtyper/` counterparts.
//!
//! Files in this module mirror upstream `rpython/rtyper/` 1:1 by name
//! and by structure (`rclass.py` → `rclass.rs`, `rpbc.py` → `rpbc.rs`).
//! The standalone `majit-rtyper` crate is still pending; the
//! per-graph type resolution machinery lives in [`legacy_resolve`]
//! inside this tree (relocated from the deleted `translate_legacy/`
//! subtree).  The dual-gate Skip arm
//! drives [`legacy_annotator::annotate`] +
//! [`legacy_resolve::resolve_types`] for graphs that the real
//! `RPythonTyper::specialize` path does not yet cover.

pub mod annlowlevel;
pub mod callparse;
pub mod cutover;
pub mod error;
pub mod extregistry;
pub mod flowspace_adapter;
pub mod legacy_annotator;
pub mod legacy_resolve;
pub mod llannotation;
pub mod llinterp;
pub mod lltypesystem;
pub mod normalizecalls;
pub mod pairtype;
pub mod pyre_call_registry;
pub mod rbool;
pub mod rbuiltin;
pub mod rclass;
pub mod rfloat;
pub mod rint;
pub mod rmodel;
pub mod rnone;
pub mod rpbc;
pub mod rstr;
pub mod rtuple;
pub mod rtyper;
pub mod unit_variant_fold;

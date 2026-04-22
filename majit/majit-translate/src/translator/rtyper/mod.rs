//! `translator/rtyper/` — RPython-orthodox `rpython/rtyper/` counterparts
//! that are not part of the LEGACY `translator_legacy/rtyper/` infra.
//!
//! Files in this module mirror upstream `rpython/rtyper/` 1:1 by name and
//! by structure (`rclass.py` → `rclass.rs`, `rpbc.py` → `rpbc.rs`).
//! Until the standalone `majit-rtyper` crate (roadmap Phase 6) lands,
//! the underlying `TypeResolutionState` machinery still lives under
//! `translator_legacy/rtyper/rtyper.rs`; these modules import it from
//! there as a temporary wiring bridge.

pub mod error;
pub mod lltypesystem;
pub mod normalizecalls;
pub mod rclass;
pub mod rpbc;
pub mod rtyper;

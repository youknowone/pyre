//! `rpython/rtyper/lltypesystem/llmemory.py` — GCREF.
//!
//! [`GCREF`] is `Ptr(GcOpaqueType('GCREF'))` (`getkind` `'ref'`). The physical
//! definition lives next to [`majit_gc::GcType`] so `majit-gc` does not
//! depend on this crate.

pub use majit_gc::{GCREF, GCREFOpaque};

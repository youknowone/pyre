//! `rpython/memory/` — the parts that belong to the translation pipeline.
//!
//! The collector itself (`rpython/memory/gc/incminimark.py`) lives in
//! `majit-gc`; what lands here is `rpython/memory/gctransform/`, which is a
//! *translation* stage: upstream it rewrites the flow graphs after rtyping so
//! that every operation which can collect is bracketed by
//! `push_roots` / `pop_roots`.
pub mod gctransform;

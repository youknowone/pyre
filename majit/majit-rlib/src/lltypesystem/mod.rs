//! `rpython/rtyper/lltypesystem` — the low-level forms an RPython value is
//! lowered to. Only the bodies this crate's own modules allocate live here; the
//! rest of the lltypesystem stays with the translator.

pub mod rlist;

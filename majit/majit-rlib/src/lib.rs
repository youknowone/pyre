//! `rpython/rlib` — the runtime support library RPython programs are written
//! against, as opposed to the translator that compiles them (`majit-translate`)
//! or the collector that runs under them (`majit-gc`).
//!
//! A module here is a port of an upstream file and is named after it, so
//! `rpython/rlib/rbigint.py` is [`rbigint`]. The one exception is
//! [`lltypesystem`], which carries the `rpython/rtyper/lltypesystem` bodies that
//! an `rlib` module's own storage is lowered to: `rbigint._digits` is a list
//! whose low-level form is a `Ptr(GcArray(Signed))`, and that body cannot live
//! in a crate above the module that allocates it. The module path spells the
//! upstream package it came from.

pub mod lltypesystem;
pub mod rbigint;

/// `lltype.malloc(T, flavor='raw')` (`rpython/rtyper/lltypesystem/lltype.py`) —
/// an allocation the collector neither traces nor reclaims.
///
/// Kept here rather than in a `lltypesystem::lltype` module because this is the
/// whole of that file this crate uses: the raw-flavor allocations rbigint falls
/// back to when no GC owns the heap.
pub fn malloc_raw<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

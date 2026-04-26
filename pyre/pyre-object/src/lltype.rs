//! `lltype.malloc` parity API — unified allocation lowering for pyre objects.
//!
//! Mirrors RPython's `lltype.malloc(T, flavor='gc')`
//! (`rpython/rtyper/lltypesystem/lltype.py:2192`), the user-facing
//! allocation primitive used throughout PyPy's interpreter
//! (`pypy/objspace/std/intobject.py:883 wrapint` etc.). In RPython's
//! translation pipeline, every `lltype.malloc(T)` call is rewritten by
//! the GC transform (`rpython/memory/gctransform/framework.py:803-853
//! gct_fv_gc_malloc`) into a managed allocation surrounded by
//! `push_roots` / `pop_roots`:
//!
//! ```text
//! v_alloc = direct_call(malloc_fast_ptr, c_const_gc,
//!                       c_type_id, c_size, ...)
//! # bracketed by push_roots(hop) / pop_roots(hop, livevars)
//! ```
//!
//! pyre has no equivalent transform stage today — its host code is
//! plain Rust compiled by cargo. This module provides the same API
//! shape (the low-level allocation primitive that the GC transform
//! consumes; PyPy interpreter source-level constructors like
//! `pypy/objspace/std/intobject.py:883 wrapint` are plain
//! `W_IntObject(x)` calls and `lltype.malloc` only emerges from the
//! rtyping stage `rpython/rtyper/rclass.py:731`) so that:
//!
//! 1. Object constructors are single allocation calls without
//!    per-callsite TLS hooks or conditional branches.
//! 2. Future GC integration replaces the body of [`malloc`] without
//!    changing any caller — the "common allocation lowering" the
//!    2026-04-25 review explicitly endorsed as an alternative to a
//!    full structural GC transform.
//!
//! Phase 1 body: `Box::into_raw(Box::new(value))` — the pre-existing
//! leak baseline. Future Phase 2 routes through a GC-managed
//! allocator with proper root push/pop (Task #145).

/// Per-type GC metadata, mirroring the compile-time constants that
/// RPython's `gct_fv_gc_malloc` (`framework.py:807-811`) closes over:
///
/// ```python
/// type_id = self.get_type_id(TYPE)
/// c_type_id = rmodel.inputconst(TYPE_ID, type_id)
/// info = self.layoutbuilder.get_info(type_id)
/// c_size = rmodel.inputconst(lltype.Signed, info.fixedsize)
/// ```
///
/// In RPython these are inputconsts woven into the `direct_call` to
/// the malloc helper. In Rust they're associated constants on the
/// payload type, surfaced through [`malloc_typed`] so the future
/// managed allocator can read them without a runtime dispatch.
///
/// `TYPE_ID` must match the id returned by `gc.register_type(...)`
/// during JitDriver init (see `pyre/pyre-jit/src/eval.rs`); a
/// `debug_assert_eq!` there guards against drift.
pub trait GcType {
    /// Backend-registered GC type id, equal to `c_type_id` in
    /// `framework.py:809`.
    const TYPE_ID: u32;
    /// Fixed payload size in bytes, equal to `info.fixedsize` in
    /// `framework.py:811`.
    const SIZE: usize;
}

/// `lltype.malloc(T, flavor='gc')` parity, *untyped* (no `GcType` impl
/// required). Allocates a fixed-size GC-managed object on the heap and
/// returns a raw pointer the caller owns until the GC takes over.
///
/// Prefer [`malloc_typed`] for any `T` with a registered GC type id —
/// the untyped variant exists only as a temporary bridge for types
/// that have not yet been wired into the per-type metadata table.
/// Non-PyObject heap allocations (Strings, raw `Vec`s manually freed
/// via `Box::from_raw`) belong on [`malloc_raw`], not here, because
/// they must NOT migrate to the managed allocator.
///
/// In Rust the construction and allocation collapse into a single
/// step: callers build the value first and pass it in, instead of
/// PyPy's allocate-then-fill-fields pattern. This is the smallest
/// adaptation of RPython's API to Rust's value-construction model.
#[inline]
pub fn malloc<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

/// Typed variant of [`malloc`]: requires `T: GcType` so the future
/// managed allocator can read `T::TYPE_ID` and `T::SIZE` without a
/// runtime registry lookup. Phase 1 body identical to [`malloc`];
/// Phase 2 will route through the GC-managed allocator with proper
/// `push_roots` / `pop_roots` brackets (`framework.py:853-856`).
///
/// New call sites should prefer [`malloc_typed`] over [`malloc`]
/// once their `T` has an assigned GC type id; the untyped variant
/// remains as a temporary bridge for types not yet registered.
#[inline]
pub fn malloc_typed<T: GcType>(value: T) -> *mut T {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        T::SIZE,
        "GcType::SIZE drift from std::mem::size_of"
    );
    Box::into_raw(Box::new(value))
}

/// `lltype.malloc(T, flavor='raw')` parity. Non-GC heap allocation;
/// caller manages lifetime via `Box::from_raw` later.
///
/// Distinct from [`malloc`] only in intent today (both call
/// `Box::into_raw`); Phase 2 GC integration will keep this on the
/// raw allocator while [`malloc`] moves to the managed allocator.
#[inline]
pub fn malloc_raw<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    // GC-flavored mallocs (`malloc` / `malloc_typed`) are leaked in
    // these tests — the Phase 2 managed allocator forbids
    // `Box::from_raw` on its output, so the tests stay Phase-2-ready
    // by never freeing GC-flavor allocations. Only `malloc_raw`
    // (RPython `flavor='raw'`) is paired with explicit
    // `Box::from_raw` cleanup.

    #[test]
    fn malloc_returns_unique_pointers() {
        let a = malloc(0u64);
        let b = malloc(0u64);
        assert_ne!(a as usize, b as usize);
    }

    #[test]
    fn malloc_writes_value() {
        let p = malloc(42u32);
        unsafe {
            assert_eq!(*p, 42);
        }
    }

    #[test]
    fn malloc_raw_independent_of_malloc() {
        let a = malloc(1u32);
        let b = malloc_raw(2u32);
        assert_ne!(a as usize, b as usize);
        unsafe {
            assert_eq!(*a, 1);
            assert_eq!(*b, 2);
            // `b` came from `malloc_raw` so explicit cleanup is sound.
            drop(Box::from_raw(b));
        }
    }

    struct DummyPayload(u64);
    impl GcType for DummyPayload {
        const TYPE_ID: u32 = 0xDEAD_BEEF;
        const SIZE: usize = std::mem::size_of::<DummyPayload>();
    }

    #[test]
    fn malloc_typed_writes_value_and_reads_type_metadata() {
        assert_eq!(<DummyPayload as GcType>::TYPE_ID, 0xDEAD_BEEF);
        assert_eq!(<DummyPayload as GcType>::SIZE, 8);
        let p = malloc_typed(DummyPayload(7));
        unsafe {
            assert_eq!((*p).0, 7);
        }
    }
}

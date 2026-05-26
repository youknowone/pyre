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
//! Current body: `Box::into_raw(Box::new(value))` — the pre-existing
//! leak baseline. Future work routes through a GC-managed
//! allocator with proper root push/pop.

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

/// Compile-time descriptor every `#[pyre_class]` type emits, consumed
/// by the JIT driver's GC registration loop in
/// `pyre/pyre-jit/src/eval.rs`.  Mirrors the per-type tuple PyPy's
/// `framework.py:807-811` materializes (TYPE_ID + fixed size + GC
/// pointer offsets) plus the static `PyType` the dispatcher uses to
/// recognise the layout at runtime.
pub struct PyreClassDescriptor {
    /// Static `PyType` pointer used by `py_type_check` and stamped
    /// into `ob_header.ob_type`.
    pub pytype_ptr: *const crate::pyobject::PyType,
    /// `GcType::TYPE_ID` for this payload.  Asserted equal to the id
    /// returned by `gc.register_type(...)` in the JIT driver.
    pub gc_type_id: u32,
    /// `GcType::SIZE` for this payload (in bytes).
    pub object_size: usize,
    /// Byte offsets of inline `PyObjectRef` fields the GC must trace.
    pub ptr_offsets: &'static [usize],
}

// Safety: every field is either a static-`'static` reference (PyType,
// ptr_offsets), a primitive, or a raw pointer to read-only static
// storage; sharing across threads is sound.
unsafe impl Sync for PyreClassDescriptor {}

/// Compile-time bridge between a `#[pyre_class]` struct and its
/// per-type static `PyType` / `PyreClassDescriptor`.  Implemented
/// automatically by `#[pyre_class]`; consumed by `py_class_typed!`
/// to thread the static `PyType` pointer through
/// `make_builtin_type_with_layout` without naming the macro-generated
/// suffixed identifier (`RANDOM_TYPE`, `RANDOM_PYRE_CLASS_DESCRIPTOR`,
/// …) at the call site.
pub trait PyreClassPyTypeOf {
    /// Static `PyType` pointer (`*const pyre_object::PyType`).  Read
    /// by `py_class_typed!` and `<W_X>::allocate` to stamp
    /// `ob_header.ob_type`.
    const PYTYPE: *const crate::pyobject::PyType;
    /// Compile-time descriptor consumed by the JIT driver's
    /// `register_pyre_class` helper in `pyre-jit/src/eval.rs`.
    const DESCRIPTOR: &'static PyreClassDescriptor;
    /// Python-visible dotted name (e.g. `"_random.Random"`) carried
    /// verbatim from `#[pyre_class("…", type_id = N)]`.  Consumed by
    /// `#[pyre_methods]` so the impl block doesn't restate it.
    const PYNAME: &'static str;
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
/// runtime registry lookup. Current body identical to [`malloc`];
/// will later route through the GC-managed allocator with proper
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
/// `Box::into_raw`); future GC integration will keep this on the
/// raw allocator while [`malloc`] moves to the managed allocator.
#[inline]
pub fn malloc_raw<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    // GC-flavored mallocs (`malloc` / `malloc_typed`) are leaked in
    // these tests — the managed allocator forbids
    // `Box::from_raw` on its output, so the tests stay forward-compatible
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

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
//! Current body: [`majit_gc::header::alloc_with_gc_header`] — the
//! `malloc_fixedsize` analog that prepends a zeroed [`GcHeader`] and
//! returns the payload pointer, so the write barrier reads a valid header
//! (`TRACK_YOUNG_PTRS = 0`) and skips these still-leaked objects. The GC
//! owns the header (incminimark defines its own `HDR`); the object model
//! delegates and stays header-agnostic. Future work routes through the
//! same allocator with nursery management and proper root push/pop.
//!
//! [`GcHeader`]: majit_gc::header::GcHeader

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
    /// `framework.py:809`.  Read at runtime so the value can be
    /// assigned by the JIT driver after `gc.register_type(...)`
    /// returns — auto-id mode delivers the result through this
    /// accessor.  Explicit `type_id = N` cells return `N` unchanged.
    fn type_id() -> u32;
    /// Fixed payload size in bytes, equal to `info.fixedsize` in
    /// `framework.py:811`.
    const SIZE: usize;
}

/// Process-wide cell that the JIT driver uses to deliver the actual
/// GC tid to a `#[pyre_class]` type after registration.  Two modes:
///
/// 1. `TypeIdCell::auto()` — initialized to [`TypeIdCell::UNASSIGNED`].
///    The driver assigns the next available tid via [`Self::set`] and
///    subsequent runtime reads return it.  This is the ergonomic
///    default for `#[pyre_class("name")]` callers who do not want to
///    reserve a slot up front.
/// 2. `TypeIdCell::with(N)` — pre-initialized with a reserved tid.
///    The driver asserts that its registration order produces the same
///    `N` (drift check).  Used by every legacy `#[pyre_class("…",
///    type_id = N)]` site so the contiguous-monotonic invariant the
///    GC's `pytype_to_tid` table relies on stays self-checking.
///
/// The cell is `repr(transparent)` over `AtomicU32` so its layout is
/// identical to the raw atomic; `const fn` constructors mean it can
/// initialize a `pub static` without `LazyLock` overhead.
#[repr(transparent)]
pub struct TypeIdCell(::std::sync::atomic::AtomicU32);

impl TypeIdCell {
    /// Sentinel meaning "no tid assigned yet".  Picked at the high end
    /// of `u32` so it cannot collide with a real registered tid; the
    /// JIT's contiguous-monotonic table is several orders of magnitude
    /// shorter than `u32::MAX`.
    pub const UNASSIGNED: u32 = u32::MAX;

    /// Cell initialized to [`Self::UNASSIGNED`] — auto-id mode.
    pub const fn auto() -> Self {
        Self(::std::sync::atomic::AtomicU32::new(Self::UNASSIGNED))
    }

    /// Cell pre-initialized with an explicit tid — explicit-id mode
    /// (drift-checked at JIT init).
    pub const fn with(n: u32) -> Self {
        Self(::std::sync::atomic::AtomicU32::new(n))
    }

    /// Current value of the cell.  Returns [`Self::UNASSIGNED`] when
    /// auto-mode and the JIT driver has not registered the type yet.
    #[inline]
    pub fn get(&self) -> u32 {
        self.0.load(::std::sync::atomic::Ordering::Acquire)
    }

    /// Write the runtime-assigned tid into the cell.  Called once per
    /// type by the JIT driver's `register_pyre_class` helper.
    #[inline]
    pub fn set(&self, n: u32) {
        self.0.store(n, ::std::sync::atomic::Ordering::Release)
    }

    /// `true` iff the cell still holds [`Self::UNASSIGNED`].  The JIT
    /// driver uses this to decide between writing (auto) and asserting
    /// (explicit).
    #[inline]
    pub fn is_unassigned(&self) -> bool {
        self.get() == Self::UNASSIGNED
    }
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
    /// Runtime-resolved GC tid cell.  Either pre-initialized with an
    /// explicit `type_id = N` (then drift-checked) or starts at
    /// [`TypeIdCell::UNASSIGNED`] and gets stamped by the JIT driver.
    pub gc_type_id: &'static TypeIdCell,
    /// `GcType::SIZE` for this payload (in bytes).
    pub object_size: usize,
    /// Byte offsets of inline `PyObjectRef` fields the GC must trace.
    pub ptr_offsets: &'static [usize],
}

// Safety: every field is either a static-`'static` reference (PyType,
// gc_type_id, ptr_offsets), a primitive, or a raw pointer to read-only
// static storage; sharing across threads is sound.
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
    majit_gc::header::alloc_with_gc_header(value)
}

/// Typed variant of [`malloc`]: requires `T: GcType` so the future
/// managed allocator can read `T::TYPE_ID` and `T::SIZE` without a
/// runtime registry lookup. Shares [`malloc`]'s body — the
/// `alloc_with_gc_header` prepend — and will later route through the
/// GC-managed allocator with proper `push_roots` / `pop_roots` brackets
/// (`framework.py:853-856`), writing the real `T::TYPE_ID` into the
/// header instead of the current zeroed placeholder.
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
    majit_gc::header::alloc_with_gc_header(value)
}

/// `lltype.malloc(T, flavor='raw')` parity. Non-GC heap allocation;
/// caller manages lifetime via `Box::from_raw` later.
///
/// Unlike [`malloc`] / [`malloc_typed`] (which now prepend a [`GcHeader`]
/// via `alloc_with_gc_header`), this stays a bare `Box::into_raw` with no
/// header, so `Box::from_raw` on its output remains sound. Used for
/// `String`s, dict `dstorage`, and other allocations that must NOT migrate
/// to the managed allocator.
///
/// [`GcHeader`]: majit_gc::header::GcHeader
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
    fn malloc_prepends_zeroed_gc_header() {
        let p = malloc(0x1234_5678_u64);
        unsafe {
            assert_eq!(*p, 0x1234_5678);
            // A zeroed GcHeader precedes the payload, so the write barrier
            // reads `*(obj - SIZE)` as TRACK_YOUNG_PTRS=0 and skips it.
            let hdr = majit_gc::header::header_of(p as usize);
            assert_eq!((*hdr).tid_and_flags, 0);
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
        fn type_id() -> u32 {
            0xDEAD_BEEF
        }
        const SIZE: usize = std::mem::size_of::<DummyPayload>();
    }

    #[test]
    fn malloc_typed_writes_value_and_reads_type_metadata() {
        assert_eq!(<DummyPayload as GcType>::type_id(), 0xDEAD_BEEF);
        assert_eq!(<DummyPayload as GcType>::SIZE, 8);
        let p = malloc_typed(DummyPayload(7));
        unsafe {
            assert_eq!((*p).0, 7);
        }
    }
}

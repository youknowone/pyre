//! W_BytesObject — Python `bytes` type (immutable).
//!
//! PyPy equivalent: pypy/objspace/std/bytesobject.py W_BytesObject
//!
//! Immutable byte sequence. Shares the same internal layout as
//! W_BytearrayObject but provides no mutation functions.

use crate::pyobject::*;

pub static BYTES_TYPE: PyType = crate::pyobject::new_pytype("bytes");

/// GC-managed byte buffer shared by `bytes` and `bytearray` bodies.
///
/// The `Vec<u8>` is a leaf (no inner `PyObjectRef`s); its GC box carries only
/// drop glue that reclaims the buffer on sweep.
pub type BytesDataStorage = Vec<u8>;

/// Runtime-assigned GC type id for [`BytesDataStorage`]. Like the set-items
/// box, this is published by `pyre-jit::eval` after the fixed-constant type
/// registrations and is never embedded in a JIT allocation descriptor.
static BYTES_DATA_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for [`BytesDataStorage`].
pub fn set_bytes_data_gc_type_id(id: u32) {
    BYTES_DATA_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for [`BytesDataStorage`].
#[majit_macros::dont_look_inside]
pub fn bytes_data_gc_type_id() -> u32 {
    BYTES_DATA_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Python bytes object — immutable byte sequence.
///
/// PyPy: W_BytesObject stores `_value` (RPython string).
/// pyre: stores a heap-allocated `Vec<u8>` in a GC-managed non-moving storage
/// box (off-GC storage epic S4), same layout as W_BytearrayObject but without
/// setitem/extend.
#[repr(C)]
pub struct W_BytesObject {
    pub ob_header: PyObject,
    pub data: *const Vec<u8>,
    pub len: usize,
    /// Strong references owned by ctypes `_objects` dictionaries.  Pyre is a
    /// tracing-GC runtime, so it has no CPython `ob_refcnt`; this trailing
    /// counter preserves the observable ctypes-owned delta used by
    /// `sys.getrefcount` compatibility without changing object identity or
    /// storing a parallel object side table.
    pub ctypes_keepalive_refs: usize,
    /// Mapdict's per-instance `dict` SPECIAL slot for a user subclass.
    /// Exact bytes objects leave this null.
    pub w_dict: PyObjectRef,
    /// Mapdict's per-instance `weakref` SPECIAL slot for a user subclass.
    /// Exact bytes objects leave this null.
    pub w_weakreflifeline: PyObjectRef,
}

/// GC type id assigned to `W_BytesObject` at JitDriver init time.
pub const W_BYTES_GC_TYPE_ID: u32 = 27;

/// Fixed payload size (`framework.py:811`).
pub const W_BYTES_OBJECT_SIZE: usize = std::mem::size_of::<W_BytesObject>();

impl crate::lltype::GcType for W_BytesObject {
    fn type_id() -> u32 {
        W_BYTES_GC_TYPE_ID
    }
    const SIZE: usize = W_BYTES_OBJECT_SIZE;
}

/// Allocate a new bytes object from a byte slice.
///
/// The `data` buffer lives in a GC-managed non-moving storage box; the sweep
/// reclaims it through the box tid's drop glue. The `W_BytesObject` body is
/// allocated in GC old-gen (`try_gc_alloc_stable_raw`) so the collector traces
/// through it and greys the box, mirroring `w_list_new`/`w_set_new`. Falls back
/// to `malloc_typed`/`malloc_raw` when no GC hook is installed (unit tests).
///
/// `dont_look_inside` (`rlib/jit.py:139`): the tracer cannot model the box
/// allocation, so the JIT residualises the call.
#[majit_macros::dont_look_inside]
pub fn w_bytes_from_bytes(bytes: &[u8]) -> PyObjectRef {
    let len = bytes.len();
    let data = crate::gc_storage::gc_alloc_storage_box(bytes.to_vec(), bytes_data_gc_type_id());
    let header = PyObject {
        ob_type: &BYTES_TYPE as *const PyType,
        w_class: get_instantiate(&BYTES_TYPE),
    };
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_BYTES_GC_TYPE_ID, W_BYTES_OBJECT_SIZE);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut W_BytesObject,
                W_BytesObject {
                    ob_header: header,
                    data,
                    len,
                    ctypes_keepalive_refs: 0,
                    w_dict: PY_NULL,
                    w_weakreflifeline: PY_NULL,
                },
            );
        }
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(W_BytesObject {
            ob_header: header,
            data,
            len,
            ctypes_keepalive_refs: 0,
            w_dict: PY_NULL,
            w_weakreflifeline: PY_NULL,
        }) as PyObjectRef
    }
}

/// Allocate a bytes-subclass instance in the managed heap.  PyPy's
/// `W_BytesObject` user subclasses carry mapdict state and therefore
/// participate in cycle collection; only exact immutable bytes may use the
/// prebuilt/immortal allocation path above.
pub fn w_bytes_subclass_from_bytes(bytes: &[u8], w_class: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(w_class);
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(
        <W_BytesObject as crate::lltype::GcType>::type_id(),
        <W_BytesObject as crate::lltype::GcType>::SIZE,
    );
    let payload = W_BytesObject {
        ob_header: PyObject {
            ob_type: &BYTES_TYPE as *const PyType,
            w_class: crate::gc_roots::shadow_stack_get(root_base),
        },
        data: crate::lltype::malloc_raw(bytes.to_vec()),
        len: bytes.len(),
        ctypes_keepalive_refs: 0,
        w_dict: PY_NULL,
        w_weakreflifeline: PY_NULL,
    };
    if raw.is_null() {
        crate::lltype::malloc_typed(payload) as PyObjectRef
    } else {
        unsafe { std::ptr::write(raw as *mut W_BytesObject, payload) };
        raw as PyObjectRef
    }
}

#[inline]
pub unsafe fn w_bytes_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BytesObject)).w_dict }
}

#[inline]
pub unsafe fn w_bytes_setdict(obj: PyObjectRef, w_dict: PyObjectRef) {
    unsafe { (*(obj as *mut W_BytesObject)).w_dict = w_dict };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

#[inline]
pub unsafe fn w_bytes_getweakref(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BytesObject)).w_weakreflifeline }
}

#[inline]
pub unsafe fn w_bytes_setweakref(obj: PyObjectRef, lifeline: PyObjectRef) {
    unsafe { (*(obj as *mut W_BytesObject)).w_weakreflifeline = lifeline };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// Allocate an empty bytes object.
pub fn w_bytes_empty() -> PyObjectRef {
    w_bytes_from_bytes(&[])
}

#[inline]
pub unsafe fn is_bytes(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BYTES_TYPE) }
}

#[inline]
pub unsafe fn w_bytes_len(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BytesObject)).len }
}

pub unsafe fn w_bytes_ctypes_keepalive_refs(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BytesObject)).ctypes_keepalive_refs }
}

pub unsafe fn w_bytes_inc_ctypes_keepalive_refs(obj: PyObjectRef) {
    let bytes = unsafe { &mut *(obj as *mut W_BytesObject) };
    bytes.ctypes_keepalive_refs = bytes.ctypes_keepalive_refs.saturating_add(1);
}

pub unsafe fn w_bytes_dec_ctypes_keepalive_refs(obj: PyObjectRef) {
    let bytes = unsafe { &mut *(obj as *mut W_BytesObject) };
    bytes.ctypes_keepalive_refs = bytes.ctypes_keepalive_refs.saturating_sub(1);
}

#[inline]
pub unsafe fn w_bytes_getitem(obj: PyObjectRef, index: usize) -> u8 {
    unsafe { w_bytes_data(obj)[index] }
}

/// Get a reference to the internal data.
pub unsafe fn w_bytes_data(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        let b = obj as *const W_BytesObject;
        let data_ref: &Vec<u8> = &*(*b).data;
        data_ref.as_slice()
    }
}

/// bytes.find(sub, start) — find first occurrence of byte value.
pub unsafe fn w_bytes_find(obj: PyObjectRef, value: u8, start: usize) -> i64 {
    unsafe {
        let data = w_bytes_data(obj);
        for i in start..data.len() {
            if data[i] == value {
                return i as i64;
            }
        }
        -1
    }
}

// ── bytes-like helpers ────────────────────────────────────────────────
//
// Many Python operations accept both bytes and bytearray ("bytes-like").
// These helpers abstract over both types for read-only operations.

/// Check if obj is bytes or bytearray (bytes-like object).
#[inline]
pub unsafe fn is_bytes_like(obj: PyObjectRef) -> bool {
    unsafe { is_bytes(obj) || crate::bytearrayobject::is_bytearray(obj) }
}

/// Get length of a bytes-like object.
#[inline]
pub unsafe fn bytes_like_len(obj: PyObjectRef) -> usize {
    unsafe {
        if is_bytes(obj) {
            w_bytes_len(obj)
        } else {
            crate::bytearrayobject::w_bytearray_len(obj)
        }
    }
}

/// Get byte at index from a bytes-like object.
#[inline]
pub unsafe fn bytes_like_getitem(obj: PyObjectRef, index: usize) -> u8 {
    unsafe {
        if is_bytes(obj) {
            w_bytes_getitem(obj, index)
        } else {
            crate::bytearrayobject::w_bytearray_getitem(obj, index)
        }
    }
}

/// Get data slice from a bytes-like object.
#[inline]
pub unsafe fn bytes_like_data(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        if is_bytes(obj) {
            w_bytes_data(obj)
        } else {
            crate::bytearrayobject::w_bytearray_data(obj)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_basic() {
        let b = w_bytes_from_bytes(b"hello");
        unsafe {
            assert!(is_bytes(b));
            assert_eq!(w_bytes_len(b), 5);
            assert_eq!(w_bytes_getitem(b, 0), b'h');
            assert_eq!(w_bytes_getitem(b, 4), b'o');
            assert_eq!(w_bytes_data(b), b"hello");
            assert_eq!(w_bytes_find(b, b'l', 0), 2);
            assert_eq!(w_bytes_find(b, b'x', 0), -1);
        }
    }

    #[test]
    fn test_bytes_empty() {
        let b = w_bytes_empty();
        unsafe {
            assert!(is_bytes(b));
            assert_eq!(w_bytes_len(b), 0);
        }
    }

    #[test]
    fn w_bytes_gc_type_id_matches_descr() {
        assert_eq!(W_BYTES_GC_TYPE_ID, 27);
        assert_eq!(
            <W_BytesObject as crate::lltype::GcType>::type_id(),
            W_BYTES_GC_TYPE_ID
        );
        assert_eq!(
            <W_BytesObject as crate::lltype::GcType>::SIZE,
            W_BYTES_OBJECT_SIZE
        );
    }
}

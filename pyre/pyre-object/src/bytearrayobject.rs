//! W_BytearrayObject — Python `bytearray` type.
//!
//! PyPy equivalent: pypy/objspace/std/bytearrayobject.py

use crate::pyobject::*;

pub static BYTEARRAY_TYPE: PyType = crate::pyobject::new_pytype("bytearray");

/// Python bytearray object.
///
/// Layout: `[ob_type | data | exports]`
#[repr(C)]
pub struct W_BytearrayObject {
    pub ob_header: PyObject,
    pub data: *mut Vec<u8>,
    /// CPython `PyByteArrayObject.ob_alloc`, including the trailing NUL byte.
    ///
    /// This cannot be derived from `Vec::capacity()`: Rust's allocator uses a
    /// different growth policy, while `bytearray.__alloc__()` and
    /// `bytearray.__sizeof__()` expose CPython's logical allocation directly.
    pub alloc: usize,
    /// Logical `ob_start - ob_bytes` offset. pyre keeps the payload itself at
    /// `Vec[0]`, but preserves this allocation state for prefix slice deletes.
    pub logical_offset: usize,
    /// `_exports` — count of active buffer exports.  Size-changing mutators
    /// are refused while this is positive (`_check_exports`).
    pub exports: i64,
    /// Mapdict `dict`/`weakref` SPECIAL slots for user subclasses.
    pub w_dict: PyObjectRef,
    pub w_weakreflifeline: PyObjectRef,
}

/// GC type id assigned to `W_BytearrayObject` at JitDriver init time.
pub const W_BYTEARRAY_GC_TYPE_ID: u32 = 28;

/// Fixed payload size (`framework.py:811`).
pub const W_BYTEARRAY_OBJECT_SIZE: usize = std::mem::size_of::<W_BytearrayObject>();

impl crate::lltype::GcType for W_BytearrayObject {
    fn type_id() -> u32 {
        W_BYTEARRAY_GC_TYPE_ID
    }
    const SIZE: usize = W_BYTEARRAY_OBJECT_SIZE;
}

/// Allocate a new bytearray from an owned byte buffer.
///
/// The `data` buffer lives in a GC-managed non-moving storage box (reusing the
/// shared `bytes` data box tid — identical `Vec<u8>` type); the sweep reclaims
/// it through the box tid's drop glue. The `W_BytearrayObject` body is allocated
/// in GC old-gen (`try_gc_alloc_stable_raw`) so the collector traces through it
/// and greys the box, mirroring `w_list_new`/`w_set_new`. Falls back to
/// `malloc_typed`/`malloc_raw` when no GC hook is installed (unit tests).
///
/// Not a residual boundary itself: its only callers are the two
/// `dont_look_inside` public constructors below, so the tracer never reaches it
/// (the `Vec<u8>`-by-value argument never crosses a residual call ABI).
fn w_bytearray_alloc(buf: Vec<u8>) -> PyObjectRef {
    let alloc = if buf.is_empty() { 0 } else { buf.len() + 1 };
    let data =
        crate::gc_storage::gc_alloc_storage_box(buf, crate::bytesobject::bytes_data_gc_type_id());
    let header = PyObject {
        ob_type: &BYTEARRAY_TYPE as *const PyType,
        w_class: get_instantiate(&BYTEARRAY_TYPE),
    };
    let raw =
        crate::gc_hook::try_gc_alloc_stable_raw(W_BYTEARRAY_GC_TYPE_ID, W_BYTEARRAY_OBJECT_SIZE);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut W_BytearrayObject,
                W_BytearrayObject {
                    ob_header: header,
                    data,
                    alloc,
                    logical_offset: 0,
                    exports: 0,
                    w_dict: PY_NULL,
                    w_weakreflifeline: PY_NULL,
                },
            );
        }
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(W_BytearrayObject {
            ob_header: header,
            data,
            alloc,
            logical_offset: 0,
            exports: 0,
            w_dict: PY_NULL,
            w_weakreflifeline: PY_NULL,
        }) as PyObjectRef
    }
}

/// Allocate a new bytearray filled with zeros.
///
/// `dont_look_inside` (`rlib/jit.py:139`): the GC-managed box allocation is not
/// phaseA-liftable, so the JIT residualises the call, matching the
/// `w_bytes_from_bytes` residual boundary.
#[majit_macros::dont_look_inside]
pub fn w_bytearray_new(size: usize) -> PyObjectRef {
    w_bytearray_alloc(vec![0u8; size])
}

/// [`w_bytearray_new`] for a `size` that came from Python, returning `None`
/// when the buffer cannot be reserved.
///
/// `bytearray(sys.maxsize)` reaches the allocator with an unsatisfiable
/// request; an infallible `vec!` aborts the process there, while the caller
/// needs to raise `MemoryError`.
#[majit_macros::dont_look_inside]
pub fn w_bytearray_try_new(size: usize) -> Option<PyObjectRef> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(size).ok()?;
    buf.resize(size, 0);
    Some(w_bytearray_alloc(buf))
}

/// Allocate a new bytearray from a byte slice.
///
/// `dont_look_inside` (`rlib/jit.py:139`): see [`w_bytearray_new`].
#[majit_macros::dont_look_inside]
pub fn w_bytearray_from_bytes(bytes: &[u8]) -> PyObjectRef {
    w_bytearray_alloc(bytes.to_vec())
}

/// Allocate a bytearray user subclass in the managed heap so its mapdict
/// edges and buffer-export cycles participate in collection.
pub fn w_bytearray_subclass_from_bytes(bytes: &[u8], w_class: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(w_class);
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(
        <W_BytearrayObject as crate::lltype::GcType>::type_id(),
        <W_BytearrayObject as crate::lltype::GcType>::SIZE,
    );
    let payload = W_BytearrayObject {
        ob_header: PyObject {
            ob_type: &BYTEARRAY_TYPE as *const PyType,
            w_class: crate::gc_roots::shadow_stack_get(root_base),
        },
        data: crate::lltype::malloc_raw(bytes.to_vec()),
        alloc: if bytes.is_empty() { 0 } else { bytes.len() + 1 },
        logical_offset: 0,
        exports: 0,
        w_dict: PY_NULL,
        w_weakreflifeline: PY_NULL,
    };
    let obj = if raw.is_null() {
        crate::lltype::malloc_typed(payload) as PyObjectRef
    } else {
        unsafe { std::ptr::write(raw as *mut W_BytearrayObject, payload) };
        raw as PyObjectRef
    };
    // `allocate_instance` registers the fresh instance on the finalizer queue
    // when its subtype has `__del__`; the `w_class` is already stamped above,
    // so the hook resolves the subclass rather than the canonical bytearray type.
    crate::gc_hook::maybe_register_finalizer(obj);
    obj
}

#[inline]
pub unsafe fn w_bytearray_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BytearrayObject)).w_dict }
}

#[inline]
pub unsafe fn w_bytearray_setdict(obj: PyObjectRef, w_dict: PyObjectRef) {
    unsafe { (*(obj as *mut W_BytearrayObject)).w_dict = w_dict };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

#[inline]
pub unsafe fn w_bytearray_getweakref(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BytearrayObject)).w_weakreflifeline }
}

#[inline]
pub unsafe fn w_bytearray_setweakref(obj: PyObjectRef, lifeline: PyObjectRef) {
    unsafe { (*(obj as *mut W_BytearrayObject)).w_weakreflifeline = lifeline };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

pub unsafe fn is_bytearray(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BYTEARRAY_TYPE) }
}

pub unsafe fn w_bytearray_len(obj: PyObjectRef) -> usize {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        (*ba.data).len()
    }
}

/// CPython 3.14 `PyByteArrayObject.ob_alloc`, including the trailing NUL.
pub unsafe fn w_bytearray_capacity(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BytearrayObject)).alloc }
}

/// Port of CPython 3.14 `bytearray_resize_lock_held`'s allocation policy.
pub unsafe fn w_bytearray_sync_alloc(obj: PyObjectRef, old_size: usize) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        let size = (*ba.data).len();
        if size == old_size {
            return;
        }
        let current = ba.alloc;
        let fits = size + ba.logical_offset < current;
        if fits && size >= current / 2 {
            return;
        }
        ba.alloc = if fits {
            size + 1
        } else if size <= current + (current >> 3) {
            size + (size >> 3) + if size < 9 { 3 } else { 6 }
        } else {
            size + 1
        };
        ba.logical_offset = 0;
    }
}

/// CPython `bytearray_setslice_linear`: a shrinking prefix slice advances
/// `ob_start` before entering the resize policy.
pub unsafe fn w_bytearray_advance_logical_start(obj: PyObjectRef, amount: usize) {
    unsafe { (*(obj as *mut W_BytearrayObject)).logical_offset += amount }
}

pub unsafe fn w_bytearray_getitem(obj: PyObjectRef, index: usize) -> u8 {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        (&*ba.data)[index]
    }
}

pub unsafe fn w_bytearray_setitem(obj: PyObjectRef, index: usize, value: u8) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        (&mut *ba.data)[index] = value;
    }
}

/// bytearray.find(sub, start) — find first occurrence of byte value.
pub unsafe fn w_bytearray_find(obj: PyObjectRef, value: u8, start: usize) -> i64 {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        let data = &*ba.data;
        for i in start..data.len() {
            if data[i] == value {
                return i as i64;
            }
        }
        -1
    }
}

/// Concatenate bytearray + bytes (b'\0' * N pattern).
pub unsafe fn w_bytearray_extend(obj: PyObjectRef, other: &[u8]) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        let old_size = (*ba.data).len();
        (*ba.data).extend_from_slice(other);
        w_bytearray_sync_alloc(obj, old_size);
    }
}

/// Get a reference to the internal data.
pub unsafe fn w_bytearray_data(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        &*ba.data
    }
}

/// Get a mutable reference to the internal data. Caller must ensure
/// the bytearray is not aliased while the returned slice is live.
pub unsafe fn w_bytearray_data_mut(obj: PyObjectRef) -> &'static mut [u8] {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        &mut *ba.data
    }
}

/// Get a mutable reference to the backing `Vec`, for length-changing
/// mutators (append / insert / remove / pop / clear).  Caller must
/// ensure the bytearray is not aliased while the reference is live and call
/// [`w_bytearray_sync_alloc`] after every actual length change.
pub unsafe fn w_bytearray_vec_mut(obj: PyObjectRef) -> &'static mut Vec<u8> {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        &mut *ba.data
    }
}

/// `_exports` — number of live buffer exports over this bytearray.
pub unsafe fn w_bytearray_exports(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_BytearrayObject)).exports }
}

/// `buffer_w` — record a new live buffer export.
pub unsafe fn w_bytearray_exports_incref(obj: PyObjectRef) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        ba.exports += 1;
    }
}

/// `bf_releasebuffer` — a consumer released its buffer export.  A release
/// without a matching acquisition is a fatal accounting bug
/// (`_exports_underflow`).
pub unsafe fn w_bytearray_exports_decref(obj: PyObjectRef) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        if ba.exports <= 0 {
            panic!(
                "bytearray bf_releasebuffer: _exports underflow: id={obj:?} exports={}",
                ba.exports
            );
        }
        ba.exports -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytearray_basic() {
        let ba = w_bytearray_new(10);
        unsafe {
            assert!(is_bytearray(ba));
            assert_eq!(w_bytearray_len(ba), 10);
            assert_eq!(w_bytearray_getitem(ba, 0), 0);
            w_bytearray_setitem(ba, 3, 1);
            assert_eq!(w_bytearray_getitem(ba, 3), 1);
            assert_eq!(w_bytearray_find(ba, 1, 0), 3);
            assert_eq!(w_bytearray_find(ba, 1, 4), -1);
        }
    }

    #[test]
    fn w_bytearray_gc_type_id_matches_descr() {
        assert_eq!(W_BYTEARRAY_GC_TYPE_ID, 28);
        assert_eq!(
            <W_BytearrayObject as crate::lltype::GcType>::type_id(),
            W_BYTEARRAY_GC_TYPE_ID
        );
        assert_eq!(
            <W_BytearrayObject as crate::lltype::GcType>::SIZE,
            W_BYTEARRAY_OBJECT_SIZE
        );
    }
}

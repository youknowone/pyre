use std::ops::{Index, IndexMut};

use crate::{PY_NULL, PyObjectRef};

/// Small-buffer capacity. Arrays up to this size avoid heap allocation.
const INLINE_CAP: usize = 8;

/// Offset of the backing pointer inside `PyObjectArray`.
pub const PYOBJECT_ARRAY_PTR_OFFSET: usize = std::mem::offset_of!(PyObjectArray, ptr);

/// Offset of the live length inside `PyObjectArray`.
pub const PYOBJECT_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(PyObjectArray, len);

/// Offset of the heap capacity inside `PyObjectArray`.
pub const PYOBJECT_ARRAY_HEAP_CAP_OFFSET: usize = std::mem::offset_of!(PyObjectArray, heap_cap);

/// Inline capacity used by `PyObjectArray`.
pub const PYOBJECT_ARRAY_INLINE_CAP: usize = INLINE_CAP;

/// Fixed-size object array with small-buffer optimization.
///
/// Arrays of up to 8 elements are stored inline (no heap allocation).
/// Larger arrays fall back to a heap allocation.
///
/// The `ptr` field (offset 0) is used by the JIT for direct memory access.
/// Call `fix_ptr()` after any struct move to keep inline storage valid.
#[repr(C)]
pub struct PyObjectArray {
    /// Raw pointer to the active storage. JIT reads this at offset 0.
    pub ptr: *mut PyObjectRef,
    /// Number of live elements.
    len: usize,
    /// Heap capacity. 0 = inline mode, >0 = heap mode.
    heap_cap: usize,
    /// Inline storage for small arrays.
    inline_buf: [PyObjectRef; INLINE_CAP],
}

impl PyObjectArray {
    pub fn filled(len: usize, value: PyObjectRef) -> Self {
        if len <= INLINE_CAP {
            let mut arr = Self {
                ptr: std::ptr::null_mut(),
                len,
                heap_cap: 0,
                inline_buf: [value; INLINE_CAP],
            };
            arr.ptr = arr.inline_buf.as_mut_ptr();
            arr
        } else {
            let mut storage = vec![value; len];
            let ptr = storage.as_mut_ptr();
            let cap = storage.capacity();
            std::mem::forget(storage);
            Self {
                ptr,
                len,
                heap_cap: cap,
                inline_buf: [PY_NULL; INLINE_CAP],
            }
        }
    }

    pub fn from_vec(mut values: Vec<PyObjectRef>) -> Self {
        let len = values.len();
        if len <= INLINE_CAP {
            let mut inline_buf = [PY_NULL; INLINE_CAP];
            inline_buf[..len].copy_from_slice(&values);
            let mut arr = Self {
                ptr: std::ptr::null_mut(),
                len,
                heap_cap: 0,
                inline_buf,
            };
            arr.ptr = arr.inline_buf.as_mut_ptr();
            arr
        } else {
            let ptr = values.as_mut_ptr();
            let cap = values.capacity();
            std::mem::forget(values);
            Self {
                ptr,
                len,
                heap_cap: cap,
                inline_buf: [PY_NULL; INLINE_CAP],
            }
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        if self.heap_cap > 0 {
            self.heap_cap
        } else {
            INLINE_CAP
        }
    }

    #[inline]
    pub fn spare_capacity(&self) -> usize {
        self.capacity().saturating_sub(self.len)
    }

    #[inline]
    pub fn is_inline(&self) -> bool {
        self.heap_cap == 0
    }

    fn grow_to_heap(&mut self, min_cap: usize) {
        let target_cap = min_cap.max(INLINE_CAP * 2);
        let mut values = Vec::with_capacity(target_cap);
        values.extend_from_slice(&self.inline_buf[..self.len]);
        self.ptr = values.as_mut_ptr();
        self.heap_cap = values.capacity();
        std::mem::forget(values);
    }

    fn grow_heap(&mut self, min_cap: usize) {
        let target_cap = min_cap.max(self.heap_cap.saturating_mul(2).max(1));
        unsafe {
            let mut values = Vec::from_raw_parts(self.ptr, self.len, self.heap_cap);
            values.reserve(target_cap.saturating_sub(values.capacity()));
            self.ptr = values.as_mut_ptr();
            self.heap_cap = values.capacity();
            std::mem::forget(values);
        }
    }

    pub fn push(&mut self, value: PyObjectRef) {
        if self.len == self.capacity() {
            if self.heap_cap == 0 {
                self.grow_to_heap(self.len + 1);
            } else {
                self.grow_heap(self.len + 1);
            }
        }

        if self.heap_cap > 0 {
            unsafe {
                *self.ptr.add(self.len) = value;
            }
        } else {
            self.inline_buf[self.len] = value;
            self.ptr = self.inline_buf.as_mut_ptr();
        }
        self.len += 1;
    }

    /// Repoint `ptr` to the current inline buffer address after a struct move.
    /// No-op for heap-backed arrays (ptr remains valid across moves).
    #[inline]
    pub fn fix_ptr(&mut self) {
        if self.heap_cap == 0 {
            self.ptr = self.inline_buf.as_mut_ptr();
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[PyObjectRef] {
        if self.heap_cap > 0 {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        } else {
            &self.inline_buf[..self.len]
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [PyObjectRef] {
        if self.heap_cap > 0 {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        } else {
            &mut self.inline_buf[..self.len]
        }
    }

    pub fn to_vec(&self) -> Vec<PyObjectRef> {
        self.as_slice().to_vec()
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.as_mut_slice().swap(a, b);
    }
}

impl Drop for PyObjectArray {
    fn drop(&mut self) {
        if self.heap_cap > 0 {
            unsafe {
                drop(Vec::from_raw_parts(self.ptr, self.len, self.heap_cap));
            }
        }
    }
}

impl Index<usize> for PyObjectArray {
    type Output = PyObjectRef;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        if self.heap_cap > 0 {
            unsafe { &*self.ptr.add(index) }
        } else {
            &self.inline_buf[index]
        }
    }
}

impl IndexMut<usize> for PyObjectArray {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if self.heap_cap > 0 {
            unsafe { &mut *self.ptr.add(index) }
        } else {
            &mut self.inline_buf[index]
        }
    }
}

// ─── GcArray: RPython GC array allocation + setarrayitem parity ──────────
//
// resume.py:1444-1537 ResumeDataDirectReader:
//   allocate_array(length, arraydescr, clear) → GCREF
//   setarrayitem_ref(array, index, fieldnum, arraydescr) → write decoded ref
//   setarrayitem_int(array, index, fieldnum, arraydescr) → write decoded int
//   setarrayitem_float(array, index, fieldnum, arraydescr) → write decoded float
//
// In pyre, GcArray is a boxed PyObjectArray on the heap. The returned
// pointer is a raw `*mut PyObjectArray` cast to usize for GcRef.

// ─── GcTypedArray: RPython typed GC array ────────────────────────────
//
// llmodel.py:788-789: bh_new_array / bh_new_array_clear
// llmodel.py:607-619: bh_setarrayitem_gc_r/i/f, bh_getarrayitem_gc_r/i/f
// resume.py:1444-1537: ResumeDataDirectReader allocate_array + setarrayitem_*
//
// RPython GC arrays are typed: ref[], int[], float[]. Each slot stores
// a raw value of the corresponding type. pyre's GcTypedArray preserves
// this distinction.

/// RPython typed GC array — descr.py:273 ArrayDescr.flag parity.
pub enum GcTypedArray {
    /// FLAG_POINTER: each slot is a PyObjectRef (GCREF).
    Ref(Vec<PyObjectRef>),
    /// FLAG_SIGNED/FLAG_UNSIGNED: each slot is a raw i64.
    Int(Vec<i64>),
    /// FLAG_FLOAT: each slot is a raw f64.
    Float(Vec<f64>),
    /// FLAG_STRUCT: Array(Struct(...)) — flat byte buffer.
    /// llmodel.py:648-665 bh_setinteriorfield_gc_* parity.
    /// Layout: num_elems elements, each item_size bytes, stored inline.
    /// Access: elem_idx * item_size + field_offset.
    Struct {
        item_size: usize,
        num_elems: usize,
        data: Vec<u8>,
    },
}

/// Array element kind — resume.py:656 arraydescr.is_array_of_* / FLAG_STRUCT parity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayKind {
    Ref,
    Int,
    Float,
    /// Array(Struct(...)) — interior fields, item_size from arraydescr.
    Struct,
}

/// resume.py:1444-1447, llmodel.py:788-790 parity.
/// RPython: `bh_new_array_clear = bh_new_array` (llmodel.py:790).
/// Both call `gc_malloc_array` which allocates from the GC nursery
/// (always zero-filled). The `clear` parameter preserves the API
/// distinction from resume.py:1444 but has no behavioral difference.
pub fn allocate_array(length: usize, kind: ArrayKind, _clear: bool) -> *mut GcTypedArray {
    // llmodel.py:790: bh_new_array_clear = bh_new_array.
    // Both allocate from gc_malloc_array (always zero-filled).
    let arr = match kind {
        ArrayKind::Ref => GcTypedArray::Ref(vec![PY_NULL; length]),
        ArrayKind::Int => GcTypedArray::Int(vec![0i64; length]),
        ArrayKind::Float => GcTypedArray::Float(vec![0.0f64; length]),
        ArrayKind::Struct => GcTypedArray::Struct {
            item_size: 0,
            num_elems: length,
            data: Vec::new(),
        },
    };
    Box::into_raw(Box::new(arr))
}

/// resume.py:749 VArrayStructInfo.allocate parity.
/// Allocate a flat byte buffer for Array(Struct(...)).
/// Layout: num_elems elements × item_size bytes, zero-filled.
/// llmodel.py: gc_malloc_array(basesize + num_elems * itemsize).
pub fn allocate_array_struct(num_elems: usize, item_size: usize) -> *mut GcTypedArray {
    let total_bytes = num_elems * item_size;
    let arr = GcTypedArray::Struct {
        item_size,
        num_elems,
        data: vec![0u8; total_bytes],
    };
    Box::into_raw(Box::new(arr))
}

/// llmodel.py:607-609 bh_setarrayitem_gc_r parity.
pub fn setarrayitem_ref(array: *mut GcTypedArray, index: usize, value: PyObjectRef) {
    if array.is_null() {
        return;
    }
    let arr = unsafe { &mut *array };
    if let GcTypedArray::Ref(v) = arr {
        if index < v.len() {
            v[index] = value;
        }
    }
}

/// llmodel.py:613-615 bh_setarrayitem_gc_i parity.
/// Write a raw i64 to an int array slot.
pub fn setarrayitem_int(array: *mut GcTypedArray, index: usize, value: i64) {
    if array.is_null() {
        return;
    }
    let arr = unsafe { &mut *array };
    match arr {
        GcTypedArray::Int(v) => {
            if index < v.len() {
                v[index] = value;
            }
        }
        // Fallback: box as W_IntObject into ref array
        GcTypedArray::Ref(v) => {
            if index < v.len() {
                v[index] = crate::intobject::w_int_new(value);
            }
        }
        _ => {}
    }
}

/// llmodel.py:618-619 bh_setarrayitem_gc_f parity.
/// Write a raw f64 to a float array slot.
pub fn setarrayitem_float(array: *mut GcTypedArray, index: usize, value: f64) {
    if array.is_null() {
        return;
    }
    let arr = unsafe { &mut *array };
    match arr {
        GcTypedArray::Float(v) => {
            if index < v.len() {
                v[index] = value;
            }
        }
        // Fallback: box as W_FloatObject into ref array
        GcTypedArray::Ref(v) => {
            if index < v.len() {
                v[index] = crate::floatobject::w_float_new(value);
            }
        }
        _ => {}
    }
}

/// resume.py:757 setinteriorfield(i, array, num, fielddescrs[j]) parity.
/// resume.py:1520-1529 ResumeDataDirectReader: dispatch on descr type.
/// llmodel.py:648-665: byte offset = elem_idx * item_size + field_offset.
///
/// For GcTypedArray::Struct: writes directly to the flat byte buffer.
/// For legacy Ref/Int/Float arrays: falls back to flat index computation.
pub fn setinteriorfield(
    array: *mut GcTypedArray,
    elem_idx: usize,
    field_offset: usize,
    field_size: usize,
    item_size: usize,
    descr_field_type: u8,
    value: i64,
) {
    if array.is_null() {
        return;
    }
    let arr = unsafe { &mut *array };
    match arr {
        GcTypedArray::Struct {
            data,
            item_size: is,
            ..
        } => {
            // llmodel.py:648-665 parity: byte_offset = elem_idx * item_size + field_offset
            let byte_offset = elem_idx * *is + field_offset;
            let end = byte_offset + field_size.min(8);
            if end <= data.len() {
                match descr_field_type {
                    2 => {
                        // bh_setinteriorfield_gc_f: write f64
                        let bits = value as u64;
                        data[byte_offset..byte_offset + 8.min(field_size)]
                            .copy_from_slice(&bits.to_ne_bytes()[..8.min(field_size)]);
                    }
                    0 => {
                        // bh_setinteriorfield_gc_r: write ref (pointer)
                        let ptr = value as usize;
                        let sz = std::mem::size_of::<usize>().min(field_size);
                        data[byte_offset..byte_offset + sz]
                            .copy_from_slice(&ptr.to_ne_bytes()[..sz]);
                    }
                    _ => {
                        // bh_setinteriorfield_gc_i: write int
                        let sz = field_size.min(8);
                        data[byte_offset..byte_offset + sz]
                            .copy_from_slice(&value.to_ne_bytes()[..sz]);
                    }
                }
            }
        }
        _ => {
            // Legacy fallback: flat index for Ref/Int/Float arrays.
            let fields_per_elem = if item_size > 0 { item_size } else { 1 };
            let flat = elem_idx * fields_per_elem + field_offset;
            match descr_field_type {
                2 => setarrayitem_float(array, flat, f64::from_bits(value as u64)),
                1 => setarrayitem_int(array, flat, value),
                _ => setarrayitem_ref(array, flat, value as PyObjectRef),
            }
        }
    }
}

/// Resume parity: get the length of a GcTypedArray.
pub fn gcarray_len(array: *const GcTypedArray) -> usize {
    if array.is_null() {
        return 0;
    }
    let arr = unsafe { &*array };
    match arr {
        GcTypedArray::Ref(v) => v.len(),
        GcTypedArray::Int(v) => v.len(),
        GcTypedArray::Float(v) => v.len(),
        GcTypedArray::Struct { num_elems, .. } => *num_elems,
    }
}

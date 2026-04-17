use std::ops::{Index, IndexMut};

use crate::{PY_NULL, PyObjectRef};

/// Offset of the backing pointer inside `PyObjectArray`.
pub const PYOBJECT_ARRAY_PTR_OFFSET: usize = std::mem::offset_of!(PyObjectArray, ptr);

/// Offset of the live length inside `PyObjectArray`.
pub const PYOBJECT_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(PyObjectArray, len);

/// Offset of the capacity inside `PyObjectArray`.
pub const PYOBJECT_ARRAY_CAP_OFFSET: usize = std::mem::offset_of!(PyObjectArray, cap);

/// pypy/interpreter/pyframe.py:110 — RPython allocates locals_cells_stack_w
/// as a plain list (GC array). This is the Rust equivalent: always heap-
/// allocated so the JIT `ptr` field stays valid across frame moves and the
/// GC can trace elements via the pointer.
///
/// The `ptr` field (offset 0) is used by the JIT for direct memory access.
#[repr(C)]
pub struct PyObjectArray {
    /// Raw pointer to the heap-allocated storage. JIT reads this at offset 0.
    pub ptr: *mut PyObjectRef,
    /// Number of live elements.
    len: usize,
    /// Heap capacity (always > 0).
    cap: usize,
}

impl PyObjectArray {
    pub fn filled(len: usize, value: PyObjectRef) -> Self {
        let mut storage = vec![value; len.max(1)];
        let ptr = storage.as_mut_ptr();
        let cap = storage.capacity();
        std::mem::forget(storage);
        Self { ptr, len, cap }
    }

    pub fn from_vec(mut values: Vec<PyObjectRef>) -> Self {
        let len = values.len();
        if values.is_empty() {
            values.push(PY_NULL);
        }
        let ptr = values.as_mut_ptr();
        let cap = values.capacity();
        std::mem::forget(values);
        Self { ptr, len, cap }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub fn spare_capacity(&self) -> usize {
        self.cap.saturating_sub(self.len)
    }

    fn grow(&mut self, min_cap: usize) {
        let target_cap = min_cap.max(self.cap.saturating_mul(2).max(4));
        unsafe {
            let mut values = Vec::from_raw_parts(self.ptr, self.len, self.cap);
            values.reserve(target_cap.saturating_sub(values.capacity()));
            self.ptr = values.as_mut_ptr();
            self.cap = values.capacity();
            std::mem::forget(values);
        }
    }

    pub fn push(&mut self, value: PyObjectRef) {
        if self.len == self.cap {
            self.grow(self.len + 1);
        }
        unsafe {
            *self.ptr.add(self.len) = value;
        }
        self.len += 1;
    }

    #[inline]
    pub fn is_inline(&self) -> bool {
        false
    }

    /// No-op — heap-backed arrays don't need pointer fixup after moves.
    /// Retained for API compatibility during transition.
    #[inline]
    pub fn fix_ptr(&mut self) {}

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[PyObjectRef] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [PyObjectRef] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn to_vec(&self) -> Vec<PyObjectRef> {
        self.as_slice().to_vec()
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.as_mut_slice().swap(a, b);
    }

    /// Insert `value` at `index`, shifting later elements right.
    /// Mirrors RPython `AbstractUnwrappedStrategy.insert` (listobject.py:1714):
    ///   `l.insert(index, self.unwrap(w_item))`
    pub fn insert(&mut self, index: usize, value: PyObjectRef) {
        debug_assert!(index <= self.len);
        if self.len == self.capacity() {
            self.grow(self.len + 1);
        }
        let len_orig = self.len;
        unsafe {
            let ptr = self.ptr;
            std::ptr::copy(ptr.add(index), ptr.add(index + 1), len_orig - index);
            *ptr.add(index) = value;
        }
        self.len += 1;
    }

    /// Remove and return the element at `index`, shifting later elements left.
    /// Mirrors RPython `AbstractUnwrappedStrategy.pop` (listobject.py:1855):
    ///   `item = l.pop(index)`
    pub fn remove(&mut self, index: usize) -> PyObjectRef {
        debug_assert!(index < self.len);
        let len = self.len;
        let slice = self.as_mut_slice();
        let value = slice[index];
        slice.copy_within(index + 1..len, index);
        self.len -= 1;
        value
    }

    /// Remove and return the last element.
    /// Mirrors RPython `AbstractUnwrappedStrategy.pop_end` (listobject.py:1848):
    ///   `return self.wrap(l.pop())`
    pub fn pop(&mut self) -> PyObjectRef {
        debug_assert!(self.len > 0);
        let value = self.as_slice()[self.len - 1];
        self.len -= 1;
        value
    }

    /// Reverse storage in-place.
    /// Mirrors RPython `AbstractUnwrappedStrategy.reverse` (listobject.py:1880):
    ///   `self.unerase(w_list.lstorage).reverse()`
    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse();
    }

    /// Replace `[start .. start+remove_count]` with `new_values` in one pass.
    /// Mirrors RPython `AbstractUnwrappedStrategy.setslice` (listobject.py:1773-1808)
    /// step==1 path: `del items[start:start+delta]` + overwrite, O(n).
    ///
    /// # Safety
    /// All pointers in `new_values` and in the existing storage must be valid.
    pub unsafe fn splice(&mut self, start: usize, remove_count: usize, new_values: &[PyObjectRef]) {
        unsafe {
            let old_len = self.len;
            let s = start.min(old_len);
            let slicelength = remove_count.min(old_len - s);
            let len2 = new_values.len();
            let new_len = old_len - slicelength + len2;
            if len2 > slicelength {
                if new_len > self.capacity() {
                    self.grow(new_len);
                }
                std::ptr::copy(
                    self.ptr.add(s + slicelength),
                    self.ptr.add(s + len2),
                    old_len - s - slicelength,
                );
                self.len = new_len;
            } else if slicelength > len2 {
                std::ptr::copy(
                    self.ptr.add(s + slicelength),
                    self.ptr.add(s + len2),
                    old_len - s - slicelength,
                );
                self.len = new_len;
            }
            if len2 > 0 {
                self.as_mut_slice()[s..s + len2].copy_from_slice(new_values);
            }
        }
    }
}

impl Drop for PyObjectArray {
    fn drop(&mut self) {
        if self.cap > 0 {
            unsafe {
                drop(Vec::from_raw_parts(self.ptr, self.len, self.cap));
            }
        }
    }
}

impl Index<usize> for PyObjectArray {
    type Output = PyObjectRef;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.ptr.add(index) }
    }
}

impl IndexMut<usize> for PyObjectArray {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.ptr.add(index) }
    }
}

// ─── FixedObjectArray: pyframe.py:112 make_sure_not_resized parity ────────
//
// RPython `locals_cells_stack_w = [None] * size; make_sure_not_resized(...)`
// becomes a fixed-length GcArray. This type mirrors that: len is immutable
// after creation, no push/grow/spare_capacity.

/// Offset of `ptr` within `FixedObjectArray`.
pub const FIXED_ARRAY_PTR_OFFSET: usize = std::mem::offset_of!(FixedObjectArray, ptr);

/// Offset of `len` within `FixedObjectArray`.
pub const FIXED_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(FixedObjectArray, len);

/// pyframe.py:110-112: fixed-length GcArray for `locals_cells_stack_w`.
///
/// Once created, the length never changes. No push, no grow.
/// Items are mutable (stack operations write via index) but the
/// array cannot be resized.
#[repr(C)]
pub struct FixedObjectArray {
    /// Raw pointer to heap-allocated storage. JIT reads at offset 0.
    pub ptr: *mut PyObjectRef,
    /// Fixed length (== allocation capacity).
    len: usize,
}

impl FixedObjectArray {
    /// pyframe.py:110: `[None] * size`
    pub fn filled(len: usize, value: PyObjectRef) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            };
        }
        let storage = vec![value; len];
        let mut boxed = storage.into_boxed_slice();
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        Self { ptr, len }
    }

    pub fn from_vec(values: Vec<PyObjectRef>) -> Self {
        if values.is_empty() {
            return Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            };
        }
        let mut boxed = values.into_boxed_slice();
        let len = boxed.len();
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        Self { ptr, len }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[PyObjectRef] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [PyObjectRef] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn to_vec(&self) -> Vec<PyObjectRef> {
        self.as_slice().to_vec()
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.as_mut_slice().swap(a, b);
    }
}

impl Drop for FixedObjectArray {
    fn drop(&mut self) {
        if self.len > 0 && !self.ptr.is_null() {
            unsafe {
                drop(Vec::from_raw_parts(self.ptr, self.len, self.len));
            }
        }
    }
}

impl Index<usize> for FixedObjectArray {
    type Output = PyObjectRef;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.ptr.add(index) }
    }
}

impl IndexMut<usize> for FixedObjectArray {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.ptr.add(index) }
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

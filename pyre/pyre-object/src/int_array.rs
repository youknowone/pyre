use std::ops::{Index, IndexMut};

use crate::object_array::{
    TypedItemsBlock, alloc_typed_items_block, dealloc_typed_items_block, grow_typed_items_block,
    typed_items_block_capacity, typed_items_block_items_base,
};

/// Small-buffer capacity constant retained for the append/pop inline-capacity
/// trace path (`is_inline()` is always false, so it is never consulted at
/// runtime).
pub const INT_ARRAY_INLINE_CAP: usize = 8;

/// Unboxed `int` list storage — `listobject.py` IntegerListStrategy
/// `lstorage = erase([int])`, i.e. a `Ptr(GcArray(Signed))`.
///
/// The data lives in a separate length-prefixed [`TypedItemsBlock`]
/// (`[capacity][i64...]`) so the JIT can address it as a GC array
/// (`GetfieldGcR(block) → GetarrayitemGcI`). `ptr` mirrors the block's items
/// base; `heap_cap` mirrors the block capacity. Both are kept in sync by every
/// method that reallocates the block.
#[repr(C)]
pub struct IntArray {
    /// `Ptr(GcArray(Signed))` — the backing block. Always non-null.
    pub block: *mut TypedItemsBlock,
    /// Items base (`= block + ITEMS_OFFSET`). Mirrors the block.
    pub ptr: *mut i64,
    /// Live length (rlist.py:116 `("length", Signed)`).
    len: usize,
    /// Allocated capacity, mirroring the block header.
    heap_cap: usize,
}

pub const INT_ARRAY_BLOCK_OFFSET: usize = std::mem::offset_of!(IntArray, block);
pub const INT_ARRAY_PTR_OFFSET: usize = std::mem::offset_of!(IntArray, ptr);
pub const INT_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(IntArray, len);
pub const INT_ARRAY_HEAP_CAP_OFFSET: usize = std::mem::offset_of!(IntArray, heap_cap);

impl IntArray {
    #[inline]
    fn sync_from_block(&mut self) {
        unsafe {
            self.ptr = typed_items_block_items_base(self.block) as *mut i64;
            self.heap_cap = typed_items_block_capacity(self.block);
        }
    }

    pub fn from_vec(values: Vec<i64>) -> Self {
        let len = values.len();
        let mut arr = Self {
            block: unsafe { alloc_typed_items_block(len) },
            ptr: std::ptr::null_mut(),
            len,
            heap_cap: 0,
        };
        arr.sync_from_block();
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), arr.ptr, len);
        }
        arr
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.heap_cap
    }

    #[inline]
    pub fn spare_capacity(&self) -> usize {
        self.capacity().saturating_sub(self.len)
    }

    /// Integer list storage is always a separate block (no inline buffer).
    #[inline]
    pub fn is_inline(&self) -> bool {
        false
    }

    fn grow(&mut self, min_cap: usize) {
        let target_cap = min_cap
            .max(self.heap_cap.saturating_mul(2))
            .max(INT_ARRAY_INLINE_CAP);
        self.block = unsafe { grow_typed_items_block(self.block, target_cap, self.len) };
        self.sync_from_block();
    }

    pub fn push(&mut self, value: i64) {
        if self.len == self.heap_cap {
            self.grow(self.len + 1);
        }
        unsafe {
            *self.ptr.add(self.len) = value;
        }
        self.len += 1;
    }

    #[inline]
    pub fn fix_ptr(&mut self) {
        self.sync_from_block();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn to_vec(&self) -> Vec<i64> {
        self.as_slice().to_vec()
    }

    pub fn insert(&mut self, index: usize, value: i64) {
        assert!(index <= self.len);
        if self.len == self.heap_cap {
            self.grow(self.len + 1);
        }
        unsafe {
            let p = self.ptr.add(index);
            std::ptr::copy(p, p.add(1), self.len - index);
            *p = value;
        }
        self.len += 1;
    }

    pub fn remove(&mut self, index: usize) -> i64 {
        assert!(index < self.len);
        let value = self.as_slice()[index];
        unsafe {
            let p = self.ptr.add(index);
            std::ptr::copy(p.add(1), p, self.len - index - 1);
        }
        self.len -= 1;
        value
    }

    pub fn pop(&mut self) -> i64 {
        assert!(self.len > 0);
        let value = self.as_slice()[self.len - 1];
        self.len -= 1;
        value
    }

    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse();
    }

    pub fn splice(&mut self, start: usize, remove_count: usize, new_values: &[i64]) {
        let old_len = self.len;
        let s = start.min(old_len);
        let slicelength = remove_count.min(old_len - s);
        let len2 = new_values.len();
        let new_len = old_len - slicelength + len2;
        if len2 > slicelength {
            if new_len > self.heap_cap {
                self.grow(new_len);
            }
            unsafe {
                std::ptr::copy(
                    self.ptr.add(s + slicelength),
                    self.ptr.add(s + len2),
                    old_len - s - slicelength,
                );
            }
            self.len = new_len;
        } else if slicelength > len2 {
            unsafe {
                std::ptr::copy(
                    self.ptr.add(s + slicelength),
                    self.ptr.add(s + len2),
                    old_len - s - slicelength,
                );
            }
            self.len = new_len;
        }
        if len2 > 0 {
            self.as_mut_slice()[s..s + len2].copy_from_slice(new_values);
        }
    }

    pub fn drain(&mut self, range: std::ops::Range<usize>) {
        let start = range.start;
        let end = range.end;
        assert!(start <= end && end <= self.len);
        let count = end - start;
        if count == 0 {
            return;
        }
        unsafe {
            let p = self.ptr.add(start);
            std::ptr::copy(p.add(count), p, self.len - end);
        }
        self.len -= count;
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl Drop for IntArray {
    fn drop(&mut self) {
        unsafe {
            dealloc_typed_items_block(self.block);
        }
    }
}

impl Index<usize> for IntArray {
    type Output = i64;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.ptr.add(index) }
    }
}

impl IndexMut<usize> for IntArray {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.ptr.add(index) }
    }
}

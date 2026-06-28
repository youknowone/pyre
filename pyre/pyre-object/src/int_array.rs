use std::ops::{Index, IndexMut};

use crate::object_array::{
    GC_INT_ARRAY_GC_TYPE_ID, TypedItemsBlock, alloc_typed_items_block, dealloc_typed_items_block,
    grow_typed_items_block, typed_items_block_capacity, typed_items_block_items_base,
};

/// Small-buffer capacity constant retained for the append/pop inline-capacity
/// trace path (`is_inline()` is always false, so it is never consulted at
/// runtime).
pub const INT_ARRAY_INLINE_CAP: usize = 8;

/// Unboxed `int` list storage — `listobject.py` IntegerListStrategy
/// `lstorage = erase([int])`, i.e. a `Ptr(GcArray(Signed))`.
///
/// `rlist.py:116` `LIST = GcStruct("list", ("length", Signed), ("items",
/// Ptr(GcArray(item))))`: the live length is `len` and the items array is the
/// length-prefixed [`TypedItemsBlock`] (`[capacity][i64...]`) reached through
/// `block`. The items base and allocated capacity are read from `block` on
/// demand (`len(l.items)` = the block's capacity header) — there is no cached
/// interior pointer, so the JIT can address the array as a GC ref
/// (`GetfieldGcR(block) → GetarrayitemGcI`) that the gcmap relocates on a move.
#[repr(C)]
pub struct IntArray {
    /// `Ptr(GcArray(Signed))` — the backing block (`l.items`). Always non-null.
    pub block: *mut TypedItemsBlock,
    /// Live length (rlist.py:116 `("length", Signed)`).
    len: usize,
}

pub const INT_ARRAY_BLOCK_OFFSET: usize = std::mem::offset_of!(IntArray, block);
pub const INT_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(IntArray, len);

impl IntArray {
    /// Items base pointer (`&l.items[0]`), derived from `block`.
    #[inline]
    fn base(&self) -> *mut i64 {
        unsafe { typed_items_block_items_base(self.block) as *mut i64 }
    }

    pub fn from_vec(values: Vec<i64>) -> Self {
        let len = values.len();
        let arr = Self {
            block: unsafe { alloc_typed_items_block(len, GC_INT_ARRAY_GC_TYPE_ID) },
            len,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), arr.base(), len);
        }
        arr
    }

    /// Allocated capacity (`len(l.items)`, rlist.py:251), read from the block
    /// header.
    #[inline]
    fn capacity(&self) -> usize {
        unsafe { typed_items_block_capacity(self.block) }
    }

    #[inline]
    pub fn spare_capacity(&self) -> usize {
        self.capacity().saturating_sub(self.len)
    }

    /// Allocated capacity (block header). The no-resize append fast path
    /// guards `len < heap_capacity()` before writing past the live length,
    /// mirroring `_ll_list_resize_ge`'s `len(items) >= length + 1` check
    /// (rlist.py:285).
    #[inline]
    pub fn heap_capacity(&self) -> usize {
        self.capacity()
    }

    /// Store the live length without touching the block. The caller must
    /// guarantee `new_len <= heap_capacity()` (the no-resize precondition);
    /// mirrors `_ll_list_resize_ge`'s `l.length = newsize` (rlist.py:293).
    /// Enforced here because this is safe/public: a `len` past the allocated
    /// capacity would make `as_slice`/`as_mut_slice` build out-of-bounds
    /// slices (UB).
    #[inline]
    pub fn set_len(&mut self, new_len: usize) {
        let cap = self.capacity();
        assert!(
            new_len <= cap,
            "IntArray::set_len precondition violated: new_len ({new_len}) > capacity ({cap})"
        );
        self.len = new_len;
    }

    /// Integer list storage is always a separate block (no inline buffer).
    #[inline]
    pub fn is_inline(&self) -> bool {
        false
    }

    fn grow(&mut self, min_cap: usize) {
        let target_cap = min_cap
            .max(self.capacity().saturating_mul(2))
            .max(INT_ARRAY_INLINE_CAP);
        self.block = unsafe {
            grow_typed_items_block(self.block, target_cap, self.len, GC_INT_ARRAY_GC_TYPE_ID)
        };
    }

    pub fn push(&mut self, value: i64) {
        if self.len == self.capacity() {
            self.grow(self.len + 1);
        }
        unsafe {
            *self.base().add(self.len) = value;
        }
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.base(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.base(), self.len) }
    }

    pub fn to_vec(&self) -> Vec<i64> {
        self.as_slice().to_vec()
    }

    pub fn insert(&mut self, index: usize, value: i64) {
        assert!(index <= self.len);
        if self.len == self.capacity() {
            self.grow(self.len + 1);
        }
        unsafe {
            let p = self.base().add(index);
            std::ptr::copy(p, p.add(1), self.len - index);
            *p = value;
        }
        self.len += 1;
    }

    pub fn remove(&mut self, index: usize) -> i64 {
        assert!(index < self.len);
        let value = self.as_slice()[index];
        unsafe {
            let p = self.base().add(index);
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
            if new_len > self.capacity() {
                self.grow(new_len);
            }
            unsafe {
                let base = self.base();
                std::ptr::copy(
                    base.add(s + slicelength),
                    base.add(s + len2),
                    old_len - s - slicelength,
                );
            }
            self.len = new_len;
        } else if slicelength > len2 {
            unsafe {
                let base = self.base();
                std::ptr::copy(
                    base.add(s + slicelength),
                    base.add(s + len2),
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
            let p = self.base().add(start);
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
        unsafe { &*self.base().add(index) }
    }
}

impl IndexMut<usize> for IntArray {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.base().add(index) }
    }
}

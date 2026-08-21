use std::ops::{Index, IndexMut};

use crate::object_array::{
    TYPED_ITEMS_BLOCK_ITEMS_OFFSET, TypedItemsBlock, alloc_typed_items_block,
    dealloc_typed_items_block, gc_float_array_gc_type_id, grow_typed_items_block,
    typed_items_block_capacity,
};

pub const FLOAT_ARRAY_INLINE_CAP: usize = 8;

/// Unboxed `float` list storage — `listobject.py` FloatListStrategy
/// `lstorage = erase([float])`, i.e. a `Ptr(GcArray(Float))`.
///
/// `rlist.py:116` `LIST = GcStruct("list", ("length", Signed), ("items",
/// Ptr(GcArray(item))))`: the live length is `len` and the items array is the
/// length-prefixed [`TypedItemsBlock`] (`[capacity][f64...]`) reached through
/// `block`. The items base and allocated capacity are read from `block` on
/// demand — there is no cached interior pointer, so the JIT addresses the array
/// as a GC ref (`GetfieldGcR(block) → GetarrayitemGcF`) the gcmap relocates.
#[repr(C)]
pub struct FloatArray {
    /// `Ptr(GcArray(Float))` — the backing block (`l.items`). Null in the empty
    /// form ([`FloatArray::empty`]), where the live length and the allocated
    /// capacity are both zero.
    pub block: *mut TypedItemsBlock,
    /// Live length (rlist.py `("length", Signed)`).
    len: usize,
}

pub const FLOAT_ARRAY_BLOCK_OFFSET: usize = std::mem::offset_of!(FloatArray, block);
pub const FLOAT_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(FloatArray, len);

impl FloatArray {
    /// Items base pointer (`&l.items[0]`), derived from `block`. The
    /// [`crate::int_array::IntArray::base`] twin — see it for why the empty
    /// form's null `block` is offset rather than branched on.
    #[inline]
    fn base(&self) -> *mut f64 {
        (self.block as *mut u8).wrapping_add(TYPED_ITEMS_BLOCK_ITEMS_OFFSET) as *mut f64
    }

    /// Storage for a list whose strategy does not read this array. The
    /// [`crate::int_array::IntArray::empty`] twin — see it for why
    /// `from_vec(Vec::new())` is not the same thing.
    pub fn empty() -> Self {
        Self {
            block: std::ptr::null_mut(),
            len: 0,
        }
    }

    pub fn from_vec(values: Vec<f64>) -> Self {
        let len = values.len();
        let arr = Self {
            block: unsafe { alloc_typed_items_block(len, gc_float_array_gc_type_id()) },
            len,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), arr.base(), len);
        }
        arr
    }

    /// Pin `block` on the shadow stack and return its slot. The
    /// [`crate::int_array::IntArray::pin_block`] twin — see it for why an
    /// old-gen, non-moving block still has to be rooted before the next GC
    /// operation.
    #[must_use]
    pub fn pin_block(&self) -> usize {
        let slot = crate::gc_roots::shadow_stack_len();
        crate::gc_roots::pin_root(self.block as crate::PyObjectRef);
        slot
    }

    /// Re-read `block` from the slot [`Self::pin_block`] returned — the
    /// `pop_roots` half of the bracket.
    pub fn reload_block(&mut self, slot: usize) {
        self.block = crate::gc_roots::shadow_stack_get(slot) as *mut TypedItemsBlock;
    }

    /// Replace the storage in place, keeping the incoming block rooted across
    /// the teardown of the outgoing one. The
    /// [`crate::int_array::IntArray::install`] twin — see it for the sweep
    /// window a bare `*self = fresh` leaves open.
    pub fn install(&mut self, fresh: FloatArray) {
        let _roots = crate::gc_roots::push_roots();
        let slot = fresh.pin_block();
        *self = fresh;
        self.reload_block(slot);
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
    /// guards `len < heap_capacity()`, mirroring `_ll_list_resize_ge`
    /// (rlist.py:285).
    #[inline]
    pub fn heap_capacity(&self) -> usize {
        self.capacity()
    }

    /// Store the live length without touching the block. The caller must
    /// guarantee `new_len <= heap_capacity()` (the no-resize precondition);
    /// mirrors `_ll_list_resize_ge`'s `l.length = newsize` (rlist.py).
    /// Enforced here because this is safe/public: a `len` past the allocated
    /// capacity would make `as_slice`/`as_mut_slice` build out-of-bounds
    /// slices (UB).
    #[inline]
    pub fn set_len(&mut self, new_len: usize) {
        let cap = self.capacity();
        assert!(
            new_len <= cap,
            "FloatArray::set_len precondition violated: new_len ({new_len}) > capacity ({cap})"
        );
        self.len = new_len;
    }

    /// Float list storage is always a separate block (no inline buffer);
    /// upstream `erase([float])` has no inline bit either.
    #[inline]
    pub fn is_inline(&self) -> bool {
        false
    }

    fn grow(&mut self, min_cap: usize) {
        let target_cap = min_cap
            .max(self.capacity().saturating_mul(2))
            .max(FLOAT_ARRAY_INLINE_CAP);
        self.block = unsafe {
            grow_typed_items_block(
                self.block,
                target_cap,
                self.len,
                gc_float_array_gc_type_id(),
            )
        };
    }

    pub fn push(&mut self, value: f64) {
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

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.as_slice().to_vec()
    }

    pub fn as_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.base(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.base(), self.len) }
    }

    pub fn insert(&mut self, index: usize, value: f64) {
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

    pub fn remove(&mut self, index: usize) -> f64 {
        assert!(index < self.len);
        let value = self.as_slice()[index];
        unsafe {
            let p = self.base().add(index);
            std::ptr::copy(p.add(1), p, self.len - index - 1);
        }
        self.len -= 1;
        value
    }

    pub fn pop(&mut self) -> f64 {
        assert!(self.len > 0);
        let value = self.as_slice()[self.len - 1];
        self.len -= 1;
        value
    }

    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse();
    }

    pub fn splice(&mut self, start: usize, remove_count: usize, new_values: &[f64]) {
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

impl Drop for FloatArray {
    fn drop(&mut self) {
        unsafe {
            dealloc_typed_items_block(self.block);
        }
    }
}

impl Index<usize> for FloatArray {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.base().add(index) }
    }
}

impl IndexMut<usize> for FloatArray {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.base().add(index) }
    }
}

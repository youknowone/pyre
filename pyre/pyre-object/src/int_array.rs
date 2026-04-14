use std::ops::{Index, IndexMut};

/// Small-buffer capacity. Arrays up to this size avoid heap allocation.
const INLINE_CAP: usize = 8;

/// Fixed-size i64 array with small-buffer optimization.
#[repr(C)]
pub struct IntArray {
    pub ptr: *mut i64,
    len: usize,
    heap_cap: usize,
    inline_buf: [i64; INLINE_CAP],
}

pub const INT_ARRAY_PTR_OFFSET: usize = std::mem::offset_of!(IntArray, ptr);
pub const INT_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(IntArray, len);
pub const INT_ARRAY_HEAP_CAP_OFFSET: usize = std::mem::offset_of!(IntArray, heap_cap);

impl IntArray {
    pub fn from_vec(mut values: Vec<i64>) -> Self {
        let len = values.len();
        if len <= INLINE_CAP {
            let mut inline_buf = [0; INLINE_CAP];
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
                inline_buf: [0; INLINE_CAP],
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

    pub fn push(&mut self, value: i64) {
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

    /// Insert `value` at `index`, shifting later elements right.
    /// Mirrors RPython `AbstractUnwrappedStrategy.insert` (listobject.py:1714):
    ///   `l.insert(index, self.unwrap(w_item))`
    pub fn insert(&mut self, index: usize, value: i64) {
        debug_assert!(index <= self.len);
        if self.len == self.capacity() {
            if self.heap_cap == 0 {
                self.grow_to_heap(self.len + 1);
            } else {
                self.grow_heap(self.len + 1);
            }
        }
        // after grow, self.ptr has capacity for len+1 elements;
        // use raw copy so we can write to the slot at index `len` (past current slice)
        unsafe {
            let ptr = self.ptr;
            std::ptr::copy(ptr.add(index), ptr.add(index + 1), self.len - index);
            *ptr.add(index) = value;
        }
        self.len += 1;
    }

    /// Remove and return the element at `index`, shifting later elements left.
    /// Mirrors RPython `AbstractUnwrappedStrategy.pop` (listobject.py:1855):
    ///   `item = l.pop(index)`
    pub fn remove(&mut self, index: usize) -> i64 {
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
    pub fn pop(&mut self) -> i64 {
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
    pub fn splice(&mut self, start: usize, remove_count: usize, new_values: &[i64]) {
        let old_len = self.len;
        let s = start.min(old_len);
        let slicelength = remove_count.min(old_len - s);
        let len2 = new_values.len();
        let new_len = old_len - slicelength + len2;
        if len2 > slicelength {
            // growing: ensure capacity, then shift right
            if new_len > self.capacity() {
                if self.heap_cap == 0 {
                    self.grow_to_heap(new_len);
                } else {
                    self.grow_heap(new_len);
                }
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
            // shrinking: shift left
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

    pub fn as_slice(&self) -> &[i64] {
        if self.heap_cap > 0 {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        } else {
            &self.inline_buf[..self.len]
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [i64] {
        if self.heap_cap > 0 {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        } else {
            &mut self.inline_buf[..self.len]
        }
    }
}

impl Drop for IntArray {
    fn drop(&mut self) {
        if self.heap_cap > 0 {
            unsafe {
                drop(Vec::from_raw_parts(self.ptr, self.len, self.heap_cap));
            }
        }
    }
}

impl Index<usize> for IntArray {
    type Output = i64;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        if self.heap_cap > 0 {
            unsafe { &*self.ptr.add(index) }
        } else {
            &self.inline_buf[index]
        }
    }
}

impl IndexMut<usize> for IntArray {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if self.heap_cap > 0 {
            unsafe { &mut *self.ptr.add(index) }
        } else {
            &mut self.inline_buf[index]
        }
    }
}

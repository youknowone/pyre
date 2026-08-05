use std::ops::{Index, IndexMut};

use crate::object_array::{
    GC_INT_ARRAY_GC_TYPE_ID, TYPED_ITEMS_BLOCK_ITEMS_OFFSET, TypedItemsBlock,
    alloc_typed_items_block, dealloc_typed_items_block, grow_typed_items_block,
    typed_items_block_capacity,
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
    /// `Ptr(GcArray(Signed))` — the backing block (`l.items`). Null in the
    /// empty form ([`IntArray::empty`]), where the live length and the
    /// allocated capacity are both zero.
    pub block: *mut TypedItemsBlock,
    /// Live length (rlist.py:116 `("length", Signed)`).
    len: usize,
}

pub const INT_ARRAY_BLOCK_OFFSET: usize = std::mem::offset_of!(IntArray, block);
pub const INT_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(IntArray, len);

impl IntArray {
    /// Items base pointer (`&l.items[0]`), derived from `block`.
    ///
    /// `wrapping_add`, so the empty form's null `block` yields the items offset
    /// itself — a non-null, 8-aligned address `as_slice` may hand to
    /// `from_raw_parts` at length zero. No other caller reaches it: the empty
    /// form's capacity is zero, so every write goes through [`Self::grow`]
    /// first.
    #[inline]
    fn base(&self) -> *mut i64 {
        (self.block as *mut u8).wrapping_add(TYPED_ITEMS_BLOCK_ITEMS_OFFSET) as *mut i64
    }

    /// Storage for a list whose strategy does not read this array: no block, no
    /// live length, no allocated capacity.
    ///
    /// This is the shape traced code already builds — `emit_typed_list_inline`
    /// writes one typed pair and leaves the other's `block` NULL from
    /// `NewWithVtable`'s zero fill. `from_vec(Vec::new())` does not produce it:
    /// `try_alloc_typed_items_block` clamps `cap` to 1 (rlist.py:251
    /// overallocation) and takes the old-gen `try_gc_alloc_stable_raw`, so each
    /// non-Integer list bought a block it never reads.
    ///
    /// The block helpers already read null as capacity zero, `grow` and
    /// `dealloc` already special-case it, and `list_object_custom_trace`
    /// forwards the owning slot only when the collector owns what it holds.
    pub fn empty() -> Self {
        Self {
            block: std::ptr::null_mut(),
            len: 0,
        }
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

    /// Pin `block` on the shadow stack and return its slot, so the block stays
    /// live across a following GC operation.
    ///
    /// `gct_fv_gc_malloc` (`framework.py:853-856`) brackets every malloc with
    /// `push_roots`/`pop_roots` over the live vars, and that bracket is about
    /// *liveness*, not only relocation: old-gen here is mark-sweep
    /// (`incminimark.py` `STATE_SWEEPING` frees every old object that did not
    /// gain `GCFLAG_VISITED` during this cycle), so **non-moving does not mean
    /// non-collected**. A block born while the cycle is still scanning gets no
    /// `VISITED` bit (`collector.rs` `oldgen_birth_flags`), and until the owning
    /// field store lands it is reachable from a Rust local only — no root, no
    /// heap edge — so this cycle's sweep is entitled to reclaim it.
    ///
    /// Every GC operation in that window is a real safepoint: with more than one
    /// registered thread `gc_op` takes `gc_op_slow`, which clears `running` and
    /// blocks on the GC mutex, letting another mutator drive a full cycle to
    /// completion. `try_gc_alloc_stable_raw` and the `try_gc_owns_object` query
    /// inside `dealloc_typed_items_block` both qualify.
    ///
    /// [`crate::gc_roots::pin_root`] publishes the raw value to the shadow stack
    /// before it performs its own forwarding query, so no safepoint sits between
    /// the allocation and the root becoming visible.
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
    /// the teardown of the outgoing one.
    ///
    /// `*self = fresh` alone is unsound here: Rust evaluates the right-hand side
    /// (which allocates `fresh.block`), then drops the value being replaced, and
    /// [`Drop`] runs `dealloc_typed_items_block` whose `try_gc_owns_object` query
    /// is a safepoint — with `fresh.block` still unrooted and unreferenced. A
    /// concurrent cycle reaching `STATE_SWEEPING` in that window frees the cell
    /// into the arena free list, and the store below then publishes freed memory
    /// as the list's storage. See [`Self::pin_block`] for why old-gen membership
    /// is no protection.
    pub fn install(&mut self, fresh: IntArray) {
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

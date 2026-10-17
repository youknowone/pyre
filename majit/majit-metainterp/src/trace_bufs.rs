//! Nursery storage for `opencoder.py` `Trace` pools other than `_ops`.
//!
//! `ListRepr` is a list header (`length` + `items`) plus a `GcArray`.
//! `_snapshot_data` and `_snapshot_array_data` are that pair: `encode_varint_signed`
//! appends, so the list is resized (`_ll_list_resize_ge` / `_ll_list_resize_hint_really`),
//! not a fixed `GcArray(Char)`. `_refs`, `_bigints`, and `_floats` use the
//! same header: `opencoder.py` `Trace.__init__` builds ordinary resizable
//! lists (`self._refs = [nullptr]`, `self._bigints = []`, `self._floats = []`).
//! `_refs` items are `GCREF` (`items_have_gc_ptrs`). `_descrs` stays a host
//! `Vec`: `DescrRef` is `Arc<dyn Descr>` and is not a GC item until the descr
//! objects themselves move (later stage).

use std::marker::PhantomData;

use majit_ir::GcRef;

const WORD: usize = std::mem::size_of::<usize>();

/// Bytes before the first item. The length word is one `usize`. On wasm32
/// that is 4 bytes, while `_bigints` / `_floats` store `i64` / `u64`.
/// The item pointer has to meet `align_of::<T>()`, and the registered
/// varsize `base_size` is this same offset.
fn items_offset<T>() -> usize {
    let align = std::mem::align_of::<T>().max(std::mem::align_of::<usize>());
    WORD.div_ceil(align) * align
}

fn host_align<T>() -> usize {
    std::mem::align_of::<T>().max(std::mem::align_of::<usize>())
}

fn gc_installed() -> bool {
    majit_gc::current_thread_gc_available()
}

fn host_layout(payload: usize) -> std::alloc::Layout {
    host_layout_aligned(payload, std::mem::align_of::<usize>())
}

fn host_layout_aligned(payload: usize, align: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(payload, align)
        .unwrap_or_else(|_| std::alloc::Layout::new::<usize>())
}

fn host_alloc(payload: usize) -> *mut u8 {
    let layout = host_layout(payload);
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    ptr
}

fn host_free(ptr: *mut u8, payload: usize) {
    host_free_aligned(ptr, payload, std::mem::align_of::<usize>());
}

fn host_alloc_aligned(payload: usize, align: usize) -> *mut u8 {
    let layout = host_layout_aligned(payload, align);
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    ptr
}

fn host_free_aligned(ptr: *mut u8, payload: usize, align: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe { std::alloc::dealloc(ptr, host_layout_aligned(payload, align)) };
}

fn list_overallocate(newsize: usize) -> usize {
    // `_ll_list_resize_hint_really` when `overallocate` is true.
    if newsize == 0 {
        return 0;
    }
    let some = if newsize < 9 { 3 } else { 6 };
    newsize + some + (newsize >> 3)
}

fn type_id(slot: &std::sync::atomic::AtomicU32, what: &str) -> u32 {
    let id = slot.load(std::sync::atomic::Ordering::Acquire);
    assert!(
        id != u32::MAX,
        "{what} is not registered; call register_trace_ops_gc_type"
    );
    id
}

static TRACE_LIST_HDR_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);
static TRACE_WORD_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);
static TRACE_REFS_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);

/// Header, plain-word array, and ref array. Char items reuse `Trace._ops`'
/// `GcArray(Char)` type (`register_trace_ops_gc_type`).
pub(crate) fn register_trace_pool_gc_types(gc: &mut dyn majit_gc::GcAllocator) {
    let hdr = gc.register_type(majit_gc::trace::TypeInfo::with_gc_ptrs(
        std::mem::size_of::<ListHdr>(),
        vec![std::mem::size_of::<usize>()],
    ));
    TRACE_LIST_HDR_GC_TYPE_ID.store(hdr, std::sync::atomic::Ordering::Release);
    let word = gc.register_type(majit_gc::trace::TypeInfo::varsize(
        items_offset::<u64>(),
        std::mem::size_of::<u64>(),
        0,
        false,
        Vec::new(),
    ));
    TRACE_WORD_GC_TYPE_ID.store(word, std::sync::atomic::Ordering::Release);
    let refs = gc.register_type(majit_gc::trace::TypeInfo::varsize(
        items_offset::<usize>(),
        std::mem::size_of::<usize>(),
        0,
        true,
        Vec::new(),
    ));
    TRACE_REFS_GC_TYPE_ID.store(refs, std::sync::atomic::Ordering::Release);
}

fn list_hdr_type_id() -> u32 {
    type_id(&TRACE_LIST_HDR_GC_TYPE_ID, "Trace list header")
}

fn word_type_id() -> u32 {
    type_id(&TRACE_WORD_GC_TYPE_ID, "Trace word GcArray")
}

fn refs_type_id() -> u32 {
    type_id(&TRACE_REFS_GC_TYPE_ID, "Trace._refs GcArray")
}

/// `ListRepr` header. `items` is `Ptr(GcArray(Char))`.
#[repr(C)]
struct ListHdr {
    length: usize,
    items: usize,
}

/// `opencoder.py` `_snapshot_data` / `_snapshot_array_data`.
///
/// One `OwnerRootGuard` holds the header. `items` is walked from the header,
/// so a minor updates both. Every access re-reads the guard.
pub struct CharList {
    root: Option<majit_gc::shadow_stack::OwnerRootGuard>,
    host: *mut ListHdr,
}

impl CharList {
    pub fn with_capacity(cap: usize) -> Self {
        if !gc_installed() {
            return Self::host_with_capacity(cap);
        }
        let mut scratch = GcRef(0);
        let hdr = unsafe {
            let mut needs = false;
            let fresh = majit_gc::alloc_fast_nursery_collecting_typed_rooted(
                list_hdr_type_id(),
                std::mem::size_of::<ListHdr>(),
                &mut scratch,
                &mut needs,
            );
            if fresh.0 == 0 {
                majit_gc::gc_alloc_failed(std::mem::size_of::<ListHdr>());
            }
            let h = fresh.0 as *mut ListHdr;
            (*h).length = 0;
            (*h).items = 0;
            let _ = needs;
            fresh
        };
        let root = majit_gc::shadow_stack::OwnerRootGuard::new(hdr);
        let mut list = Self {
            root: Some(root),
            host: std::ptr::null_mut(),
        };
        if cap > 0 {
            list.realloc_items(cap);
        }
        list
    }

    fn host_with_capacity(cap: usize) -> Self {
        let host = host_alloc(std::mem::size_of::<ListHdr>()) as *mut ListHdr;
        unsafe {
            (*host).length = 0;
            (*host).items = 0;
        }
        let mut list = Self { root: None, host };
        if cap > 0 {
            list.realloc_items(cap);
        }
        list
    }

    fn header_ref(&self) -> GcRef {
        if let Some(guard) = &self.root {
            guard.get()
        } else {
            GcRef(self.host as usize)
        }
    }

    fn header(&self) -> *mut ListHdr {
        self.header_ref().0 as *mut ListHdr
    }

    pub fn capacity(&self) -> usize {
        let items = unsafe { (*self.header()).items };
        if items == 0 {
            0
        } else {
            unsafe { *(items as *const usize) }
        }
    }

    fn chars(&self) -> *mut u8 {
        let items = unsafe { (*self.header()).items };
        if items == 0 {
            std::ptr::null_mut()
        } else {
            unsafe { (items as *mut u8).add(WORD) }
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self
    }

    fn len_cap(h: *mut ListHdr) -> (usize, usize) {
        unsafe {
            let len = (*h).length;
            let items = (*h).items;
            let cap = if items == 0 {
                0
            } else {
                *(items as *const usize)
            };
            (len, cap)
        }
    }

    /// Grow so `extra` more bytes fit. One `header()`; `_ll_list_resize_ge`
    /// only when the current items array is too short.
    #[inline]
    fn header_with_extra(&mut self, extra: usize) -> *mut ListHdr {
        let h = self.header();
        let (len, cap) = Self::len_cap(h);
        if len + extra > cap {
            self.realloc_items(list_overallocate(len + extra));
            self.header()
        } else {
            h
        }
    }

    /// `encode_varint_signed` into `chars[len..]`. Returns the new length.
    #[inline]
    unsafe fn write_varint_at(chars: *mut u8, len: usize, value: i64) -> usize {
        let mut v = value;
        let flag: u8 = if !(-(1 << 14)..(1 << 14)).contains(&v) {
            0x80
        } else {
            0
        };
        unsafe {
            *chars.add(len) = ((v & 0b0111_1111) as u8) | flag;
            v >>= 7;
            *chars.add(len + 1) = (v & 0xff) as u8;
            if flag == 0 {
                len + 2
            } else {
                v >>= 8;
                *chars.add(len + 2) = (v & 0xff) as u8;
                v >>= 8;
                *chars.add(len + 3) = (v & 0xff) as u8;
                len + 4
            }
        }
    }

    /// `encode_varint_signed` into this list (`_ll_list_resize_ge` on growth).
    ///
    /// `opencoder.py` `encode_varint_signed` appends each `chr`; `Trace.append_int`
    /// writes the same layout into a pre-grown `_ops` buffer after one length
    /// check. One `header()` here, then in-place bytes — not a stack array
    /// plus a per-byte copy, and not a shadow-stack root per byte.
    #[inline]
    pub fn append_varint(&mut self, value: i64) {
        let h = self.header_with_extra(4);
        unsafe {
            let chars = ((*h).items as *mut u8).add(WORD);
            (*h).length = Self::write_varint_at(chars, (*h).length, value);
        }
    }

    pub fn append_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let need = self.len() + bytes.len();
        if need > self.capacity() {
            self.realloc_items(list_overallocate(need));
        }
        let h = self.header();
        unsafe {
            let len = (*h).length;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.chars().add(len), bytes.len());
            (*h).length = len + bytes.len();
        }
    }

    pub fn pop(&mut self) -> u8 {
        let h = self.header();
        let b = unsafe {
            let len = (*h).length;
            assert!(len > 0, "snapshot list pop on empty");
            let b = *self.chars().add(len - 1);
            (*h).length = len - 1;
            b
        };
        // `_ll_list_resize_le`: shrink once length falls under half the allocation.
        let len = self.len();
        let cap = self.capacity();
        if cap > 0 && len < (cap >> 1).saturating_sub(5) {
            self.realloc_items(len);
        }
        b
    }

    fn realloc_items(&mut self, new_cap: usize) {
        let old_len = self.len();
        let copy_n = old_len.min(new_cap);
        let old_items = unsafe { (*self.header()).items };
        let fresh_items = if !gc_installed() {
            debug_assert!(
                self.root.is_none(),
                "host snapshot list while a GC is installed"
            );
            if new_cap == 0 {
                0
            } else {
                let payload = WORD + new_cap;
                let ptr = host_alloc(payload);
                unsafe { *(ptr as *mut usize) = new_cap };
                ptr as usize
            }
        } else if new_cap == 0 {
            0
        } else {
            let mut live = self.header_ref();
            let mut needs = false;
            let fresh = unsafe {
                majit_gc::alloc_fast_nursery_collecting_typed_rooted(
                    super::trace_ops_gc_type_id(),
                    WORD + new_cap,
                    &mut live,
                    &mut needs,
                )
            };
            if fresh.0 == 0 {
                majit_gc::gc_alloc_failed(WORD + new_cap);
            }
            unsafe {
                *(fresh.0 as *mut usize) = new_cap;
                std::ptr::write_bytes((fresh.0 as *mut u8).add(WORD), 0, new_cap);
            }
            let _ = needs;
            fresh.0
        };
        // Re-read the header. The allocating call may have moved it.
        let h = self.header();
        let old_items = if gc_installed() {
            unsafe { (*h).items }
        } else {
            old_items
        };
        unsafe {
            if copy_n > 0 && old_items != 0 && fresh_items != 0 {
                std::ptr::copy_nonoverlapping(
                    (old_items as *const u8).add(WORD),
                    (fresh_items as *mut u8).add(WORD),
                    copy_n,
                );
            }
            if gc_installed() {
                majit_gc::gc_write_barrier(GcRef(h as usize));
            }
            (*h).items = fresh_items;
        }
        if self.root.is_none() && old_items != 0 {
            let old_cap = unsafe { *(old_items as *const usize) };
            host_free(old_items as *mut u8, WORD + old_cap);
        }
    }
}

impl std::ops::Deref for CharList {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        let n = unsafe { (*self.header()).length };
        if n == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.chars(), n) }
    }
}

impl Drop for CharList {
    fn drop(&mut self) {
        if self.root.is_some() || self.host.is_null() {
            return;
        }
        unsafe {
            let items = (*self.host).items;
            if items != 0 {
                let cap = *(items as *const usize);
                host_free(items as *mut u8, WORD + cap);
            }
        }
        host_free(self.host as *mut u8, std::mem::size_of::<ListHdr>());
        self.host = std::ptr::null_mut();
    }
}

/// `ListRepr` of fixed-size items. One `OwnerRootGuard` holds the header;
/// `items` is walked from the header. Logical length lives on the header
/// (`ll_length`), capacity on the items `GcArray` length word.
pub struct WordArray<T: Copy> {
    root: Option<majit_gc::shadow_stack::OwnerRootGuard>,
    host: *mut ListHdr,
    /// `Trace._refs`: items are `GCREF` (`items_have_gc_ptrs`).
    gc_ptrs: bool,
    type_id: fn() -> u32,
    _ty: PhantomData<T>,
}

impl<T: Copy> WordArray<T> {
    pub fn empty(gc_ptrs: bool, type_id: fn() -> u32) -> Self {
        Self::with_capacity(0, gc_ptrs, type_id)
    }

    pub fn with_capacity(cap: usize, gc_ptrs: bool, type_id: fn() -> u32) -> Self {
        if !gc_installed() {
            return Self::host_with_capacity(cap, gc_ptrs, type_id);
        }
        let mut scratch = GcRef(0);
        let hdr = unsafe {
            let mut needs = false;
            let fresh = majit_gc::alloc_fast_nursery_collecting_typed_rooted(
                list_hdr_type_id(),
                std::mem::size_of::<ListHdr>(),
                &mut scratch,
                &mut needs,
            );
            if fresh.0 == 0 {
                majit_gc::gc_alloc_failed(std::mem::size_of::<ListHdr>());
            }
            let h = fresh.0 as *mut ListHdr;
            (*h).length = 0;
            (*h).items = 0;
            let _ = needs;
            fresh
        };
        let root = majit_gc::shadow_stack::OwnerRootGuard::new(hdr);
        let mut list = Self {
            root: Some(root),
            host: std::ptr::null_mut(),
            gc_ptrs,
            type_id,
            _ty: PhantomData,
        };
        if cap > 0 {
            let _ = list.realloc_items(cap, None);
        }
        list
    }

    fn host_with_capacity(cap: usize, gc_ptrs: bool, type_id: fn() -> u32) -> Self {
        let host = host_alloc(std::mem::size_of::<ListHdr>()) as *mut ListHdr;
        unsafe {
            (*host).length = 0;
            (*host).items = 0;
        }
        let mut list = Self {
            root: None,
            host,
            gc_ptrs,
            type_id,
            _ty: PhantomData,
        };
        if cap > 0 {
            let _ = list.realloc_items(cap, None);
        }
        list
    }

    fn header_ref(&self) -> GcRef {
        if let Some(guard) = &self.root {
            guard.get()
        } else {
            GcRef(self.host as usize)
        }
    }

    fn header(&self) -> *mut ListHdr {
        self.header_ref().0 as *mut ListHdr
    }

    pub fn capacity(&self) -> usize {
        let items = unsafe { (*self.header()).items };
        if items == 0 {
            0
        } else {
            unsafe { *(items as *const usize) }
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self
    }

    pub fn set(&mut self, index: usize, value: T) {
        let len = unsafe { (*self.header()).length };
        assert!(index < len, "word array index {index} len {len}");
        self.write_barrier_items();
        unsafe { *self.item_ptr().add(index) = value };
    }

    fn write_barrier_items(&self) {
        if !self.gc_ptrs {
            return;
        }
        let items = unsafe { (*self.header()).items };
        if items != 0 && self.root.is_some() {
            majit_gc::gc_write_barrier(GcRef(items));
        }
    }

    fn item_ptr(&self) -> *mut T {
        let items = unsafe { (*self.header()).items };
        if items == 0 {
            std::ptr::null_mut()
        } else {
            unsafe { (items as *mut u8).add(items_offset::<T>()) as *mut T }
        }
    }

    /// Append. Returns the stored value: a `GCREF` may have been forwarded
    /// when growth collected.
    pub fn push(&mut self, value: T) -> T {
        let len = unsafe { (*self.header()).length };
        if len == self.capacity() {
            return self.grow_push(value);
        }
        self.write_barrier_items();
        unsafe {
            *self.item_ptr().add(len) = value;
            (*self.header()).length = len + 1;
        }
        value
    }

    fn grow_push(&mut self, value: T) -> T {
        let need = unsafe { (*self.header()).length } + 1;
        let stored = self.realloc_items(list_overallocate(need), Some(value));
        let h = self.header();
        let len = unsafe { (*h).length };
        self.write_barrier_items();
        unsafe {
            *self.item_ptr().add(len) = stored;
            (*h).length = len + 1;
        }
        stored
    }

    fn realloc_items(&mut self, new_cap: usize, extra: Option<T>) -> T {
        let offset = items_offset::<T>();
        let align = host_align::<T>();
        let item_size = std::mem::size_of::<T>();
        let payload = offset + new_cap * item_size;
        let old_len = unsafe { (*self.header()).length };
        let copy_n = old_len.min(new_cap);
        let placeholder = extra.unwrap_or_else(|| unsafe { std::mem::zeroed() });
        let old_items = unsafe { (*self.header()).items };

        if !gc_installed() {
            debug_assert!(
                self.root.is_none(),
                "host trace word list used while a GC is installed"
            );
            let fresh_items = if new_cap == 0 {
                0
            } else {
                let ptr = host_alloc_aligned(payload, align);
                unsafe {
                    *(ptr as *mut usize) = new_cap;
                    if copy_n > 0 && old_items != 0 {
                        std::ptr::copy_nonoverlapping(
                            (old_items as *const u8).add(offset) as *const T,
                            ptr.add(offset) as *mut T,
                            copy_n,
                        );
                    }
                }
                ptr as usize
            };
            if old_items != 0 {
                let old_cap = unsafe { *(old_items as *const usize) };
                host_free_aligned(old_items as *mut u8, offset + old_cap * item_size, align);
            }
            unsafe {
                (*self.header()).items = fresh_items;
            }
            return placeholder;
        }

        if self.gc_ptrs {
            assert_eq!(
                item_size,
                std::mem::size_of::<usize>(),
                "GCREF item must be pointer-sized"
            );
        }
        let extra_addr = if self.gc_ptrs {
            extra.as_ref().map(gc_addr_bits).unwrap_or(0)
        } else {
            0
        };
        let mut roots = [self.header_ref(), GcRef(extra_addr)];
        let root_count = if self.gc_ptrs && extra.is_some() {
            2
        } else {
            1
        };
        let fresh_items = if new_cap == 0 {
            0
        } else {
            let mut needs = false;
            let fresh = unsafe {
                majit_gc::alloc_fast_nursery_collecting_typed_roots(
                    (self.type_id)(),
                    payload,
                    roots.as_mut_ptr(),
                    root_count,
                    &mut needs,
                )
            };
            if fresh.0 == 0 {
                majit_gc::gc_alloc_failed(payload);
            }
            unsafe {
                *(fresh.0 as *mut usize) = new_cap;
                if offset > WORD {
                    std::ptr::write_bytes((fresh.0 as *mut u8).add(WORD), 0, offset - WORD);
                }
                std::ptr::write_bytes((fresh.0 as *mut u8).add(offset), 0, new_cap * item_size);
            }
            let _ = needs;
            fresh.0
        };
        let stored = if self.gc_ptrs && extra.is_some() {
            from_gc_addr(roots[1].0)
        } else {
            placeholder
        };
        let h = self.header();
        let old_items = unsafe { (*h).items };
        unsafe {
            if copy_n > 0 && old_items != 0 && fresh_items != 0 {
                std::ptr::copy_nonoverlapping(
                    (old_items as *const u8).add(offset) as *const T,
                    (fresh_items as *mut u8).add(offset) as *mut T,
                    copy_n,
                );
            }
            majit_gc::gc_write_barrier(GcRef(h as usize));
            (*h).items = fresh_items;
        }
        stored
    }
}

impl<T: Copy> std::ops::Deref for WordArray<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        let n = unsafe { (*self.header()).length };
        if n == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.item_ptr(), n) }
    }
}

impl<T: Copy> Drop for WordArray<T> {
    fn drop(&mut self) {
        if self.root.is_some() || self.host.is_null() {
            return;
        }
        unsafe {
            let items = (*self.host).items;
            if items != 0 {
                let cap = *(items as *const usize);
                host_free_aligned(
                    items as *mut u8,
                    items_offset::<T>() + cap * std::mem::size_of::<T>(),
                    host_align::<T>(),
                );
            }
        }
        host_free(self.host as *mut u8, std::mem::size_of::<ListHdr>());
        self.host = std::ptr::null_mut();
    }
}

fn gc_addr_bits<T: Copy>(value: &T) -> usize {
    let mut bits = 0usize;
    unsafe {
        std::ptr::copy_nonoverlapping(
            (value as *const T).cast::<u8>(),
            (&mut bits as *mut usize).cast::<u8>(),
            std::mem::size_of::<usize>(),
        );
    }
    bits
}

fn from_gc_addr<T: Copy>(bits: usize) -> T {
    let mut value = unsafe { std::mem::zeroed() };
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&bits as *const usize).cast::<u8>(),
            (&mut value as *mut T).cast::<u8>(),
            std::mem::size_of::<usize>(),
        );
    }
    value
}

/// `Trace._refs`: one slot per `GCREF`. Item width is `usize`, which is
/// 4 bytes on wasm32 and 8 on the native backends.
pub fn new_refs() -> WordArray<usize> {
    let mut refs = WordArray::with_capacity(32, true, refs_type_id);
    refs.push(0);
    refs
}

pub fn new_bigints() -> WordArray<i64> {
    WordArray::empty(false, word_type_id)
}

pub fn new_floats() -> WordArray<u64> {
    WordArray::empty(false, word_type_id)
}

pub fn new_snapshot() -> CharList {
    CharList::with_capacity(128)
}

/// `blackhole.py` `_get_list_of_values` after `make_sure_not_resized`.
///
/// `rlist.py` `FixedSizeListRepr` stores the list as one `GcArray`. I/F
/// items are `Signed`; R items are `GCREF` (`items_have_gc_ptrs`). The
/// length is exact: the bytecode already named it.
///
/// Allocation is `malloc_varsize` (`incminimark.py`): tid + length, no
/// item clear for Signed. A GCREF array is cleared by
/// `gct_zero_gc_pointers_inside` / `gen_zero_gc_pointers` because the
/// GC scans it. `framework.py` `push_roots` is not part of the fast
/// bump; live previous lists travel as the overflow `livevars` span,
/// and a [`ListRootGuard`] covers a collecting call.
pub struct FixedSignedList {
    gc: GcRef,
    host: *mut u8,
    len: usize,
    gc_ptrs: bool,
}

/// `framework.py` `push_roots` / `pop_roots` for a handful of list
/// locals around a collecting call (`shadowstack.py` `incr_stack` /
/// `decr_stack`).
pub struct ListRootGuard {
    slot: majit_gc::shadow_stack::ShadowStackSlot,
    depth: usize,
}

impl ListRootGuard {
    #[inline]
    pub fn push(refs: &[GcRef]) -> Self {
        let slot = majit_gc::shadow_stack::shadow_stack_slot();
        let depth = slot.depth();
        for &g in refs {
            if g.0 != 0 {
                slot.push(g);
            }
        }
        Self { slot, depth }
    }
}

impl Drop for ListRootGuard {
    #[inline]
    fn drop(&mut self) {
        self.slot.pop_to(self.depth);
    }
}

impl FixedSignedList {
    /// Length 0, no nursery object. `rlist.py` `ll_newlist` of count 0
    /// is this sentinel; `ContinueRunningNormallyArgs` uses it for a
    /// bank the caller has not filled yet.
    #[inline(always)]
    pub fn empty(gc_ptrs: bool) -> Self {
        Self {
            gc: GcRef(0),
            host: std::ptr::null_mut(),
            len: 0,
            gc_ptrs,
        }
    }

    pub fn alloc(len: usize, gc_ptrs: bool) -> Self {
        Self::alloc_with_roots(len, gc_ptrs, &mut [])
    }

    /// `gct_fv_gc_malloc` `push_roots(hop)` livevars: registered only
    /// on `collect_and_reserve`, not on the bump.
    ///
    /// `malloc_varsize` (`incminimark.py`, `inline_simple_malloc_varsize`):
    /// the bump is `#[inline(always)]` at
    /// `alloc_fast_nursery_collecting_typed_roots`; this wrapper stays
    /// in the same function so the call site is a bump and a branch.
    #[inline(always)]
    pub fn alloc_with_roots(len: usize, gc_ptrs: bool, live: &mut [GcRef]) -> Self {
        debug_assert!(
            !gc_ptrs || std::mem::size_of::<usize>() == std::mem::size_of::<i64>(),
            "R lists store GCREF bits in i64 slots"
        );
        let offset = if gc_ptrs {
            items_offset::<usize>()
        } else {
            items_offset::<i64>()
        };
        let item = if gc_ptrs {
            std::mem::size_of::<usize>()
        } else {
            std::mem::size_of::<i64>()
        };
        let payload = offset + len * item;
        if !gc_installed() {
            let align = if gc_ptrs {
                host_align::<usize>()
            } else {
                host_align::<i64>()
            };
            let host = host_alloc_aligned(payload, align);
            unsafe {
                *(host as *mut usize) = len;
            }
            return Self {
                gc: GcRef(0),
                host,
                len,
                gc_ptrs,
            };
        }
        let type_id = if gc_ptrs {
            refs_type_id()
        } else {
            word_type_id()
        };
        let mut needs = false;
        let fresh = unsafe {
            majit_gc::alloc_fast_nursery_collecting_typed_roots(
                type_id,
                payload,
                live.as_mut_ptr(),
                live.len(),
                &mut needs,
            )
        };
        if fresh.0 == 0 {
            majit_gc::gc_alloc_failed(payload);
        }
        // `malloc_varsize` does not zero items. `gct_zero_gc_pointers_inside`
        // is elided when every GCREF slot is stored before the next
        // collection point (`_ll_alloc_and_set_nojit`). `fill_from_code`
        // is that store; this object is not rooted until the caller
        // finishes filling.
        unsafe {
            *(fresh.0 as *mut usize) = len;
            if needs {
                majit_gc::gc_write_barrier(fresh);
            }
        }
        Self {
            gc: fresh,
            host: std::ptr::null_mut(),
            len,
            gc_ptrs,
        }
    }

    #[inline(always)]
    pub fn as_gcref(&self) -> GcRef {
        if self.host.is_null() {
            self.gc
        } else {
            GcRef(0)
        }
    }

    #[inline(always)]
    pub fn reload(&mut self, forwarded: GcRef) {
        if self.host.is_null() {
            self.gc = forwarded;
        }
    }

    fn addr(&self) -> *mut u8 {
        if self.host.is_null() {
            self.gc.0 as *mut u8
        } else {
            self.host
        }
    }

    fn item_ptr(&self) -> *mut i64 {
        let offset = if self.gc_ptrs {
            items_offset::<usize>()
        } else {
            items_offset::<i64>()
        };
        unsafe { self.addr().add(offset) as *mut i64 }
    }

    /// Fill `0..len` from the bytecode that named that length
    /// (`ll_setitem_nonneg` without a bounds check). A young nursery
    /// array does not take a write barrier (`_ll_alloc_and_set_nojit`).
    #[inline]
    pub fn fill_from_code(&mut self, regs: &[i64], code: &[u8], start: usize) {
        let p = self.item_ptr();
        for i in 0..self.len {
            unsafe { *p.add(i) = regs[code[start + i] as usize] };
        }
    }

    pub fn set(&mut self, index: usize, value: i64) {
        debug_assert!(
            index < self.len,
            "fixed list index {index} len {}",
            self.len
        );
        unsafe { *self.item_ptr().add(index) = value };
    }
}

impl std::ops::Deref for FixedSignedList {
    type Target = [i64];
    fn deref(&self) -> &[i64] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.item_ptr(), self.len) }
    }
}

impl std::fmt::Debug for FixedSignedList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}

impl Drop for FixedSignedList {
    fn drop(&mut self) {
        if self.host.is_null() {
            return;
        }
        let offset = if self.gc_ptrs {
            items_offset::<usize>()
        } else {
            items_offset::<i64>()
        };
        let item = if self.gc_ptrs {
            std::mem::size_of::<usize>()
        } else {
            std::mem::size_of::<i64>()
        };
        let align = if self.gc_ptrs {
            host_align::<usize>()
        } else {
            host_align::<i64>()
        };
        host_free_aligned(self.host, offset + self.len * item, align);
        self.host = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::items_offset;

    #[test]
    fn empty_fixed_signed_list_allocates_nothing() {
        let list = super::FixedSignedList::empty(false);
        assert!(list.is_empty());
        assert_eq!(list.as_gcref(), majit_ir::GcRef(0));
    }

    #[test]
    fn sixty_four_bit_items_clear_their_alignment() {
        let align = std::mem::align_of::<i64>().max(std::mem::size_of::<usize>());
        assert_eq!(items_offset::<i64>() % align, 0);
        assert_eq!(items_offset::<u64>(), items_offset::<i64>());
        assert!(items_offset::<i64>() >= std::mem::size_of::<usize>());
        assert_eq!(items_offset::<usize>() % std::mem::align_of::<usize>(), 0);
    }

    #[test]
    fn append_varint_matches_encode_varint_signed_array() {
        let samples = [
            0i64,
            1,
            -1,
            127,
            -128,
            1 << 13,
            -(1 << 13),
            (1 << 20) - 1,
            -(1 << 20),
        ];
        let mut expected = Vec::new();
        for &v in &samples {
            let (bytes, n) = crate::opencoder::encode_varint_signed_array(v);
            expected.extend_from_slice(&bytes[..n]);
        }
        let mut one_at_a_time = super::CharList::with_capacity(0);
        for &v in &samples {
            one_at_a_time.append_varint(v);
        }
        assert_eq!(&one_at_a_time[..], expected.as_slice());
    }
}

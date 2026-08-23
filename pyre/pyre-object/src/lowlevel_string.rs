//! GC-managed low-level strings — pyre's `rstr.STR` / `rstr.UNICODE` equivalent.
//!
//! In a running interpreter a low-level string is a varsize GC leaf, matching
//! RPython's `GcStruct('rpy_string', ...)`.  Before the GC registry is installed
//! (principally unit tests), allocation falls back to an identically laid-out
//! raw block.  Both forms are laid out
//! as `{ hash: usize @0, len: usize @8, chars: [item; len] @16.. }`. The `len`
//! word is the RPython varsize length and, for the raw fallback only, also the
//! allocation capacity used to reconstruct its freeing `Layout`.
//!
//! These are the leaf primitives behind the `newstr`/`newunicode`/`strsetitem`
//! blackhole ops and the StringBuilder buffer (`current_buf`, STRINGPIECE `buf`).
//! They live here (not in `pyre-jit`) so the JIT fnaddr registry can reference
//! [`jit_ll_shrink_array`] as a residual-call target.

/// `len` field offset — one word past the `hash` field.
pub const LOWLEVEL_STRING_LEN_OFFSET: usize = std::mem::size_of::<usize>();
/// `chars` array offset — two words past the start (`hash`, then `len`).
pub const LOWLEVEL_STRING_CHARS_OFFSET: usize = 2 * std::mem::size_of::<usize>();
/// STR base size includes a trailing null byte after the chars array.
pub const LOWLEVEL_STR_BASE_SIZE: usize = LOWLEVEL_STRING_CHARS_OFFSET + 1;
/// UNICODE base size has no trailing null.
pub const LOWLEVEL_UNICODE_BASE_SIZE: usize = LOWLEVEL_STRING_CHARS_OFFSET;

static LOWLEVEL_STR_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static LOWLEVEL_UNICODE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

pub fn set_lowlevel_str_gc_type_id(id: u32) {
    debug_assert_ne!(id, 0, "0 is the unpublished sentinel");
    LOWLEVEL_STR_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Release);
    majit_gc::set_lowlevel_str_type_id(id);
}

pub fn set_lowlevel_unicode_gc_type_id(id: u32) {
    debug_assert_ne!(id, 0, "0 is the unpublished sentinel");
    LOWLEVEL_UNICODE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Release);
    majit_gc::set_lowlevel_unicode_type_id(id);
}

pub fn lowlevel_str_gc_type_id() -> u32 {
    LOWLEVEL_STR_GC_TYPE_ID.load(std::sync::atomic::Ordering::Acquire)
}

pub fn lowlevel_unicode_gc_type_id() -> u32 {
    LOWLEVEL_UNICODE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Acquire)
}

fn lowlevel_string_gc_type_id(base_size: usize, item_size: usize) -> u32 {
    match (base_size, item_size) {
        (LOWLEVEL_STR_BASE_SIZE, 1) => lowlevel_str_gc_type_id(),
        (LOWLEVEL_UNICODE_BASE_SIZE, 4) => lowlevel_unicode_gc_type_id(),
        _ => 0,
    }
}

/// Allocate a zero-filled low-level string of `length` items, storing `length`
/// in the varsize `len` word. Returns 0 on overflow / allocation failure.
pub fn bh_alloc_lowlevel_string(length: usize, base_size: usize, item_size: usize) -> i64 {
    let Some(items_size) = length.checked_mul(item_size) else {
        return 0;
    };
    let Some(total_size) = base_size.checked_add(items_size) else {
        return 0;
    };
    let tid = lowlevel_string_gc_type_id(base_size, item_size);
    let gc_ptr = if tid != 0 {
        let ptr = crate::gc_hook::try_gc_alloc_stable_raw(tid, total_size);
        ptr
    } else {
        std::ptr::null_mut()
    };
    let ptr = if !gc_ptr.is_null() {
        // Stable GC allocation is not specified to clear the payload.
        unsafe { std::ptr::write_bytes(gc_ptr, 0, total_size) };
        gc_ptr
    } else {
        let layout = std::alloc::Layout::from_size_align(total_size, std::mem::align_of::<usize>())
            .expect("low-level string layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return 0;
        }
        ptr
    };
    unsafe {
        (ptr.add(LOWLEVEL_STRING_LEN_OFFSET) as *mut usize).write(length);
    }
    ptr as i64
}

/// Read the `len` word.
pub fn bh_lowlevel_string_len(string: i64) -> usize {
    if string == 0 {
        return 0;
    }
    unsafe { *((string as *const u8).add(LOWLEVEL_STRING_LEN_OFFSET) as *const usize) }
}

/// Release a raw-fallback low-level string allocated by
/// [`bh_alloc_lowlevel_string`]. GC-managed strings are owned by the collector
/// and deliberately left untouched.
///
/// Reconstructs the exact `Layout` from the capacity word stored at
/// `LOWLEVEL_STRING_LEN_OFFSET` (the allocation stores its `length` argument
/// there, and the StringBuilder passes the buffer *capacity* as that argument).
/// `base_size`/`item_size` must match the allocation call.
pub fn bh_free_lowlevel_string(string: i64, base_size: usize, item_size: usize) {
    if string == 0 {
        return;
    }
    if majit_gc::gc_owns_object(string as usize) {
        return;
    }
    let capacity = bh_lowlevel_string_len(string);
    let total_size = base_size + capacity * item_size;
    let layout = std::alloc::Layout::from_size_align(total_size, std::mem::align_of::<usize>())
        .expect("low-level string layout");
    unsafe { std::alloc::dealloc(string as *mut u8, layout) };
}

/// `rgc.ll_shrink_array(buf, new_len)` for a raw STR low-level string — the
/// non-virtual residual target of the StringBuilder `build` tree's
/// `ll_shrink_final`.
///
/// Because the `len` word doubles as the freeing capacity, the shrink cannot
/// truncate in place: it allocates a fresh `new_len` buffer, copies `new_len`
/// chars, frees the old buffer, and returns the new one (whose `len` word is now
/// exactly `new_len`, so a caller returning it directly reports the right
/// length). STR width only (`item_size == 1`) — the width the `build` tree wires
/// today; the UNICODE builder is a future parallel set.
///
/// `extern "C"` with an `(i64, i64) -> i64` ABI so the JIT residual call reaches
/// it through the fnaddr registry.
pub extern "C" fn jit_ll_shrink_array(buf: i64, new_len: i64) -> i64 {
    if buf == 0 {
        return 0;
    }
    let new_len = if new_len < 0 { 0 } else { new_len as usize };
    const ITEM_SIZE: usize = 1;
    let new_buf = bh_alloc_lowlevel_string(new_len, LOWLEVEL_STR_BASE_SIZE, ITEM_SIZE);
    if new_buf == 0 {
        return 0;
    }
    // SAFETY: `buf` holds at least `new_len` valid chars (the builder shrinks to
    // its own `current_pos`), and `new_buf` was allocated for exactly `new_len`.
    unsafe {
        // Copy the fixed `hash` field (@0) before the variable char array, as
        // `ll_shrink_array` copies the fixed field first; a fresh allocation
        // zeroes it, so a non-zero cached hash would otherwise be lost.
        (new_buf as *mut usize).write(*(buf as *const usize));
        let src = (buf as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        let dst = (new_buf as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        std::ptr::copy_nonoverlapping(src, dst, new_len * ITEM_SIZE);
    }
    bh_free_lowlevel_string(buf, LOWLEVEL_STR_BASE_SIZE, ITEM_SIZE);
    new_buf
}

pub fn bh_lowlevel_chars_offset(item_size: usize) -> usize {
    if item_size == 1 {
        LOWLEVEL_STR_BASE_SIZE - 1
    } else {
        LOWLEVEL_UNICODE_BASE_SIZE
    }
}

pub fn bh_read_lowlevel_string(string: i64, item_size: usize) -> Vec<i64> {
    let len = bh_lowlevel_string_len(string);
    let chars_offset = bh_lowlevel_chars_offset(item_size);
    let mut chars = Vec::with_capacity(len);
    for index in 0..len {
        let addr = unsafe { (string as *const u8).add(chars_offset + index * item_size) };
        let value = unsafe {
            match item_size {
                1 => *addr as i64,
                4 => *(addr as *const u32) as i64,
                _ => *(addr as *const i64),
            }
        };
        chars.push(value);
    }
    chars
}

pub fn bh_write_lowlevel_char(string: i64, index: usize, char: i64, item_size: usize) {
    if string == 0 {
        return;
    }
    let chars_offset = bh_lowlevel_chars_offset(item_size);
    unsafe {
        let addr = (string as *mut u8).add(chars_offset + index * item_size);
        match item_size {
            1 => addr.write(char as u8),
            4 => (addr as *mut u32).write(char as u32),
            _ => (addr as *mut i64).write(char),
        }
    }
}

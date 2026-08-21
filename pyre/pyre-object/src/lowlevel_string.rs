//! Raw low-level string buffers — pyre's `rstr.STR` / `rstr.UNICODE` equivalent.
//!
//! A low-level string is a raw `std::alloc` block, off-GC and immobile, laid out
//! as `{ hash: usize @0, len: usize @8, chars: [item; len] @16.. }`. The `len`
//! word doubles as the allocation *capacity*: [`bh_free_lowlevel_string`]
//! reconstructs the freeing `Layout` from it, so a buffer's length can never be
//! truncated in place — a smaller logical length requires a fresh allocation
//! (see [`jit_ll_shrink_array`]).
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

/// Allocate a zero-filled low-level string of `length` items, storing `length`
/// in the `len` word. Returns 0 on overflow / allocation failure.
pub fn bh_alloc_lowlevel_string(length: usize, base_size: usize, item_size: usize) -> i64 {
    let Some(items_size) = length.checked_mul(item_size) else {
        return 0;
    };
    let Some(total_size) = base_size.checked_add(items_size) else {
        return 0;
    };
    let layout = std::alloc::Layout::from_size_align(total_size, std::mem::align_of::<usize>())
        .expect("low-level string layout");
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return 0;
    }
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

/// Free a low-level string allocated by [`bh_alloc_lowlevel_string`].
///
/// Reconstructs the exact `Layout` from the capacity word stored at
/// `LOWLEVEL_STRING_LEN_OFFSET` (the allocation stores its `length` argument
/// there, and the StringBuilder passes the buffer *capacity* as that argument).
/// `base_size`/`item_size` must match the allocation call.
pub fn bh_free_lowlevel_string(string: i64, base_size: usize, item_size: usize) {
    if string == 0 {
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
        let src = (buf as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        let dst = (new_buf as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        std::ptr::copy_nonoverlapping(src, dst, new_len * ITEM_SIZE);
    }
    bh_free_lowlevel_string(buf, LOWLEVEL_STR_BASE_SIZE, ITEM_SIZE);
    new_buf
}

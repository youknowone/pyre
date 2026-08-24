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

/// STR / UNICODE width `(base_size, item_size)` of the low-level string at
/// `buf`, read from its GC type id.
///
/// A GC-managed buffer carries the width in its header tid — the collector's own
/// record of the element size — exactly as `rgc.ll_shrink_array` recovers the
/// element type from the array it is handed. A raw-fallback buffer (no GC owner,
/// principally unit tests) has no header to read and defaults to STR, the only
/// width the raw path allocates. The `gc_owns_object` guard before the header
/// read mirrors `majit_gc::gc_finalizer_has_run`.
fn shrink_array_width(buf: i64) -> (usize, usize) {
    if buf != 0 && majit_gc::gc_owns_object(buf as usize) {
        let tid = unsafe { (*majit_gc::header::header_of(buf as usize)).type_id() };
        if tid != 0 && tid == lowlevel_unicode_gc_type_id() {
            return (LOWLEVEL_UNICODE_BASE_SIZE, 4);
        }
    }
    (LOWLEVEL_STR_BASE_SIZE, 1)
}

/// Width-parametric `rgc.ll_shrink_array(buf, new_len)` core.
///
/// Because the `len` word doubles as the freeing capacity, the shrink cannot
/// truncate in place: it allocates a fresh `new_len` buffer, copies `new_len`
/// items, frees the old buffer, and returns the new one (whose `len` word is now
/// exactly `new_len`, so a caller returning it directly reports the right
/// length). `base_size`/`item_size` select STR (`LOWLEVEL_STR_BASE_SIZE`, 1) vs
/// UNICODE (`LOWLEVEL_UNICODE_BASE_SIZE`, 4); the `chars` array offset is the
/// same two words in both widths.
fn shrink_lowlevel_array(buf: i64, new_len: i64, base_size: usize, item_size: usize) -> i64 {
    if buf == 0 {
        return 0;
    }
    let new_len = if new_len < 0 { 0 } else { new_len as usize };
    let new_buf = bh_alloc_lowlevel_string(new_len, base_size, item_size);
    if new_buf == 0 {
        return 0;
    }
    // SAFETY: `buf` holds at least `new_len` valid items (the builder shrinks to
    // its own `current_pos`), and `new_buf` was allocated for exactly `new_len`.
    unsafe {
        // Copy the fixed `hash` field (@0) before the variable item array, as
        // `ll_shrink_array` copies the fixed field first; a fresh allocation
        // zeroes it, so a non-zero cached hash would otherwise be lost.
        (new_buf as *mut usize).write(*(buf as *const usize));
        let src = (buf as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        let dst = (new_buf as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        std::ptr::copy_nonoverlapping(src, dst, new_len * item_size);
    }
    bh_free_lowlevel_string(buf, base_size, item_size);
    new_buf
}

/// `rgc.ll_shrink_array(buf, new_len)` — the non-virtual residual target of the
/// StringBuilder / UnicodeBuilder `build` tree's `ll_shrink_final`.
///
/// Selects the STR / UNICODE width from `buf`'s GC type id
/// ([`shrink_array_width`]) so a single residual target serves both builder
/// widths; the jtransform retarget names this one symbol for either. Only the
/// non-virtual buffer reaches here — a virtual buffer is folded by
/// `opt_call_shrink_array`.
///
/// `extern "C"` with an `(i64, i64) -> i64` ABI so the JIT residual call reaches
/// it through the fnaddr registry.
pub extern "C" fn jit_ll_shrink_array(buf: i64, new_len: i64) -> i64 {
    let (base_size, item_size) = shrink_array_width(buf);
    shrink_lowlevel_array(buf, new_len, base_size, item_size)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrink_lowlevel_array_str_truncates_and_preserves_hash() {
        let buf = bh_alloc_lowlevel_string(8, LOWLEVEL_STR_BASE_SIZE, 1);
        assert_ne!(buf, 0);
        // A non-zero cached hash must survive the realloc-shrink.
        unsafe { (buf as *mut usize).write(0xdead_beef) };
        for index in 0..8 {
            bh_write_lowlevel_char(buf, index, (b'a' + index as u8) as i64, 1);
        }
        let new_buf = shrink_lowlevel_array(buf, 5, LOWLEVEL_STR_BASE_SIZE, 1);
        assert_ne!(new_buf, 0);
        assert_eq!(bh_lowlevel_string_len(new_buf), 5);
        assert_eq!(unsafe { *(new_buf as *const usize) }, 0xdead_beef);
        assert_eq!(
            bh_read_lowlevel_string(new_buf, 1),
            vec![97, 98, 99, 100, 101]
        );
        bh_free_lowlevel_string(new_buf, LOWLEVEL_STR_BASE_SIZE, 1);
    }

    #[test]
    fn shrink_lowlevel_array_unicode_copies_width_4_items() {
        let buf = bh_alloc_lowlevel_string(4, LOWLEVEL_UNICODE_BASE_SIZE, 4);
        assert_ne!(buf, 0);
        for index in 0..4 {
            bh_write_lowlevel_char(buf, index, 0x1_0000 + index as i64, 4);
        }
        let new_buf = shrink_lowlevel_array(buf, 2, LOWLEVEL_UNICODE_BASE_SIZE, 4);
        assert_ne!(new_buf, 0);
        assert_eq!(bh_lowlevel_string_len(new_buf), 2);
        assert_eq!(
            bh_read_lowlevel_string(new_buf, 4),
            vec![0x1_0000, 0x1_0001]
        );
        bh_free_lowlevel_string(new_buf, LOWLEVEL_UNICODE_BASE_SIZE, 4);
    }

    #[test]
    fn jit_ll_shrink_array_defaults_to_str_for_a_raw_buffer() {
        // A raw-fallback buffer is not GC-owned, so the width defaults to STR.
        let buf = bh_alloc_lowlevel_string(6, LOWLEVEL_STR_BASE_SIZE, 1);
        assert_ne!(buf, 0);
        for index in 0..6 {
            bh_write_lowlevel_char(buf, index, (b'A' + index as u8) as i64, 1);
        }
        let new_buf = jit_ll_shrink_array(buf, 3);
        assert_ne!(new_buf, 0);
        assert_eq!(bh_lowlevel_string_len(new_buf), 3);
        assert_eq!(bh_read_lowlevel_string(new_buf, 1), vec![65, 66, 67]);
        bh_free_lowlevel_string(new_buf, LOWLEVEL_STR_BASE_SIZE, 1);
    }
}

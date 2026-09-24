//! `PYRE_MAX_MEMORY`: a hard ceiling on the process's total memory.
//!
//! Default 8 GiB. `0` is unbounded. A `K`/`M`/`G` suffix (optional `i`/`B`)
//! scales a byte count by 1024, 1024² or 1024³.
//!
//! mimalloc 3.x `mi_reserve_os_memory` + `mi_option_limit_os_alloc`
//! (`mi_option_disallow_os_alloc`) only refuses a new arena and the arena's
//! own OS fallback (`arena.c` `mi_arenas_try_alloc` / `mi_arena_os_alloc_aligned`).
//! Segment and ordinary `_mi_os_alloc` paths do not consult it, so a reserved
//! arena does not cap `Vec` buffers or GC blocks. The ceiling is therefore a
//! live-byte counter around the global allocator: every `std::alloc` caller
//! is covered, including the GC nursery (`alloc_arena`), old-gen arenas
//! (`allocate_new_arena`), rawmalloc blocks and `header` blocks.
//! `setrlimit(RLIMIT_AS)` is not used: macOS does not enforce it.
//!
//! A refused charge returns null. Infallible callers then abort in
//! `handle_alloc_error`; `try_reserve` and the bigint path turn the null
//! into `MemoryError`. `max_heap_size` is left at upstream's default so
//! major-collection thresholds stay put below the ceiling.

use std::alloc::{GlobalAlloc, Layout};
use std::sync::Once;

/// Default process ceiling, used when `PYRE_MAX_MEMORY` is unset.
pub const DEFAULT_MAX_MEMORY: u64 = 8 * 1024 * 1024 * 1024;

static INSTALLED: Once = Once::new();

/// Parse `PYRE_MAX_MEMORY`. `None` is the 8 GiB default. `0` is unbounded.
pub fn parse_max_memory(raw: Option<&str>) -> Result<u64, String> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_MAX_MEMORY);
    };
    let text = raw.trim();
    if text.is_empty() {
        return Err("empty value".into());
    }
    if text.starts_with('-') {
        return Err(format!("negative value {text:?}"));
    }
    let mut body = text;
    if body.len() > 1 && matches!(body.as_bytes().last(), Some(b'b' | b'B')) {
        body = &body[..body.len() - 1];
    }
    if body.len() > 1 && matches!(body.as_bytes().last(), Some(b'i' | b'I')) {
        body = &body[..body.len() - 1];
    }
    let (number, factor) = match body.as_bytes().last().copied() {
        Some(b'k' | b'K') => (&body[..body.len() - 1], 1024u64),
        Some(b'm' | b'M') => (&body[..body.len() - 1], 1024 * 1024),
        Some(b'g' | b'G') => (&body[..body.len() - 1], 1024 * 1024 * 1024),
        _ => (body, 1u64),
    };
    if number.is_empty() {
        return Err(format!("missing number in {text:?}"));
    }
    let value: u64 = number
        .parse()
        .map_err(|_| format!("not a byte count: {text:?}"))?;
    if value == 0 {
        return Ok(0);
    }
    value
        .checked_mul(factor)
        .ok_or_else(|| format!("overflows a byte count: {text:?}"))
}

/// Read `PYRE_MAX_MEMORY` and arm the ceiling. Invalid values exit the process.
/// Safe to call once, at the start of `main`, before the program allocates.
pub fn install() {
    let raw = std::env::var("PYRE_MAX_MEMORY").ok();
    let limit = match parse_max_memory(raw.as_deref()) {
        Ok(limit) => limit,
        Err(error) => {
            eprintln!("pyre: PYRE_MAX_MEMORY: {error}");
            std::process::exit(2);
        }
    };
    INSTALLED.call_once(|| apply(limit));
}

fn apply(limit: u64) {
    let bytes = usize::try_from(limit).unwrap_or(usize::MAX);
    majit_gc::arm_process_memory_ceiling(bytes);
}

fn refuse() -> *mut u8 {
    majit_gc::note_alloc_refused();
    std::ptr::null_mut()
}

/// Global allocator for the pyre binaries. A charge past the ceiling returns
/// null after one stderr line; it does not abort.
pub struct ProcessAllocator;

impl ProcessAllocator {
    pub const fn new() -> Self {
        Self
    }
}

unsafe impl GlobalAlloc for ProcessAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let charge = layout.size();
        if !majit_gc::try_charge(charge) {
            return refuse();
        }
        let ptr = unsafe { inner_alloc(layout) };
        if ptr.is_null() {
            majit_gc::uncharge(charge);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let charge = layout.size();
        if !majit_gc::try_charge(charge) {
            return refuse();
        }
        let ptr = unsafe { inner_alloc_zeroed(layout) };
        if ptr.is_null() {
            majit_gc::uncharge(charge);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { inner_dealloc(ptr, layout) };
        majit_gc::uncharge(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old = layout.size();
        if new_size > old && !majit_gc::try_charge(new_size - old) {
            return refuse();
        }
        let new_ptr = unsafe { inner_realloc(ptr, layout, new_size) };
        if new_ptr.is_null() {
            if new_size > old {
                majit_gc::uncharge(new_size - old);
            }
            return std::ptr::null_mut();
        }
        if new_size < old {
            majit_gc::uncharge(old - new_size);
        }
        new_ptr
    }
}

#[cfg(feature = "mimalloc")]
unsafe fn inner_alloc(layout: Layout) -> *mut u8 {
    unsafe { mimalloc::MiMalloc.alloc(layout) }
}

#[cfg(feature = "mimalloc")]
unsafe fn inner_alloc_zeroed(layout: Layout) -> *mut u8 {
    unsafe { mimalloc::MiMalloc.alloc_zeroed(layout) }
}

#[cfg(feature = "mimalloc")]
unsafe fn inner_dealloc(ptr: *mut u8, layout: Layout) {
    unsafe { mimalloc::MiMalloc.dealloc(ptr, layout) }
}

#[cfg(feature = "mimalloc")]
unsafe fn inner_realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    unsafe { mimalloc::MiMalloc.realloc(ptr, layout, new_size) }
}

#[cfg(not(feature = "mimalloc"))]
unsafe fn inner_alloc(layout: Layout) -> *mut u8 {
    unsafe { std::alloc::System.alloc(layout) }
}

#[cfg(not(feature = "mimalloc"))]
unsafe fn inner_alloc_zeroed(layout: Layout) -> *mut u8 {
    unsafe { std::alloc::System.alloc_zeroed(layout) }
}

#[cfg(not(feature = "mimalloc"))]
unsafe fn inner_dealloc(ptr: *mut u8, layout: Layout) {
    unsafe { std::alloc::System.dealloc(ptr, layout) }
}

#[cfg(not(feature = "mimalloc"))]
unsafe fn inner_realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_eight_gib_and_zero_is_unbounded() {
        assert_eq!(parse_max_memory(None).unwrap(), DEFAULT_MAX_MEMORY);
        assert_eq!(DEFAULT_MAX_MEMORY, 8 * 1024 * 1024 * 1024);
        assert_eq!(parse_max_memory(Some("0")).unwrap(), 0);
        assert_eq!(parse_max_memory(Some("0M")).unwrap(), 0);
    }

    #[test]
    fn suffixes_scale_by_powers_of_two() {
        assert_eq!(parse_max_memory(Some("512")).unwrap(), 512);
        assert_eq!(parse_max_memory(Some("512K")).unwrap(), 512 * 1024);
        assert_eq!(parse_max_memory(Some("512M")).unwrap(), 512 * 1024 * 1024);
        assert_eq!(
            parse_max_memory(Some("2G")).unwrap(),
            2 * 1024 * 1024 * 1024
        );
        assert_eq!(parse_max_memory(Some("1MB")).unwrap(), 1024 * 1024);
        assert_eq!(parse_max_memory(Some("1MiB")).unwrap(), 1024 * 1024);
        assert_eq!(parse_max_memory(Some(" 4k ")).unwrap(), 4 * 1024);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_max_memory(Some("")).is_err());
        assert!(parse_max_memory(Some("nope")).is_err());
        assert!(parse_max_memory(Some("-1")).is_err());
        assert!(parse_max_memory(Some("999999999999G")).is_err());
    }
}

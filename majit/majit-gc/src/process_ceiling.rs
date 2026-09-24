//! Process-wide memory ceiling shared by the global allocator and JIT code
//! mappings.
//!
//! `incminimark.py` `max_heap_size` is not set from this ceiling. A non-zero
//! `max_heap_size` pulls `next_major_collection_threshold` down
//! (`set_max_heap_size`), which changes when major collections run even while
//! the heap is far below the ceiling. The process charge is the ceiling;
//! a refused `std::alloc` or `mmap` returns failure, and the bigint / bytes
//! paths turn that into `MemoryError` the way `raw_malloc` does.
//!
//! The failure line is formatted when the ceiling is armed, so the refusal
//! path does not allocate.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::Mutex;

static CEILING: AtomicUsize = AtomicUsize::new(0);
static CHARGED: AtomicUsize = AtomicUsize::new(0);
static REPORTED: AtomicBool = AtomicBool::new(false);
static LINE: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// Record `bytes` as the process ceiling. `0` clears it (unbounded).
pub fn arm_process_memory_ceiling(bytes: usize) {
    CEILING.store(bytes, Ordering::Relaxed);
    if bytes == 0 {
        *LINE.lock() = Vec::new();
        return;
    }
    let line = format!(
        "pyre: process memory limit of {bytes} bytes exceeded (PYRE_MAX_MEMORY; 0 = unbounded)\n"
    );
    *LINE.lock() = line.into_bytes();
}

pub fn process_memory_ceiling() -> usize {
    CEILING.load(Ordering::Relaxed)
}

/// Account `size` bytes against the ceiling.
///
/// `true` when the bytes fit or no ceiling is armed. A `false` answer has
/// not been charged. Unbounded (`0`) does not touch the counter.
pub fn try_charge(size: usize) -> bool {
    if size == 0 {
        return true;
    }
    let limit = CEILING.load(Ordering::Relaxed);
    if limit == 0 {
        return true;
    }
    let prev = CHARGED.fetch_add(size, Ordering::Relaxed);
    if prev.saturating_add(size) > limit {
        let _ = CHARGED.fetch_sub(size, Ordering::Relaxed);
        return false;
    }
    true
}

/// Return `size` bytes previously accepted by [`try_charge`].
pub fn uncharge(size: usize) {
    if size == 0 || CEILING.load(Ordering::Relaxed) == 0 {
        return;
    }
    // Saturating: a block allocated before the ceiling was armed was never
    // charged, and freeing it must not wrap the counter.
    let _ = CHARGED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |charged| {
        Some(charged.saturating_sub(size))
    });
}

/// Print the ceiling line once. Does not abort: the caller returns failure
/// (`null` from the global allocator, or an `mmap` error) so a fallible
/// allocation becomes `MemoryError` and an infallible one reaches
/// `handle_alloc_error`.
pub fn note_alloc_refused() {
    if CEILING.load(Ordering::Relaxed) == 0 || REPORTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let line = LINE.lock();
    if line.is_empty() {
        emit(b"pyre: process memory limit exceeded (PYRE_MAX_MEMORY; 0 = unbounded)\n");
    } else {
        emit(&line);
    }
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
fn emit(bytes: &[u8]) {
    unsafe {
        libc::write(2, bytes.as_ptr().cast(), bytes.len());
    }
}

#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
fn emit(bytes: &[u8]) {
    use std::io::Write;
    let _ = std::io::stderr().write_all(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block allocated before arming is freed after it: the counter must
    /// stay at zero instead of wrapping and refusing every later charge.
    #[test]
    fn uncharge_of_an_uncharged_block_does_not_wrap() {
        arm_process_memory_ceiling(1 << 20);
        CHARGED.store(0, Ordering::Relaxed);
        uncharge(64);
        assert_eq!(CHARGED.load(Ordering::Relaxed), 0);
        assert!(try_charge(4096));
        uncharge(4096);
        arm_process_memory_ceiling(0);
    }
}

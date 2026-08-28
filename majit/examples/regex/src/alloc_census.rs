//! A counting global allocator, so a change to the JIT's per-deopt cost can be
//! graded on a machine that is doing other work.
//!
//! Wall clock cannot grade it here. The branching row's own history is the
//! evidence: the same binary read 56,208 and 203,740 chars/s an hour apart,
//! and an interleaved A/B of two binaries built from the same tree minutes
//! apart read a 4.9x spread between rounds on the *same* arm. A difference
//! below that spread is not measurable by timing, however many repeats are
//! taken.
//!
//! What this counts is measurable, because it does not depend on the machine
//! at all: the same binary over the same input allocates the same number of
//! bytes every time. That is the right instrument for this particular JIT
//! defect class, because the costs in question ARE allocation — a jitframe
//! `alloc_zeroed`ed and freed per guard failure, blackhole frames boxed per
//! resume, position tables sized by a monotonic counter. Bytes moved is what
//! those cost, and `chars/s` is only the machine's opinion of it today.
//!
//! Off by default: `--features alloc-census`. A `#[global_allocator]` is
//! process-wide, so the four relaxed atomics per allocation would otherwise
//! sit inside the timed rows this file reports.
//!
//! ```sh
//! cargo run -p regex --release --no-default-features \
//!     --features dynasm,alloc-census
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Calls to `alloc` and `alloc_zeroed`.
static ALLOCS: AtomicU64 = AtomicU64::new(0);
/// Bytes those calls asked for.
static BYTES: AtomicU64 = AtomicU64::new(0);
/// The `alloc_zeroed` subset, in bytes.
///
/// Separate because a zeroing allocation is charged twice: once for the
/// allocator's own bookkeeping and once for the `memset` the caller would
/// otherwise have written itself. `alloc_off_gc_jitframe` was the one this
/// example was aimed at, and watching this counter fall is how its removal was
/// graded.
static ZEROED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Calls to `dealloc`.
static FREES: AtomicU64 = AtomicU64::new(0);

pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        ZEROED_BYTES.fetch_add(layout.size() as u64, Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        FREES.fetch_add(1, Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Counted as one allocation of the growth, not of the whole block: a
        // `Vec` doubling from 1 MiB to 2 MiB moves 1 MiB of new bytes, and
        // charging it 2 MiB would make a growing vector look like a fresh one.
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(new_size.saturating_sub(layout.size()) as u64, Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// The four counters, read together.
#[derive(Clone, Copy)]
pub struct Census {
    pub allocs: u64,
    pub bytes: u64,
    pub zeroed_bytes: u64,
    pub frees: u64,
}

pub fn read() -> Census {
    Census {
        allocs: ALLOCS.load(Relaxed),
        bytes: BYTES.load(Relaxed),
        zeroed_bytes: ZEROED_BYTES.load(Relaxed),
        frees: FREES.load(Relaxed),
    }
}

impl Census {
    /// What happened between `self` and `later`.
    pub fn since(self, later: Census) -> Census {
        Census {
            allocs: later.allocs - self.allocs,
            bytes: later.bytes - self.bytes,
            zeroed_bytes: later.zeroed_bytes - self.zeroed_bytes,
            frees: later.frees - self.frees,
        }
    }

    /// One line, per input character, which is the unit every other row in
    /// this file is reported in.
    pub fn per_char(&self, chars: usize, runs: usize) -> String {
        let n = (chars * runs) as f64;
        format!(
            "{:>8.1} allocs/char, {:>9.1} B/char ({:>9.1} B zeroed), {:>8.1} frees/char",
            self.allocs as f64 / n,
            self.bytes as f64 / n,
            self.zeroed_bytes as f64 / n,
            self.frees as f64 / n,
        )
    }
}

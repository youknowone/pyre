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

use std::alloc::{GlobalAlloc, Layout};

/// The allocator the counters sit in front of. See the `fast-alloc` feature.
#[cfg(feature = "fast-alloc")]
const INNER: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(not(feature = "fast-alloc"))]
const INNER: std::alloc::System = std::alloc::System;
use std::cell::Cell;
use std::sync::atomic::AtomicUsize;
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

/// The largest request size [`SIZES`] tracks exactly.
///
/// Every allocation this example makes per input character is a small object —
/// a frame, a resume entry, a boxed interpreter — so an exact size is a usable
/// fingerprint for the site that asked for it, in a way a power-of-two bucket
/// is not: 336 and 344 are different structs and land in the same bucket.
const MAX_TRACKED_SIZE: usize = 4096;

/// Allocation count per exact request size, `[0, MAX_TRACKED_SIZE]`.
static SIZES: [AtomicU64; MAX_TRACKED_SIZE + 1] =
    [const { AtomicU64::new(0) }; MAX_TRACKED_SIZE + 1];

/// Allocations larger than [`MAX_TRACKED_SIZE`], which the histogram reports as
/// one row rather than pretending to attribute them.
static OVERSIZE: AtomicU64 = AtomicU64::new(0);

/// Optional exact-size attribution, configured once by `main` before the
/// timed rows. Zero means disabled. The small fixed sample is enough to name
/// a repeated allocation site without making a full benchmark spend minutes
/// symbolizing every allocation.
static TRACE_SIZE: AtomicUsize = AtomicUsize::new(0);
static TRACE_LEFT: AtomicUsize = AtomicUsize::new(0);
static TRACE_INITIAL_SKIP: AtomicUsize = AtomicUsize::new(0);
static TRACE_SKIP: AtomicUsize = AtomicUsize::new(0);

std::thread_local! {
    /// Backtrace capture and formatting allocate. Bypass both the census and
    /// attribution while inside the probe so the allocator cannot recurse.
    static IN_TRACE: Cell<bool> = const { Cell::new(false) };
}

pub fn configure_trace_from_env() {
    let Ok(value) = std::env::var("PYRE_CENSUS_TRACE_SIZE") else {
        return;
    };
    let size: usize = value
        .parse()
        .unwrap_or_else(|_| panic!("PYRE_CENSUS_TRACE_SIZE is not a usize: {value:?}"));
    assert!(size > 0, "PYRE_CENSUS_TRACE_SIZE must be greater than zero");
    TRACE_SIZE.store(size, Relaxed);
    let skip = std::env::var("PYRE_CENSUS_TRACE_SKIP")
        .ok()
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("PYRE_CENSUS_TRACE_SKIP is not a usize: {value:?}"))
        })
        .unwrap_or(0);
    TRACE_INITIAL_SKIP.store(skip, Relaxed);
    TRACE_SKIP.store(skip, Relaxed);
    TRACE_LEFT.store(8, Relaxed);
}

pub fn rearm_trace(label: &str) {
    if TRACE_SIZE.load(Relaxed) != 0 {
        eprintln!("[alloc-census] rearm row={label}");
        TRACE_SKIP.store(TRACE_INITIAL_SKIP.load(Relaxed), Relaxed);
        TRACE_LEFT.store(8, Relaxed);
    }
}

#[inline(never)]
fn trace_size(size: usize) {
    if size != TRACE_SIZE.load(Relaxed) {
        return;
    }
    if TRACE_SKIP
        .fetch_update(Relaxed, Relaxed, |left| (left > 0).then(|| left - 1))
        .is_ok()
    {
        return;
    }
    if TRACE_LEFT
        .fetch_update(Relaxed, Relaxed, |left| left.checked_sub(1))
        .is_err()
    {
        return;
    }
    let already = IN_TRACE.try_with(|flag| flag.replace(true)).unwrap_or(true);
    if already {
        return;
    }
    eprintln!(
        "[alloc-census] allocation size={size}\n{}",
        std::backtrace::Backtrace::force_capture()
    );
    let _ = IN_TRACE.try_with(|flag| flag.set(false));
}

#[inline]
fn record_size(size: usize) {
    if size <= MAX_TRACKED_SIZE {
        SIZES[size].fetch_add(1, Relaxed);
    } else {
        OVERSIZE.fetch_add(1, Relaxed);
    }
}

/// Count one allocation of `size`.
#[inline]
fn observe(size: usize, zeroed: bool) {
    ALLOCS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size as u64, Relaxed);
    if zeroed {
        ZEROED_BYTES.fetch_add(size as u64, Relaxed);
    }
    record_size(size);
}

pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if IN_TRACE.try_with(Cell::get).unwrap_or(false) {
            return unsafe { INNER.alloc(layout) };
        }
        observe(layout.size(), false);
        trace_size(layout.size());
        unsafe { INNER.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if IN_TRACE.try_with(Cell::get).unwrap_or(false) {
            return unsafe { INNER.alloc_zeroed(layout) };
        }
        observe(layout.size(), true);
        trace_size(layout.size());
        unsafe { INNER.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        FREES.fetch_add(1, Relaxed);
        unsafe { INNER.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if IN_TRACE.try_with(Cell::get).unwrap_or(false) {
            return unsafe { INNER.realloc(ptr, layout, new_size) };
        }
        // Counted as one allocation of the growth, not of the whole block: a
        // `Vec` doubling from 1 MiB to 2 MiB moves 1 MiB of new bytes, and
        // charging it 2 MiB would make a growing vector look like a fresh one.
        let growth = new_size.saturating_sub(layout.size());
        observe(growth, false);
        trace_size(growth);
        unsafe { INNER.realloc(ptr, layout, new_size) }
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

/// A snapshot of [`SIZES`], so two of them can be subtracted the way
/// [`Census::since`] subtracts the scalars.
pub struct SizeHistogram {
    counts: Vec<u64>,
    oversize: u64,
}

pub fn read_sizes() -> SizeHistogram {
    SizeHistogram {
        counts: SIZES.iter().map(|c| c.load(Relaxed)).collect(),
        oversize: OVERSIZE.load(Relaxed),
    }
}

impl SizeHistogram {
    /// The rows that grew between `self` and `later`, biggest allocation count
    /// first, as `(size, count)`.
    pub fn since(&self, later: &SizeHistogram) -> Vec<(usize, u64)> {
        let mut rows: Vec<(usize, u64)> = later
            .counts
            .iter()
            .zip(&self.counts)
            .enumerate()
            .filter_map(|(size, (after, before))| {
                let delta = after - before;
                (delta > 0).then_some((size, delta))
            })
            .collect();
        rows.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        rows
    }

    pub fn oversize_since(&self, later: &SizeHistogram) -> u64 {
        later.oversize - self.oversize
    }
}

/// The histogram as lines, densest site first, cut off once the tail stops
/// being worth a row.
///
/// Per input character rather than absolute, because that is the unit every
/// other row this example prints uses, and because it is what stays comparable
/// across input lengths.
pub fn report_rows(
    rows: &[(usize, u64)],
    oversize: u64,
    chars: usize,
    runs: usize,
    top: usize,
) -> Vec<String> {
    let n = (chars * runs) as f64;
    let total: u64 = rows.iter().map(|&(_, c)| c).sum::<u64>() + oversize;
    if total == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for &(size, count) in rows.iter().take(top) {
        out.push(format!(
            "{size:>6} B  {:>7.2} allocs/char  {:>8.1} B/char  {:>5.1}%",
            count as f64 / n,
            (size as u64 * count) as f64 / n,
            100.0 * count as f64 / total as f64,
        ));
    }
    if rows.len() > top {
        let shown: u64 = rows.iter().take(top).map(|&(_, c)| c).sum();
        let rest: u64 = rows.iter().map(|&(_, c)| c).sum::<u64>() - shown;
        out.push(format!(
            "{:>6}    {:>7.2} allocs/char  {:>8} {:>5.1}%",
            format!("+{}", rows.len() - top),
            rest as f64 / n,
            "",
            100.0 * rest as f64 / total as f64,
        ));
    }
    if oversize > 0 {
        out.push(format!(
            "{:>6}    {:>7.2} allocs/char  (over {} B, not attributed)",
            ">max",
            oversize as f64 / n,
            MAX_TRACKED_SIZE,
        ));
    }
    out
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

//! Per-run JIT census for an embedder.
//!
//! An interpreter that drives [`JitDriver`] needs to know what its JIT actually
//! did: a green test suite, agreement with the untraced tier and an exact
//! answer are all satisfied by an interpreter that never compiled anything.
//! The evidence lives behind four driver callbacks and a set of process-global
//! diagnostic slots. [`Census`] wires those up once so an embedder does not
//! rebuild the counters, the callback closures and the serializing lock per
//! module.
//!
//! # The window is the point
//!
//! The counters behind this module are process-global and cumulative, because
//! the callbacks are `'static` closures that cannot borrow a caller's local
//! state. An absolute read therefore answers "since this process started",
//! which is not the question a run asks. [`Census::begin`] opens a window and
//! [`Census::counts`] reports the DELTA across it.
//!
//! Two rules make that delta trustworthy, and both were learned from getting
//! them wrong:
//!
//! * The serializing lock is a process-global [`Mutex`], not a `thread_local!`.
//!   Its job is to keep two concurrently running consumers — parallel test
//!   threads, most often — out of each other's window. A thread-local lock is
//!   uncontended by construction and silently does nothing at all.
//! * The cumulative counters are diffed, never zeroed, so a window cannot
//!   destroy an enclosing reader's totals. The "last compiled body" fields
//!   cannot be diffed, so [`Census::begin`] resets those instead — and it holds
//!   the window lock across the reset AND the read, which is what makes that
//!   reset safe. A reset outside a held lock hands each of two overlapping
//!   consumers a fraction of the truth.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::{JitDriver, JitState, LoopBodyShape};

/// Loops compiled, from the compile-loop callback.
static COMPILES: AtomicUsize = AtomicUsize::new(0);
/// Guards that failed back into the runtime, from the guard-failure callback.
static GUARD_FAILURES: AtomicUsize = AtomicUsize::new(0);
/// Traces abandoned before they closed, from the trace-abort callback.
static TRACE_ABORTS: AtomicUsize = AtomicUsize::new(0);
/// Calls that entered compiled code, from the compiled-entry callback.
static COMPILED_ENTRIES: AtomicUsize = AtomicUsize::new(0);
/// Recorded and optimized op counts, summed over every compiled loop.
static TRACE_OPS_BEFORE: AtomicUsize = AtomicUsize::new(0);
static TRACE_OPS_AFTER: AtomicUsize = AtomicUsize::new(0);
/// Driver-internal totals taken off a driver by [`Census::absorb`]. Neither has
/// a callback, so a driver that is dropped without being absorbed takes its
/// tallies with it.
static ABSORBED_BRIDGES: AtomicUsize = AtomicUsize::new(0);
static ABSORBED_PANICS: AtomicUsize = AtomicUsize::new(0);

/// The last compiled body's optimized op count and shape.
///
/// Not cumulative, so [`Census::begin`] resets them rather than diffing them.
/// The shape is held as its two flags rather than the struct so the recording
/// stays lock-free on the compile path.
static LAST_OPS_AFTER: AtomicUsize = AtomicUsize::new(0);
static LAST_HAS_JUMP: AtomicBool = AtomicBool::new(false);
static LAST_ALWAYS_FAILS: AtomicBool = AtomicBool::new(false);

/// Serializes measurement windows against each other.
///
/// Process-global on purpose: the counters it protects are, and the consumer
/// this exists for is a parallel test runner entering the same JIT from several
/// threads at once.
static WINDOW_LOCK: Mutex<()> = Mutex::new(());

/// One reading of what the JIT did.
///
/// Every field except the two `last_*` ones is a count over the window it was
/// read in. `bridges_compiled` and `internal_compile_panics` are the exception
/// in a second way as well: they have no callback and only appear here for a
/// driver that was handed to [`Census::absorb`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CensusCounts {
    pub loops_compiled: usize,
    /// Only what [`Census::absorb`] collected. Zero otherwise, which is not the
    /// same as "no bridge was compiled".
    pub bridges_compiled: usize,
    pub loops_aborted: usize,
    pub guard_failures: usize,
    /// Non-zero means a trace was dropped by a panic inside compilation and the
    /// tier silently fell back to the untraced path for it. Nothing else
    /// reports this: a run that stops compiling still answers correctly, so
    /// every other counter here stays plausible. Only what
    /// [`Census::absorb`] collected, as with `bridges_compiled`.
    pub internal_compile_panics: usize,
    pub trace_ops_before: usize,
    pub trace_ops_after: usize,
    /// The only field that separates "an artifact exists" from "an artifact
    /// ran".
    pub compiled_entries: usize,
    /// Optimized op count of the LAST compiled body in the window.
    ///
    /// `loops_compiled > 0` is necessary but not sufficient evidence of a
    /// working tier: an entirely empty dispatch still compiles a trace, one
    /// whose whole optimized body is `Finish()`. A compile counter counts
    /// TRACES, not WORK.
    pub last_ops_after: usize,
    /// Shape of the LAST compiled body in the window. Its [`Default`] value is
    /// also what "nothing compiled in this window" reads as — see
    /// [`LoopBodyShape::why_not`], whose no-`Jump` arm is phrased for
    /// both cases.
    pub last_loop_body_shape: LoopBodyShape,
}

impl fmt::Display for CensusCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "loops_compiled={} bridges_compiled={} loops_aborted={} \
             guard_failures={} internal_compile_panics={} trace_ops_before={} \
             trace_ops_after={} compiled_entries={} last_ops_after={} \
             last_closes_a_loop={}",
            self.loops_compiled,
            self.bridges_compiled,
            self.loops_aborted,
            self.guard_failures,
            self.internal_compile_panics,
            self.trace_ops_before,
            self.trace_ops_after,
            self.compiled_entries,
            self.last_ops_after,
            self.last_loop_body_shape.closes_a_loop(),
        )
    }
}

/// An open measurement window over the process-global JIT counters.
///
/// Hold one for as long as the run being measured, then read it:
///
/// ```ignore
/// let census = Census::begin();
/// let answer = run_the_interpreter();
/// assert!(census.counts().loops_compiled > 0, "{census}");
/// ```
///
/// ⚠ NOT REENTRANT. [`Census::begin`] takes a plain [`Mutex`], so opening a
/// second window on a thread that already holds one deadlocks. Where a test
/// helper opens the window, every test that can compile has to go through that
/// helper and none of them may call another.
///
/// ⚠ Bind it to a NAMED local, `_census` included. `let _ = Census::begin();`
/// drops the window on the spot and leaves the run unserialized, which fails
/// only under contention and only sometimes.
pub struct Census {
    /// Cumulative counters at [`Census::begin`], subtracted at read time.
    base: RawCounts,
    /// Abort-reason tallies at [`Census::begin`], for the same reason.
    aborts_before: Vec<(&'static str, u64)>,
    /// Held for the window's whole life. Dropping the [`Census`] closes it.
    _window: MutexGuard<'static, ()>,
}

/// The cumulative half of a reading, snapshotted for the later subtraction.
#[derive(Clone, Copy, Debug, Default)]
struct RawCounts {
    compiles: usize,
    bridges: usize,
    aborts: usize,
    guard_failures: usize,
    panics: usize,
    ops_before: usize,
    ops_after: usize,
    compiled_entries: usize,
}

impl RawCounts {
    fn read() -> Self {
        Self {
            compiles: COMPILES.load(Ordering::Relaxed),
            bridges: ABSORBED_BRIDGES.load(Ordering::Relaxed),
            aborts: TRACE_ABORTS.load(Ordering::Relaxed),
            guard_failures: GUARD_FAILURES.load(Ordering::Relaxed),
            panics: ABSORBED_PANICS.load(Ordering::Relaxed),
            ops_before: TRACE_OPS_BEFORE.load(Ordering::Relaxed),
            ops_after: TRACE_OPS_AFTER.load(Ordering::Relaxed),
            compiled_entries: COMPILED_ENTRIES.load(Ordering::Relaxed),
        }
    }
}

impl Census {
    /// Wire every driver callback the census reads into the process-global
    /// counters.
    ///
    /// The four hooks are `set_on_compile_loop`, `set_on_guard_failure`,
    /// `set_on_trace_abort` and `set_on_compiled_entry`. Each holds ONE
    /// closure, so this replaces whatever was installed before it; an embedder
    /// that was recording the compiled body's shape by hand reads
    /// [`CensusCounts::last_loop_body_shape`] instead of installing its own.
    ///
    /// Install it on every driver whose activity should be counted — the
    /// counters are shared, the callbacks are not.
    pub fn install<S: JitState>(driver: &mut JitDriver<S>) {
        driver.set_on_compile_loop(|_green_key, ops_before, ops_after, opcodes| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
            TRACE_OPS_BEFORE.fetch_add(ops_before, Ordering::Relaxed);
            TRACE_OPS_AFTER.fetch_add(ops_after, Ordering::Relaxed);
            LAST_OPS_AFTER.store(ops_after, Ordering::Relaxed);
            let shape = LoopBodyShape::of(opcodes);
            LAST_HAS_JUMP.store(shape.has_jump, Ordering::Relaxed);
            LAST_ALWAYS_FAILS.store(shape.has_always_fails, Ordering::Relaxed);
        });
        driver.set_on_guard_failure(|_green_key, _fail_index, _count| {
            GUARD_FAILURES.fetch_add(1, Ordering::Relaxed);
        });
        driver.set_on_trace_abort(|_green_key, _permanent| {
            TRACE_ABORTS.fetch_add(1, Ordering::Relaxed);
        });
        driver.set_on_compiled_entry(|_green_key, _target_pc| {
            COMPILED_ENTRIES.fetch_add(1, Ordering::Relaxed);
        });
    }

    /// Open a measurement window: serialize against any other window, snapshot
    /// the cumulative counters, and reset the non-cumulative ones.
    ///
    /// Blocks until any window open on another thread closes. See the type's
    /// reentrancy and binding warnings.
    #[must_use = "the window closes when the returned Census is dropped"]
    pub fn begin() -> Self {
        let window = WINDOW_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        Self {
            base: RawCounts::read(),
            aborts_before: abort_reasons(),
            _window: window,
        }
    }

    /// What the JIT did since [`Census::begin`].
    pub fn counts(&self) -> CensusCounts {
        let now = RawCounts::read();
        CensusCounts {
            loops_compiled: now.compiles.saturating_sub(self.base.compiles),
            bridges_compiled: now.bridges.saturating_sub(self.base.bridges),
            loops_aborted: now.aborts.saturating_sub(self.base.aborts),
            guard_failures: now.guard_failures.saturating_sub(self.base.guard_failures),
            internal_compile_panics: now.panics.saturating_sub(self.base.panics),
            trace_ops_before: now.ops_before.saturating_sub(self.base.ops_before),
            trace_ops_after: now.ops_after.saturating_sub(self.base.ops_after),
            compiled_entries: now
                .compiled_entries
                .saturating_sub(self.base.compiled_entries),
            last_ops_after: LAST_OPS_AFTER.load(Ordering::Relaxed),
            last_loop_body_shape: LoopBodyShape {
                has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
                has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
            },
        }
    }

    /// The abort reasons that fired since [`Census::begin`], as `label=delta`.
    ///
    /// Empty when nothing aborted, so a caller can print it unconditionally in
    /// a failure message and a quiet window stays quiet. See [`abort_reasons`]
    /// for what the labels do and do not distinguish.
    pub fn abort_reasons_since(&self) -> String {
        render_abort_delta(&self.aborts_before, &abort_reasons())
    }

    /// Take a driver's own tallies before it is dropped.
    ///
    /// Bridge compiles and swallowed compilation panics have no callback, so
    /// they live on the driver and die with it. This adds them to the shared
    /// counters, which makes them visible to any window still open.
    ///
    /// ⚠ Call it ONCE, at the driver's end of life. The driver's tallies are
    /// cumulative, so absorbing a driver that will keep running counts its
    /// history again at the next call.
    pub fn absorb<S: JitState>(driver: &JitDriver<S>) {
        let stats = driver.get_stats();
        ABSORBED_BRIDGES.fetch_add(stats.bridges_compiled, Ordering::Relaxed);
        ABSORBED_PANICS.fetch_add(stats.internal_compile_panics as usize, Ordering::Relaxed);
    }

    /// Absolute counters since process start, or since the last
    /// [`Census::reset`]. Takes no lock, so it is safe to call from inside an
    /// open window — and is a total, not a window, which is why the assertions
    /// a run makes belong on [`Census::counts`] instead.
    pub fn totals() -> CensusCounts {
        let now = RawCounts::read();
        CensusCounts {
            loops_compiled: now.compiles,
            bridges_compiled: now.bridges,
            loops_aborted: now.aborts,
            guard_failures: now.guard_failures,
            internal_compile_panics: now.panics,
            trace_ops_before: now.ops_before,
            trace_ops_after: now.ops_after,
            compiled_entries: now.compiled_entries,
            last_ops_after: LAST_OPS_AFTER.load(Ordering::Relaxed),
            last_loop_body_shape: LoopBodyShape {
                has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
                has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
            },
        }
    }

    /// Zero every counter this module owns.
    ///
    /// For a caller that reads totals rather than windows. It moves the ground
    /// under any window that is already open, so a windowed reader should have
    /// no reason to call it — [`Census::begin`] already gives that reader a
    /// fresh zero without disturbing anyone else.
    pub fn reset() {
        for counter in [
            &COMPILES,
            &GUARD_FAILURES,
            &TRACE_ABORTS,
            &COMPILED_ENTRIES,
            &TRACE_OPS_BEFORE,
            &TRACE_OPS_AFTER,
            &ABSORBED_BRIDGES,
            &ABSORBED_PANICS,
            &LAST_OPS_AFTER,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
    }
}

impl fmt::Display for Census {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.counts())
    }
}

/// Snapshot the abort-reason tallies as `(label, count)` pairs.
///
/// [`CensusCounts::loops_aborted`] counts aborts without saying why, and the
/// reason is not carried on the trace-abort callback — it survives only in the
/// `MC_DIAG` slots. Without this a gate can assert `loops_aborted == 0`, watch
/// it fail, and have no way to learn which reason fired.
///
/// Selection is by the `abrt_` label prefix rather than by the slot range those
/// labels currently occupy: a hard-coded range names the wrong counters the
/// moment a slot is added, and it does so silently.
///
/// ⚠ `abrt_bridge` is the UNCLASSIFIED bucket wearing a specific-sounding name.
/// Every abort whose reason was never staged falls back to the generic reason,
/// whose id is that slot, so a count there is not evidence of bridge activity.
/// [`Census::abort_reasons_since`] relabels it in its output for that reason.
///
/// ⚠ The slots are process-global, cumulative, and have no reset, so diff two
/// snapshots rather than reading one.
pub fn abort_reasons() -> Vec<(&'static str, u64)> {
    crate::MC_DIAG_LABELS
        .iter()
        .enumerate()
        .filter(|(_, label)| label.starts_with("abrt_"))
        .map(|(slot, label)| (*label, crate::mc_diag(slot)))
        .collect()
}

/// Render `after - before` over two [`abort_reasons`] snapshots, dropping the
/// slots that did not move.
fn render_abort_delta(before: &[(&'static str, u64)], after: &[(&'static str, u64)]) -> String {
    after
        .iter()
        .zip(before)
        .filter_map(|((label, now), (_, then))| {
            let delta = now.saturating_sub(*then);
            if delta == 0 {
                return None;
            }
            let name = if *label == "abrt_bridge" {
                "unclassified(abrt_bridge)"
            } else {
                label
            };
            Some(format!("{name}={delta}"))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window read is a DELTA, so activity that happened before it opened
    /// belongs to nobody's window.
    #[test]
    fn a_window_reports_only_what_happened_inside_it() {
        let census = Census::begin();
        assert_eq!(census.counts(), CensusCounts::default());

        COMPILES.fetch_add(2, Ordering::Relaxed);
        TRACE_OPS_AFTER.fetch_add(11, Ordering::Relaxed);
        assert_eq!(census.counts().loops_compiled, 2);
        assert_eq!(census.counts().trace_ops_after, 11);

        drop(census);
        // A second window opened after the first closed starts from zero
        // without either of them having zeroed a shared counter.
        let next = Census::begin();
        assert_eq!(next.counts().loops_compiled, 0);
        assert!(Census::totals().loops_compiled >= 2);
    }

    /// The `last_*` fields cannot be diffed, so the window resets them — and a
    /// window with no compile in it must not inherit the previous one's body.
    #[test]
    fn the_last_body_fields_do_not_leak_across_windows() {
        {
            let _census = Census::begin();
            LAST_OPS_AFTER.store(42, Ordering::Relaxed);
            LAST_HAS_JUMP.store(true, Ordering::Relaxed);
        }
        let census = Census::begin();
        let counts = census.counts();
        assert_eq!(counts.last_ops_after, 0);
        assert!(!counts.last_loop_body_shape.closes_a_loop());
        assert!(counts.last_loop_body_shape.why_not().is_some());
    }

    #[test]
    fn abort_reason_labels_are_selected_by_prefix_not_by_slot_range() {
        let reasons = abort_reasons();
        assert!(!reasons.is_empty());
        assert!(reasons.iter().all(|(label, _)| label.starts_with("abrt_")));
    }

    #[test]
    fn a_quiet_abort_window_renders_empty_and_a_busy_one_names_its_slot() {
        let before = vec![("abrt_too_long", 3u64), ("abrt_bridge", 1u64)];
        assert_eq!(render_abort_delta(&before, &before), "");

        let after = vec![("abrt_too_long", 5u64), ("abrt_bridge", 4u64)];
        assert_eq!(
            render_abort_delta(&before, &after),
            "abrt_too_long=2 unclassified(abrt_bridge)=3"
        );
    }

    #[test]
    fn display_names_every_field_it_reports() {
        let text = CensusCounts::default().to_string();
        for key in [
            "loops_compiled=",
            "bridges_compiled=",
            "loops_aborted=",
            "guard_failures=",
            "internal_compile_panics=",
            "trace_ops_before=",
            "trace_ops_after=",
            "compiled_entries=",
            "last_ops_after=",
            "last_closes_a_loop=",
        ] {
            assert!(text.contains(key), "{key} missing from {text}");
        }
    }
}

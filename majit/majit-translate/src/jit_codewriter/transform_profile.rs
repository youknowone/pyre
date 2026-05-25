//! Env-gated per-phase accumulator for `transform_graph_to_jitcode`.
//!
//! Enabled when `PYRE_PROFILE_DRAIN` is set; aggregates wall-clock time
//! across every drained graph so the per-phase hotspot is identifiable
//! without `O(graphs)` log noise.

use std::cell::RefCell;
use std::time::{Duration, Instant};

thread_local! {
    static TOTALS: RefCell<Vec<(&'static str, Duration, usize)>> = const { RefCell::new(Vec::new()) };
}

pub fn enabled() -> bool {
    std::env::var_os("PYRE_PROFILE_DRAIN").is_some()
}

pub fn record(phase: &'static str, dur: Duration) {
    TOTALS.with(|t| {
        let mut t = t.borrow_mut();
        if let Some(slot) = t.iter_mut().find(|(name, _, _)| *name == phase) {
            slot.1 += dur;
            slot.2 += 1;
        } else {
            t.push((phase, dur, 1));
        }
    });
}

pub fn time_phase<R>(phase: &'static str, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let start = Instant::now();
    let out = f();
    record(phase, start.elapsed());
    out
}

pub struct PhaseScope {
    phase: &'static str,
    start: Option<Instant>,
}

impl PhaseScope {
    pub fn new(phase: &'static str) -> Self {
        Self {
            phase,
            start: enabled().then(Instant::now),
        }
    }
}

impl Drop for PhaseScope {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            record(self.phase, start.elapsed());
        }
    }
}

pub fn dump_transform_phase_totals() {
    TOTALS.with(|t| {
        let mut t = t.borrow_mut();
        let mut grand = Duration::ZERO;
        for (_, d, _) in t.iter() {
            grand += *d;
        }
        eprintln!(
            "[PYRE_PROFILE_DRAIN] PHASE TOTALS ({:.3}s wall across all phases):",
            grand.as_secs_f64(),
        );
        let mut rows: Vec<_> = t.iter().cloned().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, dur, n) in rows {
            eprintln!(
                "[PYRE_PROFILE_DRAIN]   {:>8.3}s  {:>5} calls  {}",
                dur.as_secs_f64(),
                n,
                name,
            );
        }
        t.clear();
    });
}

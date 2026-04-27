//! Port of `rpython/jit/metainterp/jitprof.py:52-122 Profiler`.
//!
//! RPython carries `counters: list[int]` indexed by `Counters.*` (rlib/jit.py:
//! 1414-1442) and a separate `calls: int` for the CALL+RECORDED_OPS path.
//! Pyre stores each counter as its own `AtomicUsize` field on
//! [`JitProfiler`] so cross-crate callers (TraceCtx in `pyre-jit-trace`,
//! heapcache in `majit-trace`, the vector pass in `optimizeopt`) can hit
//! `MetaInterpStaticData.profiler` through the shared `Arc` without any
//! extra synchronisation.
//!
//! `Ordering::Relaxed` is sufficient for every counter: there is no causal
//! relationship between any two counter updates, and we only ever publish
//! totals via [`JitProfiler::snapshot`] which itself is `Relaxed`.

use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;

use crate::pyjitpl::counters;

/// jitprof.py:52-122 `Profiler` — every `Counters.*` slot is one
/// `AtomicUsize`, plus the standalone `calls` counter that
/// `count_ops` increments on the CALL_*+RECORDED_OPS path
/// (jitprof.py:121-122).
///
/// `field_for_kind` maps a `Counters.*` id (see `pyjitpl::counters`) to
/// the matching `AtomicUsize`; unknown ids are silently ignored,
/// matching upstream's permissive `self.counters[kind] += 1` (an
/// out-of-range id raises `IndexError` upstream, but that has only ever
/// fired for hand-rolled counter ids that are not part of the
/// canonical `Counters` enum — pyre never produces such ids).
#[derive(Default, Debug)]
pub struct JitProfiler {
    /// jit.py:1416 `Counters.TRACING` — RPython tracks this as wall-clock
    /// time + entry count via `_start`/`_end` (jitprof.py:75-93). Pyre
    /// only records the entry count today; the wall-clock split lands
    /// when start_tracing/end_tracing hooks are wired.
    pub tracing: AtomicUsize,
    /// jit.py:1417 `Counters.BACKEND` — same shape as TRACING.
    pub backend: AtomicUsize,
    /// jit.py:1418 `Counters.OPS` — every executed op
    /// (`execute_and_record_varargs` / `execute_and_record`,
    /// pyjitpl.py:2629/2645).
    pub ops: AtomicUsize,
    /// jit.py:1419 `Counters.HEAPCACHED_OPS` — folded-away ops that the
    /// heapcache resolved without recording (pyjitpl.py:388/397/562/...).
    pub heapcached_ops: AtomicUsize,
    /// jit.py:1420 `Counters.RECORDED_OPS` — ops that survived the
    /// heapcache and reached `_record_helper` / `_record_helper_varargs`
    /// (pyjitpl.py:2658/2669).
    pub recorded_ops: AtomicUsize,
    /// jit.py:1421 `Counters.GUARDS` — guards counted by the trace
    /// recorder (pyjitpl.py:2581).
    pub guards: AtomicUsize,
    /// jit.py:1422 `Counters.OPT_OPS` — every op the optimizer emits
    /// (optimizer.py:626 inside `_emit_operation`).
    pub opt_ops: AtomicUsize,
    /// jit.py:1423 `Counters.OPT_GUARDS` — guards emitted by the
    /// optimizer (optimizer.py:629).
    pub opt_guards: AtomicUsize,
    /// jit.py:1424 `Counters.OPT_GUARDS_SHARED` — guards that share
    /// resume data with a previous guard via descriptor fusion
    /// (optimizer.py:673-674).
    pub opt_guards_shared: AtomicUsize,
    /// jit.py:1425 `Counters.OPT_FORCINGS`.
    pub opt_forcings: AtomicUsize,
    /// jit.py:1426 `Counters.OPT_VECTORIZE_TRY` — entries into the
    /// vector pass (vector.py:139).
    pub opt_vectorize_try: AtomicUsize,
    /// jit.py:1427 `Counters.OPT_VECTORIZED` — successful vectorise
    /// (vector.py:146).
    pub opt_vectorized: AtomicUsize,
    /// jit.py:1428 `Counters.ABORT_TOO_LONG`.
    pub abort_too_long: AtomicUsize,
    /// jit.py:1429 `Counters.ABORT_BRIDGE`.
    pub abort_bridge: AtomicUsize,
    /// jit.py:1430 `Counters.ABORT_BAD_LOOP`.
    pub abort_bad_loop: AtomicUsize,
    /// jit.py:1431 `Counters.ABORT_ESCAPE`.
    pub abort_escape: AtomicUsize,
    /// jit.py:1432 `Counters.ABORT_FORCE_QUASIIMMUT`.
    pub abort_force_quasiimmut: AtomicUsize,
    /// jit.py:1433 `Counters.ABORT_SEGMENTED_TRACE`.
    pub abort_segmented_trace: AtomicUsize,
    /// jit.py:1434 `Counters.FORCE_VIRTUALIZABLES`.
    pub force_virtualizables: AtomicUsize,
    /// jit.py:1435 `Counters.NVIRTUALS`.
    pub nvirtuals: AtomicUsize,
    /// jit.py:1436 `Counters.NVHOLES`.
    pub nvholes: AtomicUsize,
    /// jit.py:1437 `Counters.NVREUSED`.
    pub nvreused: AtomicUsize,
    /// jitprof.Profiler.calls — `count_ops` increments this when the op
    /// is a CALL_* and `kind == RECORDED_OPS` (jitprof.py:121-122).
    pub calls: AtomicUsize,
}

impl JitProfiler {
    /// jitprof.py:118-122 `Profiler.count_ops(opnum, kind=Counters.OPS)`.
    ///
    /// ```python
    /// def count_ops(self, opnum, kind=Counters.OPS):
    ///     self.counters[kind] += 1
    ///     if OpHelpers.is_call(opnum) and kind == Counters.RECORDED_OPS:
    ///         self.calls += 1
    /// ```
    ///
    /// `kind` is a [`crate::pyjitpl::counters`] id. Unknown ids are a
    /// silent no-op (see field doc).
    pub fn count_ops(&self, opnum: OpCode, kind: i32) {
        if let Some(field) = self.field_for_kind(kind) {
            field.fetch_add(1, Ordering::Relaxed);
        }
        if opnum.is_call() && kind == counters::RECORDED_OPS {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// jitprof.py:101-102 `Profiler.count(kind, inc=1)`.
    ///
    /// ```python
    /// def count(self, kind, inc=1):
    ///     self.counters[kind] += inc
    /// ```
    ///
    /// Used for non-op events (ABORT_*, NV*, OPT_VECTORIZE_*, ...).
    /// Unknown ids are a silent no-op (matching the OPS variant above).
    pub fn count(&self, kind: i32, inc: usize) {
        if let Some(field) = self.field_for_kind(kind) {
            field.fetch_add(inc, Ordering::Relaxed);
        }
    }

    /// jitprof.py:104-113 `Profiler.get_counter(num)` — single-counter
    /// readback via `Counters.*` id. `None` for unknown ids.
    pub fn get_counter(&self, kind: i32) -> Option<usize> {
        self.field_for_kind(kind)
            .map(|field| field.load(Ordering::Relaxed))
    }

    /// Snapshot every counter at a moment.
    ///
    /// Each load is `Relaxed`, so the returned snapshot is **not**
    /// guaranteed to be a single coherent read across counters — only
    /// each individual counter's value is consistent with itself.
    /// Mirrors RPython where `_print_stats` (jitprof.py:130-174) reads
    /// each `cnt[Counters.X]` one-by-one without any locking.
    pub fn snapshot(&self) -> JitProfilerSnapshot {
        JitProfilerSnapshot {
            tracing: self.tracing.load(Ordering::Relaxed),
            backend: self.backend.load(Ordering::Relaxed),
            ops: self.ops.load(Ordering::Relaxed),
            heapcached_ops: self.heapcached_ops.load(Ordering::Relaxed),
            recorded_ops: self.recorded_ops.load(Ordering::Relaxed),
            guards: self.guards.load(Ordering::Relaxed),
            opt_ops: self.opt_ops.load(Ordering::Relaxed),
            opt_guards: self.opt_guards.load(Ordering::Relaxed),
            opt_guards_shared: self.opt_guards_shared.load(Ordering::Relaxed),
            opt_forcings: self.opt_forcings.load(Ordering::Relaxed),
            opt_vectorize_try: self.opt_vectorize_try.load(Ordering::Relaxed),
            opt_vectorized: self.opt_vectorized.load(Ordering::Relaxed),
            abort_too_long: self.abort_too_long.load(Ordering::Relaxed),
            abort_bridge: self.abort_bridge.load(Ordering::Relaxed),
            abort_bad_loop: self.abort_bad_loop.load(Ordering::Relaxed),
            abort_escape: self.abort_escape.load(Ordering::Relaxed),
            abort_force_quasiimmut: self.abort_force_quasiimmut.load(Ordering::Relaxed),
            abort_segmented_trace: self.abort_segmented_trace.load(Ordering::Relaxed),
            force_virtualizables: self.force_virtualizables.load(Ordering::Relaxed),
            nvirtuals: self.nvirtuals.load(Ordering::Relaxed),
            nvholes: self.nvholes.load(Ordering::Relaxed),
            nvreused: self.nvreused.load(Ordering::Relaxed),
            calls: self.calls.load(Ordering::Relaxed),
        }
    }

    fn field_for_kind(&self, kind: i32) -> Option<&AtomicUsize> {
        Some(match kind {
            counters::TRACING => &self.tracing,
            counters::BACKEND => &self.backend,
            counters::OPS => &self.ops,
            counters::HEAPCACHED_OPS => &self.heapcached_ops,
            counters::RECORDED_OPS => &self.recorded_ops,
            counters::GUARDS => &self.guards,
            counters::OPT_OPS => &self.opt_ops,
            counters::OPT_GUARDS => &self.opt_guards,
            counters::OPT_GUARDS_SHARED => &self.opt_guards_shared,
            counters::OPT_FORCINGS => &self.opt_forcings,
            counters::OPT_VECTORIZE_TRY => &self.opt_vectorize_try,
            counters::OPT_VECTORIZED => &self.opt_vectorized,
            counters::ABORT_TOO_LONG => &self.abort_too_long,
            counters::ABORT_BRIDGE => &self.abort_bridge,
            counters::ABORT_BAD_LOOP => &self.abort_bad_loop,
            counters::ABORT_ESCAPE => &self.abort_escape,
            counters::ABORT_FORCE_QUASIIMMUT => &self.abort_force_quasiimmut,
            counters::ABORT_SEGMENTED_TRACE => &self.abort_segmented_trace,
            counters::FORCE_VIRTUALIZABLES => &self.force_virtualizables,
            counters::NVIRTUALS => &self.nvirtuals,
            counters::NVHOLES => &self.nvholes,
            counters::NVREUSED => &self.nvreused,
            _ => return None,
        })
    }
}

/// Plain-old-data snapshot of [`JitProfiler`].
///
/// Used by debug printers, tests, and the `JitStats` view. Field order
/// mirrors RPython's `_print_stats` (jitprof.py:130-174) so the
/// eventual `print_stats` port can iterate on a fixed layout.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct JitProfilerSnapshot {
    pub tracing: usize,
    pub backend: usize,
    pub ops: usize,
    pub heapcached_ops: usize,
    pub recorded_ops: usize,
    pub guards: usize,
    pub opt_ops: usize,
    pub opt_guards: usize,
    pub opt_guards_shared: usize,
    pub opt_forcings: usize,
    pub opt_vectorize_try: usize,
    pub opt_vectorized: usize,
    pub abort_too_long: usize,
    pub abort_bridge: usize,
    pub abort_bad_loop: usize,
    pub abort_escape: usize,
    pub abort_force_quasiimmut: usize,
    pub abort_segmented_trace: usize,
    pub force_virtualizables: usize,
    pub nvirtuals: usize,
    pub nvholes: usize,
    pub nvreused: usize,
    pub calls: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_ops_increments_kind_bucket_and_calls_only_on_call_recorded_ops() {
        // jitprof.py:118-122 contract: counters[kind] += 1, and if the op
        // is a CALL_* AND kind == RECORDED_OPS, calls += 1.  Other
        // (kind, opnum) combinations leave `calls` untouched.
        let prof = JitProfiler::default();
        // OPS path on a non-call: ops += 1, calls unchanged.
        prof.count_ops(OpCode::IntAdd, counters::OPS);
        assert_eq!(prof.ops.load(Ordering::Relaxed), 1);
        assert_eq!(prof.calls.load(Ordering::Relaxed), 0);
        // OPS path on a CALL_*: kind != RECORDED_OPS so calls untouched
        // (jitprof.py:121 only bumps calls on the RECORDED_OPS branch).
        prof.count_ops(OpCode::CallI, counters::OPS);
        assert_eq!(prof.ops.load(Ordering::Relaxed), 2);
        assert_eq!(prof.calls.load(Ordering::Relaxed), 0);
        // RECORDED_OPS path on a non-call: recorded_ops += 1, calls untouched.
        prof.count_ops(OpCode::IntAdd, counters::RECORDED_OPS);
        assert_eq!(prof.recorded_ops.load(Ordering::Relaxed), 1);
        assert_eq!(prof.calls.load(Ordering::Relaxed), 0);
        // RECORDED_OPS + CALL_*: both recorded_ops and calls bump.
        prof.count_ops(OpCode::CallI, counters::RECORDED_OPS);
        assert_eq!(prof.recorded_ops.load(Ordering::Relaxed), 2);
        assert_eq!(prof.calls.load(Ordering::Relaxed), 1);
        // HEAPCACHED_OPS / GUARDS independent buckets.
        prof.count_ops(OpCode::PtrEq, counters::HEAPCACHED_OPS);
        prof.count_ops(OpCode::GuardTrue, counters::GUARDS);
        assert_eq!(prof.heapcached_ops.load(Ordering::Relaxed), 1);
        assert_eq!(prof.guards.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn count_routes_every_counters_id_to_its_atomic_field() {
        // Walk every Counters.* id `count` should accept and verify it
        // lands in the matching atomic.  Anchors the field_for_kind
        // dispatch table — a future Counters addition without a matching
        // arm is caught here.
        let prof = JitProfiler::default();
        for (kind, expected) in [
            (counters::TRACING, &prof.tracing),
            (counters::BACKEND, &prof.backend),
            (counters::OPS, &prof.ops),
            (counters::HEAPCACHED_OPS, &prof.heapcached_ops),
            (counters::RECORDED_OPS, &prof.recorded_ops),
            (counters::GUARDS, &prof.guards),
            (counters::OPT_OPS, &prof.opt_ops),
            (counters::OPT_GUARDS, &prof.opt_guards),
            (counters::OPT_GUARDS_SHARED, &prof.opt_guards_shared),
            (counters::OPT_FORCINGS, &prof.opt_forcings),
            (counters::OPT_VECTORIZE_TRY, &prof.opt_vectorize_try),
            (counters::OPT_VECTORIZED, &prof.opt_vectorized),
            (counters::ABORT_TOO_LONG, &prof.abort_too_long),
            (counters::ABORT_BRIDGE, &prof.abort_bridge),
            (counters::ABORT_BAD_LOOP, &prof.abort_bad_loop),
            (counters::ABORT_ESCAPE, &prof.abort_escape),
            (
                counters::ABORT_FORCE_QUASIIMMUT,
                &prof.abort_force_quasiimmut,
            ),
            (counters::ABORT_SEGMENTED_TRACE, &prof.abort_segmented_trace),
            (counters::FORCE_VIRTUALIZABLES, &prof.force_virtualizables),
            (counters::NVIRTUALS, &prof.nvirtuals),
            (counters::NVHOLES, &prof.nvholes),
            (counters::NVREUSED, &prof.nvreused),
        ] {
            let before = expected.load(Ordering::Relaxed);
            prof.count(kind, 3);
            assert_eq!(
                expected.load(Ordering::Relaxed),
                before + 3,
                "kind {kind} did not land in the expected atomic field",
            );
        }
    }

    #[test]
    fn count_with_unknown_kind_is_silent_noop() {
        // pyre's permissive variant of upstream's `counters[kind] += 1`:
        // an id outside the `Counters.*` set is silently ignored so a
        // miswired callsite cannot panic the profiler.
        let prof = JitProfiler::default();
        prof.count(-1, 5);
        prof.count(99, 5);
        prof.count_ops(OpCode::IntAdd, -7);
        let snap = prof.snapshot();
        assert_eq!(snap, JitProfilerSnapshot::default());
    }

    #[test]
    fn snapshot_reads_every_counter_independently() {
        // jitprof.py:130-174 `_print_stats` reads each cnt[Counters.X]
        // one-by-one with no locking — `snapshot` must do the same.
        let prof = JitProfiler::default();
        prof.count(counters::OPS, 7);
        prof.count(counters::OPT_OPS, 11);
        prof.count(counters::NVIRTUALS, 13);
        prof.count_ops(OpCode::CallI, counters::RECORDED_OPS);
        let snap = prof.snapshot();
        assert_eq!(snap.ops, 7);
        assert_eq!(snap.opt_ops, 11);
        assert_eq!(snap.nvirtuals, 13);
        assert_eq!(snap.recorded_ops, 1);
        assert_eq!(snap.calls, 1);
    }

    #[test]
    fn get_counter_reads_via_kind_id() {
        // jitprof.py:104-113 `get_counter(num)` — pyre returns Option to
        // signal unknown ids instead of upstream's IndexError.
        let prof = JitProfiler::default();
        prof.count(counters::ABORT_ESCAPE, 1);
        assert_eq!(prof.get_counter(counters::ABORT_ESCAPE), Some(1));
        assert_eq!(prof.get_counter(counters::OPS), Some(0));
        assert_eq!(prof.get_counter(-1), None);
    }

    #[test]
    fn metainterp_static_data_default_initialises_profiler_with_zeroed_counters() {
        // pyjitpl.py:2199-2200 contract: every freshly built
        // `MetaInterpStaticData` carries a `Profiler()` with all
        // counters at 0 and exposes it as a public field so cross-crate
        // callers can hit it through the shared `Arc`.
        let sd = crate::pyjitpl::MetaInterpStaticData::new();
        let snap = sd.profiler.snapshot();
        assert_eq!(snap, JitProfilerSnapshot::default());
        // Update through the field and confirm the same struct
        // observes the bump (no shadowing into a separate counter).
        sd.profiler.count_ops(OpCode::IntAdd, counters::OPS);
        assert_eq!(sd.profiler.ops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn metainterp_staticdata_arc_lets_cross_crate_callers_share_one_profiler() {
        // The Arc<MetaInterpStaticData> on MetaInterp.staticdata is what
        // TraceCtx / heapcache / vector pass clone to reach the same
        // counter sink (RPython parity: `self.metainterp_sd.profiler`).
        // This test exercises the same shape: clone the Arc, bump
        // through the clone, observe via the original.
        let sd = std::sync::Arc::new(crate::pyjitpl::MetaInterpStaticData::new());
        let cross_crate_handle = sd.clone();
        cross_crate_handle
            .profiler
            .count_ops(OpCode::CallI, counters::RECORDED_OPS);
        cross_crate_handle.profiler.count(counters::NVIRTUALS, 4);
        let snap = sd.profiler.snapshot();
        assert_eq!(snap.recorded_ops, 1);
        assert_eq!(snap.calls, 1);
        assert_eq!(snap.nvirtuals, 4);
    }
}

/// Warm state management — the lifecycle from interpreting to compiled code.
///
/// Manages the transition: Interpreting -> Tracing -> Compiled.
/// When the hot counter fires, we start tracing. When the trace is
/// complete, we compile it and cache the result.
///
/// Reference: rpython/jit/metainterp/warmstate.py WarmEnterState, BaseJitCell
use std::collections::HashMap;
use std::time::Duration;

use majit_codegen::LoopToken;

use crate::counter::JitCounter;
use crate::jitlog::JitLog;
use crate::recorder::TraceRecorder;

/// Flags on a JitCell, mirroring warmstate.py JC_* constants.
pub mod jc_flags {
    /// We are currently tracing from this green key.
    pub const TRACING: u8 = 0x01;
    /// Don't trace here (e.g., trace was too long last time).
    pub const DONT_TRACE_HERE: u8 = 0x02;
    /// Has a temporary procedure token (CALL_ASSEMBLER fallback).
    pub const TEMPORARY: u8 = 0x04;
    /// Tracing has occurred at least once from this key.
    pub const TRACING_OCCURRED: u8 = 0x08;
}

/// Per-greenkey cell that tracks JIT state for a specific program location.
///
/// Mirrors rpython/jit/metainterp/warmstate.py BaseJitCell.
pub struct JitCell {
    /// JC_* flags.
    pub flags: u8,
    /// Compiled loop token, if compilation has completed.
    pub loop_token: Option<LoopToken>,
}

impl JitCell {
    fn new() -> Self {
        JitCell {
            flags: 0,
            loop_token: None,
        }
    }

    pub fn is_tracing(&self) -> bool {
        self.flags & jc_flags::TRACING != 0
    }

    pub fn is_compiled(&self) -> bool {
        self.loop_token.is_some() && (self.flags & jc_flags::TEMPORARY == 0)
    }

    pub fn has_procedure_token(&self) -> bool {
        self.loop_token.is_some()
    }
}

/// The current JIT state for a particular green key.
pub enum JitState {
    /// Normal interpretation; no tracing is active.
    Interpreting,
    /// Actively recording a trace.
    Tracing(TraceRecorder),
    /// A compiled loop exists for this green key.
    Compiled(LoopToken),
}

/// Warm state manager — the orchestrator of the JIT lifecycle.
///
/// Keeps track of per-greenkey cells and the global hot counter.
/// The interpreter calls `maybe_compile()` at loop headers;
/// WarmState decides whether to start tracing, continue interpreting,
/// or dispatch to compiled code.
/// Default number of guard failures before triggering bridge compilation.
const DEFAULT_BRIDGE_THRESHOLD: u32 = 5;

pub struct WarmState {
    /// Global hot counter.
    counter: JitCounter,
    /// Per-greenkey cells, keyed by the hash of the green key.
    cells: HashMap<u64, JitCell>,
    /// Compilation threshold (copied from counter for easy access).
    threshold: u32,
    /// Guard failure threshold for triggering bridge compilation.
    bridge_threshold: u32,
    /// Next token number for compiled loops.
    next_token_number: u64,
    /// Optional profiling logger, enabled via MAJIT_STATS=1 or MAJIT_LOG=1.
    jitlog: Option<JitLog>,
}

/// Result of checking whether a green key is hot.
pub enum HotResult {
    /// Not yet hot; keep interpreting.
    NotHot,
    /// Threshold reached; start tracing.
    StartTracing(TraceRecorder),
    /// Already tracing (caller should keep feeding ops to the active recorder).
    AlreadyTracing,
    /// Compiled code exists; run it.
    RunCompiled,
}

impl WarmState {
    /// Create a new WarmState with the given threshold.
    /// Automatically enables JitLog if MAJIT_STATS=1 or MAJIT_LOG=1.
    pub fn new(threshold: u32) -> Self {
        WarmState {
            counter: JitCounter::new(threshold),
            cells: HashMap::new(),
            threshold,
            bridge_threshold: DEFAULT_BRIDGE_THRESHOLD,
            next_token_number: 0,
            jitlog: JitLog::from_env(),
        }
    }

    /// Create a new WarmState with an explicit JitLog.
    pub fn with_jitlog(threshold: u32, jitlog: Option<JitLog>) -> Self {
        WarmState {
            counter: JitCounter::new(threshold),
            cells: HashMap::new(),
            threshold,
            bridge_threshold: DEFAULT_BRIDGE_THRESHOLD,
            next_token_number: 0,
            jitlog,
        }
    }

    /// Check and possibly transition the JIT state for a given green key.
    ///
    /// Called by the interpreter at loop back-edges and function entries.
    /// Returns a `HotResult` telling the interpreter what to do next.
    pub fn maybe_compile(&mut self, green_key_hash: u64) -> HotResult {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            if cell.is_compiled() {
                return HotResult::RunCompiled;
            }
            if cell.is_tracing() {
                return HotResult::AlreadyTracing;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                return HotResult::NotHot;
            }
        }

        if !self.counter.tick(green_key_hash) {
            return HotResult::NotHot;
        }

        // Threshold reached: start tracing
        self.counter.reset(green_key_hash);
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(JitCell::new);
        cell.flags |= jc_flags::TRACING | jc_flags::TRACING_OCCURRED;

        HotResult::StartTracing(TraceRecorder::new())
    }

    /// Mark that tracing is done for a green key. Clears the TRACING flag.
    /// The caller is responsible for compiling the trace and calling
    /// `install_compiled` with the resulting LoopToken.
    pub fn finish_tracing(&mut self, green_key_hash: u64) {
        if let Some(cell) = self.cells.get_mut(&green_key_hash) {
            cell.flags &= !jc_flags::TRACING;
        }
    }

    /// Mark that tracing was aborted for a green key.
    /// Optionally sets DONT_TRACE_HERE to prevent retrying.
    pub fn abort_tracing(&mut self, green_key_hash: u64, dont_trace_here: bool) {
        if let Some(cell) = self.cells.get_mut(&green_key_hash) {
            cell.flags &= !jc_flags::TRACING;
            if dont_trace_here {
                cell.flags |= jc_flags::DONT_TRACE_HERE;
            }
        }
        if let Some(log) = &mut self.jitlog {
            log.log_abort();
        }
    }

    /// Install a compiled loop token for a green key.
    pub fn install_compiled(&mut self, green_key_hash: u64, token: LoopToken) {
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(JitCell::new);
        cell.flags &= !jc_flags::TRACING;
        cell.loop_token = Some(token);
    }

    /// Get a reference to the compiled loop token for a green key.
    pub fn get_compiled(&self, green_key_hash: u64) -> Option<&LoopToken> {
        self.cells
            .get(&green_key_hash)
            .and_then(|cell| cell.loop_token.as_ref())
    }

    /// Allocate a new unique LoopToken number.
    pub fn alloc_token_number(&mut self) -> u64 {
        let n = self.next_token_number;
        self.next_token_number += 1;
        n
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Set a new threshold. Also updates the internal counter.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
        self.counter.set_threshold(threshold);
    }

    /// Decay all counters (e.g., periodically to avoid stale counts).
    pub fn decay_counters(&mut self) {
        self.counter.decay_all();
    }

    /// Get a reference to the JitCell for a green key, if it exists.
    pub fn get_cell(&self, green_key_hash: u64) -> Option<&JitCell> {
        self.cells.get(&green_key_hash)
    }

    /// Log a successful trace compilation. No-op if JitLog is disabled.
    pub fn log_compile(
        &mut self,
        green_key: u64,
        ops_before_opt: usize,
        ops_after_opt: usize,
        opt_time: Duration,
        compile_time: Duration,
    ) {
        if let Some(log) = &mut self.jitlog {
            log.log_compile(green_key, ops_before_opt, ops_after_opt, opt_time, compile_time);
        }
    }

    /// Log a guard failure. No-op if JitLog is disabled.
    pub fn log_guard_failure(&mut self, guard_index: u32) {
        if let Some(log) = &mut self.jitlog {
            log.log_guard_failure(guard_index);
        }
    }

    /// Log a loop entry. No-op if JitLog is disabled.
    pub fn log_loop_entry(&mut self, green_key: u64) {
        if let Some(log) = &mut self.jitlog {
            log.log_loop_entry(green_key);
        }
    }

    /// Get a reference to the JitLog, if enabled.
    pub fn jitlog(&self) -> Option<&JitLog> {
        self.jitlog.as_ref()
    }

    /// Get the bridge compilation threshold.
    pub fn bridge_threshold(&self) -> u32 {
        self.bridge_threshold
    }

    /// Set the bridge compilation threshold.
    pub fn set_bridge_threshold(&mut self, threshold: u32) {
        self.bridge_threshold = threshold;
    }

    /// Check whether a guard failure count has reached the bridge threshold.
    /// Returns true if bridge compilation should be triggered.
    pub fn should_compile_bridge(&self, guard_fail_count: u32) -> bool {
        guard_fail_count >= self.bridge_threshold
    }

    /// Log a bridge compilation. No-op if JitLog is disabled.
    pub fn log_bridge_compile(&mut self, guard_index: u32) {
        if let Some(log) = &mut self.jitlog {
            log.log_bridge_compile(guard_index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_hot_initially() {
        let mut ws = WarmState::new(3);
        match ws.maybe_compile(42) {
            HotResult::NotHot => {}
            _ => panic!("expected NotHot"),
        }
    }

    #[test]
    fn test_start_tracing_at_threshold() {
        let mut ws = WarmState::new(3);
        // Tick 1, 2: not hot
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        // Tick 3: threshold reached, start tracing
        match ws.maybe_compile(42) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }
    }

    #[test]
    fn test_already_tracing() {
        let mut ws = WarmState::new(2);
        // First tick: eviction (always false). Second tick: threshold reached.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }
        // Next call sees TRACING flag
        match ws.maybe_compile(42) {
            HotResult::AlreadyTracing => {}
            _ => panic!("expected AlreadyTracing"),
        }
    }

    #[test]
    fn test_run_compiled() {
        let mut ws = WarmState::new(1);
        let token_num = ws.alloc_token_number();
        let token = LoopToken::new(token_num);
        ws.install_compiled(42, token);

        match ws.maybe_compile(42) {
            HotResult::RunCompiled => {}
            _ => panic!("expected RunCompiled"),
        }
    }

    #[test]
    fn test_finish_tracing() {
        let mut ws = WarmState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(42);

        let cell = ws.get_cell(42).unwrap();
        assert!(!cell.is_tracing());
        assert!(cell.flags & jc_flags::TRACING_OCCURRED != 0);
    }

    #[test]
    fn test_abort_tracing() {
        let mut ws = WarmState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }
        ws.abort_tracing(42, true);

        let cell = ws.get_cell(42).unwrap();
        assert!(!cell.is_tracing());
        assert!(cell.flags & jc_flags::DONT_TRACE_HERE != 0);

        // Now it should report NotHot because DONT_TRACE_HERE is set
        match ws.maybe_compile(42) {
            HotResult::NotHot => {}
            _ => panic!("expected NotHot due to DONT_TRACE_HERE"),
        }
    }

    #[test]
    fn test_abort_tracing_allows_retry() {
        let mut ws = WarmState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }
        // Abort without DONT_TRACE_HERE
        ws.abort_tracing(42, false);

        // Counter was reset during start_tracing, but hash is still in the table.
        // Need to tick again to reach threshold. The hash is found now (not evicted),
        // so one tick to reach count=1, another to reach count=2 >= threshold=2.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing on retry"),
        }
    }

    #[test]
    fn test_different_green_keys() {
        let mut ws = WarmState::new(3);
        // Key 1: tick 1 (eviction), tick 2 (count=2 < 3)
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        // Key 2: tick 1 (eviction)
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        // Key 1: tick 3 -> threshold, starts tracing
        match ws.maybe_compile(1) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing for key 1"),
        }
        // Key 2 still not hot (only 2 total ticks: eviction + one more needed)
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
    }

    #[test]
    fn test_alloc_token_number() {
        let mut ws = WarmState::new(10);
        assert_eq!(ws.alloc_token_number(), 0);
        assert_eq!(ws.alloc_token_number(), 1);
        assert_eq!(ws.alloc_token_number(), 2);
    }

    #[test]
    fn test_set_threshold() {
        let mut ws = WarmState::new(100);
        assert_eq!(ws.threshold(), 100);
        ws.set_threshold(50);
        assert_eq!(ws.threshold(), 50);
    }

    #[test]
    fn test_get_compiled() {
        let mut ws = WarmState::new(1);
        assert!(ws.get_compiled(42).is_none());

        let token = LoopToken::new(0);
        ws.install_compiled(42, token);

        let compiled = ws.get_compiled(42);
        assert!(compiled.is_some());
        assert_eq!(compiled.unwrap().number, 0);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut ws = WarmState::new(3);
        let key = 0xDEAD;

        // Phase 1: Not hot (need 3 ticks to reach threshold=3)
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));

        // Phase 2: Start tracing (third tick reaches threshold)
        let recorder = match ws.maybe_compile(key) {
            HotResult::StartTracing(rec) => rec,
            _ => panic!("expected StartTracing"),
        };

        // Phase 3: Already tracing
        assert!(matches!(ws.maybe_compile(key), HotResult::AlreadyTracing));

        // Phase 4: Finish tracing, install compiled code
        ws.finish_tracing(key);
        let token_num = ws.alloc_token_number();
        let token = LoopToken::new(token_num);
        ws.install_compiled(key, token);

        // Phase 5: Run compiled
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));

        drop(recorder);
    }

    #[test]
    fn test_bridge_threshold_default() {
        let ws = WarmState::new(3);
        assert_eq!(ws.bridge_threshold(), 5);
    }

    #[test]
    fn test_bridge_threshold_custom() {
        let mut ws = WarmState::new(3);
        ws.set_bridge_threshold(10);
        assert_eq!(ws.bridge_threshold(), 10);
    }

    #[test]
    fn test_should_compile_bridge() {
        let ws = WarmState::new(3);
        assert!(!ws.should_compile_bridge(0));
        assert!(!ws.should_compile_bridge(4));
        assert!(ws.should_compile_bridge(5));
        assert!(ws.should_compile_bridge(100));
    }
}

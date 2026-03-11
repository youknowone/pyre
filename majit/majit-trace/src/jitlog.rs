/// Structured JIT profiling and statistics logging.
///
/// Tracks compilation events, guard failures, and timing data.
/// Activated via MAJIT_LOG=1 or MAJIT_STATS=1 environment variables.
/// Prints a summary report on drop.
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Whether JIT statistics collection is enabled.
/// Checks MAJIT_STATS=1 or MAJIT_LOG=1.
pub fn stats_enabled() -> bool {
    std::env::var("MAJIT_STATS")
        .or_else(|_| std::env::var("MAJIT_LOG"))
        .is_ok_and(|v| v == "1")
}

/// Per-trace compilation record.
#[derive(Debug, Clone)]
pub struct TraceRecord {
    /// Green key hash identifying this trace.
    pub green_key: u64,
    /// Number of operations before optimization.
    pub ops_before_opt: usize,
    /// Number of operations after optimization.
    pub ops_after_opt: usize,
    /// Time spent recording + optimization.
    pub opt_time: Duration,
    /// Time spent in backend compilation.
    pub compile_time: Duration,
}

/// Structured JIT profiling logger.
///
/// Collects statistics about trace compilation, guard failures,
/// and loop entries. Prints a summary on drop when enabled.
pub struct JitLog {
    /// Successfully compiled traces.
    compiled: Vec<TraceRecord>,
    /// Number of traces that were aborted.
    aborted: u64,
    /// Guard failure counts, keyed by guard index.
    guard_failures: HashMap<u32, u64>,
    /// Loop entry counts, keyed by green key hash.
    loop_entries: HashMap<u64, u64>,
    /// Whether to print the summary on drop.
    print_on_drop: bool,
}

impl JitLog {
    /// Create a new JitLog. If `print_on_drop` is true, a summary
    /// is printed to stderr when the JitLog is dropped.
    pub fn new(print_on_drop: bool) -> Self {
        JitLog {
            compiled: Vec::new(),
            aborted: 0,
            guard_failures: HashMap::new(),
            loop_entries: HashMap::new(),
            print_on_drop,
        }
    }

    /// Create a JitLog that is enabled based on environment variables.
    /// Returns `Some(JitLog)` if MAJIT_STATS=1 or MAJIT_LOG=1, else `None`.
    pub fn from_env() -> Option<Self> {
        if stats_enabled() {
            Some(Self::new(true))
        } else {
            None
        }
    }

    /// Record a successful trace compilation.
    pub fn log_compile(
        &mut self,
        green_key: u64,
        ops_before_opt: usize,
        ops_after_opt: usize,
        opt_time: Duration,
        compile_time: Duration,
    ) {
        self.compiled.push(TraceRecord {
            green_key,
            ops_before_opt,
            ops_after_opt,
            opt_time,
            compile_time,
        });
    }

    /// Record a trace abort.
    pub fn log_abort(&mut self) {
        self.aborted += 1;
    }

    /// Record a guard failure for the given guard index.
    pub fn log_guard_failure(&mut self, guard_index: u32) {
        *self.guard_failures.entry(guard_index).or_insert(0) += 1;
    }

    /// Record a loop entry for the given green key.
    pub fn log_loop_entry(&mut self, green_key: u64) {
        *self.loop_entries.entry(green_key).or_insert(0) += 1;
    }

    /// Number of successfully compiled traces.
    pub fn traces_compiled(&self) -> usize {
        self.compiled.len()
    }

    /// Number of aborted traces.
    pub fn traces_aborted(&self) -> u64 {
        self.aborted
    }

    /// Total guard failures across all guards.
    pub fn total_guard_failures(&self) -> u64 {
        self.guard_failures.values().sum()
    }

    /// Total operations recorded across all compiled traces (before optimization).
    pub fn total_ops_before(&self) -> usize {
        self.compiled.iter().map(|r| r.ops_before_opt).sum()
    }

    /// Total operations after optimization across all compiled traces.
    pub fn total_ops_after(&self) -> usize {
        self.compiled.iter().map(|r| r.ops_after_opt).sum()
    }

    /// Total compilation time across all traces.
    pub fn total_compile_time(&self) -> Duration {
        self.compiled.iter().map(|r| r.compile_time).sum()
    }

    /// Total optimization time across all traces.
    pub fn total_opt_time(&self) -> Duration {
        self.compiled.iter().map(|r| r.opt_time).sum()
    }

    /// Get the compiled trace records.
    pub fn compiled_traces(&self) -> &[TraceRecord] {
        &self.compiled
    }

    /// Get guard failure counts.
    pub fn guard_failure_counts(&self) -> &HashMap<u32, u64> {
        &self.guard_failures
    }

    /// Get loop entry counts.
    pub fn loop_entry_counts(&self) -> &HashMap<u64, u64> {
        &self.loop_entries
    }

    /// Format the summary report as a string.
    pub fn summary(&self) -> String {
        let total_before = self.total_ops_before();
        let total_after = self.total_ops_after();
        let reduction_pct = if total_before > 0 {
            ((total_before - total_after) as f64 / total_before as f64) * 100.0
        } else {
            0.0
        };
        let compile_ms = self.total_compile_time().as_secs_f64() * 1000.0;
        let opt_ms = self.total_opt_time().as_secs_f64() * 1000.0;

        format!(
            "\
=== JIT Statistics ===
Traces compiled: {}
Traces aborted: {}
Total ops recorded: {}
Total ops after opt: {} ({:.0}% reduction)
Guard failures: {}
Optimization time: {:.1}ms
Compilation time: {:.1}ms",
            self.traces_compiled(),
            self.traces_aborted(),
            total_before,
            total_after,
            reduction_pct,
            self.total_guard_failures(),
            opt_ms,
            compile_ms,
        )
    }
}

impl Drop for JitLog {
    fn drop(&mut self) {
        if self.print_on_drop
            && (self.traces_compiled() > 0 || self.traces_aborted() > 0)
        {
            eprintln!("{}", self.summary());
        }
    }
}

/// A lightweight timer for measuring JIT phases.
pub struct JitTimer {
    start: Instant,
}

impl JitTimer {
    /// Start a new timer.
    pub fn start() -> Self {
        JitTimer {
            start: Instant::now(),
        }
    }

    /// Return elapsed duration since the timer was started.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jitlog_empty() {
        let log = JitLog::new(false);
        assert_eq!(log.traces_compiled(), 0);
        assert_eq!(log.traces_aborted(), 0);
        assert_eq!(log.total_guard_failures(), 0);
        assert_eq!(log.total_ops_before(), 0);
        assert_eq!(log.total_ops_after(), 0);
    }

    #[test]
    fn test_jitlog_compile() {
        let mut log = JitLog::new(false);
        log.log_compile(42, 100, 60, Duration::from_millis(2), Duration::from_millis(3));
        assert_eq!(log.traces_compiled(), 1);
        assert_eq!(log.total_ops_before(), 100);
        assert_eq!(log.total_ops_after(), 60);
        assert_eq!(log.total_compile_time(), Duration::from_millis(3));
        assert_eq!(log.total_opt_time(), Duration::from_millis(2));
    }

    #[test]
    fn test_jitlog_abort() {
        let mut log = JitLog::new(false);
        log.log_abort();
        log.log_abort();
        assert_eq!(log.traces_aborted(), 2);
    }

    #[test]
    fn test_jitlog_guard_failures() {
        let mut log = JitLog::new(false);
        log.log_guard_failure(0);
        log.log_guard_failure(0);
        log.log_guard_failure(1);
        assert_eq!(log.total_guard_failures(), 3);
        assert_eq!(*log.guard_failure_counts().get(&0).unwrap(), 2);
        assert_eq!(*log.guard_failure_counts().get(&1).unwrap(), 1);
    }

    #[test]
    fn test_jitlog_loop_entries() {
        let mut log = JitLog::new(false);
        log.log_loop_entry(100);
        log.log_loop_entry(100);
        log.log_loop_entry(200);
        assert_eq!(*log.loop_entry_counts().get(&100).unwrap(), 2);
        assert_eq!(*log.loop_entry_counts().get(&200).unwrap(), 1);
    }

    #[test]
    fn test_jitlog_summary_format() {
        let mut log = JitLog::new(false);
        log.log_compile(1, 50, 30, Duration::from_millis(1), Duration::from_millis(2));
        log.log_compile(2, 100, 55, Duration::from_millis(1), Duration::from_millis(3));
        log.log_abort();
        log.log_guard_failure(0);
        log.log_guard_failure(1);

        let summary = log.summary();
        assert!(summary.contains("Traces compiled: 2"));
        assert!(summary.contains("Traces aborted: 1"));
        assert!(summary.contains("Total ops recorded: 150"));
        assert!(summary.contains("Total ops after opt: 85"));
        assert!(summary.contains("Guard failures: 2"));
        assert!(summary.contains("Compilation time:"));
    }

    #[test]
    fn test_jitlog_reduction_percentage() {
        let mut log = JitLog::new(false);
        log.log_compile(1, 200, 100, Duration::ZERO, Duration::ZERO);
        let summary = log.summary();
        assert!(summary.contains("50% reduction"));
    }

    #[test]
    fn test_jitlog_zero_ops_no_panic() {
        let log = JitLog::new(false);
        let summary = log.summary();
        assert!(summary.contains("0% reduction"));
    }

    #[test]
    fn test_jitlog_multiple_compiles_accumulate() {
        let mut log = JitLog::new(false);
        for i in 0..5 {
            log.log_compile(
                i,
                10 * (i as usize + 1),
                5 * (i as usize + 1),
                Duration::from_micros(100 * (i + 1)),
                Duration::from_micros(200 * (i + 1)),
            );
        }
        assert_eq!(log.traces_compiled(), 5);
        // 10+20+30+40+50 = 150
        assert_eq!(log.total_ops_before(), 150);
        // 5+10+15+20+25 = 75
        assert_eq!(log.total_ops_after(), 75);
    }

    #[test]
    fn test_jit_timer() {
        let timer = JitTimer::start();
        // Just verify it doesn't panic and returns a non-negative duration
        let elapsed = timer.elapsed();
        assert!(elapsed.as_nanos() < 1_000_000_000); // less than 1 second
    }

    #[test]
    fn test_compiled_traces_accessor() {
        let mut log = JitLog::new(false);
        log.log_compile(42, 100, 60, Duration::from_millis(1), Duration::from_millis(2));
        let traces = log.compiled_traces();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].green_key, 42);
        assert_eq!(traces[0].ops_before_opt, 100);
        assert_eq!(traces[0].ops_after_opt, 60);
    }
}

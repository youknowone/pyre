// Runtime utilities for interpreters using the majit JIT framework.
//
// Provides JIT hints, driver configuration, per-thread state,
// error types, and statistics.
//
// Reference: rpython/rlib/jit.py

/// Hint to the JIT that this value should be treated as a compile-time constant.
///
/// In RPython this is `jit.promote(x)`. During tracing, the tracer records
/// a `GUARD_VALUE` that specializes on the current concrete value.
/// In normal (non-tracing) mode, this is a no-op that returns the value unchanged.
#[inline(always)]
pub fn hint_promote<T: Copy>(val: T) -> T {
    val
}

/// Hint that a string value should be promoted (specialized).
///
/// Like `hint_promote` but for string-typed values that need special
/// handling in the optimizer (string interning, hash caching, etc.).
#[inline(always)]
pub fn hint_promote_string<T: Copy>(val: T) -> T {
    val
}

/// Hint that the following call's result only depends on its arguments.
/// The JIT can cache the result (CSE / constant folding).
///
/// In RPython this is `@jit.elidable`. Functions marked elidable are
/// recorded as `CALL_PURE_*` in the trace, enabling the optimizer to
/// eliminate calls where all arguments are known constants.
#[inline(always)]
pub fn hint_elidable() {}

/// Hint to not trace into the following function.
///
/// In RPython this is `@jit.dont_look_inside`. The function will be
/// called as a residual (opaque) call during tracing.
#[inline(always)]
pub fn hint_dont_look_inside() {}

/// Hint that a loop in the following function should be unrolled.
///
/// In RPython this is `@jit.unroll_safe`. Without this hint, loops
/// inside traced functions would cause the tracer to abort.
#[inline(always)]
pub fn hint_unroll_safe() {}

/// Hint that the following function is loop-invariant.
///
/// In RPython this is `@jit.loop_invariant`. The result is cached
/// for the duration of one loop iteration.
#[inline(always)]
pub fn hint_loop_invariant() {}

/// Hint that a conditional branch is expected to take the given path.
///
/// In RPython this is `jit.conditional_call()`. Helps the JIT generate
/// better guard placement.
#[inline(always)]
pub fn hint_conditional_call() {}

/// Hint that a virtual reference may be used.
///
/// In RPython this is `jit.virtual_ref(obj)`. Creates a virtual reference
/// that the optimizer can keep virtual (avoiding allocation) as long as
/// the reference doesn't escape the trace.
#[inline(always)]
pub fn hint_virtual_ref<T>(val: T) -> T {
    val
}

/// Hint that a virtual reference is no longer needed.
///
/// In RPython this is `jit.virtual_ref_finish(vref, obj)`.
#[inline(always)]
pub fn hint_virtual_ref_finish() {}

/// Hint: access the virtualizable directly as a struct, bypassing JIT tracking.
/// RPython equivalent: `jit.hint(frame, access_directly=True)`
#[inline(always)]
pub fn hint_access_directly<T>(x: T) -> T {
    x
}

/// Hint: this virtualizable was freshly allocated, so direct access is safe.
/// RPython equivalent: `jit.hint(frame, access_directly=True, fresh_virtualizable=True)`
#[inline(always)]
pub fn hint_fresh_virtualizable<T>(x: T) -> T {
    x
}

/// Hint: force the virtualizable now (performance hint for generators etc).
/// RPython equivalent: `jit.hint(frame, force_virtualizable=True)`
#[inline(always)]
pub fn hint_force_virtualizable<T>(x: T) -> T {
    x
}

/// Hint that a value is expected to be a compile-time constant.
///
/// Unlike `hint_promote`, this doesn't generate a guard — it's a
/// lighter hint used when the value is expected to already be constant
/// (e.g., after a previous promote).
#[inline(always)]
pub fn hint_isconstant<T: Copy>(val: T) -> bool {
    // In non-JIT mode, nothing is a JIT constant
    false
}

/// Hint that a value should be treated as an "is virtual" check.
///
/// In RPython this is `jit.isvirtual(x)`. Returns true if the JIT
/// has virtualized the given object (only meaningful during tracing).
#[inline(always)]
pub fn hint_isvirtual<T>(_val: &T) -> bool {
    false
}

/// Tell the JIT that we're at a merge point (loop header).
/// `green_key` identifies the position in the interpreter.
///
/// Prefer the [`jit_merge_point!`] macro for multi-variable drivers.
#[inline(always)]
pub fn jit_merge_point_single(_green_key: u64) {}

/// Tell the JIT that we can close a loop here.
///
/// Prefer the [`can_enter_jit!`] macro for multi-variable drivers.
#[inline(always)]
pub fn can_enter_jit_single(_green_key: u64) {}

/// Mark a JIT merge point (loop header).
///
/// This is where the JIT checks whether the current loop is hot and
/// potentially enters compiled code. Green variables form the lookup key;
/// red variables carry mutable state.
///
/// Usage:
/// ```ignore
/// jit_merge_point!(DriverStruct, green1, green2; red1, red2);
/// ```
///
/// In non-JIT mode this is a no-op that uses `black_box` to keep the
/// bindings visible to the tracer.
#[macro_export]
macro_rules! jit_merge_point {
    ($driver:ty, $($green:expr),* ; $($red:expr),*) => {
        $(let _ = ::core::hint::black_box(&$green);)*
        $(let _ = ::core::hint::black_box(&$red);)*
    };
}

/// Mark a potential JIT entry point (loop back-edge).
///
/// Placed at the back-edge of a loop so the JIT knows this is a valid
/// point to enter already-compiled machine code.
///
/// Usage:
/// ```ignore
/// can_enter_jit!(DriverStruct, green1, green2; red1, red2);
/// ```
///
/// In non-JIT mode this is a no-op.
#[macro_export]
macro_rules! can_enter_jit {
    ($driver:ty, $($green:expr),* ; $($red:expr),*) => {
        $(let _ = ::core::hint::black_box(&$green);)*
        $(let _ = ::core::hint::black_box(&$red);)*
    };
}

/// Configuration for a JIT driver instance.
pub struct JitDriverConfig {
    /// Name of this JIT driver (for debugging/logging).
    pub name: &'static str,
    /// Threshold for tracing (number of loop iterations before tracing starts).
    pub threshold: u32,
    /// Maximum trace length.
    pub trace_limit: u32,
    /// Whether to enable the optimizer.
    pub enable_optimizer: bool,
}

impl Default for JitDriverConfig {
    fn default() -> Self {
        JitDriverConfig {
            name: "default",
            threshold: 1039, // prime number, like PyPy
            trace_limit: 6000,
            enable_optimizer: true,
        }
    }
}

/// Per-thread JIT state, holding the warm state and compiled code cache.
pub struct JitState {
    pub config: JitDriverConfig,
    // The warm state and backend would be stored here
    // but they live in their respective crates.
}

impl JitState {
    /// Create a new JIT state with the given configuration.
    pub fn new(config: JitDriverConfig) -> Self {
        JitState { config }
    }

    /// Create a new JIT state with default configuration.
    pub fn with_defaults() -> Self {
        JitState {
            config: JitDriverConfig::default(),
        }
    }
}

/// Errors that can occur during JIT compilation.
#[derive(Debug)]
pub enum JitError {
    /// The trace was too long.
    TraceTooLong,
    /// The trace contained an invalid operation.
    InvalidTrace(String),
    /// Compilation failed in the backend.
    CompilationFailed(String),
    /// The loop was invalidated.
    LoopInvalidated,
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::TraceTooLong => write!(f, "trace too long"),
            JitError::InvalidTrace(msg) => write!(f, "invalid trace: {msg}"),
            JitError::CompilationFailed(msg) => write!(f, "compilation failed: {msg}"),
            JitError::LoopInvalidated => write!(f, "loop invalidated"),
        }
    }
}

impl std::error::Error for JitError {}

/// JIT compilation statistics.
#[derive(Debug, Default, Clone)]
pub struct JitStats {
    pub traces_recorded: u64,
    pub traces_compiled: u64,
    pub traces_aborted: u64,
    pub guards_failed: u64,
    pub bridges_compiled: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = JitDriverConfig::default();
        assert_eq!(config.name, "default");
        assert_eq!(config.threshold, 1039);
        assert_eq!(config.trace_limit, 6000);
        assert!(config.enable_optimizer);
    }

    #[test]
    fn test_stats_default() {
        let stats = JitStats::default();
        assert_eq!(stats.traces_recorded, 0);
        assert_eq!(stats.traces_compiled, 0);
        assert_eq!(stats.traces_aborted, 0);
        assert_eq!(stats.guards_failed, 0);
        assert_eq!(stats.bridges_compiled, 0);
    }

    #[test]
    fn test_hint_promote_identity() {
        assert_eq!(hint_promote(42i64), 42);
        assert_eq!(hint_promote(3.14f64), 3.14);
        assert_eq!(hint_promote(true), true);
    }

    #[test]
    fn test_jit_state_with_defaults() {
        let state = JitState::with_defaults();
        assert_eq!(state.config.threshold, 1039);
    }

    #[test]
    fn test_jit_state_custom_config() {
        let config = JitDriverConfig {
            name: "test",
            threshold: 100,
            trace_limit: 500,
            enable_optimizer: false,
        };
        let state = JitState::new(config);
        assert_eq!(state.config.name, "test");
        assert_eq!(state.config.threshold, 100);
        assert_eq!(state.config.trace_limit, 500);
        assert!(!state.config.enable_optimizer);
    }

    #[test]
    fn test_jit_error_display() {
        assert_eq!(JitError::TraceTooLong.to_string(), "trace too long");
        assert_eq!(
            JitError::InvalidTrace("bad op".into()).to_string(),
            "invalid trace: bad op"
        );
        assert_eq!(
            JitError::CompilationFailed("oom".into()).to_string(),
            "compilation failed: oom"
        );
        assert_eq!(JitError::LoopInvalidated.to_string(), "loop invalidated");
    }

    #[test]
    fn test_stats_clone() {
        let mut stats = JitStats::default();
        stats.traces_recorded = 10;
        stats.traces_compiled = 5;
        let cloned = stats.clone();
        assert_eq!(cloned.traces_recorded, 10);
        assert_eq!(cloned.traces_compiled, 5);
    }
}

/// Tracing engine for the JIT.
///
/// Provides:
/// - Hot counter for detecting frequently-executed loops
/// - Trace recorder for building IR from interpreter execution
/// - Warm state management (interpreter → tracing → compiled)
///
/// Reference: rpython/jit/metainterp/pyjitpl.py, warmstate.py, counter.py
pub mod counter;
pub mod heapcache;
pub mod history;
pub mod logger;
pub mod opencoder;
pub mod recorder;
pub mod warmstate;

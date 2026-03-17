/// Tracing engine for the JIT.
///
/// Provides:
/// - Hot counter for detecting frequently-executed loops
/// - Trace recorder for building IR from interpreter execution
/// - Warm state management (interpreter → tracing → compiled)
///
/// Reference: rpython/jit/metainterp/pyjitpl.py, warmstate.py, counter.py
pub mod counter;
pub mod encoding;
pub mod heapcache;
pub mod jitlog;
pub mod recorder;
pub mod trace;
pub mod warmenterstate;
/// Backward-compat alias — examples still import `warmstate::*`.
pub use warmenterstate as warmstate;

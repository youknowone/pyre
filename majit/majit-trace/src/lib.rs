/// Tracing engine for the JIT.
///
/// Provides:
/// - Hot counter for detecting frequently-executed loops
/// - Trace recorder for building IR from interpreter execution
/// - Warm state management (interpreter → tracing → compiled)
///
/// Reference: rpython/jit/metainterp/pyjitpl.py, warmstate.py, counter.py
pub mod counter;
#[expect(
    clippy::too_many_arguments,
    reason = "heap-cache transfer functions retain RPython's explicit operation operands and descriptor state so the port remains structurally auditable against heapcache.py"
)]
pub mod heapcache;
pub mod logger;

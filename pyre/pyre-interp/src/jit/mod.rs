pub mod descr;
pub mod frame_layout;
pub mod helpers;
pub mod state;
pub mod trace;

/// Auto-generated trace functions from majit-analyze.
/// These replace the manual tracing code in state.rs.
#[allow(dead_code, unused_imports)]
pub mod generated {
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}

pub mod codewriter;
pub mod descr;
pub mod executor;
pub mod frame_layout;
pub mod metainterp;
pub mod helpers;
pub mod state;
pub mod trace;
pub mod virtualizable_spec;

/// Auto-generated trace functions from majit-analyze.
#[allow(dead_code, unused_imports)]
pub mod generated {
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}

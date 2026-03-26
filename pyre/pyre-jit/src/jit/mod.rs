pub mod codewriter;
pub use pyre_jit_trace::descr;
pub mod executor;
pub use pyre_jit_trace::frame_layout;
pub use pyre_jit_trace::helpers;
pub mod metainterp;
pub mod state;
pub mod trace;
pub use pyre_jit_trace::virtualizable_spec;

/// Auto-generated trace functions from majit-analyze.
#[allow(dead_code, unused_imports)]
pub mod generated {
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}

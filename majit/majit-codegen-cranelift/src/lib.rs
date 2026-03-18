/// Cranelift-based JIT code generation backend for majit.
///
/// This crate implements the `majit_codegen::Backend` trait using Cranelift
/// to translate majit IR traces into native machine code.
pub mod compiler;
pub mod guard;

pub use compiler::{
    CallAssemblerDescr, CraneliftBackend, execute_call_assembler_direct, force_token_to_dead_frame,
    get_float_from_deadframe, get_int_from_deadframe, get_latest_descr_from_deadframe,
    get_ref_from_deadframe, get_savedata_ref_from_deadframe, grab_savedata_ref_from_deadframe,
    jit_exc_is_pending, jit_exc_raise, register_call_assembler_bridge,
    register_call_assembler_force, set_gil_hooks, set_savedata_ref_on_deadframe,
};

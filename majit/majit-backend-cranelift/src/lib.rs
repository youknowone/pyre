// These interfaces preserve the shared backend's explicit lowering inputs and
// recovery-layout ownership. Interior-mutability accessors are synchronized by
// the backend's compile lock; changing them to `&mut self` would break the
// Backend trait's shared-reference contract.
#![allow(
    clippy::approx_constant,
    clippy::empty_line_after_doc_comments,
    clippy::inconsistent_digit_grouping,
    clippy::manual_checked_ops,
    clippy::mut_from_ref,
    clippy::needless_range_loop,
    clippy::redundant_locals,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unnecessary_mut_passed,
    clippy::useless_vec
)]

/// Cranelift-based JIT code generation backend for majit.
///
/// This crate implements the `majit_backend::Backend` trait using Cranelift
/// to translate majit IR traces into native machine code.
pub mod compiler;
pub mod guard;

pub use compiler::{
    CallAssemblerDescr, CraneliftBackend, FrameRestore, JitFrameLayoutInfo,
    force_token_to_dead_frame, get_float_from_deadframe, get_int_from_deadframe,
    get_latest_descr_from_deadframe, get_ref_from_deadframe, get_savedata_ref_from_deadframe,
    install_gc_standalone, jit_exc_class_raw, jit_exc_clear, jit_exc_is_pending, jit_exc_raise,
    jit_exc_value_peek, jit_exc_value_raw, register_call_assembler_blackhole,
    register_call_assembler_bridge, register_call_assembler_force,
    register_call_assembler_unbox_int, register_jitframe_layout, register_materialize_str_call,
    register_materialize_str_plain, register_prologue_probe_addr, register_recovery_layout,
    register_resumedata_deopt, register_stack_check_addresses, set_gil_hooks,
    set_jitframe_gc_type_id, set_savedata_ref_on_deadframe, take_pending_frame_restore,
};

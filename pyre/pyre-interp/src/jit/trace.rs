//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! This stays as the stable entry surface used by the interpreter loop while
//! trace recording executes directly through `TraceFrameState`.

use majit_meta::{TraceAction, TraceCtx};
use pyre_bytecode::CodeObject;

use crate::jit::state::{PyreSym, TraceFrameState};

pub fn trace_bytecode(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    code: &CodeObject,
    pc: usize,
    concrete_frame: usize,
) -> TraceAction {
    let mut frame_state = TraceFrameState::from_concrete(ctx, sym.frame, concrete_frame, pc + 1);
    frame_state.trace_code_step(code, pc)
}

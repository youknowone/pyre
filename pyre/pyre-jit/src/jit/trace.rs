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
    let mut frame_state = TraceFrameState::from_sym(ctx, sym, concrete_frame, pc + 1);

    // PyPy interp_jit.py:89 — promote(valuestackdepth).
    // hint(self.valuestackdepth, promote=True) forces valuestackdepth
    // to be a compile-time constant, enabling constant-folding of
    // stack offset calculations.
    frame_state.promote_valuestackdepth(concrete_frame);

    let action = frame_state.trace_code_step(code, pc);

    // RPython pyjitpl.py reached_loop_header(): when a back-edge closes
    // the loop, the compiled trace must be registered under the back-edge's
    // green key, not the function-entry key.  Without this, func-entry
    // tracing stores the loop under PC=0, causing infinite re-entry.
    if matches!(
        action,
        TraceAction::CloseLoop | TraceAction::CloseLoopWithArgs { .. }
    ) {
        let back_edge_key = crate::eval::make_green_key(
            code as *const CodeObject,
            pc,
        );
        ctx.set_green_key(back_edge_key);
    }

    action
}

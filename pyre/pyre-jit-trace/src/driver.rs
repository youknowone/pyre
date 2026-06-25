//! JIT driver access from pyre-jit-trace.
//!
//! make_green_key is a pure function. driver_pair() is accessed through
//! callbacks since the JIT_DRIVER TLS lives in pyre-jit/eval.rs.

use crate::callbacks;
use crate::state::PyreJitState;

/// RPython green_key = pypyjit greens `[next_instr, is_being_profiled,
/// pycode]` (interp_jit.py:67-70). pyre's portal greens are always exactly
/// `(PyCode*, next_instr)`, and the JIT path never runs under a profiler,
/// so `is_being_profiled` folds to 0 — the trace-side call sites have no
/// frame to read it from. The returned u64 is the full
/// `JitCell.get_uhash` over the typed green tuple (warmstate.py:584-593),
/// so this legacy hash flow and the typed marker-path lookup
/// (`lookup_chain_with_key`) agree on the same cell.
#[inline(always)]
pub fn make_green_key(code_ptr: *const (), pc: usize) -> u64 {
    majit_ir::pypyjit_greenkey_uhash(pc, false, code_ptr as u64)
}

/// Type alias for the JIT driver pair. Must match pyre-jit/eval.rs JitDriverPair.
pub type JitDriverPair = (
    majit_metainterp::JitDriver<PyreJitState>,
    std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>,
);

/// Get the JIT driver pair through callbacks.
#[inline]
pub fn driver_pair() -> &'static mut JitDriverPair {
    let ptr = (callbacks::get().driver_pair)();
    unsafe { &mut *(ptr as *mut JitDriverPair) }
}

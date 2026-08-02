//! Global Interpreter Lock — `pypy/module/thread/gil.py`.
//!
//! The lock itself lives in `majit_gc::rgil` (the `rpython/translator/c/src`
//! side). What this module adds is the half that belongs to the object space:
//! a thread holds the GIL for as long as it runs pyre code, so without a
//! periodic hand-off a compute-bound thread would never let another one run.
//! `GILReleaseAction` is that hand-off — an action registered on the ticker
//! which yields the GIL every `sys.getcheckinterval()` bytecodes.

use crate::executioncontext::{
    AsyncAction, AsyncActionOps, ExecutionContext, PeriodicAsyncAction, PeriodicAsyncActionOps,
};
use crate::pyframe::PyFrame;
use pyre_object::PyObjectRef;

/// gil.py:44-50 `GILReleaseAction` — "an action called every
/// `sys.checkinterval` bytecodes. It releases the GIL to give some other
/// thread a chance to run."
pub struct GilReleaseAction {
    base: PeriodicAsyncAction,
}

impl GilReleaseAction {
    fn new(space: PyObjectRef) -> Box<Self> {
        Box::new(Self {
            base: *PeriodicAsyncAction::new(space),
        })
    }
}

impl AsyncActionOps for GilReleaseAction {
    /// gil.py:48-50 `perform`: `rgil.yield_thread()`.
    fn perform(
        &mut self,
        _ec: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<(), crate::PyError> {
        majit_gc::rgil::yield_thread();
        Ok(())
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base.base
    }
}

impl PeriodicAsyncActionOps for GilReleaseAction {}

/// gil.py:20-23 `GILThreadLocals.initialize` — "add the GIL-releasing callback
/// as an action on the space".
///
/// `use_bytecode_counter=True` is what puts it at the end of the periodic list
/// (executioncontext.py:503-504: "hack to put the release-the-GIL one at the
/// end of the list"), behind the signal check. Idempotent; the action is leaked
/// deliberately because the actionflag holds a pointer into it for the whole
/// run, exactly as `install_signal_handling` does.
pub fn initialize(ec: &mut ExecutionContext) {
    if ec.gil_release_action.is_some() {
        return;
    }
    let action: &'static mut GilReleaseAction = Box::leak(GilReleaseAction::new(ec.space));
    let async_ptr: *mut dyn AsyncActionOps = &mut *action;
    action.register_periodic_action(&mut ec.actionflag, true);
    ec.gil_release_action = Some(async_ptr);
}

/// gil.py:25-34 `GILThreadLocals.setup_threads` — "enable threads in the object
/// space, if they haven't already been". Returns whether this call is the one
/// that set them up.
pub fn setup_threads(ec: &mut ExecutionContext) -> bool {
    let first = ec.gil_release_action.is_none();
    majit_gc::rgil::allocate();
    initialize(ec);
    first
}

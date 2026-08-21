//! App-level GC hooks — PyPy: `pypy/module/gc/hook.py`.

use super::{new_collect_stats, new_collect_step_stats_full, new_minor_stats};
use crate::executioncontext::{
    ActionFlagOps, AsyncAction, AsyncActionControl, AsyncActionOps, ExecutionContext,
};
use crate::pyframe::PyFrame;
use pyre_object::*;
use std::sync::OnceLock;

struct GcMinorHookAction {
    base: AsyncAction,
    depth: usize,
    count: i64,
    duration: f64,
    duration_min: f64,
    duration_max: f64,
    total_memory_used: usize,
    pinned_objects: usize,
}

impl GcMinorHookAction {
    fn new() -> Self {
        Self {
            base: AsyncAction::default(),
            depth: 0,
            count: 0,
            duration: 0.0,
            duration_min: f64::INFINITY,
            duration_max: 0.0,
            total_memory_used: 0,
            pinned_objects: 0,
        }
    }

    fn reset(&mut self) {
        self.count = 0;
        self.duration = 0.0;
        self.duration_min = f64::INFINITY;
        self.duration_max = 0.0;
    }

    fn do_perform(&mut self) -> Result<(), crate::PyError> {
        // `self` is a field of this very allocation, so read the callback
        // through the pointer rather than borrowing the whole singleton.
        let Some(hooks) = app_hooks_ptr() else {
            return Ok(());
        };
        let count = self.count;
        let duration = self.duration;
        let duration_min = self.duration_min;
        let duration_max = self.duration_max;
        let total_memory_used = self.total_memory_used;
        let pinned_objects = self.pinned_objects;
        self.reset();

        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(unsafe { (*hooks).w_on_gc_minor });
        let callable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let stats = new_minor_stats(
            count,
            duration,
            duration_min,
            duration_max,
            total_memory_used,
            pinned_objects,
        )?;
        pyre_object::gc_roots::pin_root(stats);
        let stats_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &[pyre_object::gc_roots::shadow_stack_get(stats_slot)],
        )?;
        Ok(())
    }
}

impl AsyncActionOps for GcMinorHookAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, crate::PyError> {
        if self.depth != 0 {
            return Ok(AsyncActionControl::Continue);
        }
        self.depth += 1;
        let result = self.do_perform();
        self.depth -= 1;
        result.map(|()| AsyncActionControl::Continue)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

struct GcCollectStepHookAction {
    base: AsyncAction,
    depth: usize,
    count: i64,
    duration: f64,
    duration_min: f64,
    duration_max: f64,
    oldstate: u8,
    newstate: u8,
}

impl GcCollectStepHookAction {
    fn new() -> Self {
        Self {
            base: AsyncAction::default(),
            depth: 0,
            count: 0,
            duration: 0.0,
            duration_min: f64::INFINITY,
            duration_max: 0.0,
            oldstate: 0,
            newstate: 0,
        }
    }

    fn reset(&mut self) {
        self.count = 0;
        self.duration = 0.0;
        self.duration_min = f64::INFINITY;
        self.duration_max = 0.0;
    }

    fn do_perform(&mut self) -> Result<(), crate::PyError> {
        // `self` is a field of this very allocation, so read the callback
        // through the pointer rather than borrowing the whole singleton.
        let Some(hooks) = app_hooks_ptr() else {
            return Ok(());
        };
        let count = self.count;
        let duration = self.duration;
        let duration_min = self.duration_min;
        let duration_max = self.duration_max;
        let oldstate = self.oldstate;
        let newstate = self.newstate;
        self.reset();

        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(unsafe { (*hooks).w_on_gc_collect_step });
        let callable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let stats = new_collect_step_stats_full(
            count,
            duration,
            duration_min,
            duration_max,
            oldstate,
            newstate,
            super::is_done_states(oldstate, newstate),
        )?;
        pyre_object::gc_roots::pin_root(stats);
        let stats_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &[pyre_object::gc_roots::shadow_stack_get(stats_slot)],
        )?;
        Ok(())
    }
}

impl AsyncActionOps for GcCollectStepHookAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, crate::PyError> {
        if self.depth != 0 {
            return Ok(AsyncActionControl::Continue);
        }
        self.depth += 1;
        let result = self.do_perform();
        self.depth -= 1;
        result.map(|()| AsyncActionControl::Continue)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

struct GcCollectHookAction {
    base: AsyncAction,
    depth: usize,
    count: i64,
    num_major_collects: usize,
    arenas_count_before: usize,
    arenas_count_after: usize,
    arenas_bytes: usize,
    rawmalloc_bytes_before: usize,
    rawmalloc_bytes_after: usize,
    pinned_objects: usize,
}

impl GcCollectHookAction {
    fn new() -> Self {
        Self {
            base: AsyncAction::default(),
            depth: 0,
            count: 0,
            num_major_collects: 0,
            arenas_count_before: 0,
            arenas_count_after: 0,
            arenas_bytes: 0,
            rawmalloc_bytes_before: 0,
            rawmalloc_bytes_after: 0,
            pinned_objects: 0,
        }
    }

    fn do_perform(&mut self) -> Result<(), crate::PyError> {
        // `self` is a field of this very allocation, so read the callback
        // through the pointer rather than borrowing the whole singleton.
        let Some(hooks) = app_hooks_ptr() else {
            return Ok(());
        };
        let count = self.count;
        let num_major_collects = self.num_major_collects;
        let arenas_count_before = self.arenas_count_before;
        let arenas_count_after = self.arenas_count_after;
        let arenas_bytes = self.arenas_bytes;
        let rawmalloc_bytes_before = self.rawmalloc_bytes_before;
        let rawmalloc_bytes_after = self.rawmalloc_bytes_after;
        let pinned_objects = self.pinned_objects;
        self.count = 0;

        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(unsafe { (*hooks).w_on_gc_collect });
        let callable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let stats = new_collect_stats(
            count,
            num_major_collects,
            arenas_count_before,
            arenas_count_after,
            arenas_bytes,
            rawmalloc_bytes_before,
            rawmalloc_bytes_after,
            pinned_objects,
        )?;
        pyre_object::gc_roots::pin_root(stats);
        let stats_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &[pyre_object::gc_roots::shadow_stack_get(stats_slot)],
        )?;
        Ok(())
    }
}

impl AsyncActionOps for GcCollectHookAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, crate::PyError> {
        if self.depth != 0 {
            return Ok(AsyncActionControl::Continue);
        }
        self.depth += 1;
        let result = self.do_perform();
        self.depth -= 1;
        result.map(|()| AsyncActionControl::Continue)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

/// `hook.py W_AppLevelHooks`. The callback references live directly
/// on the singleton owner, so the generated type tracer forwards them; the
/// three action objects are embedded exactly as its `gc_minor`,
/// `gc_collect_step`, and `gc_collect` attributes are upstream.
#[crate::pyre_class("GcHooks")]
pub struct W_AppLevelHooks {
    pub w_on_gc_minor: PyObjectRef,
    pub w_on_gc_collect_step: PyObjectRef,
    pub w_on_gc_collect: PyObjectRef,
    gc_minor_enabled: bool,
    gc_collect_step_enabled: bool,
    gc_collect_enabled: bool,
    gc_minor: GcMinorHookAction,
    gc_collect_step: GcCollectStepHookAction,
    gc_collect: GcCollectHookAction,
}

impl W_AppLevelHooks {
    fn write_barrier(&mut self) {
        pyre_object::gc_hook::try_gc_write_barrier_managed(self as *mut Self as *mut u8);
    }
}

#[crate::pyre_methods]
impl W_AppLevelHooks {
    #[getter]
    fn on_gc_minor(&self) -> PyObjectRef {
        self.w_on_gc_minor
    }

    #[setter]
    fn set_on_gc_minor(&mut self, w_obj: PyObjectRef) {
        self.gc_minor_enabled = !unsafe { is_none(w_obj) };
        self.w_on_gc_minor = w_obj;
        self.write_barrier();
    }

    #[getter]
    fn on_gc_collect_step(&self) -> PyObjectRef {
        self.w_on_gc_collect_step
    }

    #[setter]
    fn set_on_gc_collect_step(&mut self, w_obj: PyObjectRef) {
        self.gc_collect_step_enabled = !unsafe { is_none(w_obj) };
        self.w_on_gc_collect_step = w_obj;
        self.write_barrier();
    }

    #[getter]
    fn on_gc_collect(&self) -> PyObjectRef {
        self.w_on_gc_collect
    }

    #[setter]
    fn set_on_gc_collect(&mut self, w_obj: PyObjectRef) {
        self.gc_collect_enabled = !unsafe { is_none(w_obj) };
        self.w_on_gc_collect = w_obj;
        self.write_barrier();
    }

    fn set(&mut self, w_obj: PyObjectRef) -> Result<(), crate::PyError> {
        // hook.py:100-107 — fetch all three first, so a missing later
        // attribute leaves the existing hook set untouched.
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_a = crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            "on_gc_minor",
        )?;
        pyre_object::gc_roots::pin_root(w_a);
        let a_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_b = crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            "on_gc_collect_step",
        )?;
        pyre_object::gc_roots::pin_root(w_b);
        let b_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_c = crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            "on_gc_collect",
        )?;
        pyre_object::gc_roots::pin_root(w_c);
        let c_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        self.set_on_gc_minor(pyre_object::gc_roots::shadow_stack_get(a_slot));
        self.set_on_gc_collect_step(pyre_object::gc_roots::shadow_stack_get(b_slot));
        self.set_on_gc_collect(pyre_object::gc_roots::shadow_stack_get(c_slot));
        Ok(())
    }

    fn reset(&mut self) {
        self.set_on_gc_minor(w_none());
        self.set_on_gc_collect_step(w_none());
        self.set_on_gc_collect(w_none());
    }
}

static HOOKS_OBJECT: OnceLock<usize> = OnceLock::new();

/// Root the `space.fromcache(W_AppLevelHooks)` singleton independently of
/// the `gc` module dictionary.  Upstream ownership is the process-wide object
/// space, so deleting/rebinding `gc.hooks` must not let the action owner (or
/// its callback fields) be swept while the actionflag still stores pointers
/// into it. `allocate_stable` makes relocation impossible; the visitor marks
/// the owner and its registered type trace reaches the three callbacks.
pub fn walk_hook_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let Some(&addr) = HOOKS_OBJECT.get() else {
        return;
    };
    let mut root = majit_ir::GcRef(addr);
    visitor(&mut root);
    debug_assert_eq!(root.0, addr, "allocate_stable GcHooks moved");
}

/// The singleton as a raw pointer, without borrowing it.
///
/// The three hook actions are *fields* of `W_AppLevelHooks`, so every
/// `AsyncActionOps::perform` already holds `&mut` over part of this
/// allocation. `W_AppLevelHooks::from_obj` hands out `&'static mut Self`,
/// which would overlap that borrow — and so would a shared `&`. Callers read
/// and write through this pointer instead, narrowing any reference they do
/// form to the single field they touch. Holding it across an allocating call
/// is safe because `initialize` uses `allocate_stable`, which is what
/// `walk_hook_roots`'s `debug_assert_eq!` above pins down.
fn app_hooks_ptr() -> Option<*mut W_AppLevelHooks> {
    let &addr = HOOKS_OBJECT.get()?;
    let obj = addr as PyObjectRef;
    unsafe { pyre_object::py_type_check(obj, &APPLEVELHOOKS_TYPE) }
        .then_some(obj as *mut W_AppLevelHooks)
}

/// Create the space-owned singleton and bind its three actions to the shared
/// `space.actionflag`. The main ExecutionContext calls this during bootstrap;
/// worker ECs carry [`crate::executioncontext::SpaceActionFlag`] references to
/// that same flag, so a collection performed by a worker fires and dispatches
/// the same action indexes there, matching PyPy's process-owned object space.
pub fn initialize(
    space: PyObjectRef,
    actionflag: &mut (dyn ActionFlagOps + 'static),
) -> PyObjectRef {
    *HOOKS_OBJECT.get_or_init(|| {
        let _ = type_object();
        let none = w_none();
        let obj = W_AppLevelHooks::allocate_stable(W_AppLevelHooks {
            ob: PyObject::default(),
            w_on_gc_minor: none,
            w_on_gc_collect_step: none,
            w_on_gc_collect: none,
            gc_minor_enabled: false,
            gc_collect_step_enabled: false,
            gc_collect_enabled: false,
            gc_minor: GcMinorHookAction::new(),
            gc_collect_step: GcCollectStepHookAction::new(),
            gc_collect: GcCollectHookAction::new(),
        });

        let hooks = W_AppLevelHooks::from_obj(obj).expect("fresh GcHooks layout");
        hooks
            .gc_minor
            .register_nonperiodic_action(space, actionflag);
        hooks
            .gc_collect_step
            .register_nonperiodic_action(space, actionflag);
        hooks
            .gc_collect
            .register_nonperiodic_action(space, actionflag);
        majit_gc::hook::register_gc_hooks(majit_gc::hook::GcHookCallbacks {
            is_gc_minor_enabled,
            is_gc_collect_step_enabled,
            is_gc_collect_enabled,
            on_gc_minor,
            on_gc_collect_step,
            on_gc_collect,
        });
        obj as usize
    }) as PyObjectRef
}

pub fn hooks_object() -> PyObjectRef {
    if let Some(&addr) = HOOKS_OBJECT.get() {
        return addr as PyObjectRef;
    }
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    assert!(
        !ec.is_null(),
        "gc.hooks initialized without an ExecutionContext"
    );
    unsafe {
        initialize(
            (*ec).space,
            &mut (*ec).actionflag as &mut (dyn ActionFlagOps + 'static),
        )
    }
}

fn is_gc_minor_enabled() -> bool {
    app_hooks_ptr().is_some_and(|hooks| unsafe { (*hooks).gc_minor_enabled })
}

fn is_gc_collect_step_enabled() -> bool {
    app_hooks_ptr().is_some_and(|hooks| unsafe { (*hooks).gc_collect_step_enabled })
}

fn is_gc_collect_enabled() -> bool {
    app_hooks_ptr().is_some_and(|hooks| unsafe { (*hooks).gc_collect_enabled })
}

fn on_gc_minor(duration: f64, total_memory_used: usize, pinned_objects: usize) {
    let Some(hooks) = app_hooks_ptr() else { return };
    let action = unsafe { &mut (*hooks).gc_minor };
    action.count += 1;
    action.duration += duration;
    action.duration_min = action.duration_min.min(duration);
    action.duration_max = action.duration_max.max(duration);
    action.total_memory_used = total_memory_used;
    action.pinned_objects = pinned_objects;
    action.fire();
}

fn on_gc_collect_step(duration: f64, oldstate: u8, newstate: u8) {
    let Some(hooks) = app_hooks_ptr() else { return };
    let action = unsafe { &mut (*hooks).gc_collect_step };
    action.count += 1;
    action.duration += duration;
    action.duration_min = action.duration_min.min(duration);
    action.duration_max = action.duration_max.max(duration);
    action.oldstate = oldstate;
    action.newstate = newstate;
    action.fire();
}

#[allow(clippy::too_many_arguments)]
fn on_gc_collect(
    num_major_collects: usize,
    arenas_count_before: usize,
    arenas_count_after: usize,
    arenas_bytes: usize,
    rawmalloc_bytes_before: usize,
    rawmalloc_bytes_after: usize,
    pinned_objects: usize,
) {
    let Some(hooks) = app_hooks_ptr() else { return };
    let action = unsafe { &mut (*hooks).gc_collect };
    action.count += 1;
    action.num_major_collects = num_major_collects;
    action.arenas_count_before = arenas_count_before;
    action.arenas_count_after = arenas_count_after;
    action.arenas_bytes = arenas_bytes;
    action.rawmalloc_bytes_before = rawmalloc_bytes_before;
    action.rawmalloc_bytes_after = rawmalloc_bytes_after;
    action.pinned_objects = pinned_objects;
    action.fire();
}

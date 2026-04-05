//! JIT-enabled evaluation — the sole entry point for JIT execution.
//!
//! This module owns the JitDriver, tracing hooks, and compiled-code
//! execution. pyre-interpreter provides the pure interpreter (eval_frame_plain)
//! and the opcode trait implementations on PyFrame.
//!
//! Equivalent to PyPy's `pypyjit/interp_jit.py` — the JIT is injected
//! from outside the interpreter.

use crate::jit::state::{PyreEnv, PyreJitState};
use crate::jit::trace::trace_bytecode;
use pyre_interpreter::PyExecutionContext;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{PyResult, StepResult, execute_opcode_step};
use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;

use majit_backend::Backend;
use majit_gc::trace::TypeInfo;
use majit_ir::Value;
use majit_metainterp::blackhole::ExceptionState;
use majit_metainterp::{CompiledExitLayout, DetailedDriverRunOutcome, JitState};

/// resume.py:1312 blackhole_from_resumedata parity: preserve per-frame
/// resume data from the last guard failure. rd_numb provides frame
/// boundaries (jitcode_index, pc); values are resolved from deadframe.
thread_local! {
    static LAST_GUARD_FRAMES: std::cell::RefCell<Option<Vec<crate::call_jit::ResumedFrame>>> =
        const { std::cell::RefCell::new(None) };
}

/// Take the last guard frames (consuming them).
pub(crate) fn take_last_guard_frames() -> Option<Vec<crate::call_jit::ResumedFrame>> {
    LAST_GUARD_FRAMES.with(|c| c.borrow_mut().take())
}

/// RPython jitexc.py:53 ContinueRunningNormally parity.
enum LoopResult {
    Done(PyResult),
    ContinueRunningNormally,
}

/// Action from handle_jit_outcome for eval_loop_jit dispatch.
enum JitAction {
    Return(PyResult),
    Continue,
    /// RPython jitexc.py:53: guard-restored → restart portal.
    ContinueRunningNormally,
}

use crate::jit::descr::{JITFRAME_GC_TYPE_ID, W_FLOAT_GC_TYPE_ID, W_INT_GC_TYPE_ID};
use crate::jit::virtualizable_gen::build_virtualizable_info;
use majit_gc::collector::MiniMarkGC;
use majit_metainterp::JitDriver;
use pyre_object::floatobject::W_FloatObject;
use pyre_object::intobject::W_IntObject;
use pyre_object::{w_bool_from, w_int_new, w_none, w_str_new, w_tuple_new};

const JIT_THRESHOLD: u32 = 200;
type JitDriverPair = (
    JitDriver<PyreJitState>,
    majit_metainterp::virtualizable::VirtualizableInfo,
);

thread_local! {
    static JIT_DRIVER: UnsafeCell<JitDriverPair> = UnsafeCell::new({
        let info = build_virtualizable_info();
        let mut d = JitDriver::new(JIT_THRESHOLD);
        d.set_virtualizable_info(info.clone());
        let mut gc = MiniMarkGC::new();
        let w_int_tid = gc.register_type(TypeInfo::simple(std::mem::size_of::<W_IntObject>()));
        debug_assert_eq!(w_int_tid, W_INT_GC_TYPE_ID);
        let w_float_tid =
            gc.register_type(TypeInfo::simple(std::mem::size_of::<W_FloatObject>()));
        debug_assert_eq!(w_float_tid, W_FLOAT_GC_TYPE_ID);
        // jitframe.py:49 — rgc.register_custom_trace_hook(JITFRAME, jitframe_trace)
        let jitframe_tid = gc.register_type(majit_metainterp::jitframe::jitframe_type_info());
        debug_assert_eq!(jitframe_tid, JITFRAME_GC_TYPE_ID);
        d.set_gc_allocator(Box::new(gc));
        d.register_raw_int_box_helper(pyre_object::intobject::jit_w_int_new as *const ());
        d.set_intval_descr(pyre_jit_trace::descr::int_intval_descr());
        d.register_raw_int_force_helper(crate::call_jit::jit_force_recursive_call_raw_1 as *const ());
        d.register_raw_int_force_helper(crate::call_jit::jit_force_self_recursive_call_raw_1 as *const ());
        d.register_create_frame_raw(
            crate::call_jit::jit_create_callee_frame_1 as *const (),
            crate::call_jit::jit_create_callee_frame_1_raw_int as *const (),
        );
        d.register_create_frame_raw(
            crate::call_jit::jit_create_self_recursive_callee_frame_1 as *const (),
            crate::call_jit::jit_create_self_recursive_callee_frame_1_raw_int as *const (),
        );
        // resume.py:1367 — BlackholeAllocator for virtual materialization.
        d.register_blackhole_allocator(PyreBlackholeAllocator);
        // warmspot.py:1039 handle_jitexception_from_blackhole parity:
        // portal_runner is called when ContinueRunningNormally is raised
        // at a recursive portal level during blackhole execution.
        d.register_portal_runner(pyre_portal_runner);
        // PyPy interp_jit.py:75 — JitDriver(is_recursive=True)
        d.set_is_recursive(true);
        // warmstate.py:259 trace_eagerness=200 (RPython default).
        // warmspot.py:449 — portal function returns a Python object (int).
        // pyre's portal always returns PyObjectRef, but the JIT unboxes
        // int results to raw i64 via unbox_finish_result, so the static
        // result_type is Int.
        d.set_result_type(majit_ir::Type::Int);
        (d, info)
    });
}

#[inline]
pub fn driver_pair() -> &'static mut JitDriverPair {
    JIT_DRIVER.with(|cell| unsafe { &mut *cell.get() })
}

// GREEN_KEY_ALIASES removed: compile.py:269 parity — cross-loop cut
// traces are now stored directly under the inner loop's green_key
// (cut_inner_green_key) in compile_loop, matching RPython's
// jitcell_token = cross_loop.jitcell_token. No alias dispatch needed.

/// Return a raw pointer to the thread-local VirtualizableInfo.
/// Used by the blackhole to implement BC_GETFIELD_VABLE_* bytecodes.
pub(crate) fn get_virtualizable_info() -> *const majit_metainterp::virtualizable::VirtualizableInfo
{
    let pair = driver_pair();
    &pair.1 as *const _
}

/// pypy/module/pypyjit/interp_jit.py → PyPyJitDriver(JitDriver).
///
/// RPython: reds = ['frame', 'ec'], greens = ['next_instr', 'is_being_profiled', 'pycode'],
///          virtualizables = ['frame']
#[derive(Clone, Copy)]
pub struct PyPyJitDriver;

impl PyPyJitDriver {
    pub fn new(
        get_printable_location: Option<fn(usize, bool, pyre_object::PyObjectRef) -> String>,
        get_location: Option<fn(usize, bool, pyre_object::PyObjectRef) -> pyre_object::PyObjectRef>,
        get_unique_id: Option<fn(usize, bool, pyre_object::PyObjectRef) -> usize>,
        should_unroll_one_iteration: Option<fn(usize, bool, pyre_object::PyObjectRef) -> bool>,
        name: Option<&'static str>,
        is_recursive: bool,
    ) -> Self {
        let _ = (
            get_printable_location,
            get_location,
            get_unique_id,
            should_unroll_one_iteration,
            name,
            is_recursive,
        );
        PyPyJitDriver
    }

    /// interp_jit.py:85-87 — jit_merge_point inside dispatch loop.
    /// Delegates to the real JitDriver via driver_pair().
    pub fn jit_merge_point(
        &self,
        frame: &mut PyFrame,
        ec: *const PyExecutionContext,
        next_instr: usize,
        pycode: pyre_object::PyObjectRef,
        is_being_profiled: bool,
    ) {
        let _ = (ec, pycode, is_being_profiled);
        // The actual merge point is handled inside eval_loop_jit's
        // jit_merge_point_hook. This method exists for API parity.
        let _ = (frame, next_instr);
    }

    /// interp_jit.py:114-117 — can_enter_jit at back-edge.
    /// Delegates to the real JitDriver via driver_pair().
    pub fn can_enter_jit(
        &self,
        frame: &mut PyFrame,
        ec: *const PyExecutionContext,
        next_instr: usize,
        pycode: pyre_object::PyObjectRef,
        is_being_profiled: bool,
    ) {
        let _ = (ec, is_being_profiled, pycode);
        // The actual can_enter_jit is handled inside eval_loop_jit's
        // maybe_compile_and_run on StepResult::CloseLoop.
        let _ = (frame, next_instr);
    }
}

pub const pypyjitdriver: PyPyJitDriver = PyPyJitDriver;

/// interp_jit.py:77 — class __extend__(PyFrame)
///
/// In RPython, __extend__ adds methods to PyFrame. In Rust, PyFrame methods
/// are defined directly; this struct provides the interp_jit.py API surface.
pub struct __extend__;

impl __extend__ {
    /// interp_jit.py:79-96 — dispatch(self, pycode, next_instr, ec).
    ///
    /// RPython:
    ///   while True:
    ///       pypyjitdriver.jit_merge_point(ec=ec, frame=self, ...)
    ///       next_instr = self.handle_bytecode(co_code, next_instr, ec)
    ///   except Yield: ...
    ///   except ExitFrame: ...
    ///
    /// In pyre, the JIT-instrumented dispatch loop is eval_loop_jit().
    /// pycode and ec are stored on the frame; eval_loop_jit reads them
    /// from frame.code and frame.execution_context respectively.
    pub fn dispatch(
        frame: &mut PyFrame,
        _pycode: pyre_object::PyObjectRef,
        next_instr: usize,
        _ec: *const PyExecutionContext,
    ) -> PyResult {
        frame.next_instr = next_instr;
        match eval_loop_jit(frame) {
            LoopResult::Done(result) => result,
            LoopResult::ContinueRunningNormally => Ok(w_none()),
        }
    }

    /// interp_jit.py:98-117 — jump_absolute(self, jumpto, ec).
    ///
    /// RPython:
    ///   if we_are_jitted():
    ///       decr_by = _get_adapted_tick_counter()
    ///       self.last_instr = intmask(jumpto)
    ///       ec.bytecode_trace(self, decr_by)
    ///       jumpto = r_uint(self.last_instr)   # re-read after trace hook
    ///   pypyjitdriver.can_enter_jit(...)
    ///   return jumpto
    pub fn jump_absolute(
        frame: &mut PyFrame,
        mut jumpto: usize,
        ec: *mut PyExecutionContext,
    ) -> usize {
        if majit_metainterp::we_are_jitted() {
            let decr_by = _get_adapted_tick_counter();
            frame.next_instr = jumpto;
            if !ec.is_null() {
                unsafe {
                    (*ec).bytecode_trace(frame as *mut PyFrame, decr_by);
                }
            }
            // Re-read: trace/profile hook may have changed the jump target
            // (interp_jit.py:112 — jumpto = r_uint(self.last_instr))
            jumpto = frame.next_instr;
        }
        // can_enter_jit is handled by eval_loop_jit's StepResult::CloseLoop
        // path which calls maybe_compile_and_run.
        jumpto
    }
}

/// interp_jit.py:119-131 — _get_adapted_tick_counter().
///
/// Normally the tick counter is decremented by 100 for every Python opcode.
/// Here, to better support JIT compilation of small loops, we decrement it
/// by a possibly smaller constant.  We get the maximum 100 when the
/// (unoptimized) trace length is at least 3200 (a bit randomly).
#[inline]
fn _get_adapted_tick_counter() -> usize {
    let (driver, _) = driver_pair();
    let trace_length = driver.current_trace_length();
    // current_trace_length() returns -1 when not tracing
    let decr_by = if trace_length < 0 {
        100 // also if current_trace_length() returned -1
    } else {
        (trace_length as usize) / 32
    };
    decr_by.clamp(1, 100)
}

#[derive(Clone, Copy)]
pub struct W_NotFromAssembler {
    space: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
}

impl W_NotFromAssembler {
    pub fn __init__(
        &mut self,
        space: pyre_object::PyObjectRef,
        w_callable: pyre_object::PyObjectRef,
    ) {
        self.space = space;
        self.w_callable = w_callable;
    }

    pub fn descr_call(&self, __args__: &[pyre_object::PyObjectRef]) -> Self {
        _call_not_in_trace(self.space, self.w_callable, __args__);
        *self
    }
}

pub fn not_from_assembler_new(
    space: pyre_object::PyObjectRef,
    _w_subtype: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
) -> W_NotFromAssembler {
    let _ = _w_subtype;
    W_NotFromAssembler { space, w_callable }
}

#[allow(unused_variables)]
pub fn _call_not_in_trace(
    space: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) {
    let _ = space;
    let _ = pyre_interpreter::baseobjspace::call_function(w_callable, args);
}

#[inline]
fn green_key_from_pycode(next_instr: usize, w_pycode: pyre_object::PyObjectRef) -> Option<u64> {
    // Safety: this follows existing wrappers that treat `W_CodeObject`
    // as an owned pointer to a `CodeObject`.
    let code_ptr = unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) };
    if code_ptr.is_null() {
        return None;
    }
    Some(make_green_key(
        code_ptr as *const pyre_interpreter::CodeObject,
        next_instr,
    ))
}

/// RPython interp_jit.py helper: get_printable_location.
pub fn get_printable_location(
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> String {
    let mut opcode = "<eof>".to_string();
    let mut code_name = "<unknown>".to_string();
    let code_ptr = unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) };
    if !code_ptr.is_null() {
        let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
        code_name = code.obj_name.to_string();
        if let Some((instr, _)) = pyre_interpreter::decode_instruction_at(code, next_instr) {
            opcode = format!("{:?}", instr);
        }
    }
    format!("{code_name} #{next_instr} {opcode}")
}

/// RPython interp_jit.py helper: get_unique_id.
pub fn get_unique_id(
    _next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> usize {
    // A stable process-local unique-id equivalent using the code pointer.
    unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) as usize }
}

/// RPython interp_jit.py helper: get_location.
pub fn get_location(
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    let (filename, line, name, opcode) =
        match unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) } {
            x if x.is_null() => (
                "<unknown>".to_string(),
                0,
                "<unknown>".to_string(),
                "<eof>".to_string(),
            ),
            code_ptr => {
                let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
                let (_opcode, opname) =
                    match pyre_interpreter::decode_instruction_at(code, next_instr) {
                        Some((instruction, _)) => {
                            (format!("{instruction:?}"), format!("{:?}", instruction))
                        }
                        None => ("<eof>".to_string(), "<eof>".to_string()),
                    };
                let line = code
                    .locations
                    .get(next_instr)
                    .and_then(|(start, _)| Some(start.line.get() as usize))
                    .unwrap_or_else(|| {
                        code.first_line_number
                            .map(|line| line.get())
                            .unwrap_or(0)
                            .saturating_add(next_instr)
                    });
                (
                    code.source_path.to_string(),
                    line,
                    code.obj_name.to_string(),
                    opname,
                )
            }
        };
    let _ = opcode;
    w_tuple_new(vec![
        w_str_new(&filename),
        w_int_new(line as i64),
        w_str_new(&name),
        w_int_new(next_instr as i64),
        w_str_new(&opcode),
    ])
}

/// RPython interp_jit.py helper: should_unroll_one_iteration.
pub fn should_unroll_one_iteration(
    _next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> bool {
    match unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) } {
        ptr if ptr.is_null() => false,
        code_ptr => {
            let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
            code.flags.contains(pyre_interpreter::CodeFlags::GENERATOR)
        }
    }
}

/// interp_jit.py:216 — get_jitcell_at_key.
///
/// Returns True if a jitcell exists for this green key, regardless of
/// whether machine code has been compiled. A cell is created when the
/// counter first ticks, so this returns True even before compilation.
pub fn get_jitcell_at_key(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    let key = green_key_from_pycode(next_instr, w_pycode);
    let (driver, _) = driver_pair();
    w_bool_from(key.is_some_and(|green_key| {
        driver
            .meta_interp_mut()
            .warm_state_mut()
            .get_cell(green_key)
            .is_some()
    }))
}

/// RPython interp_jit.py helper: dont_trace_here.
pub fn dont_trace_here(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    let Some(green_key) = green_key_from_pycode(next_instr, w_pycode) else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .disable_noninlinable_function(green_key);
}

/// RPython interp_jit.py helper: mark_as_being_traced.
pub fn mark_as_being_traced(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    let Some(green_key) = green_key_from_pycode(next_instr, w_pycode) else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .mark_as_being_traced(green_key);
}

/// RPython interp_jit.py helper: trace_next_iteration.
pub fn trace_next_iteration(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    let Some(green_key) = green_key_from_pycode(next_instr, w_pycode) else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .trace_next_iteration(green_key);
}

/// RPython interp_jit.py helper: trace_next_iteration_hash.
pub fn trace_next_iteration_hash(_space: pyre_object::PyObjectRef, green_key_hash: usize) {
    let _ = _space;
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .trace_next_iteration(green_key_hash as u64);
}

/// RPython interp_jit.py helper: residual_call.
pub fn residual_call(
    _space: pyre_object::PyObjectRef,
    callable: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) -> pyre_object::PyObjectRef {
    let _ = _space;
    pyre_interpreter::baseobjspace::call_function(callable, args)
}

/// interp_jit.py:138-167 — set_param(space, __args__).
///
/// Configure the tunable JIT parameters.
///   * set_param(name=value, ...)            # as keyword arguments
///   * set_param("name=value,name=value")    # as a user-supplied string
///   * set_param("off")                      # disable the jit
///   * set_param("default")                  # restore all defaults
pub fn set_param(
    _space: pyre_object::PyObjectRef,
    __args__: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let _ = _space;
    let (driver, _) = driver_pair();

    // Separate positional args from kwargs dict (last arg with __pyre_kw__ marker).
    let (pos_args, kwds) = split_kwargs(__args__);

    // interp_jit.py:147-148
    if pos_args.len() > 1 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "set_param() takes at most 1 non-keyword argument, {} given",
            pos_args.len()
        )));
    }

    // interp_jit.py:151-156 — positional string → jit.set_user_param(None, text)
    if pos_args.len() == 1 {
        let w_text = pos_args[0];
        if !unsafe { pyre_object::is_str(w_text) } {
            return Ok(w_none());
        }
        let text = unsafe { pyre_object::w_str_get_value(w_text) };
        // rlib/jit.py:842-845
        if text == "off" {
            let ws = driver.meta_interp_mut().warm_state_mut();
            ws.set_param("threshold", -1);
            ws.set_param("function_threshold", -1);
        } else if text == "default" {
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .set_default_params();
        } else {
            // rlib/jit.py:850-862 — "name=value,name=value"
            let ws = driver.meta_interp_mut().warm_state_mut();
            for s in text.split(',') {
                let s = s.trim();
                if s.is_empty() {
                    continue;
                }
                // rlib/jit.py:853 — len(parts) != 2 → raise ValueError
                let Some((name, value)) = s.split_once('=') else {
                    return Err(pyre_interpreter::PyError::new(
                        pyre_interpreter::PyErrorKind::ValueError,
                        "error in JIT parameters string".to_string(),
                    ));
                };
                let value = value.trim();
                if name == "enable_opts" {
                    ws.set_param_enable_opts(value);
                } else if let Ok(parsed) = value.parse::<i64>() {
                    ws.set_param(name, parsed);
                } else {
                    return Err(pyre_interpreter::PyError::new(
                        pyre_interpreter::PyErrorKind::ValueError,
                        "error in JIT parameters string".to_string(),
                    ));
                }
            }
        }
    }

    // interp_jit.py:157-167 — keyword arguments
    if let Some(kw_dict) = kwds {
        let ws = driver.meta_interp_mut().warm_state_mut();
        let d = unsafe { &*(kw_dict as *const pyre_object::dictobject::W_DictObject) };
        for &(k, v) in unsafe { &*d.entries } {
            if !unsafe { pyre_object::is_str(k) } {
                continue;
            }
            let key = unsafe { pyre_object::w_str_get_value(k) };
            if key == "__pyre_kw__" {
                continue;
            }
            // interp_jit.py:158-159
            if key == "enable_opts" {
                if unsafe { pyre_object::is_str(v) } {
                    ws.set_param_enable_opts(unsafe { pyre_object::w_str_get_value(v) });
                }
                continue;
            }
            // interp_jit.py:160-167 — validate parameter name
            if !is_known_jit_param(key) {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "no JIT parameter '{key}'"
                )));
            }
            if unsafe { pyre_object::is_int(v) } {
                ws.set_param(key, unsafe { pyre_object::w_int_get_value(v) });
            }
        }
    }

    Ok(w_none())
}

/// rlib/jit.py:588-605 PARAMETERS — valid parameter names.
fn is_known_jit_param(name: &str) -> bool {
    matches!(
        name,
        "threshold"
            | "function_threshold"
            | "trace_eagerness"
            | "decay"
            | "trace_limit"
            | "inlining"
            | "loop_longevity"
            | "retrace_limit"
            | "pureop_historylength"
            | "max_retrace_guards"
            | "max_unroll_loops"
            | "disable_unrolling"
            | "enable_opts"
            | "max_unroll_recursion"
            | "vec"
            | "vec_all"
            | "vec_cost"
    )
}

/// Split args into (positional, optional kwargs dict).
fn split_kwargs(
    args: &[pyre_object::PyObjectRef],
) -> (
    &[pyre_object::PyObjectRef],
    Option<pyre_object::PyObjectRef>,
) {
    if let Some(&last) = args.last() {
        if !last.is_null()
            && unsafe { pyre_object::is_dict(last) }
            && unsafe {
                pyre_object::w_dict_lookup(last, pyre_object::w_str_new("__pyre_kw__")).is_some()
            }
        {
            return (&args[..args.len() - 1], Some(last));
        }
    }
    (args, None)
}

/// interp_jit.py:259 — releaseall(space).
///
/// Mark all current machine code objects as ready to release.
/// They will be released at the next GC (unless in use on a thread stack).
///
/// RPython: jit_hooks.stats_memmgr_release_all(None) → memory manager
/// marks loops for release, does NOT invalidate warm-state cells.
pub fn releaseall(_space: pyre_object::PyObjectRef) {
    let _ = _space;
    let (driver, _) = driver_pair();
    // Mark compiled loops for release without clearing warm-state.
    // Matches RPython's release_all_loops() semantics: the loops are
    // freed at the next collection, not immediately invalidated.
    driver.mark_all_loops_for_release();
}

fn init_callbacks() {
    use pyre_jit_trace::callbacks::{self, CallJitCallbacks};
    thread_local! {
        static INIT: Cell<bool> = const { Cell::new(false) };
    }
    INIT.with(|c| {
        if !c.get() {
            c.set(true);
            let cb = Box::leak(Box::new(CallJitCallbacks {
                callee_frame_helper: crate::call_jit::callee_frame_helper,
                callable_prefers_function_entry: crate::call_jit::callable_prefers_function_entry,
                recursive_force_cache_safe: crate::call_jit::recursive_force_cache_safe,
                jit_drop_callee_frame: crate::call_jit::jit_drop_callee_frame as *const (),
                jit_force_callee_frame: crate::call_jit::jit_force_callee_frame as *const (),
                jit_force_recursive_call_1: crate::call_jit::jit_force_recursive_call_1
                    as *const (),
                jit_force_recursive_call_argraw_boxed_1:
                    crate::call_jit::jit_force_recursive_call_argraw_boxed_1 as *const (),
                jit_force_self_recursive_call_argraw_boxed_1:
                    crate::call_jit::jit_force_self_recursive_call_argraw_boxed_1 as *const (),
                jit_create_callee_frame_1: crate::call_jit::jit_create_callee_frame_1 as *const (),
                jit_create_callee_frame_1_raw_int:
                    crate::call_jit::jit_create_callee_frame_1_raw_int as *const (),
                jit_create_self_recursive_callee_frame_1:
                    crate::call_jit::jit_create_self_recursive_callee_frame_1 as *const (),
                jit_create_self_recursive_callee_frame_1_raw_int:
                    crate::call_jit::jit_create_self_recursive_callee_frame_1_raw_int as *const (),
                driver_pair: || JIT_DRIVER.with(|cell| cell.get() as *mut u8),
            }));
            callbacks::init(cb);
        }
    });
}

// JIT_TRACING_DEPTH removed — now MetaInterp.tracing_call_depth field.
// RPython portal_call_depth parity: state colocated with tracing context.

/// Read the call depth from pyre-interpreter's CALL_DEPTH TLS.
/// Replaces the separate JIT_CALL_DEPTH — single source of truth.
#[inline(always)]
fn call_depth() -> u32 {
    pyre_interpreter::call::call_depth()
}

/// RPython green_key = (pycode, next_instr).
/// Each (code, pc) pair has independent warmup counter and compiled loop.
#[inline(always)]
pub fn make_green_key(code_ptr: *const pyre_interpreter::CodeObject, pc: usize) -> u64 {
    (code_ptr as u64).wrapping_mul(1000003) ^ (pc as u64)
}

// JIT_CALL_DEPTH removed — pyre-interpreter::call::CALL_DEPTH is the single
// source of truth. call_depth() reads it. No more Box<dyn Any> allocation.

/// RPython compile.py:204-207 (record_loop_or_bridge) parity:
/// Register the compiled loop's invalidation flag with all quasi-immutable
/// dependencies collected during optimization. The optimizer records
/// namespace pointers in quasi_immutable_deps when processing
/// QUASIIMMUT_FIELD ops. After compilation, this function reads them
/// from MetaInterp and registers watchers so GUARD_NOT_INVALIDATED
/// fails when the namespace mutates.
fn register_quasi_immutable_deps(green_key: u64) {
    let (driver, _) = driver_pair();
    let deps: Vec<(u64, u32)> =
        std::mem::take(&mut driver.meta_interp_mut().last_quasi_immutable_deps);
    if deps.is_empty() {
        return;
    }
    let Some(token) = driver.get_loop_token(green_key) else {
        return;
    };
    let flag = token.invalidation_flag();
    for (ns_ptr, slot) in deps {
        let ns = unsafe { &mut *(ns_ptr as *mut pyre_interpreter::PyNamespace) };
        ns.register_slot_watcher(slot as usize, &flag);
    }
}

/// RPython rstack.stack_almost_full() parity.
#[inline]
fn stack_almost_full() -> bool {
    call_depth() > 20
}

/// Evaluate a Python frame with JIT compilation.
///
/// This is the main entry point for pyre-jit.
pub fn eval_with_jit(frame: &mut PyFrame) -> PyResult {
    eval_with_jit_inner(frame)
}

fn eval_with_jit_inner(frame: &mut PyFrame) -> PyResult {
    // PYRE_JIT=0 disables JIT entirely, falling back to plain interpreter.
    static PYRE_JIT_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *PYRE_JIT_DISABLED.get_or_init(|| std::env::var("PYRE_JIT").as_deref() == Ok("0")) {
        return pyre_interpreter::eval::eval_frame_plain(frame);
    }
    pyre_interpreter::call::register_eval_override(eval_with_jit);
    pyre_interpreter::call::register_inline_call_override(
        crate::call_jit::maybe_handle_inline_concrete_call,
    );
    #[cfg(not(target_arch = "wasm32"))]
    crate::call_jit::install_jit_call_bridge();
    init_callbacks();
    #[cfg(not(target_arch = "wasm32"))]
    majit_backend_cranelift::register_rebuild_state_after_failure(rebuild_state_after_failure);
    frame.fix_array_ptrs();
    // Set CURRENT_FRAME so zero-arg super() can find __class__ in the caller.
    let _frame_guard = pyre_interpreter::eval::install_current_frame(frame);

    // RPython blackhole.py parity: during bridge tracing, concrete
    // (force helper) calls must use the plain interpreter to avoid
    // corrupting the bridge trace's symbolic state via eval_loop_jit's
    // jit_merge_point_hook. RPython's blackhole interpreter has no
    // JIT hooks; pyre's equivalent is eval_frame_plain.
    {
        let (drv, _) = driver_pair();
        if drv.is_bridge_tracing() {
            return pyre_interpreter::eval::eval_frame_plain(frame);
        }
    }

    // RPython warmspot.py ll_portal_runner:
    //   maybe_compile_and_run(increment_threshold, *args)
    //   return portal_ptr(*args)
    //
    // maybe_compile_and_run = try_function_entry_jit: checks for compiled
    // code (dispatch) or threshold (start tracing). Internally guards on
    // JC_TRACING (driver.is_tracing()) to avoid re-entry during tracing.
    //
    // portal_ptr = eval_loop_jit at depth 0 (has jit_merge_point +
    // can_enter_jit back-edge), plain interpreter at depth > 0.
    if let Some(result) = try_function_entry_jit(frame) {
        return result;
    }
    handle_jitexception(frame)
}

/// RPython warmspot.py:941 portal_runner for force callbacks.
///
/// Unlike eval_with_jit, this does NOT register callbacks (already done)
/// and is safe to call from force helpers during compiled code execution.
/// RPython: bhimpl_recursive_call → portal_runner → CAN enter JIT.
pub(crate) fn portal_runner_for_force(frame: &mut PyFrame) -> PyResult {
    frame.fix_array_ptrs();
    if let Some(result) = try_function_entry_jit(frame) {
        return result;
    }
    handle_jitexception(frame)
}

/// warmspot.py:970-983 ContinueRunningNormally → portal_ptr(*args) parity.
///
/// Called from handle_jitexception_in_portal (via portal_runner callback)
/// when ContinueRunningNormally is raised at a recursive portal level.
/// Extracts the red_ref values (frame locals as PyObjectRef pointers)
/// and calls the portal function (eval_with_jit) with those values.
///
/// Returns Ok((return_type, value)) or Err(JitException) if the portal
/// itself raises a JitException (warmspot.py:979-980 loop back).
fn pyre_portal_runner(
    exc: &majit_metainterp::jitexc::JitException,
) -> Result<(majit_metainterp::blackhole::BhReturnType, i64), majit_metainterp::jitexc::JitException>
{
    use majit_metainterp::blackhole::BhReturnType;
    use majit_metainterp::jitexc::JitException;

    let JitException::ContinueRunningNormally { red_ref, .. } = exc else {
        // Not ContinueRunningNormally — shouldn't reach here.
        return Ok((BhReturnType::Void, 0));
    };

    // warmspot.py:972-975: extract args from ContinueRunningNormally.
    // pyre's red args: [frame_ptr, next_instr, vsd, locals...]
    // red_ref[0] is the frame pointer.
    if red_ref.is_empty() {
        return Ok((BhReturnType::Void, 0));
    }

    let frame_ptr = red_ref[0] as *mut PyFrame;
    if frame_ptr.is_null() {
        return Ok((BhReturnType::Void, 0));
    }

    // warmspot.py:976-978: result = portal_ptr(*args)
    // In pyre, this means running the frame through eval_with_jit.
    let frame = unsafe { &mut *frame_ptr };
    match crate::eval::eval_with_jit(frame) {
        Ok(result) => {
            // warmspot.py:982: result = unspecialize_value(result)
            Ok((BhReturnType::Ref, result as i64))
        }
        Err(py_err) => {
            // blackhole.py:1773-1775: regular exception from portal_ptr.
            // _handle_jitexception catches it with `except Exception as e`
            // and converts via get_llexception(cpu, e) → lle.
            // This lle becomes current_exc in the blackhole chain.
            //
            // In Rust, we return Err(ExitFrameWithExceptionRef) which
            // handle_jitexception_in_portal converts to Err(exc_value),
            // and handle_jitexception sets current_exc = exc_value.
            let exc_obj = py_err.exc_object;
            Err(JitException::ExitFrameWithExceptionRef(majit_ir::GcRef(
                exc_obj as usize,
            )))
        }
    }
}

/// RPython warmspot.py:961-1007 handle_jitexception parity.
#[inline(always)]
fn handle_jitexception(frame: &mut PyFrame) -> PyResult {
    loop {
        match eval_loop_jit(frame) {
            LoopResult::Done(result) => return result,
            LoopResult::ContinueRunningNormally => {
                frame.fix_array_ptrs();
                // RPython warmspot.py:976-978:
                //   result = portal_ptr(*args)
                //   return result
                // Requires all guards to have working blackhole (no Failed).
                // Until resume data is complete, fall through to eval_loop_jit.
                continue;
            }
        }
    }
}

fn debug_first_arg_int(frame: &PyFrame) -> Option<i64> {
    if frame.locals_cells_stack_w.len() == 0 {
        return None;
    }
    let value = frame.locals_cells_stack_w[0];
    if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
        return None;
    }
    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
}

/// JIT-enabled evaluation loop (PyPy interp_jit.py dispatch()).
///
/// Calls merge_point on EVERY iteration (PyPy line 85-87), not just
/// when tracing. This matches PyPy's jit_merge_point placement.
/// RPython interp_jit.py dispatch() parity.
///
/// The hot loop mirrors RPython's structure exactly:
///   while True:
///       jit_merge_point(...)      # thin inline check
///       next_instr = handle_bytecode(...)
///
/// JIT hooks are thin inline checks; all heavy logic is in #[cold] helpers.
fn eval_loop_jit(frame: &mut PyFrame) -> LoopResult {
    let code = unsafe { &*frame.code };
    let env = PyreEnv;
    let (driver, info) = driver_pair();
    let is_portal: bool = &*code.obj_name != "<module>";

    loop {
        if frame.next_instr >= code.instructions.len() {
            return LoopResult::Done(Ok(w_none()));
        }

        let pc = frame.next_instr;
        let Some((instruction, op_arg)) = pyre_interpreter::decode_instruction_at(code, pc) else {
            return LoopResult::Done(Ok(w_none()));
        };

        // ── jit_merge_point (RPython interp_jit.py:85-87) ──
        // Runtime no-op. Only handles trace feed when tracing is active.
        if is_portal {
            let tracing_depth = driver.meta_interp().tracing_call_depth;
            if let Some(depth) = tracing_depth {
                if call_depth() == depth {
                    if let Some(loop_result) =
                        jit_merge_point_hook(frame, code, pc, driver, info, &env)
                    {
                        return loop_result;
                    }
                }
            } else if driver.is_tracing() {
                // First merge_point after trace start — depth not yet set.
                if let Some(loop_result) = jit_merge_point_hook(frame, code, pc, driver, info, &env)
                {
                    return loop_result;
                }
            }
        }

        // ── inline replay (tracing bookkeeping) ──
        if frame.pending_inline_resume_pc == Some(pc) {
            if matches!(
                instruction,
                pyre_interpreter::bytecode::Instruction::Call { .. }
            ) {
                frame.pending_inline_resume_pc = None;
                continue;
            }
        }
        if let pyre_interpreter::bytecode::Instruction::Call { argc } = instruction {
            if !frame.pending_inline_results.is_empty() {
                frame.next_instr = pc + 1;
                if pyre_interpreter::call::replay_pending_inline_call(
                    frame,
                    argc.get(op_arg) as usize,
                ) {
                    continue;
                }
                frame.next_instr = pc;
            }
        }

        // ── handle_bytecode (RPython interp_jit.py:90) ──
        frame.next_instr += 1;
        let next_instr = frame.next_instr;
        if let pyre_interpreter::bytecode::Instruction::Call { argc } = instruction {
            if pyre_interpreter::call::replay_pending_inline_call(frame, argc.get(op_arg) as usize)
            {
                continue;
            }
        }
        match execute_opcode_step(frame, code, instruction, op_arg, next_instr) {
            Ok(StepResult::Continue) => {}
            Ok(StepResult::CloseLoop { loop_header_pc, .. }) if is_portal => {
                // ── can_enter_jit (RPython interp_jit.py:114) ──
                // RPython interp_jit.py:114 → warmstate.py:446
                let green_key = make_green_key(frame.code, loop_header_pc);
                if let Some(loop_result) =
                    maybe_compile_and_run(frame, green_key, loop_header_pc, driver, info, &env)
                {
                    return loop_result;
                }
            }
            Ok(StepResult::CloseLoop { .. }) => {}
            Ok(StepResult::Return(result)) => return LoopResult::Done(Ok(result)),
            Ok(StepResult::Yield(result)) => return LoopResult::Done(Ok(result)),
            Err(err) => {
                if pyre_interpreter::eval::handle_exception(frame, &err) {
                    continue;
                }
                return LoopResult::Done(Err(err));
            }
        }
    }
}

/// RPython jit_merge_point slow path — only called when tracing is active.
#[cold]
#[inline(never)]
fn jit_merge_point_hook(
    frame: &mut PyFrame,
    code: &pyre_interpreter::CodeObject,
    pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    let concrete_frame = frame as *mut PyFrame as usize;
    let green_key = make_green_key(frame.code, pc);
    let mut jit_state = build_jit_state(frame, info);
    let current_depth = call_depth();
    let was_tracing = driver.is_tracing();
    if let Some(outcome) = driver.jit_merge_point_keyed(
        green_key,
        pc,
        &mut jit_state,
        env,
        || {},
        |ctx, sym| {
            let (driver, _) = driver_pair();
            driver.meta_interp_mut().tracing_call_depth = Some(current_depth);
            // RPython parity: codewriter.make_jitcodes() runs before tracing
            // starts, populating all_liveness. In pyre, JitCode compilation is
            // lazy — ensure the code's JitCode (with liveness) exists before
            // tracing so get_list_of_active_boxes can use it.
            crate::jit::codewriter::ensure_jitcode_for(code);
            let snapshot = Box::new(frame.snapshot_for_tracing());
            let _ = concrete_frame;
            let (action, _executed_frame) = trace_bytecode(ctx, sym, code, pc, snapshot);
            action
        },
    ) {
        match handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
            JitAction::Return(result) => return Some(LoopResult::Done(result)),
            JitAction::ContinueRunningNormally => return Some(LoopResult::ContinueRunningNormally),
            JitAction::Continue => {}
        }
    }
    // Trace completed or aborted — clear tracing depth.
    if !driver.is_tracing() {
        driver.meta_interp_mut().tracing_call_depth = None;
        // compile.py:269: cross-loop cut stores under inner key.
        // Use the actual compiled key for post-compilation steps.
        let compiled_key = driver.last_compiled_key().unwrap_or(green_key);
        register_quasi_immutable_deps(compiled_key);
        // RPython pyjitpl.py:3048-3061 raise_continue_running_normally:
        // after trace compilation, restart so maybe_compile_and_run
        // (try_function_entry_jit) dispatches to compiled code.
        if was_tracing {
            return Some(LoopResult::ContinueRunningNormally);
        }
    }
    None
}

/// RPython warmstate.py:446-511 maybe_compile_and_run.
///
/// Entry point to the JIT. Called at can_enter_jit (back-edge).
///
/// RPython order: cell lookup (JC_TRACING → skip, JC_COMPILED → enter)
/// BEFORE counter.tick(). This prevents compiled loops from occupying
/// counter hash-table slots and evicting non-compiled loops (the 5-way
/// associative cache has only 5 slots per bucket).
#[cold]
#[inline(never)]
fn maybe_compile_and_run(
    frame: &mut PyFrame,
    green_key: u64,
    loop_header_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    // pyre-local extension: PYRE_NO_JIT disables JIT entirely.
    // No RPython counterpart — kept for development debugging only.
    // TODO: remove when JIT is stable enough to not need a kill switch.
    static NO_JIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *NO_JIT.get_or_init(|| std::env::var_os("PYRE_NO_JIT").is_some()) {
        return None;
    }
    // warmstate.py:473-477: JC_TRACING → skip entirely (no counter tick)
    if driver.is_tracing() {
        return None;
    }
    // warmstate.py:503-511: procedure_token exists → EnterJitAssembler.
    // RPython enters assembler unconditionally when a compiled loop is
    // available for this green_key.
    if driver.has_compiled_loop(green_key) {
        return execute_assembler(frame, green_key, loop_header_pc, driver, info, env);
    }
    // warmstate.py:496-511: counter.tick → threshold reached → bound_reached
    if driver
        .meta_interp_mut()
        .warm_state_mut()
        .counter
        .tick(green_key)
    {
        if driver.meta_interp().is_tracing_key(green_key) {
            return None;
        }
        return bound_reached(frame, green_key, loop_header_pc, driver, info, env);
    }
    None
}

/// compile.py:701-717 handle_fail outcome.
/// compile.py:701-717: handle_fail NEVER returns in RPython — it raises
/// ContinueRunningNormally or DoneWithThisFrame. In pyre, we return the
/// equivalent BlackholeResult.
/// compile.py:701-717 handle_fail outcome.
enum HandleFailOutcome {
    /// Bridge compiled successfully — continue in compiled code.
    BridgeCompiled,
    /// Resume in blackhole interpreter.
    ResumeInBlackhole,
}

/// compile.py:701-717 handle_fail.
///
/// Single function containing the complete guard failure handling:
/// compile.py:701-717 handle_fail.
///
/// RPython: handle_fail NEVER returns — both paths raise
/// ContinueRunningNormally or DoneWithThisFrame.
/// pyre: returns BlackholeResult (equivalent to RPython's exceptions).
fn handle_fail(
    frame: &mut PyFrame,
    green_key: u64,
    trace_id: u64,
    fail_index: u32,
    should_bridge: bool,
    owning_key: u64,
    descr_addr: usize,
    exit_layout: &CompiledExitLayout,
    raw_values: &[i64],
    _info: &majit_metainterp::virtualizable::VirtualizableInfo,
) -> HandleFailOutcome {
    // compile.py:702-703: must_compile() AND not stack_almost_full()
    if should_bridge && !stack_almost_full() {
        let is_tracing = {
            let (driver, _) = driver_pair();
            driver.is_tracing()
        };
        if !is_tracing {
            // compile.py:704: self.start_compiling() (set ST_BUSY_FLAG)
            {
                let (driver, _) = driver_pair();
                driver.meta_interp_mut().start_guard_compiling(descr_addr);
            }
            // compile.py:706-708: _trace_and_compile_from_bridge(deadframe)
            // force_plain_eval prevents concrete calls during bridge
            // tracing from re-entering compiled code.
            let compiled = {
                let _plain = pyre_interpreter::call::force_plain_eval();
                crate::call_jit::trace_and_compile_from_bridge(
                    owning_key,
                    trace_id,
                    fail_index,
                    frame,
                    raw_values,
                    exit_layout,
                )
            };
            // compile.py:709: done_compiling (clear ST_BUSY_FLAG)
            {
                let (driver, _) = driver_pair();
                driver.meta_interp_mut().done_guard_compiling(descr_addr);
            }
            if compiled {
                // compile.py:708: bridge compiled → ContinueRunningNormally.
                // RPython: the bridge is attached to the guard descr;
                // re-entering compiled code will follow the bridge.
                return HandleFailOutcome::BridgeCompiled;
            }
        }
    }
    // compile.py:710-716 / pyjitpl.py:2906 (SwitchToBlackhole):
    // resume_in_blackhole(metainterp_sd, jitdriver_sd, self, deadframe)
    HandleFailOutcome::ResumeInBlackhole
}

/// compile.py:710-716 resume_in_blackhole parity.
///
/// RPython: resume_in_blackhole → blackhole_from_resumedata →
/// consume_one_section → _run_forever → raises.
/// pyre: build_resumed_frames → resume_in_blackhole → returns result.
///
/// The RPython orthodox path (blackhole_resume_via_rd_numb →
/// consume_one_section) is not yet compatible with pyre's JitCode
/// liveness format. Use pyre's build_resumed_frames path.
fn resume_in_blackhole_from_exit_layout(
    frame: &mut PyFrame,
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) -> crate::call_jit::BlackholeResult {
    use crate::call_jit::BlackholeResult;

    build_blackhole_frames_from_deadframe(raw_values, exit_layout);
    let guard_frames = take_last_guard_frames();
    if let Some(ref frames) = guard_frames {
        crate::call_jit::resume_in_blackhole(frame, frames)
    } else {
        BlackholeResult::Failed
    }
}

/// RPython warmstate.py:387-423 execute_assembler.
///
/// Run compiled machine code for a given green_key. Handles the
/// fail_descr outcomes: DoneWithThisFrame, GuardFailure, etc.
#[cold]
#[inline(never)]
fn execute_assembler(
    frame: &mut PyFrame,
    green_key: u64,
    entry_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    frame.next_instr = entry_pc;

    let mut jit_state = build_jit_state(frame, info);

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][execute-assembler] key={} pc={} arg0={:?}",
            green_key,
            entry_pc,
            debug_first_arg_int(frame),
        );
    }

    // warmstate.py:395 func_execute_token(loop_token, *args) → deadframe
    let outcome = driver.run_compiled_detailed_with_bridge_keyed(
        green_key,
        entry_pc,
        &mut jit_state,
        env,
        || {},
    );

    if majit_metainterp::majit_log_enabled() {
        let kind = match &outcome {
            DetailedDriverRunOutcome::Finished { .. } => "finished",
            DetailedDriverRunOutcome::Jump { .. } => "jump",
            DetailedDriverRunOutcome::Abort { .. } => "abort",
            DetailedDriverRunOutcome::GuardFailure { .. } => "guard-failure",
        };
        eprintln!(
            "[jit][execute-assembler] outcome key={} pc={} kind={}",
            green_key, entry_pc, kind
        );
    }

    // warmstate.py:402-422 handle fail_descr outcome
    match outcome {
        // warmstate.py:402-415 fast path: DoneWithThisFrame
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            ..
        } => {
            let raw_int_result = raw_int_result || driver.has_raw_int_finish(green_key);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][handle-outcome] finished key={} raw_flag={} typed_values={:?}",
                    green_key, raw_int_result, typed_values
                );
            }
            let [value] = typed_values.as_slice() else {
                return Some(LoopResult::Done(Err(
                    pyre_interpreter::PyError::type_error(
                        "compiled finish did not produce a single object return value",
                    ),
                )));
            };
            let result = match value {
                majit_ir::Value::Int(raw) => {
                    let _ = raw_int_result;
                    pyre_object::intobject::w_int_new(*raw)
                }
                majit_ir::Value::Ref(value) => value.as_usize() as pyre_object::PyObjectRef,
                majit_ir::Value::Float(f) => pyre_object::floatobject::w_float_new(*f),
                majit_ir::Value::Void => {
                    return Some(LoopResult::Done(Err(
                        pyre_interpreter::PyError::type_error(
                            "compiled finish produced a void return value",
                        ),
                    )));
                }
            };
            Some(LoopResult::Done(Ok(result)))
        }
        // warmstate.py:416-422 general: handle_fail
        // compile.py:701-717 → bridge or blackhole
        DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            should_bridge,
            owning_key,
            descr_addr,
            ref raw_values,
            ref exit_layout,
        } => {
            match handle_fail(
                frame,
                green_key,
                trace_id,
                fail_index,
                should_bridge,
                owning_key,
                descr_addr,
                exit_layout,
                raw_values,
                info,
            ) {
                HandleFailOutcome::BridgeCompiled => Some(LoopResult::ContinueRunningNormally),
                HandleFailOutcome::ResumeInBlackhole => {
                    // compile.py:710-716 / pyjitpl.py:2906 SwitchToBlackhole
                    let bh_result =
                        resume_in_blackhole_from_exit_layout(frame, raw_values, exit_layout);
                    match bh_result {
                        crate::call_jit::BlackholeResult::ContinueRunningNormally => {
                            Some(LoopResult::ContinueRunningNormally)
                        }
                        crate::call_jit::BlackholeResult::DoneWithThisFrame(r) => {
                            Some(LoopResult::Done(r))
                        }
                        crate::call_jit::BlackholeResult::Failed => {
                            if majit_metainterp::majit_log_enabled() {
                                eprintln!(
                                    "[jit][BUG] blackhole failed key={} — invalidating",
                                    green_key,
                                );
                            }
                            driver.invalidate_loop(green_key);
                            None
                        }
                    }
                }
            }
        }
        DetailedDriverRunOutcome::Jump { .. } | DetailedDriverRunOutcome::Abort { .. } => None,
    }
}

/// RPython warmstate.py:425-444 bound_reached.
///
/// Called when counter threshold fires and no compiled code exists.
/// Starts tracing via back_edge_or_run_compiled_keyed.
#[cold]
#[inline(never)]
fn bound_reached(
    frame: &mut PyFrame,
    green_key: u64,
    loop_header_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][bound-reached] key={} pc={} arg0={:?}",
            green_key,
            loop_header_pc,
            debug_first_arg_int(frame),
        );
    }
    // warmstate.py:429: jitcounter.decay_all_counters()
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .counter
        .decay_all_counters();
    // warmstate.py:430
    if stack_almost_full() {
        return None;
    }
    // warmstate.py:437-444: MetaInterp.compile_and_run_once
    frame.next_instr = loop_header_pc;
    let mut jit_state = build_jit_state(frame, info);
    // warmstate.py:473-477 JC_TRACING
    if driver.meta_interp().is_tracing_key(green_key) {
        return None;
    }
    // warmstate.py:503-511: procedure_token → EnterJitAssembler.
    let outcome = if driver.has_compiled_loop(green_key) {
        Some(driver.run_compiled_detailed_with_bridge_keyed(
            green_key,
            loop_header_pc,
            &mut jit_state,
            env,
            || {},
        ))
    } else if !driver.is_tracing() {
        // warmstate.py:437-444 compile_and_run_once parity:
        // start tracing AND trace synchronously in a single call.
        let had_compiled = driver.has_compiled_loop(green_key);
        driver.bound_reached(green_key, loop_header_pc, &mut jit_state, env);
        // force_start_tracing may return RunCompiled (retargeted trace
        // already compiled for this cell). In that case, enter compiled.
        // compile.py:269: actual key may be inner key after cross-loop cut.
        let actual_key = driver.last_compiled_key().unwrap_or(green_key);
        if !driver.is_tracing() && driver.has_compiled_loop(actual_key) {
            Some(driver.run_compiled_detailed_with_bridge_keyed(
                actual_key,
                loop_header_pc,
                &mut jit_state,
                env,
                || {},
            ))
        } else if driver.is_tracing() {
            // RPython pyjitpl.py:2876-2888 _compile_and_run_once:
            // interpret() traces the entire loop synchronously.
            // Set tracing_call_depth so inner function calls (which
            // run their own eval_loop_jit) don't trigger jit_merge_point_hook.
            driver.meta_interp_mut().tracing_call_depth = Some(call_depth());
            let code = unsafe { &*frame.code };
            let concrete_frame = Box::new(frame.snapshot_for_tracing());
            let outcome = driver.jit_merge_point_keyed(
                green_key,
                loop_header_pc,
                &mut jit_state,
                env,
                || {},
                |ctx, sym| {
                    use pyre_jit_trace::trace::trace_bytecode;
                    crate::jit::codewriter::ensure_jitcode_for(code);
                    let (action, _) =
                        trace_bytecode(ctx, sym, code, loop_header_pc, concrete_frame);
                    action
                },
            );
            driver.meta_interp_mut().tracing_call_depth = None;
            let compiled_key = driver.last_compiled_key().unwrap_or(green_key);
            if !had_compiled && driver.has_compiled_loop(compiled_key) {
                register_quasi_immutable_deps(compiled_key);
                driver
                    .meta_interp_mut()
                    .warm_state_mut()
                    .counter
                    .set_compiled_hint(compiled_key, true);
            }
            // pyjitpl.py:3048-3061 raise_continue_running_normally:
            // after compilation, restart so execute_assembler runs.
            if !driver.is_tracing() {
                return Some(LoopResult::ContinueRunningNormally);
            }
            outcome
        } else {
            None
        }
    } else {
        None
    };
    if let Some(outcome) = outcome {
        // compile.py:701-717 handle_fail: bridge/blackhole decision.
        if let DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            should_bridge,
            owning_key,
            descr_addr,
            ref raw_values,
            ref exit_layout,
        } = outcome
        {
            match handle_fail(
                frame,
                green_key,
                trace_id,
                fail_index,
                should_bridge,
                owning_key,
                descr_addr,
                exit_layout,
                raw_values,
                info,
            ) {
                HandleFailOutcome::BridgeCompiled => {
                    return Some(LoopResult::ContinueRunningNormally);
                }
                HandleFailOutcome::ResumeInBlackhole => {
                    match resume_in_blackhole_from_exit_layout(frame, raw_values, exit_layout) {
                        crate::call_jit::BlackholeResult::ContinueRunningNormally => {
                            return Some(LoopResult::ContinueRunningNormally);
                        }
                        crate::call_jit::BlackholeResult::DoneWithThisFrame(r) => {
                            return Some(LoopResult::Done(r));
                        }
                        crate::call_jit::BlackholeResult::Failed => {}
                    }
                }
            }
        } else {
            match handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
                JitAction::Return(result) => return Some(LoopResult::Done(result)),
                JitAction::ContinueRunningNormally | JitAction::Continue => {}
            }
        }
    }
    driver.meta_interp_mut().tracing_call_depth = None;
    None
}

/// RPython warmstate.py maybe_compile_and_run parity.
///
/// Called at every portal entry (function call). Must be fast for the
/// common case (no compiled code, not tracing, threshold not reached).
pub fn try_function_entry_jit(frame: &mut PyFrame) -> Option<PyResult> {
    if std::env::var_os("MAJIT_DUMP_BYTECODE").is_some() {
        let code = unsafe { &*frame.code };
        if code.obj_name.as_str() == "fannkuch" && frame.next_instr == 0 {
            use std::sync::OnceLock;
            static DUMPED: OnceLock<()> = OnceLock::new();
            if DUMPED.get().is_none() {
                let _ = DUMPED.set(());
                let mut state = pyre_interpreter::OpArgState::default();
                eprintln!("-- fannkuch bytecode dump --");
                for (pc, unit) in code.instructions.iter().copied().enumerate() {
                    let (instr, oparg) = state.get(unit);
                    eprintln!("{pc:03}: {instr:?} oparg={oparg:?}");
                }
                for pc in [
                    72usize, 99, 129, 131, 141, 155, 168, 179, 234, 245, 447, 449,
                ] {
                    eprintln!(
                        "decode[{pc}] = {:?}",
                        pyre_interpreter::decode_instruction_at(code, pc)
                    );
                }
            }
        }
    }
    let green_key = make_green_key(frame.code, frame.next_instr);
    let (driver, info) = driver_pair();

    // RPython warmstate.py maybe_compile_and_run fast path:
    // if no compiled loop and not tracing, just tick the counter.
    if !driver.has_compiled_loop(green_key) && !driver.is_tracing() {
        let should_trace = driver
            .meta_interp_mut()
            .warm_state_mut()
            .should_trace_function_entry(green_key);
        if !should_trace {
            return None;
        }
    }

    // RPython warmstate.py:473-477: per-cell JC_TRACING.
    if driver.meta_interp().is_tracing_key(green_key) {
        return None;
    }
    if driver.has_compiled_loop(green_key) {
        // Same gate as maybe_compile_and_run: only enter compiled code
        // when a compiled loop exists for this green_key.
        // warmstate.py:503-511: procedure_token → enter unconditionally.
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][func-entry] run compiled key={} arg0={:?} depth={} raw_finish_known={}",
                green_key,
                debug_first_arg_int(frame),
                call_depth(),
                driver.has_raw_int_finish(green_key)
            );
        }
        let env = PyreEnv;
        let mut jit_state = build_jit_state(frame, info);
        let outcome = driver.run_compiled_detailed_with_bridge_keyed(
            green_key,
            frame.next_instr,
            &mut jit_state,
            &env,
            || {},
        );
        if majit_metainterp::majit_log_enabled() {
            let kind = match &outcome {
                DetailedDriverRunOutcome::Finished { .. } => "finished",
                DetailedDriverRunOutcome::Jump { .. } => "jump",
                DetailedDriverRunOutcome::Abort { .. } => "abort",
                DetailedDriverRunOutcome::GuardFailure { .. } => "guard-failure",
            };
            eprintln!(
                "[jit][func-entry] compiled outcome key={} arg0={:?} kind={}",
                green_key,
                debug_first_arg_int(frame),
                kind
            );
        }

        // compile.py:701-717 handle_fail parity.
        if let DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            should_bridge,
            owning_key,
            descr_addr,
            ref raw_values,
            ref exit_layout,
        } = outcome
        {
            match handle_fail(
                frame,
                green_key,
                trace_id,
                fail_index,
                should_bridge,
                owning_key,
                descr_addr,
                exit_layout,
                raw_values,
                info,
            ) {
                HandleFailOutcome::BridgeCompiled => {
                    // Bridge compiled → ContinueRunningNormally → re-enter
                    // compiled code which will follow the new bridge.
                    // Fall through to eval_loop_jit below.
                }
                HandleFailOutcome::ResumeInBlackhole => {
                    match resume_in_blackhole_from_exit_layout(frame, raw_values, exit_layout) {
                        crate::call_jit::BlackholeResult::DoneWithThisFrame(r) => {
                            return Some(r);
                        }
                        crate::call_jit::BlackholeResult::ContinueRunningNormally => {
                            // Fall through to eval_loop_jit
                        }
                        crate::call_jit::BlackholeResult::Failed => {
                            if majit_metainterp::majit_log_enabled() {
                                eprintln!(
                                    "[jit][BUG] blackhole failed key={} — invalidating",
                                    green_key,
                                );
                            }
                            let (driver, _) = driver_pair();
                            driver.invalidate_loop(green_key);
                        }
                    }
                }
            }
        } else {
            match handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
                JitAction::Return(result) => return Some(result),
                JitAction::ContinueRunningNormally | JitAction::Continue => {}
            }
        }

        // After compiled code guard-restored fallback, re-establish the
        // frame's array pointer.
        frame.fix_array_ptrs();
        return None;
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] probe key={} arg0={:?} tracing={}",
            green_key,
            debug_first_arg_int(frame),
            driver.is_tracing(),
        );
    }

    if driver.is_tracing() {
        return None;
    }

    // RPython warmstate.py:446 maybe_compile_and_run: standard counter
    // threshold for ALL functions. No self-recursive boosting.
    let should_trace = driver
        .meta_interp_mut()
        .warm_state_mut()
        .should_trace_function_entry(green_key);
    let count = driver
        .meta_interp()
        .warm_state_ref()
        .function_entry_count(green_key);
    let function_threshold = driver.meta_interp().warm_state_ref().function_threshold();
    let boosted = driver.is_function_boosted(green_key);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] count key={} arg0={:?} count={} boosted={} threshold={}",
            green_key,
            debug_first_arg_int(frame),
            count,
            boosted,
            function_threshold
        );
    }
    if !should_trace {
        return None;
    }
    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame, info);
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .reset_function_entry_count(green_key);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] start tracing key={} arg0={:?} count={} boosted={}",
            green_key,
            debug_first_arg_int(frame),
            count,
            boosted
        );
    }
    driver.force_start_tracing(green_key, frame.next_instr, &mut jit_state, &env);
    if driver.is_tracing() {
        // RPython warmstate.py:429 decay_all_counters:
        // called once after tracing starts to prevent burst compilation.
        driver.meta_interp_mut().warm_state_mut().decay_counters();
    }
    None
}

fn handle_jit_outcome(
    outcome: DetailedDriverRunOutcome,
    _jit_state: &PyreJitState,
    frame: &mut PyFrame,
    _info: &majit_metainterp::virtualizable::VirtualizableInfo,
    green_key: u64,
) -> JitAction {
    match outcome {
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            ..
        } => {
            let (driver, _) = driver_pair();
            let raw_int_result = raw_int_result || driver.has_raw_int_finish(green_key);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][handle-outcome] finished key={} raw_flag={} typed_values={:?}",
                    green_key, raw_int_result, typed_values
                );
            }
            let [value] = typed_values.as_slice() else {
                return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                    "compiled finish did not produce a single object return value",
                )));
            };
            let value = match value {
                majit_ir::Value::Int(raw) => {
                    // RPython parity: top-level finish values are already typed
                    // (MIFrame.make_result_of_lastop / finishframe).  An INT
                    // result register represents a Python int payload, not an
                    // object pointer.  Re-box at the interpreter boundary
                    // regardless of whether majit marked the trace as a
                    // raw-int-finish optimization.
                    let _ = raw_int_result;
                    pyre_object::intobject::w_int_new(*raw)
                }
                majit_ir::Value::Ref(value) => value.as_usize() as pyre_object::PyObjectRef,
                majit_ir::Value::Float(f) => {
                    // Re-box the raw float into a W_FloatObject.
                    pyre_object::floatobject::w_float_new(*f)
                }
                majit_ir::Value::Void => {
                    return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                        "compiled finish produced a void return value",
                    )));
                }
            };
            JitAction::Return(Ok(value))
        }
        DetailedDriverRunOutcome::Jump { .. } => {
            let _ = frame;
            JitAction::Continue
        }
        DetailedDriverRunOutcome::GuardFailure { .. } => {
            // Guard failure handled by handle_fail() before reaching here.
            // If we reach handle_jit_outcome with a GuardFailure, state was
            // already restored — proceed to blackhole resume.
            JitAction::ContinueRunningNormally
        }
        DetailedDriverRunOutcome::Abort { .. } => JitAction::Continue,
    }
}

/// resume.py:1441-1442 allocate_struct(typedescr) → cpu.bh_new(typedescr).
fn allocate_struct(typedescr: &dyn majit_ir::SizeDescr) -> usize {
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_new(typedescr) as usize
}

/// resume.py:1437-1439 allocate_with_vtable(descr) → exec_new_with_vtable(cpu, descr).
fn allocate_with_vtable(descr: &dyn majit_ir::SizeDescr) -> usize {
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_new_with_vtable(descr) as usize
}

/// resume.py:945-956 getvirtual_ptr parity.
///
/// Lazily materializes a virtual from rd_virtuals[vidx].
/// Pattern: check cache → allocate_with_vtable/allocate_struct → cache → setfields.
/// RPython caches the REAL object pointer before filling fields, enabling
/// recursive/shared virtual resolution without NULL placeholders.
fn materialize_virtual_from_rd(
    vidx: usize,
    dead_frame: &[Value],
    num_failargs: i32,
    rd_consts: &[(i64, majit_ir::Type)],
    rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
    virtuals_cache: &mut HashMap<usize, Value>,
) -> Value {
    // resume.py:951: v = self.virtuals_cache.get_ptr(index)
    if let Some(cached) = virtuals_cache.get(&vidx) {
        return cached.clone();
    }
    let Some(virtuals) = rd_virtuals else {
        return Value::Ref(majit_ir::GcRef::NULL);
    };
    let Some(entry) = virtuals.get(vidx) else {
        return Value::Ref(majit_ir::GcRef::NULL);
    };
    // resume.py:1552-1588 decode_* parity.
    fn decode_tagged_fieldnum(
        tagged: i16,
        dead_frame: &[Value],
        num_failargs: i32,
        rd_consts: &[(i64, majit_ir::Type)],
        rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Option<Value> {
        if tagged == majit_ir::resumedata::UNINITIALIZED_TAG {
            return None;
        }
        let (val, tagbits) = majit_metainterp::resume::untag(tagged);
        Some(match tagbits {
            majit_ir::resumedata::TAGBOX => {
                // resume.py:1556-1564: negative index → num + count
                let idx = if val < 0 {
                    (val + num_failargs) as usize
                } else {
                    val as usize
                };
                dead_frame.get(idx).cloned().unwrap_or(Value::Int(0))
            }
            majit_ir::resumedata::TAGINT => Value::Int(val as i64),
            majit_ir::resumedata::TAGCONST => {
                // resume.py:1552-1564: type-aware constant decode
                if tagged == majit_ir::resumedata::NULLREF {
                    return Some(Value::Ref(majit_ir::GcRef::NULL));
                }
                let ci = (val - majit_ir::resumedata::TAG_CONST_OFFSET) as usize;
                let (c, tp) = rd_consts
                    .get(ci)
                    .copied()
                    .unwrap_or((0, majit_ir::Type::Int));
                match tp {
                    majit_ir::Type::Ref => Value::Ref(majit_ir::GcRef(c as usize)),
                    majit_ir::Type::Float => Value::Float(f64::from_bits(c as u64)),
                    _ => Value::Int(c),
                }
            }
            majit_ir::resumedata::TAGVIRTUAL => {
                return Some(materialize_virtual_from_rd(
                    val as usize,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                ));
            }
            _ => Value::Int(0),
        })
    }
    fn box_opt_value(v: &Option<Value>) -> pyre_object::PyObjectRef {
        match v {
            Some(Value::Ref(gc)) => gc.0 as pyre_object::PyObjectRef,
            Some(Value::Int(n)) => pyre_object::intobject::w_int_new(*n),
            Some(Value::Float(f)) => pyre_object::floatobject::w_float_new(*f),
            _ => std::ptr::null_mut(),
        }
    }
    // resume.py:643-760: dispatch by virtual kind.
    match entry {
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            kind, fieldnums, ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            kind, fieldnums, ..
        } => {
            let clear = matches!(entry, majit_ir::RdVirtualInfo::VArrayInfoClear { .. });
            // resume.py:650-670: allocate_array(len, arraydescr, clear)
            let arr_kind = match kind {
                2 => pyre_object::ArrayKind::Float,
                1 => pyre_object::ArrayKind::Int,
                _ => pyre_object::ArrayKind::Ref,
            };
            let array = pyre_object::allocate_array(fieldnums.len(), arr_kind, clear);
            // resume.py:654: cache BEFORE filling — recursive/shared virtuals
            // may reference this vidx during element decoding.
            let result = Value::Ref(majit_ir::GcRef(array as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py:656-670: element kind dispatch + UNINITIALIZED skip.
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue; // resume.py:659: skip UNINITIALIZED
                }
                let v = decode_tagged_fieldnum(
                    fnum,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                if let Some(val) = v {
                    // resume.py:656-670: dispatch by element kind.
                    match val {
                        Value::Float(f) => pyre_object::setarrayitem_float(array, i, f),
                        Value::Int(n) => pyre_object::setarrayitem_int(array, i, n),
                        _ => pyre_object::setarrayitem_ref(array, i, box_opt_value(&Some(val))),
                    }
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            size,
            field_types,
            item_size,
            field_offsets,
            field_sizes,
            fieldnums,
            ..
        } => {
            // resume.py:749: allocate_array(self.size, self.arraydescr, clear=True)
            let fields_per_elem = if *size > 0 {
                fieldnums.len() / *size
            } else {
                0
            };
            let is = if *item_size > 0 {
                *item_size
            } else {
                // Fallback: estimate from fields_per_elem * 8 bytes per field.
                fields_per_elem * 8
            };
            let array = pyre_object::allocate_array_struct(*size, is);
            // resume.py:751: virtuals_cache.set_ptr BEFORE setfields
            let result = Value::Ref(majit_ir::GcRef(array as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py:752-759: nested loop — for i in range(size), for j in range(fielddescrs)
            let mut p = 0;
            for i in 0..*size {
                for j in 0..fields_per_elem {
                    if p >= fieldnums.len() {
                        break;
                    }
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    let v = decode_tagged_fieldnum(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    if let Some(val) = v {
                        // resume.py:757,1520-1529: setinteriorfield(i, array, num, fielddescrs[j])
                        let ft = field_types.get(j).copied().unwrap_or(0);
                        let fo = field_offsets.get(j).copied().unwrap_or(j * 8);
                        let fs = field_sizes.get(j).copied().unwrap_or(8);
                        let raw = match val {
                            Value::Int(i) => i,
                            Value::Float(f) => f.to_bits() as i64,
                            Value::Ref(r) => r.0 as i64,
                            Value::Void => 0,
                        };
                        pyre_object::setinteriorfield(array, i, fo, fs, is, ft, raw);
                    }
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            size,
            offsets,
            entry_sizes,
            fieldnums,
        } => {
            // resume.py:701-708: allocate_raw_buffer + setrawbuffer_item
            let buffer = unsafe {
                std::alloc::alloc_zeroed(
                    std::alloc::Layout::from_size_align(*size, 8)
                        .unwrap_or(std::alloc::Layout::new::<u8>()),
                )
            };
            // resume.py:704: cache BEFORE filling fields.
            let result = Value::Int(buffer as i64);
            virtuals_cache.insert(vidx, result.clone());
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if let Some(val) = decode_tagged_fieldnum(
                    fnum,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                ) {
                    let offset = offsets.get(i).copied().unwrap_or(i * 8);
                    let esz = entry_sizes.get(i).copied().unwrap_or(8);
                    if offset + esz <= *size {
                        let raw = match val {
                            Value::Int(n) => n,
                            Value::Float(f) => f.to_bits() as i64,
                            _ => 0,
                        };
                        unsafe {
                            match esz {
                                1 => *(buffer.add(offset) as *mut u8) = raw as u8,
                                2 => *(buffer.add(offset) as *mut u16) = raw as u16,
                                4 => *(buffer.add(offset) as *mut u32) = raw as u32,
                                _ => *(buffer.add(offset) as *mut i64) = raw,
                            }
                        }
                    }
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            // resume.py:723-727: base_buffer + offset
            if let Some(fnum) = fieldnums.first() {
                if let Some(Value::Int(base)) = decode_tagged_fieldnum(
                    *fnum,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                ) {
                    let result = Value::Int(base + *offset as i64);
                    // resume.py:727: virtuals_cache.set_int(index, buffer)
                    virtuals_cache.insert(vidx, result.clone());
                    return result;
                }
            }
            return Value::Int(0);
        }
        majit_ir::RdVirtualInfo::Empty => {
            panic!("[jit] materialize_virtual: rd_virtuals[{vidx}] is Empty");
        }
        _ => {} // Instance/Struct: fall through
    }
    // Instance/Struct: extract fields for ob_type-based materialization.
    // resume.py:593 fielddescrs + fieldnums
    enum VirtualKind<'a> {
        /// resume.py:612 VirtualInfo — allocate_with_vtable(descr=self.descr).
        Instance {
            descr: &'a Option<majit_ir::DescrRef>,
            known_class: Option<i64>,
        },
        /// resume.py:628 VStructInfo — allocate_struct(self.typedescr).
        Struct {
            typedescr: &'a Option<majit_ir::DescrRef>,
            type_id: u32,
        },
    }
    let (kind, fielddescrs, fieldnums, descr_size) = match entry {
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            known_class,
            fielddescrs,
            fieldnums,
            descr_size,
            ..
        } => (
            VirtualKind::Instance {
                descr,
                known_class: *known_class,
            },
            fielddescrs.as_slice(),
            fieldnums.as_slice(),
            *descr_size,
        ),
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            type_id,
            fielddescrs,
            fieldnums,
            descr_size,
            ..
        } => (
            VirtualKind::Struct {
                typedescr,
                type_id: *type_id,
            },
            fielddescrs.as_slice(),
            fieldnums.as_slice(),
            *descr_size,
        ),
        _ => unreachable!(),
    };

    // resume.py:617-621 VirtualInfo.allocate / resume.py:634-637 VStructInfo.allocate
    //   Phase 1: allocate (allocate_with_vtable or allocate_struct)
    //   Phase 2: virtuals_cache.set_ptr(index, struct)  ← BEFORE setfields
    //   Phase 3: self.setfields(decoder, struct)         ← fields filled AFTER

    // Phase 1: allocate.
    let obj_ptr: usize = match kind {
        // resume.py:617-621: VirtualInfo.allocate(descr) → allocate_with_vtable.
        VirtualKind::Instance { descr, known_class } => {
            let ob_type = known_class.unwrap_or(0);
            let int_type_addr = &pyre_object::INT_TYPE as *const _ as i64;
            let float_type_addr = &pyre_object::FLOAT_TYPE as *const _ as i64;
            if ob_type == int_type_addr {
                let obj = Box::new(pyre_object::intobject::W_IntObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: ob_type as *const pyre_object::pyobject::PyType,
                    },
                    intval: 0,
                });
                Box::into_raw(obj) as usize
            } else if ob_type == float_type_addr {
                let obj = Box::new(pyre_object::floatobject::W_FloatObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: ob_type as *const pyre_object::pyobject::PyType,
                    },
                    floatval: 0.0,
                });
                Box::into_raw(obj) as usize
            } else if ob_type != 0 {
                // resume.py:619: allocate_with_vtable(descr=self.descr).
                if let Some(d) = descr {
                    allocate_with_vtable(
                        d.as_size_descr()
                            .expect("VirtualInfo descr must be SizeDescr"),
                    )
                } else {
                    // Fallback: no live descr (decoded from EncodedResumeData).
                    debug_assert!(descr_size > 0, "VirtualInfo must have descr_size");
                    let size = if descr_size > 0 { descr_size } else { 16 };
                    let fallback =
                        majit_ir::make_size_descr_with_vtable(0, size, 0, ob_type as usize);
                    allocate_with_vtable(fallback.as_size_descr().unwrap())
                }
            } else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit] materialize_virtual: vidx={vidx} Instance with no known_class",
                    );
                }
                return Value::Ref(majit_ir::GcRef::NULL);
            }
        }
        // resume.py:633-637: VStructInfo.allocate → allocate_struct.
        // resume.py:618-619: VirtualInfo.allocate → allocate_with_vtable.
        // VirtualKind::Struct may contain objects with vtable that
        // should be VirtualInfo — use vtable presence to dispatch
        // until VirtualKind classification is fixed upstream.
        VirtualKind::Struct { typedescr, type_id } => {
            if let Some(td) = typedescr {
                let sd = td
                    .as_size_descr()
                    .expect("VStruct typedescr must be SizeDescr");
                if sd.vtable() != 0 {
                    allocate_with_vtable(sd)
                } else {
                    allocate_struct(sd)
                }
            } else if descr_size > 0 {
                let fallback = majit_ir::make_size_descr_full(0, descr_size, type_id);
                let sd = fallback.as_size_descr().unwrap();
                if sd.vtable() != 0 {
                    allocate_with_vtable(sd)
                } else {
                    allocate_struct(sd)
                }
            } else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!("[jit] materialize_virtual: vidx={vidx} Struct with no typedescr",);
                }
                return Value::Ref(majit_ir::GcRef::NULL);
            }
        }
    };

    // Phase 2: cache REAL object pointer BEFORE setting fields.
    // resume.py:620: decoder.virtuals_cache.set_ptr(index, struct)
    let obj_ref = Value::Ref(majit_ir::GcRef(obj_ptr));
    virtuals_cache.insert(vidx, obj_ref.clone());

    // Phase 3: setfields — decode each field and write to object.
    // resume.py:596-603: for each fielddescr, decoder.setfield(struct, num, descr)
    let is_instance = matches!(kind, VirtualKind::Instance { .. });
    match kind {
        VirtualKind::Instance { known_class, .. }
            if known_class == Some(&pyre_object::INT_TYPE as *const _ as i64) =>
        {
            // W_IntObject fast path: find intval field (offset 8).
            // fielddescrs may include ob_type (offset 0) first.
            let intval_idx = fielddescrs
                .iter()
                .position(|fd| fd.offset == 8)
                .unwrap_or(0);
            if let Some(&tagged) = fieldnums.get(intval_idx) {
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let intval = match val {
                    Value::Int(n) => n,
                    Value::Ref(gc) if !gc.is_null() => unsafe {
                        pyre_object::intobject::w_int_get_value(gc.0 as pyre_object::PyObjectRef)
                    },
                    _ => 0,
                };
                unsafe {
                    (*(obj_ptr as *mut pyre_object::intobject::W_IntObject)).intval = intval;
                }
            }
        }
        VirtualKind::Instance { known_class, .. }
            if known_class == Some(&pyre_object::FLOAT_TYPE as *const _ as i64) =>
        {
            // W_FloatObject fast path: find floatval field (offset 8).
            let floatval_idx = fielddescrs
                .iter()
                .position(|fd| fd.offset == 8)
                .unwrap_or(0);
            if let Some(&tagged) = fieldnums.get(floatval_idx) {
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let floatval = match val {
                    Value::Float(f) => f,
                    Value::Int(bits) => f64::from_bits(bits as u64),
                    _ => 0.0,
                };
                unsafe {
                    (*(obj_ptr as *mut pyre_object::floatobject::W_FloatObject)).floatval =
                        floatval;
                }
            }
        }
        _ => {
            // resume.py:598-602 AbstractVirtualStructInfo.setfields:
            // for each fielddescr, decoder.setfield(struct, num, descr)
            for (i, &tagged) in fieldnums.iter().enumerate() {
                if tagged == majit_ir::resumedata::NULLREF
                    || tagged == majit_ir::resumedata::UNINITIALIZED_TAG
                {
                    continue;
                }
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let raw = match val {
                    Value::Int(n) => n,
                    Value::Float(f) => f.to_bits() as i64,
                    Value::Ref(gc) => gc.0 as i64,
                    _ => 0,
                };
                let Some(descr) = fielddescrs.get(i) else {
                    debug_assert!(false, "fielddescrs missing for field {}", i);
                    continue;
                };
                // Skip vtable slot (offset 0) for Instance — already set by allocate_with_vtable.
                if descr.offset == 0 && is_instance {
                    continue;
                }
                unsafe {
                    let addr = (obj_ptr as *mut u8).add(descr.offset);
                    match descr.field_type {
                        majit_ir::Type::Ref => {
                            let p = match val {
                                Value::Ref(gc) => gc.0 as i64,
                                Value::Int(n) => n,
                                _ => 0,
                            };
                            std::ptr::write(addr as *mut i64, p);
                        }
                        majit_ir::Type::Float => {
                            let bits = match val {
                                Value::Float(f) => f.to_bits(),
                                Value::Int(n) => n as u64,
                                _ => 0,
                            };
                            std::ptr::write(addr as *mut u64, bits);
                        }
                        _ => match descr.field_size {
                            1 => std::ptr::write(addr, raw as u8),
                            2 => std::ptr::write(addr as *mut u16, raw as u16),
                            4 => std::ptr::write(addr as *mut u32, raw as u32),
                            _ => std::ptr::write(addr as *mut i64, raw),
                        },
                    }
                }
            }
        }
    }
    obj_ref
}

/// resume.py:1552-1588 ResumeDataDirectReader decode_int/decode_ref parity.
///
/// Decode a tagged value from rd_numb into a concrete Value.
/// Handles TAGBOX (deadframe), TAGINT (inline), TAGCONST (constant pool),
/// and TAGVIRTUAL (lazy materialization via materialize_virtual_from_rd).
fn decode_tagged_value(
    tagged: i16,
    dead_frame: &[Value],
    num_failargs: i32,
    rd_consts: &[(i64, majit_ir::Type)],
    rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
    virtuals_cache: &mut HashMap<usize, Value>,
) -> Value {
    let (val, tagbits) = majit_metainterp::resume::untag(tagged);
    match tagbits {
        majit_metainterp::resume::TAGBOX => {
            let idx = if val < 0 {
                (val + num_failargs) as usize
            } else {
                val as usize
            };
            dead_frame.get(idx).cloned().unwrap_or(Value::Int(0))
        }
        majit_metainterp::resume::TAGINT => Value::Int(val as i64),
        majit_metainterp::resume::TAGCONST => {
            let (c, tp) = rd_consts
                .get((val - majit_metainterp::resume::TAG_CONST_OFFSET) as usize)
                .copied()
                .unwrap_or((0, majit_ir::Type::Int));
            match tp {
                majit_ir::Type::Ref => Value::Ref(majit_ir::GcRef(c as usize)),
                majit_ir::Type::Float => Value::Float(f64::from_bits(c as u64)),
                _ => Value::Int(c),
            }
        }
        majit_metainterp::resume::TAGVIRTUAL => {
            // resume.py:1572: decode_ref(TAGVIRTUAL) → getvirtual_ptr(num)
            materialize_virtual_from_rd(
                val as usize,
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            )
        }
        _ => Value::Int(0),
    }
}

fn decode_virtual_int_payload(value: &Value) -> i64 {
    match value {
        Value::Int(n) => *n,
        Value::Ref(gc) if !gc.is_null() => unsafe {
            pyre_object::intobject::w_int_get_value(gc.0 as pyre_object::PyObjectRef)
        },
        _ => 0,
    }
}

fn decode_virtual_float_payload_bits(value: &Value) -> i64 {
    match value {
        Value::Float(f) => f.to_bits() as i64,
        Value::Int(n) => *n,
        Value::Ref(gc) if !gc.is_null() => unsafe {
            pyre_object::floatobject::w_float_get_value(gc.0 as pyre_object::PyObjectRef).to_bits()
                as i64
        },
        _ => 0,
    }
}

fn decode_exit_layout_values(raw_values: &[i64], layout: &CompiledExitLayout) -> Vec<Value> {
    layout
        .exit_types
        .iter()
        .enumerate()
        .map(|(index, tp)| {
            let raw = raw_values.get(index).copied().unwrap_or(0);
            match tp {
                majit_ir::Type::Int => Value::Int(raw),
                majit_ir::Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                majit_ir::Type::Float => Value::Float(f64::from_bits(raw as u64)),
                majit_ir::Type::Void => Value::Void,
            }
        })
        .collect()
}

/// Phase A: decode rd_numb + materialize virtuals + restore frame state.
/// RPython: this corresponds to rebuild_from_resumedata (resume.py:1042)
/// which decodes the deadframe into typed values and writes them to the
/// virtualizable/MIFrames. Returns typed values for Phase B and resume PC.
pub(crate) fn decode_and_restore_guard_failure(
    jit_state: &mut PyreJitState,
    meta: &crate::jit::state::PyreMeta,
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) -> Option<(Vec<Value>, usize)> {
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] exit-layout trace_id={} fail_idx={} source_op={:?} rd_numb={} recovery={} resume_layout={}",
            exit_layout.trace_id,
            exit_layout.fail_index,
            exit_layout.source_op_index,
            exit_layout.rd_numb.as_ref().map(|v| v.len()).unwrap_or(0),
            exit_layout.recovery_layout.is_some(),
            exit_layout.resume_layout.is_some(),
        );
    }
    if majit_metainterp::majit_log_enabled() {
        let nraw = raw_values.len();
        let slots: Vec<String> = (0..nraw)
            .map(|i| format!("{:#x}", raw_values[i] as usize))
            .collect();
        eprintln!(
            "[jit] guard-fail: fail_idx={} types={:?} raw_len={} raw=[{}]",
            exit_layout.fail_index,
            exit_layout.exit_types,
            nraw,
            slots.join(", ")
        );
    }
    // resume.py:1042 rebuild_from_resumedata: decode rd_numb into typed values.
    let mut typed = {
        let rd_numb = exit_layout.rd_numb.as_deref().unwrap_or(&[]);
        let rd_consts = exit_layout.rd_consts.as_deref().unwrap_or(&[]);
        if rd_numb.is_empty() {
            decode_exit_layout_values(raw_values, exit_layout)
        } else {
            rebuild_typed_from_rd_numb(raw_values, rd_numb, rd_consts, exit_layout)
        }
    };
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] rebuilt typed prefix: {:?}",
            typed.iter().take(6).collect::<Vec<_>>()
        );
    }
    // resume.py:945/993: materialize virtuals + replay pending fields.
    rebuild_state_after_failure_with_exit_layout(&mut typed, raw_values, exit_layout);
    replay_pending_fields(&typed, exit_layout);

    // resume.py:1312 blackhole_from_resumedata parity:
    // Build per-frame ResumedFrame chain from rd_numb decoded frames.
    let resumed_frames = {
        let rd_numb = exit_layout.rd_numb.as_deref().unwrap_or(&[]);
        let rd_consts = exit_layout.rd_consts.as_deref().unwrap_or(&[]);
        if !rd_numb.is_empty() {
            build_resumed_frames(raw_values, rd_numb, rd_consts, exit_layout)
        } else {
            // Fallback for guards with empty rd_numb: single frame from typed.
            // typed still has old format [frame, ni, vsd, locals..., stack...]
            let frame_ptr = match typed.first() {
                Some(Value::Ref(r)) => r.as_usize() as *mut pyre_interpreter::pyframe::PyFrame,
                Some(Value::Int(v)) => *v as *mut pyre_interpreter::pyframe::PyFrame,
                _ => std::ptr::null_mut(),
            };
            let py_pc = match typed.get(1) {
                Some(Value::Int(v)) => *v as usize,
                _ => 0,
            };
            let vsd = match typed.get(2) {
                Some(Value::Int(v)) => *v as usize,
                _ => 0,
            };
            let code = if !frame_ptr.is_null() {
                unsafe { (*frame_ptr).code }
            } else {
                std::ptr::null()
            };
            let slot_values = if typed.len() > 3 {
                typed[3..].to_vec()
            } else {
                Vec::new()
            };
            vec![crate::call_jit::ResumedFrame {
                code,
                py_pc,
                rd_numb_pc: None,
                frame_ptr,
                vsd,
                values: slot_values,
            }]
        }
    };
    LAST_GUARD_FRAMES.with(|c| *c.borrow_mut() = Some(resumed_frames));

    // virtualizable.py:126: write fields from resumedata to frame.
    let restored = jit_state.restore_guard_failure_values(meta, &typed, &ExceptionState::default());
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] guard-fail restored: ni={} vsd={}",
            jit_state.next_instr(),
            jit_state.valuestackdepth(),
        );
    }

    if restored {
        Some((typed, jit_state.next_instr()))
    } else {
        None
    }
}

/// compile.py:710 resume_in_blackhole(deadframe) →
/// resume.py:1312 blackhole_from_resumedata(deadframe) parity:
/// Build LAST_GUARD_FRAMES directly from deadframe.
/// RPython does NOT call rebuild_from_resumedata (guard restore)
/// before the blackhole path — the blackhole chain consumes
/// deadframe values directly via consume_one_section.
pub(crate) fn build_blackhole_frames_from_deadframe(
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) {
    let rd_numb = exit_layout.rd_numb.as_deref().unwrap_or(&[]);
    let rd_consts = exit_layout.rd_consts.as_deref().unwrap_or(&[]);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][deadframe] fail_index={} rd_numb_len={} exit_types={}",
            exit_layout.fail_index,
            rd_numb.len(),
            exit_layout.exit_types.len(),
        );
    }
    let resumed_frames = if !rd_numb.is_empty() {
        build_resumed_frames(raw_values, rd_numb, rd_consts, exit_layout)
    } else {
        let typed = decode_exit_layout_values(raw_values, exit_layout);
        build_blackhole_frames_fallback(&typed)
    };
    LAST_GUARD_FRAMES.with(|c| *c.borrow_mut() = Some(resumed_frames));
}

/// resume.py:1312 blackhole_from_resumedata parity:
/// Build resumed frames directly from deadframe without thread-local storage.
/// Used by call_assembler_guard_failure path where the thread-local may be
/// consumed by try_function_entry_jit's resume path.
pub(crate) fn build_resumed_frames_from_deadframe(
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) -> Vec<crate::call_jit::ResumedFrame> {
    let rd_numb = exit_layout.rd_numb.as_deref().unwrap_or(&[]);
    let rd_consts = exit_layout.rd_consts.as_deref().unwrap_or(&[]);
    if !rd_numb.is_empty() {
        build_resumed_frames(raw_values, rd_numb, rd_consts, exit_layout)
    } else {
        let typed = decode_exit_layout_values(raw_values, exit_layout);
        build_blackhole_frames_fallback(&typed)
    }
}

fn build_blackhole_frames_fallback(typed: &[Value]) -> Vec<crate::call_jit::ResumedFrame> {
    let frame_ptr = match typed.first() {
        Some(Value::Ref(r)) => r.as_usize() as *mut pyre_interpreter::pyframe::PyFrame,
        Some(Value::Int(v)) => *v as *mut pyre_interpreter::pyframe::PyFrame,
        _ => std::ptr::null_mut(),
    };
    let py_pc = match typed.get(1) {
        Some(Value::Int(v)) => *v as usize,
        _ => 0,
    };
    let code = if !frame_ptr.is_null() {
        unsafe { (*frame_ptr).code }
    } else {
        std::ptr::null()
    };
    // typed still has old format [frame, ni, vsd, ...]. Extract vsd and strip header.
    let vsd = match typed.get(2) {
        Some(Value::Int(v)) => *v as usize,
        _ => 0,
    };
    let slot_values = if typed.len() > 3 {
        typed[3..].to_vec()
    } else {
        Vec::new()
    };
    vec![crate::call_jit::ResumedFrame {
        code,
        py_pc,
        rd_numb_pc: None,
        frame_ptr,
        vsd,
        values: slot_values,
    }]
}

/// Decode rd_numb to produce typed values via
/// `majit_ir::resumedata::rebuild_from_numbering`. Each slot is TAGBOX
/// (deadframe), TAGCONST (constant), TAGINT (small int), or TAGVIRTUAL
/// (virtual to materialize). Single-frame only (no per-jitcode liveness).
fn rebuild_typed_from_rd_numb(
    raw_values: &[i64],
    rd_numb: &[u8],
    rd_consts: &[(i64, majit_ir::Type)],
    exit_layout: &CompiledExitLayout,
) -> Vec<Value> {
    use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};

    let (_num_failargs, vable_values, _vref_values, frames) =
        rebuild_from_numbering(rd_numb, rd_consts);

    // resume.py:1045 consume_vref_and_vable_boxes parity.
    // vable_array format: [frame_ptr, ni, vsd, locals..., stack...]
    // (opencoder.py:722 moves virtualizable_ptr to front).
    if majit_metainterp::majit_log_enabled() && !vable_values.is_empty() {
        eprintln!(
            "[jit] guard-fail: vable_values={} items: {:?}",
            vable_values.len(),
            vable_values.iter().take(6).collect::<Vec<_>>()
        );
    }

    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    let mut virtuals_cache: HashMap<usize, Value> = HashMap::new();

    // resume.py:1083 + pyjitpl.py:3400-3428 parity:
    // Decode vable_values into typed prefix [frame_ptr, ni, vsd, locals..., stack...].
    // In RPython, virtualizable_boxes are restored first, then synchronize_virtualizable
    // writes them back to the actual frame object.
    fn decode_rv(
        rv: &majit_ir::resumedata::RebuiltValue,
        dead_frame_typed: &[Value],
        exit_layout: &CompiledExitLayout,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Value {
        use majit_ir::resumedata::RebuiltValue;
        match rv {
            RebuiltValue::Box(idx) => dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0)),
            RebuiltValue::Int(i) => Value::Int(*i as i64),
            RebuiltValue::Const(c, tp) => match tp {
                majit_ir::Type::Int => Value::Int(*c),
                majit_ir::Type::Ref => Value::Ref(majit_ir::GcRef(*c as usize)),
                majit_ir::Type::Float => Value::Float(f64::from_bits(*c as u64)),
                _ => Value::Int(*c),
            },
            RebuiltValue::Virtual(vidx) => materialize_virtual_from_rd(
                *vidx,
                dead_frame_typed,
                exit_layout.exit_types.len() as i32,
                exit_layout.rd_consts.as_deref().unwrap_or(&[]),
                exit_layout.rd_virtuals.as_deref(),
                virtuals_cache,
            ),
            _ => Value::Int(0),
        }
    }
    // resume.py:1042-1057 rebuild_from_resumedata parity:
    // RPython produces TWO streams:
    //   1. virtualizable_boxes (consume_vref_and_vable → synchronize_virtualizable)
    //   2. frame registers (consume_boxes per frame)
    // pyjitpl.py:3419-3430: virtualizable_boxes restored, then
    // synchronize_virtualizable writes them back to the heap.
    // Frame registers fill frame.registers_i/r/f independently.

    // vable_values = [frame_ptr(0), ni(1), code(2), vsd(3), ns(4), array...]
    // fail_args header = [frame, ni, vsd] (3-slot; code/ns immutable → skip)
    let header: Vec<Value> = if vable_values.len() >= 5 {
        vec![
            decode_rv(
                &vable_values[0],
                &dead_frame_typed,
                exit_layout,
                &mut virtuals_cache,
            ),
            decode_rv(
                &vable_values[1], // ni
                &dead_frame_typed,
                exit_layout,
                &mut virtuals_cache,
            ),
            // skip vable_values[2]=code (immutable)
            decode_rv(
                &vable_values[3], // vsd
                &dead_frame_typed,
                exit_layout,
                &mut virtuals_cache,
            ),
            // skip vable_values[4]=namespace (immutable)
        ]
    } else {
        Vec::new()
    };

    // resume.py:1049-1056: rebuild_from_resumedata iterates all frames
    // via newframe()+consume_boxes(). For guard-failure restore into JIT
    // state (restore_guard_failure_values), only the outermost frame's
    // values matter — inner frames are handled by build_resumed_frames →
    // resume_in_blackhole. rd_numb frames are innermost-first; last = outermost.
    let mut typed = header;
    if let Some(outermost) = frames.last() {
        _prepare_next_section(
            outermost,
            raw_values,
            &dead_frame_typed,
            exit_layout,
            &mut typed,
            &mut virtuals_cache,
        );
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] guard-fail: rd_numb decoded {} slots from {} frame(s)",
            typed.len(),
            frames.len(),
        );
    }

    typed
}

/// Decode rd_numb into per-frame ResumedFrame chain via
/// `majit_ir::resumedata::rebuild_from_numbering`.
/// Single-frame only (RPython's blackhole_from_resumedata uses
/// per-jitcode liveness for multi-frame decode).
fn build_resumed_frames(
    raw_values: &[i64],
    rd_numb: &[u8],
    rd_consts: &[(i64, majit_ir::Type)],
    exit_layout: &CompiledExitLayout,
) -> Vec<crate::call_jit::ResumedFrame> {
    use majit_ir::resumedata::rebuild_from_numbering;

    // resume.py:1049 parity: consume_boxes(f.get_current_position_info(), ...)
    // uses per-jitcode liveness (all_liveness from codewriter) to split
    // multi-frame sections. Requires encoder and decoder to use the SAME
    // liveness source. Currently pyre's encoder uses snapshot-based data
    // while the JitCode liveness is a separate system. Until they are
    // unified, use single-frame fallback (None) which works for all
    // current single-function traces.
    // TODO: unify rd_numb encoding with JitCode liveness for multi-frame.
    let (_num_failargs, vable_values, _vref_values, frames) =
        rebuild_from_numbering(rd_numb, rd_consts);

    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][resume] exit_types={:?} dead_frame={:?} vable={} frames={}",
            exit_layout.exit_types,
            dead_frame_typed,
            vable_values.len(),
            frames.len()
        );
    }
    let mut virtuals_cache: HashMap<usize, Value> = HashMap::new();

    // resume.py:1045 consume_vref_and_vable parity:
    // Reconstruct header [frame_ptr, ni, vsd] from vable_values.
    fn resolve_rebuilt_value(
        rv: &majit_ir::resumedata::RebuiltValue,
        dead_frame_typed: &[Value],
        exit_layout: &CompiledExitLayout,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Value {
        use majit_ir::resumedata::RebuiltValue;
        match rv {
            RebuiltValue::Box(idx) => dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0)),
            RebuiltValue::Int(i) => Value::Int(*i as i64),
            RebuiltValue::Const(c, tp) => match tp {
                majit_ir::Type::Int => Value::Int(*c),
                majit_ir::Type::Ref => Value::Ref(majit_ir::GcRef(*c as usize)),
                _ => Value::Int(*c),
            },
            RebuiltValue::Virtual(vidx) => materialize_virtual_from_rd(
                *vidx,
                dead_frame_typed,
                exit_layout.exit_types.len() as i32,
                exit_layout.rd_consts.as_deref().unwrap_or(&[]),
                exit_layout.rd_virtuals.as_deref(),
                virtuals_cache,
            ),
            _ => Value::Int(0),
        }
    }
    // resume.py:1045 consume_vref_and_vable: vable header is extracted
    // AFTER _prepare_next_section materializes virtuals. The post-section
    // block below is the authoritative extraction. vable_values is always
    // non-empty for guards with complete resume data (resume.py:397 asserts
    // resume_position >= 0). The no-snapshot fallback in store_final_boxes_in_guard
    // now encodes fail_args[0..3] as vable_array to maintain this invariant.

    let mut all_values: Vec<Vec<Value>> = Vec::with_capacity(frames.len());
    for frame in &frames {
        // RPython parity: no header prepend. Values = slot registers only.
        let mut values = Vec::new();
        _prepare_next_section(
            frame,
            raw_values,
            &dead_frame_typed,
            exit_layout,
            &mut values,
            &mut virtuals_cache,
        );
        all_values.push(values);
    }
    // RPython parity: _prepare_next_section + materialize_virtual_from_rd
    // is the authoritative path for virtual materialization.
    // rebuild_state_after_failure_with_exit_layout (recovery_layout) is a
    // pyre-specific legacy path that is NOT in RPython and can produce
    // stale values. Disabled for RPython parity.
    // resume.py:993 _prepare_pendingfields: apply ONCE for the whole reader.
    // No header — values = slot registers only.
    if let Some(first_values) = all_values.first() {
        replay_pending_fields(first_values, exit_layout);
    }

    // opencoder.py:722 _list_of_boxes_virtualizable: snapshot reorders
    // virtualizable_ptr from end to front.
    // vable_values = [frame_ptr(0), ni(1), code(2), vsd(3), ns(4), array...]
    let (vable_frame_ptr, _vable_ni, vable_vsd) = if vable_values.len() >= 5 {
        let frame_val = resolve_rebuilt_value(
            &vable_values[0],
            &dead_frame_typed,
            exit_layout,
            &mut virtuals_cache,
        );
        let ni_val = resolve_rebuilt_value(
            &vable_values[1], // ni
            &dead_frame_typed,
            exit_layout,
            &mut virtuals_cache,
        );
        // skip vable_values[2]=code (immutable)
        let vsd_val = resolve_rebuilt_value(
            &vable_values[3], // vsd
            &dead_frame_typed,
            exit_layout,
            &mut virtuals_cache,
        );
        // skip vable_values[4]=namespace (immutable)
        let fp = match &frame_val {
            Value::Ref(r) => r.as_usize() as *mut pyre_interpreter::pyframe::PyFrame,
            Value::Int(v) => *v as *mut pyre_interpreter::pyframe::PyFrame,
            _ => std::ptr::null_mut(),
        };
        let ni = match &ni_val {
            Value::Int(v) => *v as usize,
            _ => 0,
        };
        let vsd = match &vsd_val {
            Value::Int(v) => *v as usize,
            _ => 0,
        };
        (fp, ni, vsd)
    } else {
        (std::ptr::null_mut(), 0, 0)
    };

    // TODO: resume.py:1399 consume_vable_info writes ALL vable fields
    // back to the virtualizable (= PyFrame in pyre) BEFORE the blackhole
    // runs. This is not yet implemented — blocked by OpRef dedup aliasing
    // between vable and frame sections (see KNOWN DEVIATION in
    // resumedata.rs:225). Until resolved, blackhole resume may produce
    // null locals when frame section has partial liveness.

    let mut result = Vec::with_capacity(frames.len());
    for (idx, (frame, values)) in frames.iter().zip(all_values.into_iter()).enumerate() {
        // frame_ptr from vable for single-frame or outermost (caller).
        let frame_ptr = if frames.len() == 1 || idx == frames.len() - 1 {
            vable_frame_ptr
        } else {
            std::ptr::null_mut()
        };
        // resume.py:1338 read_jitcode_pos_pc parity:
        // py_pc comes from rd_numb frame header (frame.pc = orgpc).
        // pc=0 is valid (function start). pc=-1 = no-snapshot sentinel.
        let py_pc = if frame.pc >= 0 {
            frame.pc as usize
        } else {
            // No-snapshot guard: fall back to vable ni.
            _vable_ni
        };
        // resume.py:1339 jitcodes[jitcode_pos]:
        // Outermost frame: code from frame_ptr. Inner frames: code from
        // jitcode_index registry (no live PyFrame for inlined calls).
        let code = if !frame_ptr.is_null() {
            unsafe { (*frame_ptr).code }
        } else {
            pyre_jit_trace::state::code_for_jitcode_index(frame.jitcode_index)
                .unwrap_or(std::ptr::null())
        };
        // Per-frame VSD: outermost uses vable_vsd, inner frames derive
        // from their code's nlocals + snapshot stack depth.
        let vsd = if frames.len() == 1 || idx == frames.len() - 1 {
            vable_vsd
        } else if !code.is_null() {
            let nlocals = unsafe { &*code }.varnames.len();
            nlocals + values.len().saturating_sub(nlocals)
        } else {
            values.len()
        };
        result.push(crate::call_jit::ResumedFrame {
            code,
            py_pc,
            rd_numb_pc: if frame.pc >= 0 {
                Some(frame.pc as usize)
            } else {
                None
            },
            frame_ptr,
            vsd,
            values,
        });
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] build_resumed_frames: {} frame(s) from rd_numb",
            result.len()
        );
    }

    result
}

/// resume.py:1017-1026 _prepare_next_section: decode one frame's slots
/// from rd_numb tagged values into typed Value vector.
fn _prepare_next_section(
    frame: &majit_ir::resumedata::RebuiltFrame,
    raw_values: &[i64],
    dead_frame_typed: &[Value],
    exit_layout: &CompiledExitLayout,
    typed: &mut Vec<Value>,
    virtuals_cache: &mut HashMap<usize, Value>,
) {
    use majit_ir::resumedata::RebuiltValue;
    let rd_consts = exit_layout.rd_consts.as_deref().unwrap_or(&[]);
    let rd_virtuals = exit_layout.rd_virtuals.as_deref();
    let num_failargs = exit_layout.exit_types.len() as i32;
    for val in &frame.values {
        typed.push(match val {
            RebuiltValue::Box(idx) => dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0)),
            RebuiltValue::Const(c, tp) => match tp {
                majit_ir::Type::Int => Value::Int(*c),
                majit_ir::Type::Ref => Value::Ref(majit_ir::GcRef(*c as usize)),
                majit_ir::Type::Float => Value::Float(f64::from_bits(*c as u64)),
                majit_ir::Type::Void => Value::Void,
            },
            RebuiltValue::Int(i) => Value::Int(*i as i64),
            // resume.py:1572: decode_ref(TAGVIRTUAL) → getvirtual_ptr(num)
            RebuiltValue::Virtual(vidx) => materialize_virtual_from_rd(
                *vidx,
                dead_frame_typed,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            ),
            // resume.py:131 UNINITIALIZED parity: dead/uninitialized slots
            // stay at default. In pyre, PY_NULL via Value::Void.
            RebuiltValue::Unassigned => Value::Void,
        });
    }
}

/// Guard failure recovery: reconstruct virtual objects from their
/// field values stored as extra fail_args after null (NONE) slots.
///
/// When the optimizer places a virtual in fail_args, it sets the
/// resume.py:945/993 parity: virtual materialization via rd_virtuals.
/// Called from Cranelift's guard failure handler via TLS callback.
/// RPython uses rd_virtuals/rd_pendingfields for precise materialization.
/// No heuristic pair decode — recovery_layout handles everything.
fn rebuild_state_after_failure(_outputs: &mut [i64], _types: &[majit_ir::Type]) {
    // RPython: materialization happens in rebuild_from_resumedata via
    // getvirtual_ptr (resume.py:945) and _prepare_pendingfields (resume.py:993).
    // The Cranelift callback is a no-op; precise materialization is done in
    // rebuild_state_after_failure_with_exit_layout using recovery_layout.
}

/// virtual's slot to NONE and appends field values (ob_type, intval).
/// On guard failure, we detect contiguous null Ref slots at the end
/// of the locals/stack region and pair them with trailing Int fields.
///
/// resume.py:993-1007 _prepare_pendingfields: replay deferred field writes.
///
/// After virtual materialization, pending SETFIELD_GC/SETARRAYITEM_GC
/// ops stored in rd_pendingfields are replayed on the materialized objects.
/// This ensures lazy field writes that were deferred during optimization
/// take effect when the guard fires.
fn replay_pending_fields(typed: &[Value], exit_layout: &CompiledExitLayout) {
    let Some(ref recovery) = exit_layout.recovery_layout else {
        return;
    };
    if recovery.pending_field_layouts.is_empty() {
        return;
    }

    let resolve_value = |src: &majit_backend::ExitValueSourceLayout| -> Option<i64> {
        match src {
            majit_backend::ExitValueSourceLayout::ExitValue(idx) => {
                typed.get(*idx).map(|v| match v {
                    Value::Int(i) => *i,
                    Value::Float(f) => f.to_bits() as i64,
                    Value::Ref(r) => r.0 as i64,
                    _ => 0,
                })
            }
            majit_backend::ExitValueSourceLayout::Constant(c) => Some(*c),
            _ => None,
        }
    };

    for pf in &recovery.pending_field_layouts {
        let Some(target_ptr) = resolve_value(&pf.target) else {
            continue;
        };
        let Some(value_raw) = resolve_value(&pf.value) else {
            continue;
        };
        if target_ptr == 0 {
            continue; // null target — skip
        }
        // resume.py:1003-1007 _prepare_pendingfields parity:
        //   if itemindex < 0: setfield(struct, fieldnum, descr)
        //   else:             setarrayitem(struct, itemindex, fieldnum, descr)
        //
        // resume.py:1509-1518 setfield / 1520-1530 setarrayitem:
        //   descr.is_pointer_field() → bh_setfield_gc_r / bh_setarrayitem_gc_r
        //   descr.is_float_field()   → bh_setfield_gc_f / bh_setarrayitem_gc_f
        //   else                     → bh_setfield_gc_i / bh_setarrayitem_gc_i
        let addr = if pf.is_array_item {
            // setarrayitem: base + offset + item_index * item_size
            let item_index = pf.item_index.unwrap_or(0);
            target_ptr as usize + pf.field_offset + item_index * pf.field_size
        } else {
            // setfield: base + offset
            target_ptr as usize + pf.field_offset
        };
        unsafe {
            match pf.field_type {
                majit_ir::Type::Ref => {
                    // bh_setfield_gc_r: store pointer
                    std::ptr::write(addr as *mut usize, value_raw as usize);
                }
                majit_ir::Type::Float => {
                    // bh_setfield_gc_f: store f64
                    std::ptr::write(addr as *mut u64, value_raw as u64);
                }
                majit_ir::Type::Int | majit_ir::Type::Void => {
                    // bh_setfield_gc_i: store integer (size-aware)
                    match pf.field_size {
                        8 => std::ptr::write(addr as *mut i64, value_raw),
                        4 => std::ptr::write(addr as *mut i32, value_raw as i32),
                        2 => std::ptr::write(addr as *mut i16, value_raw as i16),
                        1 => std::ptr::write(addr as *mut u8, value_raw as u8),
                        _ => std::ptr::write(addr as *mut i64, value_raw),
                    }
                }
            }
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit] replay_pending_field: type={:?} offset={} size={} target={:#x} value={:#x}",
                pf.field_type,
                pf.field_offset,
                pf.field_size,
                target_ptr as usize,
                value_raw as usize
            );
        }
    }
}

/// resume.py:945/993 parity: materialize virtuals via rd_virtuals.
/// RPython uses recovery_layout (rd_virtuals + rd_pendingfields)
/// for precise materialization. No heuristic pair decode, no w_none().
fn rebuild_state_after_failure_with_exit_layout(
    typed: &mut Vec<Value>,
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) {
    if let Some(ref recovery) = exit_layout.recovery_layout {
        if !recovery.virtual_layouts.is_empty() {
            rebuild_state_after_failure_from_recovery_layout(
                typed,
                raw_values,
                exit_layout,
                recovery,
            );
        }
    }
}

/// resume.py parity: materialize virtuals using recovery layout (rd_virtuals).
/// Field values are read from raw_values (deadframe) since rd_numb-decoded
/// typed array only contains frame slots, not the appended virtual fields.
fn rebuild_state_after_failure_from_recovery_layout(
    typed: &mut Vec<Value>,
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
    recovery: &majit_backend::ExitRecoveryLayout,
) -> bool {
    let w_int_type = &pyre_object::pyobject::INT_TYPE as *const _ as usize;
    let w_float_type = &pyre_object::pyobject::FLOAT_TYPE as *const _ as usize;

    // resume.py parity: resolve field values from deadframe (raw_values),
    // not from rd_numb-decoded typed array. Virtual field values live at
    // deadframe positions beyond the frame slots.
    // resume.py:1252 parity: resolve field values, including nested virtuals.
    let resolve_value = |src: &majit_backend::ExitValueSourceLayout,
                         materialized: &[usize]|
     -> Option<i64> {
        match src {
            majit_backend::ExitValueSourceLayout::ExitValue(idx) => raw_values.get(*idx).copied(),
            majit_backend::ExitValueSourceLayout::Constant(c) => Some(*c),
            majit_backend::ExitValueSourceLayout::Virtual(vidx) => {
                materialized.get(*vidx).map(|&ptr| ptr as i64)
            }
            _ => None,
        }
    };

    // Materialize each virtual described in the recovery layout.
    let mut materialized: Vec<usize> = Vec::new();
    for vl in &recovery.virtual_layouts {
        match vl {
            majit_backend::ExitVirtualLayout::Object {
                descr,
                fields,
                fielddescrs,
                descr_size,
                ..
            } => {
                let field_vals: Vec<i64> = fields
                    .iter()
                    .map(|(_, src)| resolve_value(src, &materialized).unwrap_or(0))
                    .collect();
                let vtable = descr
                    .as_ref()
                    .and_then(|d| d.as_size_descr())
                    .map(|sd| sd.vtable())
                    .unwrap_or(0);
                let ob_type_idx = fielddescrs.iter().position(|fd| fd.offset == 0);
                let ob_type = if vtable != 0 {
                    vtable
                } else {
                    ob_type_idx
                        .and_then(|idx| field_vals.get(idx).copied())
                        .unwrap_or(0) as usize
                };
                let payload_idx = fielddescrs
                    .iter()
                    .position(|fd| fd.offset != 0)
                    .or_else(|| (!field_vals.is_empty()).then_some(0));
                let payload_src = payload_idx.and_then(|idx| fields.get(idx).map(|(_, src)| src));
                let val_raw = payload_idx
                    .and_then(|idx| field_vals.get(idx).copied())
                    .unwrap_or(0);
                // resume.py:617-621 VirtualInfo.allocate + setfields
                let obj = if ob_type == w_float_type {
                    pyre_object::floatobject::w_float_new(f64::from_bits(
                        decode_recovery_float_payload_bits(
                            val_raw,
                            payload_src,
                            typed,
                            raw_values,
                            exit_layout,
                        ) as u64,
                    )) as usize
                } else if ob_type == w_int_type {
                    pyre_object::intobject::w_int_new(decode_recovery_payload_from_source(
                        val_raw,
                        payload_src,
                        typed,
                        raw_values,
                        exit_layout,
                    )) as usize
                } else if ob_type != 0 || *descr_size > 0 {
                    // General struct: allocate + setfields using fielddescrs
                    let size = if *descr_size > 0 { *descr_size } else { 16 };
                    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
                    let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut u8 };
                    if ob_type != 0 {
                        unsafe { *(ptr as *mut i64) = ob_type as i64 };
                    }
                    // resume.py:597-602 setfields using fielddescrs
                    for (i, &val) in field_vals.iter().enumerate() {
                        if let Some(fd) = fielddescrs.get(i) {
                            if fd.offset == 0 && ob_type != 0 {
                                continue;
                            }
                            let addr = unsafe { ptr.add(fd.offset) };
                            unsafe {
                                match fd.field_type {
                                    majit_ir::Type::Ref => std::ptr::write(addr as *mut i64, val),
                                    majit_ir::Type::Float => {
                                        std::ptr::write(addr as *mut u64, val as u64)
                                    }
                                    _ => match fd.field_size {
                                        1 => std::ptr::write(addr, val as u8),
                                        2 => std::ptr::write(addr as *mut u16, val as u16),
                                        4 => std::ptr::write(addr as *mut u32, val as u32),
                                        _ => std::ptr::write(addr as *mut i64, val),
                                    },
                                }
                            }
                        }
                    }
                    ptr as usize
                } else {
                    return false;
                };
                materialized.push(obj);
            }
            majit_backend::ExitVirtualLayout::Struct {
                typedescr,
                fields,
                fielddescrs,
                descr_size,
                ..
            } => {
                let field_vals: Vec<i64> = fields
                    .iter()
                    .map(|(_, src)| resolve_value(src, &materialized).unwrap_or(0))
                    .collect();
                let vtable = typedescr
                    .as_ref()
                    .and_then(|d| d.as_size_descr())
                    .map(|sd| sd.vtable())
                    .unwrap_or(0);
                let ob_type_idx = fielddescrs.iter().position(|fd| fd.offset == 0);
                let ob_type = if vtable != 0 {
                    vtable
                } else {
                    ob_type_idx
                        .and_then(|idx| field_vals.get(idx).copied())
                        .unwrap_or(0) as usize
                };
                let payload_idx = fielddescrs
                    .iter()
                    .position(|fd| fd.offset != 0)
                    .or_else(|| (!field_vals.is_empty()).then_some(0));
                let payload_src = payload_idx.and_then(|idx| fields.get(idx).map(|(_, src)| src));
                let val_raw = payload_idx
                    .and_then(|idx| field_vals.get(idx).copied())
                    .unwrap_or(0);
                // resume.py:617-621 VirtualInfo.allocate + setfields
                let obj = if ob_type == w_float_type {
                    pyre_object::floatobject::w_float_new(f64::from_bits(
                        decode_recovery_float_payload_bits(
                            val_raw,
                            payload_src,
                            typed,
                            raw_values,
                            exit_layout,
                        ) as u64,
                    )) as usize
                } else if ob_type == w_int_type {
                    pyre_object::intobject::w_int_new(decode_recovery_payload_from_source(
                        val_raw,
                        payload_src,
                        typed,
                        raw_values,
                        exit_layout,
                    )) as usize
                } else if ob_type != 0 || *descr_size > 0 {
                    // General struct: allocate + setfields using fielddescrs
                    let size = if *descr_size > 0 { *descr_size } else { 16 };
                    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
                    let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut u8 };
                    if ob_type != 0 {
                        unsafe { *(ptr as *mut i64) = ob_type as i64 };
                    }
                    // resume.py:597-602 setfields using fielddescrs
                    for (i, &val) in field_vals.iter().enumerate() {
                        if let Some(fd) = fielddescrs.get(i) {
                            if fd.offset == 0 && ob_type != 0 {
                                continue;
                            }
                            let addr = unsafe { ptr.add(fd.offset) };
                            unsafe {
                                match fd.field_type {
                                    majit_ir::Type::Ref => std::ptr::write(addr as *mut i64, val),
                                    majit_ir::Type::Float => {
                                        std::ptr::write(addr as *mut u64, val as u64)
                                    }
                                    _ => match fd.field_size {
                                        1 => std::ptr::write(addr, val as u8),
                                        2 => std::ptr::write(addr as *mut u16, val as u16),
                                        4 => std::ptr::write(addr as *mut u32, val as u32),
                                        _ => std::ptr::write(addr as *mut i64, val),
                                    },
                                }
                            }
                        }
                    }
                    ptr as usize
                } else {
                    return false;
                };
                materialized.push(obj);
            }
            majit_backend::ExitVirtualLayout::Array {
                clear, kind, items, ..
            } => {
                // resume.py:650-670: allocate_array(len, arraydescr, clear)
                let arr_kind = match kind {
                    2 => pyre_object::ArrayKind::Float,
                    1 => pyre_object::ArrayKind::Int,
                    _ => pyre_object::ArrayKind::Ref,
                };
                let array = pyre_object::allocate_array(items.len(), arr_kind, *clear);
                for (i, src) in items.iter().enumerate() {
                    if let Some(val) = resolve_value(src, &materialized) {
                        // resume.py:656-670: element kind dispatch
                        match kind {
                            2 => pyre_object::setarrayitem_float(
                                array,
                                i,
                                f64::from_bits(val as u64),
                            ),
                            1 => pyre_object::setarrayitem_int(array, i, val),
                            _ => pyre_object::setarrayitem_ref(
                                array,
                                i,
                                val as pyre_object::PyObjectRef,
                            ),
                        }
                    }
                }
                // resume.py:654: return array GcRef directly
                materialized.push(array as usize);
            }
            majit_backend::ExitVirtualLayout::ArrayStruct {
                field_types,
                item_size,
                field_offsets,
                field_sizes,
                element_fields,
                ..
            } => {
                // resume.py:749: allocate_array(self.size, self.arraydescr, clear=True)
                let size = element_fields.len();
                let is = if *item_size > 0 {
                    *item_size
                } else {
                    let fpe = element_fields.first().map(|ef| ef.len()).unwrap_or(0);
                    fpe * 8
                };
                let array = pyre_object::allocate_array_struct(size, is);
                for (i, ef) in element_fields.iter().enumerate() {
                    for (j, (_, src)) in ef.iter().enumerate() {
                        if let Some(val) = resolve_value(src, &materialized) {
                            // resume.py:757: setinteriorfield(i, array, num, fielddescrs[j])
                            let ft = field_types.get(j).copied().unwrap_or(0);
                            let fo = field_offsets.get(j).copied().unwrap_or(j * 8);
                            let fs = field_sizes.get(j).copied().unwrap_or(8);
                            pyre_object::setinteriorfield(array, i, fo, fs, is, ft, val as i64);
                        }
                    }
                }
                materialized.push(array as usize);
            }
            majit_backend::ExitVirtualLayout::RawBuffer { size, entries } => {
                let buffer = unsafe {
                    std::alloc::alloc_zeroed(
                        std::alloc::Layout::from_size_align(*size, 8)
                            .unwrap_or(std::alloc::Layout::new::<u8>()),
                    )
                };
                for &(offset, esz, ref src) in entries {
                    if let Some(val) = resolve_value(src, &materialized) {
                        if offset + esz <= *size {
                            unsafe {
                                match esz {
                                    1 => *(buffer.add(offset) as *mut u8) = val as u8,
                                    2 => *(buffer.add(offset) as *mut u16) = val as u16,
                                    4 => *(buffer.add(offset) as *mut u32) = val as u32,
                                    _ => *(buffer.add(offset) as *mut i64) = val,
                                }
                            }
                        }
                    }
                }
                materialized.push(buffer as usize);
            }
            majit_backend::ExitVirtualLayout::RawSlice { offset, base } => {
                // resume.py:723-727: base_buffer + offset
                let base_val = resolve_value(base, &materialized).unwrap_or(0);
                materialized.push((base_val + *offset as i64) as usize);
            }
        }
    }

    // Replace Virtual(idx) frame slots with materialized pointers.
    let mut replaced = 0usize;
    if let Some(frame) = recovery.frames.last() {
        for (slot_idx, src) in frame.slots.iter().enumerate() {
            if let majit_backend::ExitValueSourceLayout::Virtual(vidx) = src {
                if let Some(&ptr) = materialized.get(*vidx) {
                    if slot_idx < typed.len() {
                        typed[slot_idx] = Value::Ref(majit_ir::GcRef(ptr));
                        replaced += 1;
                    }
                }
            }
        }
    }

    // [pyre safety] target_slot fallback: Virtual markers in frame.slots
    // can be lost during to_exit_recovery_layout_with_caller_prefix merge.
    // Use target_slot from ExitVirtualLayout to place remaining objects.
    if replaced == 0 && !materialized.is_empty() {
        for (vidx, vl) in recovery.virtual_layouts.iter().enumerate() {
            let target = match vl {
                majit_backend::ExitVirtualLayout::Object { target_slot, .. }
                | majit_backend::ExitVirtualLayout::Struct { target_slot, .. } => *target_slot,
                _ => None,
            };
            if let Some(slot_idx) = target {
                if slot_idx < typed.len()
                    && matches!(typed[slot_idx], Value::Ref(majit_ir::GcRef(0)))
                {
                    if let Some(&ptr) = materialized.get(vidx) {
                        if ptr != 0 {
                            typed[slot_idx] = Value::Ref(majit_ir::GcRef(ptr));
                            replaced += 1;
                        }
                    }
                }
            }
        }
    }
    replaced > 0 || materialized.is_empty()
}

fn decode_recovery_payload_from_source(
    raw: i64,
    src: Option<&majit_backend::ExitValueSourceLayout>,
    typed: &[Value],
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) -> i64 {
    match src {
        Some(majit_backend::ExitValueSourceLayout::ExitValue(_)) => {
            maybe_unbox_known_int_ref(raw, typed, raw_values, &exit_layout.exit_types)
        }
        _ => raw,
    }
}

fn decode_recovery_float_payload_bits(
    raw: i64,
    src: Option<&majit_backend::ExitValueSourceLayout>,
    typed: &[Value],
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) -> i64 {
    match src {
        Some(majit_backend::ExitValueSourceLayout::ExitValue(_)) => {
            maybe_unbox_known_float_ref_bits(raw, typed, raw_values, &exit_layout.exit_types)
        }
        _ => raw,
    }
}

fn decode_recovery_int_payload(
    typed_raw: i64,
    typed: &[Value],
    raw_slot: Option<i64>,
    raw_values: &[i64],
    exit_types: &[majit_ir::Type],
) -> i64 {
    raw_slot
        .map(|raw| maybe_unbox_known_int_ref(raw, typed, raw_values, exit_types))
        .unwrap_or(typed_raw)
}

fn maybe_unbox_known_int_ref(
    raw: i64,
    typed: &[Value],
    raw_values: &[i64],
    exit_types: &[majit_ir::Type],
) -> i64 {
    let obj = raw as usize as pyre_object::PyObjectRef;
    if obj.is_null() || !raw_matches_known_ref(raw, typed, raw_values, exit_types) {
        return raw;
    }
    if unsafe { pyre_object::pyobject::is_int(obj) } {
        unsafe { pyre_object::intobject::w_int_get_value(obj) }
    } else {
        raw
    }
}

fn maybe_unbox_known_float_ref_bits(
    raw: i64,
    typed: &[Value],
    raw_values: &[i64],
    exit_types: &[majit_ir::Type],
) -> i64 {
    let obj = raw as usize as pyre_object::PyObjectRef;
    if obj.is_null() || !raw_matches_known_ref(raw, typed, raw_values, exit_types) {
        return raw;
    }
    if unsafe { pyre_object::pyobject::is_float(obj) } {
        unsafe { pyre_object::floatobject::w_float_get_value(obj).to_bits() as i64 }
    } else {
        raw
    }
}

fn raw_matches_known_ref(
    raw: i64,
    typed: &[Value],
    raw_values: &[i64],
    exit_types: &[majit_ir::Type],
) -> bool {
    typed.iter().skip(3).any(|value| match value {
        Value::Ref(gc) => gc.0 as i64 == raw,
        _ => false,
    }) || raw_values
        .iter()
        .zip(exit_types.iter())
        .skip(3)
        .any(|(value, tp)| matches!(tp, majit_ir::Type::Ref) && *value == raw)
}

pub(crate) fn build_jit_state(
    frame: &PyFrame,
    virtualizable_info: &majit_metainterp::virtualizable::VirtualizableInfo,
) -> PyreJitState {
    let mut jit_state = PyreJitState {
        frame: frame as *const PyFrame as usize,
    };
    assert!(
        jit_state.sync_from_virtualizable(virtualizable_info),
        "build_jit_state: frame must be a valid PyFrame with readable fields"
    );
    jit_state
}

fn sync_jit_state_to_frame(
    jit_state: &PyreJitState,
    frame: &mut PyFrame,
    virtualizable_info: &majit_metainterp::virtualizable::VirtualizableInfo,
) {
    // Heap IS the source of truth — read back from the frame pointer.
    let _ = jit_state.sync_to_virtualizable(virtualizable_info);
    frame.next_instr = jit_state.next_instr();
    frame.valuestackdepth = jit_state.valuestackdepth();
}

/// resume.py:1437-1541 — BlackholeAllocator for pyre's object model.
///
/// Used by ResumeDataDirectReader during guard failure blackhole resume
/// to allocate virtual objects and replay pending field writes.
/// RPython delegates to self.cpu (metainterp_sd.cpu) for allocation.
pub(crate) struct PyreBlackholeAllocator;

impl majit_metainterp::resume::BlackholeAllocator for PyreBlackholeAllocator {
    fn allocate_struct(&self, descr_index: u32) -> i64 {
        // resume.py:1441 allocate_struct — same as allocate_with_vtable
        // for pyre (no RPython struct/GC distinction).
        self.allocate_with_vtable(descr_index)
    }

    fn allocate_with_vtable(&self, descr_index: u32) -> i64 {
        // resume.py:1437-1439 allocate_with_vtable
        // Allocate a fresh GC object by type_id. Fields are zeroed; the
        // caller fills them via setfield_typed. Must return a NEW object
        // (not from small-int pool) because setfield_typed mutates in-place.
        use pyre_jit_trace::descr::{W_FLOAT_GC_TYPE_ID, W_INT_GC_TYPE_ID};
        match descr_index {
            W_INT_GC_TYPE_ID => {
                let obj = Box::new(pyre_object::intobject::W_IntObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: &pyre_object::pyobject::INT_TYPE as *const _,
                    },
                    intval: 0,
                });
                Box::into_raw(obj) as i64
            }
            W_FLOAT_GC_TYPE_ID => {
                let obj = Box::new(pyre_object::floatobject::W_FloatObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: &pyre_object::pyobject::FLOAT_TYPE as *const _,
                    },
                    floatval: 0.0,
                });
                Box::into_raw(obj) as i64
            }
            // resume.py:1437 allocate_with_vtable must return a valid object.
            // Panic on unknown type_id so the issue is visible immediately.
            _ => panic!(
                "allocate_with_vtable: unsupported gc type_id {}",
                descr_index
            ),
        }
    }

    fn setfield_typed(
        &self,
        struct_ptr: i64,
        value: i64,
        _descr: u32,
        field_offset: usize,
        field_size: usize,
    ) {
        // resume.py:1509-1528 setfield — write field at byte offset
        if struct_ptr != 0 && field_offset > 0 {
            unsafe {
                let ptr = (struct_ptr as *mut u8).add(field_offset);
                match field_size {
                    8 => (ptr as *mut i64).write(value),
                    4 => (ptr as *mut i32).write(value as i32),
                    2 => (ptr as *mut i16).write(value as i16),
                    1 => ptr.write(value as u8),
                    _ => (ptr as *mut i64).write(value),
                }
            }
        }
    }

    fn setarrayitem_typed(&self, array: i64, index: usize, value: i64, _descr: u32) {
        // resume.py:1009-1015 setarrayitem dispatch by type
        if array != 0 {
            // pyre list items are PyObjectRef (pointer-sized)
            let item_size = std::mem::size_of::<usize>();
            unsafe {
                let base = array as *mut u8;
                let ptr = base.add(index * item_size) as *mut i64;
                ptr.write(value);
            }
        }
    }

    fn box_int(&self, value: i64) -> i64 {
        pyre_object::intobject::w_int_new(value) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_interpreter::{function_get_code, is_function};

    #[test]
    fn test_eval_simple_addition() {
        let source = "x = 1 + 2";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(x), 3);
        }
    }

    #[test]
    fn test_eval_while_loop() {
        let source = "\
i = 0
s = 0
while i < 100:
    s = s + i
    i = i + 1";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 4950);
        }
    }

    #[test]
    fn test_eval_with_jit_redecodes_opargs_after_extended_arg_jumps() {
        let source = "\
def fannkuch(n):
    p = [0] * n
    q = [0] * n
    s = [0] * n
    i = 0
    while i < n:
        p[i] = i
        q[i] = i
        s[i] = i
        i = i + 1
    maxflips = 0
    checksum = 0
    sign = 1
    while True:
        q0 = p[0]
        if q0 != 0:
            i = 1
            while i < n:
                q[i] = p[i]
                i = i + 1
            flips = 1
            while True:
                qq = q[q0]
                if qq == 0:
                    break
                q[q0] = q0
                if q0 >= 3:
                    i = 1
                    j = q0 - 1
                    while i < j:
                        t = q[i]
                        q[i] = q[j]
                        q[j] = t
                        i = i + 1
                        j = j - 1
                q0 = qq
                flips = flips + 1
            if flips > maxflips:
                maxflips = flips
            checksum = checksum + sign * flips
        if sign == 1:
            t = p[0]
            p[0] = p[1]
            p[1] = t
            sign = -1
        else:
            t = p[1]
            p[1] = p[2]
            p[2] = t
            sign = 1
            i = 2
            while i < n:
                sx = s[i]
                if sx != 0:
                    s[i] = sx - 1
                    break
                if i == n - 1:
                    return 999
                s[i] = i
                t = p[0]
                j = 0
                while j < i + 1:
                    p[j] = p[j + 1]
                    j = j + 1
                p[i + 1] = t
                i = i + 1

r = fannkuch(6)";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        if std::env::var_os("MAJIT_DUMP_BYTECODE").is_some() {
            let mut state = pyre_interpreter::OpArgState::default();
            for (pc, unit) in code.instructions.iter().copied().enumerate() {
                let (instr, oparg) = state.get(unit);
                eprintln!("{pc:03}: {instr:?} oparg={oparg:?}");
            }
            for pc in [72usize, 99, 129, 131, 141, 168, 179, 447, 449] {
                eprintln!(
                    "decode[{pc}] = {:?}",
                    pyre_interpreter::decode_instruction_at(&code, pc)
                );
            }
        }
        let mut frame = PyFrame::new(code);
        let result = eval_with_jit(&mut frame);
        if std::env::var_os("MAJIT_DUMP_BYTECODE").is_some() {
            let mut keys: Vec<_> = unsafe { (*frame.namespace).keys().cloned().collect() };
            keys.sort();
            eprintln!("module result: {:?}", result);
            eprintln!("module namespace keys: {:?}", keys);
        }
        unsafe {
            let r = *(*frame.namespace).get("r").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(r), 999);
        }
    }

    #[test]
    fn test_recursive_fib_callable_prefers_function_entry() {
        let source = "\
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let fib = *(*frame.namespace).get("fib").unwrap();
            assert!(crate::call_jit::callable_prefers_function_entry(fib));
        }
    }

    #[test]
    fn test_nonrecursive_helper_does_not_prefer_function_entry() {
        let source = "\
def add(a, b):
    return a + b
";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let add = *(*frame.namespace).get("add").unwrap();
            assert!(!crate::call_jit::callable_prefers_function_entry(add));
        }
    }

    #[test]
    fn test_recursive_global_reads_do_not_reuse_force_cache_across_global_mutation() {
        let source = "\
factor = 1
def g(n):
    if n < 2:
        return n * factor
    return g(n - 1) + g(n - 2) + factor

first = g(12)
factor = 2
second = g(12)";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let first = *(*frame.namespace).get("first").unwrap();
            let second = *(*frame.namespace).get("second").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(first), 376);
            assert_eq!(pyre_object::intobject::w_int_get_value(second), 752);
        }
    }

    #[test]
    fn test_inline_residual_user_call_with_many_args_stays_correct() {
        let source = "\
def helper(a, b, c, d, e):
    return a + b + c + d + e

def outer(x):
    return helper(x, x, x, x, x)

s = 0
i = 0
while i < 300:
    s = s + outer(i)
    i = i + 1";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 224_250);
        }
    }

    #[test]
    fn test_nested_direct_helper_calls_stay_correct() {
        let source = "\
def add(a, b):
    return a + b

def mul(a, b):
    return a * b

def square(x):
    return mul(x, x)

def compute(i):
    return add(square(i), i)

s = 0
i = 0
while i < 300:
    s = add(s, compute(i))
    i = add(i, 1)";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 8_999_900);
        }
    }

    #[test]
    fn test_dynamic_int_list_indexing_stays_correct() {
        let source = "\
q = [0, 1, 2, 3, 4]
i = 0
s = 0
while i < 200:
    q0 = i % 5
    s = s + q[q0]
    q[q0] = q[q0] + 1
    i = i + 1";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            let q = *(*frame.namespace).get("q").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 4_300);
            assert_eq!(
                pyre_object::intobject::w_int_get_value(
                    pyre_object::listobject::w_list_getitem(q, 0).unwrap()
                ),
                40
            );
            assert_eq!(
                pyre_object::intobject::w_int_get_value(
                    pyre_object::listobject::w_list_getitem(q, 4).unwrap()
                ),
                44
            );
        }
    }
}

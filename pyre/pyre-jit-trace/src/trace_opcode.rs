//! MIFrame opcode handlers for trace-time JIT.
//!
//! Contains all `impl MIFrame` methods and trait implementations
//! (SharedOpcodeHandler, LocalOpcodeHandler, etc.).

use crate::state::*;

use std::borrow::Cow;

use majit_ir::{DescrRef, GcRef, OpCode, OpRef, Type, Value};
use majit_metainterp::{
    CANNOT_RAISE_NO_HEAP_EFFECT_INFO, TraceAction, TraceCtx, default_effect_info,
};

use pyre_interpreter::bytecode::{BinaryOperator, CodeObject, ComparisonOperator, Instruction};

extern "C" fn trace_function_get_defaults(func: i64) -> i64 {
    unsafe { function_get_defaults(func as PyObjectRef) as i64 }
}

extern "C" fn trace_function_get_kwdefaults(func: i64) -> i64 {
    let kwdefaults = unsafe { pyre_interpreter::function_get_kwdefaults(func as PyObjectRef) };
    pyre_interpreter::baseobjspace::unwrap_cell(kwdefaults) as i64
}

extern "C" fn trace_dict_lookup_jit(dict: i64, key: i64) -> i64 {
    unsafe {
        pyre_object::w_dict_lookup(dict as PyObjectRef, key as PyObjectRef).unwrap_or(PY_NULL)
            as i64
    }
}

/// floatobject.py:561 `descr_pow` → `_pow(space, x, y)` parity.
///
/// `_pow` in floatobject.py:799-881 takes two raw floats and returns a
/// raw float (can raise OverflowError / ValueError / ZeroDivisionError).
/// The JIT trace records this as `CALL_F(float_pow_jit, lhs, rhs)`
/// (pyjitpl.py:2119-2121 CALL_F branch taken because
/// `check_forces_virtual_or_virtualizable()` is False for ll_math_pow,
/// and `exc=True` because EF_CAN_RAISE), followed by `GUARD_NO_EXCEPTION`
/// via `handle_possible_exception` (pyjitpl.py:1950-1955, 3395).
///
/// ll_math_pow (ll_math.py:260) is the can-raise helper (EF_CAN_RAISE),
/// NOT elidable and NOT force-virtual. Using Rust's native `x.powf(y)`
/// would drop the Python exception semantics (negative base fractional
/// exponent → ValueError, 0.0 raised to negative → ZeroDivisionError,
/// overflow → OverflowError). Using CALL_MAY_FORCE_F would be wrong
/// because the optimizer postpones that family until GUARD_NOT_FORCED
/// arrives (heap.py CALL_MAY_FORCE branch), which is the virtualizable
/// protocol — ll_math_pow does not touch virtualizables.
///
/// Extracted to module level for stable function pointer identity.
///
/// Must match `float_pow_impl` semantics in `baseobjspace.rs`: any
/// divergence would cause the JIT compiled code to produce a different
/// result from the interpreter for the same input (correctness bug).
pub(crate) extern "C" fn float_pow_jit(x: f64, y: f64) -> f64 {
    match pyre_interpreter::float_pow_raw(x, y) {
        Ok(z) => z,
        Err(err) => {
            // llmodel.py:194-199 _store_exception parity: set JIT exception
            // state so the following GuardNoException sees it and fails,
            // propagating the raise into the meta-interpreter.
            let exc_obj = err.to_exc_object();
            #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
            majit_backend_cranelift::jit_exc_raise(exc_obj as i64);
            #[cfg(all(feature = "dynasm", not(target_arch = "wasm32")))]
            majit_backend_dynasm::jit_exc_raise(exc_obj as i64);
            #[cfg(target_arch = "wasm32")]
            majit_backend_wasm::jit_exc_raise(exc_obj as i64);
            let _ = exc_obj; // suppress unused warning when no backend
            // Return value is discarded by GuardNoException path; use NaN
            // as a safe sentinel in case the guard is elided.
            f64::NAN
        }
    }
}
use pyre_interpreter::eval::{get_current_exception, set_current_exception};

/// Runtime helper for traced `RAISE_VARARGS`.
///
/// The trace records the Python `CALL` that constructs an exception object,
/// then `RAISE_VARARGS` itself must materialize a real JIT exception before
/// `handle_possible_exception` emits `GuardException`. Without this explicit
/// helper the compiled bridge contains only the constructor call plus
/// `GuardException`, so the guard sees no pending exception and incorrectly
/// resumes down the normal path.
pub(crate) extern "C" fn raise_exception_jit(exc_obj: i64) {
    #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
    majit_backend_cranelift::jit_exc_raise(exc_obj);
    #[cfg(all(feature = "dynasm", not(target_arch = "wasm32")))]
    majit_backend_dynasm::jit_exc_raise(exc_obj);
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::jit_exc_raise(exc_obj);
    let _ = exc_obj;
}

/// Runtime helper for traced `RAISE_VARARGS`.
///
/// Mirrors `eval.rs:1035-1129` on the compiled path:
/// normalize the exception operand, normalize/attach the optional
/// cause, and publish the final exception via `jit_exc_raise` so the
/// following `GUARD_EXCEPTION` sees it.
pub(crate) extern "C" fn normalize_raise_varargs_jit(
    frame_ptr: i64,
    exc_obj: i64,
    cause_obj: i64,
) -> i64 {
    let frame_ptr = frame_ptr as *const pyre_interpreter::pyframe::PyFrame;
    let exc = exc_obj as pyre_object::PyObjectRef;
    let raw_cause = cause_obj as pyre_object::PyObjectRef;

    // pyopcode.py:704-722 — cause and exc normalization both run against
    // `self.space`/`frame.execution_context`. Pin the caller's frame
    // context for the whole body so the cause-class-call and the
    // exc-class-call observe the same namespace / thread state.
    let frame_ctx = if frame_ptr.is_null() {
        std::ptr::null()
    } else {
        unsafe { (*frame_ptr).execution_context }
    };
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    if !frame_ctx.is_null() {
        pyre_interpreter::call::set_last_exec_ctx(frame_ctx);
    }

    let cause = if raw_cause.is_null() {
        None
    } else {
        // pyopcode.py:706-707 — cause class-call must mirror the exc
        // class-call (pyopcode.py:711-713) on compiled traces. Force
        // both onto the plain interpreter path so the constructor
        // cannot re-enter the tracer.
        let result = {
            let _plain_guard = pyre_interpreter::call::force_plain_eval();
            normalize_raise_cause(raw_cause)
        };
        match result {
            Ok(cause) => Some(cause),
            Err(err) => {
                pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
                let exc = err.to_exc_object();
                raise_exception_jit(exc as i64);
                return exc as i64;
            }
        }
    };

    let mut final_exc: pyre_object::PyObjectRef = unsafe {
        if pyre_object::is_exception(exc) {
            exc
        } else if pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(exc) {
            if frame_ctx.is_null() {
                pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
                let err =
                    PyError::runtime_error("raise helper missing current frame").to_exc_object();
                raise_exception_jit(err as i64);
                return err as i64;
            }
            let result = {
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                pyre_interpreter::call::call_function_impl_result(exc, &[])
            };
            match result {
                Ok(obj) if pyre_object::is_exception(obj) => obj,
                Ok(_) => {
                    PyError::type_error("exceptions must derive from BaseException").to_exc_object()
                }
                Err(err) => err.to_exc_object(),
            }
        } else {
            PyError::type_error("exceptions must derive from BaseException").to_exc_object()
        }
    };

    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);

    if let Err(err) = attach_raise_cause(final_exc, cause) {
        final_exc = err.to_exc_object();
    }
    raise_exception_jit(final_exc as i64);
    final_exc as i64
}

/// Runtime helper for traced `PUSH_EXC_INFO`: read the per-thread
/// `CURRENT_EXCEPTION` slot so the compiled bridge preserves
/// `pyopcode.py:786` / `eval.rs:1220-1229` semantics (save the
/// previous sys_exc_info before `CURRENT_EXCEPTION` is overwritten).
pub(crate) extern "C" fn trace_get_current_exception_jit() -> i64 {
    pyre_interpreter::eval::get_current_exception() as i64
}

/// Runtime helper for traced `PUSH_EXC_INFO` / `POP_EXCEPT`: write the
/// per-thread `CURRENT_EXCEPTION` slot so the compiled bridge preserves
/// `pyopcode.py:786/:778` / `eval.rs:1220-1229 / :1243-1249` semantics.
pub(crate) extern "C" fn trace_set_current_exception_jit(exc: i64) {
    pyre_interpreter::eval::set_current_exception(exc as pyre_object::PyObjectRef);
}
use pyre_interpreter::eval::{attach_raise_cause, normalize_raise_cause};
use pyre_interpreter::truth_value as objspace_truth_value;
use pyre_interpreter::{
    OpcodeStepExecutor, PyError, SharedOpcodeHandler, call_function, decode_instruction_at,
    execute_opcode_step, function_get_defaults, function_get_globals, function_get_globals_obj,
    is_builtin_code, is_function, range_iter_continues,
};

use pyre_object::PyObjectRef;
use pyre_object::listobject::w_list_getitem;
use pyre_object::methodobject::{METHOD_TYPE, is_method, w_method_get_func, w_method_get_self};
use pyre_object::pyobject::{
    FLOAT_TYPE, INT_TYPE, LIST_TYPE, LONG_TYPE, PyType, TUPLE_TYPE, get_instantiate, is_float,
    is_int, is_list, is_long, is_tuple,
};
use pyre_object::rangeobject::RANGE_ITER_TYPE;
use pyre_object::specialisedtupleobject::{
    SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
};
use pyre_object::tupleobject::w_tuple_getitem;
use pyre_object::{
    PY_NULL, w_list_can_append_without_realloc, w_list_is_inline_storage, w_list_len,
    w_list_uses_float_storage, w_list_uses_int_storage, w_list_uses_object_storage, w_tuple_len,
};

fn trace_abort_error(reason: &'static str) -> PyError {
    PyError::internal_trace_abort(reason)
}

fn is_trace_abort_error(err: &PyError) -> bool {
    err.kind == pyre_interpreter::PyErrorKind::TraceAbort
}

fn trace_plain_int_payload(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    item: OpRef,
    concrete_item: PyObjectRef,
) -> OpRef {
    if frame.value_type(item) == Type::Int {
        return item;
    }
    unsafe {
        if is_long(concrete_item) {
            return crate::state::trace_unbox_long_with_resume(
                frame,
                item,
                &LONG_TYPE as *const _ as i64,
            );
        }
    }
    frame.trace_guarded_int_payload(ctx, item)
}

fn trace_set_tuple_w_class(ctx: &mut TraceCtx, tuple: OpRef, descr: DescrRef) {
    let w_class = get_instantiate(&TUPLE_TYPE);
    if w_class.is_null() {
        return;
    }
    // Pyre object-model adaptation: `ob_type` is the JIT-visible
    // RPython vtable, but Python-level `type()` reads `w_class`.
    // PyPy's specialised tuple variants all share the public tuple typedef.
    let w_class = ctx.const_ref(w_class as i64);
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[tuple, w_class], descr.clone());
    ctx.heapcache_setfield_cached(tuple, descr.index(), w_class);
}

/// Emit `GetfieldGcR(w_class) → PtrEq(expected) → GuardTrue` so the trace
/// only stays specialised for instances whose Python-level `w_class`
/// matches `expected_typeobj`. Mirrors the `type(w) is W_IntObject` /
/// `type(w) is W_FloatObject` half of `listobject.py:2390 is_plain_int1`
/// and `specialisedtupleobject.py:176`. Without this guard a later
/// int/float subclass with the same payload layout would re-enter a
/// trace specialised for the exact payload type and silently lose
/// subclass identity when the trace rewraps via `wrapint` / `wrapfloat`.
///
/// Only `Type::Ref` items can carry a divergent `w_class`: a raw
/// `Type::Int` / `Type::Float` trace value is an unboxed payload produced
/// by arithmetic or a guarded unbox, and its concrete shadow can only be
/// `Int` / `Float` (the `write_int_reg` / `write_ref_reg` sanitizers
/// collapse a boxed subclass to `Null`), never a subclass pointer. Reading
/// `w_class` off such a value would force the box that OptVirtualize is
/// meant to remove, so skip the guard there.
fn trace_guard_exact_w_class(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    expected_typeobj: PyObjectRef,
) {
    if expected_typeobj.is_null() || frame.value_type(obj) != Type::Ref {
        return;
    }
    if ctx.heap_cache().is_unescaped(obj) {
        return;
    }
    let descr = crate::descr::w_class_descr();
    let actual = crate::state::opimpl_getfield_gc_r(ctx, obj, descr);
    let expected = ctx.const_ref(expected_typeobj as i64);
    let eq = ctx.record_op(OpCode::PtrEq, &[actual, expected]);
    frame.generate_guard(ctx, OpCode::GuardTrue, &[eq]);
}

fn positional_defaults_to_load(
    callable: PyObjectRef,
    code: &CodeObject,
    nargs: usize,
) -> Option<Vec<PyObjectRef>> {
    let nparams = code.arg_count as usize;
    if nargs >= nparams {
        return None;
    }

    let defaults = unsafe { function_get_defaults(callable) };
    if defaults.is_null() {
        return None;
    }

    let defaults = pyre_interpreter::baseobjspace::unwrap_cell(defaults);
    let ndefaults = if unsafe { pyre_object::is_tuple(defaults) } {
        unsafe { w_tuple_len(defaults) }
    } else {
        0
    };
    if ndefaults == 0 {
        return None;
    }

    let first_default = nparams.saturating_sub(ndefaults);
    if nargs < first_default {
        return None;
    }

    let defaults_to_load = nparams - first_default;
    let default_start = ndefaults - defaults_to_load;
    let mut loaded = Vec::with_capacity(nparams - nargs);
    for i in nargs..nparams {
        let default_idx = default_start + (i - first_default);
        loaded.push(unsafe { w_tuple_getitem(defaults, default_idx as i64) }.unwrap_or(PY_NULL));
    }
    Some(loaded)
}

fn fill_positional_defaults_for_trace_call<'a>(
    callable: PyObjectRef,
    code: &CodeObject,
    args: &'a [PyObjectRef],
) -> Cow<'a, [PyObjectRef]> {
    let Some(defaults) = positional_defaults_to_load(callable, code, args.len()) else {
        return Cow::Borrowed(args);
    };
    let mut full = Vec::with_capacity(args.len() + defaults.len());
    full.extend_from_slice(args);
    full.extend(defaults);
    Cow::Owned(full)
}

fn const_step_one_slice_bounds(
    concrete_obj: PyObjectRef,
    concrete_key: PyObjectRef,
    concrete_value: PyObjectRef,
) -> Option<(i64, i64, i64, i64, bool)> {
    unsafe {
        if concrete_obj.is_null()
            || concrete_key.is_null()
            || concrete_value.is_null()
            || !is_list(concrete_obj)
            || !pyre_object::sliceobject::is_slice(concrete_key)
            || !is_list(concrete_value)
        {
            return None;
        }
        let step = pyre_object::sliceobject::w_slice_get_step(concrete_key);
        let step_is_none = pyre_object::is_none(step);
        let step_is_one = if step_is_none {
            true
        } else if is_int(step) {
            pyre_object::w_int_get_value(step) == 1
        } else {
            false
        };
        if !step_is_one {
            return None;
        }
        let start = pyre_object::sliceobject::w_slice_get_start(concrete_key);
        let stop = pyre_object::sliceobject::w_slice_get_stop(concrete_key);
        if pyre_object::is_none(start)
            || pyre_object::is_none(stop)
            || !is_int(start)
            || !is_int(stop)
        {
            return None;
        }
        let start = pyre_object::w_int_get_value(start);
        let stop = pyre_object::w_int_get_value(stop);
        if start < 0 || stop < start {
            return None;
        }
        let len = w_list_len(concrete_obj) as i64;
        // PyPy `W_ListObject.descr_setitem` routes slices through
        // `_unpack_slice` before `setslice`, so storage-level code sees
        // adjusted positive-step bounds, not the raw slice fields.  This
        // helper only accepts non-negative constant bounds, so adjustment
        // reduces to CPython/PyPy's upper clamp.
        Some((start, stop, start.min(len), stop.min(len), step_is_none))
    }
}

fn concrete_list_strategy_id(concrete: PyObjectRef) -> Option<i64> {
    unsafe {
        if w_list_uses_object_storage(concrete) {
            Some(0)
        } else if w_list_uses_int_storage(concrete) {
            Some(1)
        } else if w_list_uses_float_storage(concrete) {
            Some(2)
        } else {
            None
        }
    }
}

use crate::descr::{
    float_floatval_descr, int_intval_descr, list_strategy_descr, ob_type_descr,
    slice_w_start_descr, slice_w_step_descr, slice_w_stop_descr, w_float_size_descr,
    w_int_size_descr,
};
use crate::frame_layout::{
    PYFRAME_DEBUGDATA_OFFSET, PYFRAME_LASTBLOCK_OFFSET, PYFRAME_PYCODE_OFFSET,
};
use crate::helpers::TraceHelperAccess;

fn emit_call_assembler_callee_frame(
    this: &mut MIFrame,
    ctx: &mut TraceCtx,
    callable: OpRef,
    args: &[OpRef],
    concrete_callable: PyObjectRef,
    w_callee_code: *const (),
    callee_code: &CodeObject,
    is_self_recursive: bool,
    self_recursive_raw_int_arg: Option<OpRef>,
) -> Result<(OpRef, bool), PyError> {
    let needs_positional_defaults = is_self_recursive
        && positional_defaults_to_load(concrete_callable, callee_code, args.len()).is_some();

    if is_self_recursive && !needs_positional_defaults && args.len() == 1 {
        if let Some(raw_arg) = self_recursive_raw_int_arg {
            let nlocals = callee_code.varnames.len();
            let ncells = pyre_interpreter::ncells(callee_code);
            let max_stack = callee_code.max_stackdepth as usize;
            let callee_globals = unsafe { function_get_globals(concrete_callable) };
            // R3.3b: also resolve the canonical W_DictObject sibling so the
            // inline new-PyFrame helper populates `PyFrame.w_globals_obj`
            // in lockstep with the raw slot, matching what production
            // Rust constructors do at heap allocation.
            let callee_globals_obj =
                unsafe { pyre_interpreter::function_get_globals_obj(concrete_callable) };
            let stores_global = unsafe {
                pyre_interpreter::w_code_frame_stores_global(
                    w_callee_code as PyObjectRef,
                    callee_globals,
                )
            };
            if ncells == 0 && !stores_global {
                let pycode_const = ctx.const_ref(w_callee_code as i64);
                let w_globals_const = ctx.const_ref(callee_globals as i64);
                let w_globals_obj_const = ctx.const_ref(callee_globals_obj as i64);
                let ec = this.ensure_execution_context(ctx);
                let frame = crate::helpers::emit_new_pyframe_inline_self_recursive(
                    ctx,
                    raw_arg,
                    nlocals + ncells + max_stack,
                    nlocals + ncells,
                    pycode_const,
                    w_globals_const,
                    w_globals_obj_const,
                    ec,
                );
                return Ok((frame, false));
            }
        }
    }

    if args.len() == 1 {
        let (helper, helper_arg_types, helper_args) = if is_self_recursive {
            if let Some(raw_arg) = self_recursive_raw_int_arg {
                let (helper, helper_arg_types) =
                    one_arg_callee_frame_helper(Type::Int, !needs_positional_defaults);
                let helper_args = if needs_positional_defaults {
                    vec![this.frame(), callable, raw_arg]
                } else {
                    vec![this.frame(), raw_arg]
                };
                (helper, helper_arg_types, helper_args)
            } else {
                let (helper, helper_arg_types) = one_arg_callee_frame_helper(
                    this.value_type(args[0]),
                    !needs_positional_defaults,
                );
                let helper_args = if needs_positional_defaults {
                    vec![this.frame(), callable, args[0]]
                } else {
                    vec![this.frame(), args[0]]
                };
                (helper, helper_arg_types, helper_args)
            }
        } else {
            let (helper, helper_arg_types) =
                one_arg_callee_frame_helper(this.value_type(args[0]), false);
            (
                helper,
                helper_arg_types,
                vec![this.frame(), callable, args[0]],
            )
        };
        let frame = ctx.call_ref_typed_with_effect(
            helper,
            &helper_args,
            &helper_arg_types,
            default_effect_info(),
        );
        return Ok((frame, true));
    }

    if let Some(frame_helper) = (crate::callbacks::get().callee_frame_helper)(args.len()) {
        let mut helper_args = vec![this.frame(), callable];
        helper_args.extend_from_slice(args);
        let helper_arg_types = frame_callable_arg_types(args.len());
        let frame = ctx.call_ref_typed_with_effect(
            frame_helper,
            &helper_args,
            &helper_arg_types,
            default_effect_info(),
        );
        return Ok((frame, true));
    }

    Err(PyError::type_error(
        "call_assembler: no frame helper for nargs",
    ))
}

/// pyjitpl.py:1188-1199 `_opimpl_setfield_vable` parity helper.
///
/// `PyreSym.vable_*` is a pyre-only parallel symbolic cache that
/// RPython does not have — RPython's `metainterp.virtualizable_boxes`
/// is the single canonical source for vable static state.  Setfield-
/// vable opcodes (`_opimpl_setfield_vable`) and the JUMP-time
/// shadow flush (`flush_to_frame`) publish into the boxes shadow so
/// future readers (JUMP-arg dedup, `close_loop_args_at`)
/// observe the same identity as `s.vable_*`.  Callers gate on
/// `s.owns_virtualizable_shadow()` before calling — upstream
/// `_opimpl_setfield_vable` short-circuits on
/// `_nonstandard_virtualizable` so callee inline frames never reach
/// the `metainterp.virtualizable_boxes[index] = valuebox` write, and
/// the call-site gate is pyre's analog (callee inline syms allocated
/// via `PyreSym::new_uninit` keep `vable_array_base` /
/// `bridge_local_oprefs` `None` so `owns_virtualizable_shadow()`
/// returns false for them).
///
/// Snapshot capture (`flush_to_frame_for_guard`) intentionally does
/// NOT mirror into the shared `ctx.virtualizable_boxes`.  Upstream
/// `rpython/jit/metainterp/pyjitpl.py:2586 capture_resumedata` reads
/// `metainterp.virtualizable_boxes` and hands it to
/// `rpython/jit/metainterp/opencoder.py:718
/// _list_of_boxes_virtualizable(boxes)` with no fallback heap source.
/// Pyre matches that single-source model in spirit for the two
/// per-opcode-advancing fields: `last_instr` / `valuestackdepth` are
/// rewritten in `s.vable_*` to their pre-opcode value at `orgpc - 1`
/// before the snapshot is built.  The other four scalars (`pycode`,
/// `debugdata`, `lastblock`, `w_globals`) keep whatever OpRef
/// `init_vable_indices` seeded at trace start because the tracer
/// never reaches their mutators under CPython 3.14 bytecode
/// (`pycode` / `w_globals`: only `pyframe.rs::frame_reinit`;
/// `debugdata`: only `getorcreate_debug_data` on debug paths;
/// `lastblock`: only `pyopcode.py:1268
/// SETUP_FINALLY/SETUP_EXCEPT/POP_BLOCK` which CPython 3.14 no
/// longer emits — try/except/finally goes through the zero-cost
/// `co_exceptiontable` consulted only on raise).  Convergence to
/// RPython's pure single-source model requires emitting
/// `_opimpl_setfield_vable` for those handlers if/when they are
/// re-introduced, after which the heap remains authoritative through
/// `metainterp.virtualizable_boxes` rather than through the snapshot
/// reader's own state.
///
/// The snapshot does not mirror its `s.vable_*` overrides into the
/// shared shadow because the shared shadow is the JUMP/JIT-time view
/// (live virtualizable values that `close_loop_args_at`'s JUMP-arg
/// derivation consumes), while `s.vable_last_instr/vsd`
/// carry the pre-opcode override that the snapshot reader needs.
/// The two stores stay distinct deliberately: `record_branch_guard`
/// saves `s.vable_last_instr/vsd` before flushing and restores them
/// after the snapshot is built, but the shared shadow has no
/// symmetric save/restore — mirroring the override there would leak
/// the pre-opcode value into the JUMP path.
///
/// `static_field_name` matches the canonical PyFrame virtualizable
/// spec at `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`
/// (`last_instr`, `pycode`, `valuestackdepth`, `debugdata`,
/// `lastblock`, `w_globals`).
///
/// No-op when `virtualizable_boxes` is not yet seeded
/// (non-virtualizable trace, or before `init_virtualizable_boxes`).
pub(crate) fn mirror_vable_static_to_boxes(
    ctx: &mut TraceCtx,
    static_field_name: &str,
    opref: OpRef,
    concrete: Value,
) {
    if !ctx.has_virtualizable_boxes() {
        return;
    }
    let idx = ctx
        .virtualizable_info()
        .and_then(|info| info.static_field_index_by_name(static_field_name));
    if let Some(idx) = idx {
        ctx.set_virtualizable_entry_at(idx, opref, concrete);
    }
}

/// Resolve the mutable frame-mirror index for a stack slot.
///
/// RPython `pyjitpl.py` keeps each kind-specific register bank indexed by
/// post-regalloc register number. Pyre still uses `registers_r` as a
/// semantic mirror for `locals_cells_stack_w`, so stack writers must not use
/// post-regalloc colors here: stack colors can legally coalesce with dead
/// local colors and would overwrite the local mirror before loop-close and
/// guard snapshots consume it. The encoder builds the color-indexed Ref bank
/// separately from liveness and the virtualizable shadow.
pub(crate) fn stack_slot_reg_idx(sym: &PyreSym, stack_idx: usize) -> usize {
    sym.nlocals + stack_idx
}

/// Write a Ref-boxed value to the symbolic operand stack at depth
/// offset `stack_idx`. Centralizes the dual-shadow update that
/// `push_typed_value`, `finishframe_exception`'s exception/lasti push,
/// the `caller_result_stack_idx` writeback (metainterp.rs:475+) and
/// inline-call setup all duplicated:
///
/// - `registers_r[reg_idx]` — the semantic frame mirror slot
///   (`reg_idx == nlocals + stack_idx`).
/// - `virtualizable_boxes[NUM_VABLE_SCALARS + semantic_idx]` —
///   `locals_cells_stack_w` heap mirror, ALWAYS semantic-indexed
///   (`pyjitpl.py:1242-1247 _opimpl_setarrayitem_vable`).
/// - `symbolic_stack_types[stack_idx]` set to `Type::Ref` (every slot
///   of `locals_cells_stack_w` is W_Root per
///   `virtualizable.py:86-98 read_boxes`).
/// - `concrete_stack[stack_idx]` set to `concrete` for Box-identity
///   tracking.
///
/// Caller is responsible for:
/// - Wrapping Int/Float values via `wrapint` / `wrapfloat` BEFORE
///   calling so `boxed` is always Ref-typed.
/// - Advancing `valuestackdepth` (push) or leaving it (positional
///   write into an existing slot).
/// - Emitting the separate `_opimpl_setfield_vable_i(vsd, depth±1)`
///   IR op via `mirror_vable_static_to_boxes` when the operation
///   logically advances the frame's vsd field (push / pop).
pub(crate) fn write_stack_slot(
    sym: &mut PyreSym,
    ctx: &mut TraceCtx,
    stack_idx: usize,
    boxed: OpRef,
    concrete: ConcreteValue,
) {
    let semantic_idx = sym.nlocals + stack_idx;
    let reg_idx = stack_slot_reg_idx(sym, stack_idx);
    // Portal frames carry the authoritative
    // stack shadow on `virtualizable_boxes` (`pyjitpl.py:1242
    // _opimpl_setarrayitem_vable`). The companion read paths
    // (`read_stack_slot`, `read_live`, and the
    // `get_list_of_active_boxes` snapshot fallback) source
    // their portal-frame view from the vable shadow, so the pyre-only
    // `registers_r[reg_idx]` semantic-mirror write is dead for portal
    // frames.  Non-portal frames retain the lazy-fill mirror because
    // their read path still consults `registers_r[reg_idx]`.
    if !sym.owns_virtualizable_shadow() {
        if reg_idx >= sym.registers_r.len() {
            sym.registers_r.resize(reg_idx + 1, OpRef::NONE);
        }
        sym.registers_r[reg_idx] = boxed;
    }
    if stack_idx >= sym.symbolic_stack_types.len() {
        sym.symbolic_stack_types.resize(stack_idx + 1, Type::Ref);
    }
    sym.symbolic_stack_types[stack_idx] = Type::Ref;
    if stack_idx >= sym.concrete_stack.len() {
        sym.concrete_stack
            .resize(stack_idx + 1, ConcreteValue::Null);
    }
    sym.concrete_stack[stack_idx] = concrete;
    if sym.owns_virtualizable_shadow() {
        let flat_idx = crate::virtualizable_gen::NUM_VABLE_SCALARS + semantic_idx;
        // pyjitpl.py:1242-1247 _opimpl_setarrayitem_vable: a Ref/Null
        // concrete carries a real W_Root heap pointer; update both
        // halves of the shadow. Int/Float concrete means pyre's lazy
        // wrapint/wrapfloat emitted a NewWithVtable OpRef without
        // allocating yet — update only the OpRef half so
        // synchronize_virtualizable keeps writing the existing W_Root.
        match concrete.to_ir_ref_value() {
            Some(v) => {
                ctx.set_virtualizable_entry_at(flat_idx, boxed, v);
            }
            None => {
                ctx.set_virtualizable_box_at(flat_idx, boxed);
            }
        }
    }
}

/// Write an inline callee frame's live state back to its heap `PyFrame`
/// before a loop-token CALL_ASSEMBLER (opimpl_jit_merge_point
/// portal_call_depth>0 → do_recursive_call, pyjitpl.py:1579-1602).
///
/// The callee's compiled loop reads `locals_cells_stack_w` /
/// `last_instr` / `valuestackdepth` from the frame object at entry
/// (virtualizable.py:86-98 read_boxes), but the inlined prefix advanced
/// those values only in the symbolic register banks; the runtime frame
/// still holds its creation-time state (call args). Emit the
/// virtualizable write_boxes shape (virtualizable.py:99-110):
/// SETFIELD_GC for the per-call statics + SETARRAYITEM_GC per live
/// boxed array slot (unboxed int/float slots stay residual helper CALLs
/// that box runtime-side). Slots never touched symbolically
/// (`OpRef::NONE`) keep their runtime creation value.
///
/// The other four statics (`pycode`, `debugdata`, `lastblock`,
/// `w_globals`) are correct from frame creation and have no mutators
/// under CPython 3.14 bytecode (see `flush_to_frame_for_guard`).
pub(crate) fn gen_writeback_inline_frame_to_heap(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    frame_opref: OpRef,
    target_pc: usize,
    valuestackdepth: usize,
) {
    let info = crate::frame_layout::build_pyframe_virtualizable_info();

    // last_instr = target_pc - 1 so the compiled loop's next_instr()
    // lands on the merge point (pyjitpl.py:2973 reached_loop_header pin).
    let last_instr = ctx.const_int(target_pc as i64 - 1);
    if let Some(idx) = info.static_field_index_by_name("last_instr") {
        let descr = info.static_field_descr(idx);
        ctx.vable_setfield_descr(frame_opref, last_instr, descr);
    }
    let vsd = ctx.const_int(valuestackdepth as i64);
    if let Some(idx) = info.static_field_index_by_name("valuestackdepth") {
        let descr = info.static_field_descr(idx);
        ctx.vable_setfield_descr(frame_opref, vsd, descr);
    }

    // locals_cells_stack_w items. Boxed (Ref) slots are written back with
    // an inline `SetarrayitemGc` into `locals_cells_stack_w`, the same
    // primitive `gen_store_back_in_vable` (trace_ctx.rs) uses for the
    // vable array items. The array base, item descr
    // (`pyobject_gcarray_descr`) and flat slot index match the
    // `trace_array_getitem_value` read path the compiled loop uses at
    // entry, so the optimizer's heapcache pairs them. Among the boxed
    // values, any virtual is forced by `OptVirtualize` when stored
    // through the array (write_boxes parity, virtualizable.py:99-110).
    //
    // Unboxed int/float slots have no W_Root to store, so they stay
    // residual `jit_frame_set_slot_{int,float}` CALLs that box runtime-
    // side (w_int_new / w_float_new) — keeping the boxing out of the
    // trace. GC visibility of the stored refs between these stores and
    // the compiled loop's entry loads is covered by
    // `walk_jit_callee_frame_roots` (pyre-jit::call_jit) — the heap frame
    // sits on no `CURRENT_FRAME` chain while compiled code runs.
    let cb = crate::callbacks::get();
    let mut array_ref = OpRef::NONE;
    for slot in 0..valuestackdepth {
        let Some(&value) = sym.registers_r.get(slot) else {
            break;
        };
        if value == OpRef::NONE {
            continue;
        }
        let index = ctx.const_int(slot as i64);
        match ctx.get_opref_type(value) {
            Some(Type::Int) => ctx.call_void_typed(
                cb.jit_frame_set_slot_int,
                &[frame_opref, index, value],
                &[Type::Ref, Type::Int, Type::Int],
            ),
            Some(Type::Float) => ctx.call_void_typed(
                cb.jit_frame_set_slot_float,
                &[frame_opref, index, value],
                &[Type::Ref, Type::Int, Type::Float],
            ),
            _ => {
                if array_ref == OpRef::NONE {
                    array_ref = frame_locals_cells_stack_array(ctx, frame_opref);
                }
                ctx.vable_setarrayitem_descr(array_ref, index, value, pyobject_gcarray_descr());
            }
        }
    }
}

/// Read the symbolic OpRef at depth offset `stack_idx`, with lazy
/// heap-fill from `locals_cells_stack_w` when the slot is empty.
/// Symmetric counterpart of `write_stack_slot`.
///
/// Reads the semantic frame mirror via `stack_slot_reg_idx`.  On
/// NONE-fill, the IR `getarrayitem` op still uses the SEMANTIC array
/// index (`locals_cells_stack_w[nlocals + stack_idx]`) — the heap layout
/// the array descr describes — and stores the result in the mirror slot
/// subsequent stack reads consult.
///
/// `init_symbolic` (state.rs:2785) leaves
/// `locals_cells_stack_array_ref = OpRef::NONE` for active-owner
/// traces because their locals come from `OpRef::input_arg_ref` and
/// stack writes route through the vable shadow, so the lazy-fill
/// path is normally unused.  In the rare case it does fire (e.g.
/// `pop_value` / `swap_stack_slots` reading a stack slot whose
/// `registers_r` entry was never written), emit the
/// `getfield_raw` for the array base on demand and cache it on the
/// sym so subsequent fills reuse the same op.  Without this guard,
/// `trace_array_getitem_value(NONE, idx)` would record a malformed
/// `GetarrayitemGcR` with a NONE base operand.
pub(crate) fn read_stack_slot(sym: &mut PyreSym, ctx: &mut TraceCtx, stack_idx: usize) -> OpRef {
    let semantic_idx = sym.nlocals + stack_idx;
    // For portal frames, read the stack slot
    // directly from the `virtualizable_boxes` shadow — PyPy-orthodox
    // (`pyjitpl.py:1230 _opimpl_getarrayitem_vable`).  Empirical verification
    // (`PYRE_PATH3_VERIFY_STACK_READ`) showed zero mismatch between
    // the vable shadow and the legacy `registers_r[reg_idx]` semantic-mirror
    // value across 9 benches.  Routing through vable retires one dependency
    // on the `registers_r` semantic-mirror deviation.
    //
    // Non-portal frames keep the `registers_r` lazy-fill path below — they
    // don't own a vable shadow.  Their semantic-mirror is not yet retired.
    if sym.owns_virtualizable_shadow() {
        let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
        if let Some(v) = ctx.virtualizable_box_at(nvs + semantic_idx) {
            return v;
        }
    }
    let reg_idx = stack_slot_reg_idx(sym, stack_idx);
    if reg_idx >= sym.registers_r.len() {
        sym.registers_r.resize(reg_idx + 1, OpRef::NONE);
    }
    if sym.registers_r[reg_idx] == OpRef::NONE {
        if sym.locals_cells_stack_array_ref == OpRef::NONE {
            sym.locals_cells_stack_array_ref = frame_locals_cells_stack_array(ctx, sym.frame);
        }
        let idx_const = ctx.const_int(semantic_idx as i64);
        sym.registers_r[reg_idx] =
            trace_array_getitem_value(ctx, sym.locals_cells_stack_array_ref, idx_const);
    }
    sym.registers_r[reg_idx]
}

/// Swap two operand-stack slots — third member of the
/// `read_stack_slot` / `write_stack_slot` family. Pre-fills both
/// slots through `read_stack_slot`, swaps the registers_r entries,
/// `symbolic_stack_types`, `concrete_stack`, and the vable shadow's
/// `(OpRef, Value)` pairs atomically.
///
/// `virtualizable_boxes` is the single source of truth for the frame's
/// Ref array (opencoder.py:718); reading each half via
/// `concrete_of_opref` separately would drop non-const Box identity
/// into the sentinel fallback, hence the `virtualizable_entry_at`
/// pair-read+pair-write.
///
/// `reg_top` / `reg_other` are semantic frame-mirror indices.
pub(crate) fn swap_stack_slots(
    sym: &mut PyreSym,
    ctx: &mut TraceCtx,
    top_idx: usize,
    other_idx: usize,
) {
    let _ = read_stack_slot(sym, ctx, top_idx);
    let _ = read_stack_slot(sym, ctx, other_idx);
    let semantic_top = sym.nlocals + top_idx;
    let semantic_other = sym.nlocals + other_idx;
    let reg_top = stack_slot_reg_idx(sym, top_idx);
    let reg_other = stack_slot_reg_idx(sym, other_idx);
    if reg_top != reg_other {
        sym.registers_r.swap(reg_top, reg_other);
    }
    if top_idx < sym.symbolic_stack_types.len() && other_idx < sym.symbolic_stack_types.len() {
        sym.symbolic_stack_types.swap(top_idx, other_idx);
    }
    if top_idx < sym.concrete_stack.len() && other_idx < sym.concrete_stack.len() {
        sym.concrete_stack.swap(top_idx, other_idx);
    }
    if sym.owns_virtualizable_shadow() {
        let flat_top = crate::virtualizable_gen::NUM_VABLE_SCALARS + semantic_top;
        let flat_other = crate::virtualizable_gen::NUM_VABLE_SCALARS + semantic_other;
        if let (Some((op_top, val_top)), Some((op_other, val_other))) = (
            ctx.virtualizable_entry_at(flat_top),
            ctx.virtualizable_entry_at(flat_other),
        ) {
            ctx.set_virtualizable_entry_at(flat_top, op_other, val_other);
            ctx.set_virtualizable_entry_at(flat_other, op_top, val_top);
        } else {
            panic!(
                "swap_stack_slots: missing virtualizable_boxes entries for stack slots {top_idx} and {other_idx}"
            );
        }
    }
}

impl MIFrame {
    #[allow(dead_code)]
    fn active_execution_context(&self) -> *const pyre_interpreter::PyExecutionContext {
        let exec_ctx = self.sym().concrete_execution_context;
        if !exec_ctx.is_null() {
            return exec_ctx;
        }
        if self.concrete_frame_addr != 0 {
            let frame = unsafe {
                &*(self.concrete_frame_addr as *const pyre_interpreter::pyframe::PyFrame)
            };
            return frame.execution_context;
        }
        std::ptr::null()
    }

    /// Get the concrete return value from the frame's stack top.
    #[allow(dead_code)]
    fn concrete_stack_value_at_return(&self) -> Option<PyObjectRef> {
        // MIFrame Box tracking: read from concrete_stack
        let s = self.sym();
        if s.valuestackdepth > 0 {
            let v = s.concrete_value_at(s.valuestackdepth - 1);
            if !v.is_null() {
                return Some(v.to_pyobj());
            }
        }
        None
    }

    pub(crate) fn next_instruction_consumes_comparison_truth(&self) -> bool {
        let code = unsafe { &*(*self.sym().jitcode).raw_code() };
        // RPython optimize_goto_if_not works on the semantic successor,
        // not on bytecode trivia like EXTENDED_ARG/NOT_TAKEN/CACHE.
        let mut pc = self.fallthrough_pc;
        loop {
            match decode_instruction_at(code, pc) {
                Some((instruction, _))
                    if instruction_is_trivia_between_compare_and_branch(instruction) =>
                {
                    pc += 1
                }
                Some((instruction, _)) => {
                    return instruction_consumes_comparison_truth(instruction);
                }
                None => return false,
            }
        }
    }

    pub fn from_sym(
        ctx: &mut TraceCtx,
        sym: &mut PyreSym,
        concrete_frame: usize,
        fallthrough_pc: usize,
        opcode_start_pc: usize,
    ) -> Self {
        // sym was initialized when its owning MetaInterpFrame was pushed
        // (trace.rs root push / metainterp.rs perform_call). MIFrame is a
        // borrowed per-instruction view; no re-initialization here.
        // RPython pyjitpl.py: orgpc = opcode start PC passed to each handler.
        let orgpc = opcode_start_pc;
        Self {
            ctx,
            sym,
            fallthrough_pc,
            concrete_frame_addr: concrete_frame,
            orgpc,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
        }
    }

    #[doc(hidden)]
    pub fn capture_current_fail_args(&mut self) -> Vec<OpRef> {
        self.with_ctx(|this, ctx| this.current_fail_args(ctx))
    }

    #[doc(hidden)]
    pub fn capture_guard_class(
        &mut self,
        obj: OpRef,
        expected_type: *const pyre_object::pyobject::PyType,
    ) {
        self.with_ctx(|this, ctx| this.guard_class(ctx, obj, expected_type));
    }

    #[doc(hidden)]
    pub fn capture_trace_guarded_int_payload(&mut self, int_obj: OpRef) -> OpRef {
        self.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, int_obj))
    }

    #[doc(hidden)]
    pub fn capture_record_branch_guard(
        &mut self,
        exitvalue: OpRef,
        condition: OpRef,
        branch_concrete: bool,
        next_instr: usize,
    ) {
        self.with_ctx(|this, ctx| {
            this.record_branch_guard(ctx, exitvalue, condition, branch_concrete, next_instr);
        });
    }

    #[doc(hidden)]
    pub fn capture_generate_guard(&mut self, opcode: OpCode, args: &[OpRef]) {
        self.with_ctx(|this, ctx| this.generate_guard(ctx, opcode, args));
    }

    #[doc(hidden)]
    pub fn capture_concrete_branch_truth_for_value(
        &mut self,
        value: OpRef,
        concrete_val: pyre_object::PyObjectRef,
    ) -> Result<bool, PyError> {
        self.concrete_branch_truth_for_value(value, concrete_val)
    }

    #[doc(hidden)]
    pub fn capture_trace_dynamic_list_index(
        &mut self,
        key: OpRef,
        len: OpRef,
        concrete_key: i64,
    ) -> OpRef {
        self.with_ctx(|this, ctx| this.trace_dynamic_list_index(ctx, key, len, concrete_key))
    }

    #[doc(hidden)]
    pub fn capture_generated_list_getitem_by_strategy(
        &mut self,
        list: OpRef,
        key: OpRef,
        concrete_key: i64,
        strategy_id: i64,
    ) -> OpRef {
        self.with_ctx(|this, ctx| {
            crate::generated_list_getitem_by_strategy(
                this,
                ctx,
                list,
                key,
                concrete_key,
                strategy_id,
            )
        })
    }

    #[doc(hidden)]
    pub fn capture_list_append_value(
        &mut self,
        list: OpRef,
        value: OpRef,
        concrete_list: pyre_object::PyObjectRef,
        concrete_value: pyre_object::PyObjectRef,
    ) -> Result<(), PyError> {
        self.list_append_value(list, value, concrete_list, concrete_value)
    }

    #[doc(hidden)]
    pub fn capture_direct_len_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: pyre_object::PyObjectRef,
    ) -> Result<OpRef, PyError> {
        self.direct_len_value(callable, value, concrete_value)
    }

    #[doc(hidden)]
    pub fn capture_iter_next_value(
        &mut self,
        iter: OpRef,
        concrete_iter: pyre_object::PyObjectRef,
    ) -> Result<FrontendOp, PyError> {
        self.iter_next_value(iter, concrete_iter)
    }

    #[doc(hidden)]
    pub fn capture_close_loop_args_at(&mut self, target_pc: Option<usize>) -> Vec<OpRef> {
        self.with_ctx(|this, ctx| this.close_loop_args_at(ctx, target_pc))
    }

    #[doc(hidden)]
    pub fn symbolic_nlocals(&self) -> usize {
        self.sym().nlocals
    }

    #[doc(hidden)]
    pub fn symbolic_valuestackdepth(&self) -> usize {
        self.sym().valuestackdepth
    }

    /// Read `PyFrame.valuestackdepth` directly from the concrete frame at
    /// `concrete_frame_addr`.  The orthodox
    /// PyPy-parity replacement for `self.sym().valuestackdepth`.
    ///
    /// RPython has no symbolic mirror of the Python stack — `MIFrame` only
    /// holds the per-jitcode-invocation register banks (`registers_r/i/f`),
    /// and the user-side stack lives in `PyFrame.locals_cells_stack_w` /
    /// `PyFrame.valuestackdepth` accessed via IR `getfield/setfield` on the
    /// virtualizable.  Pyre's `PyreSym.valuestackdepth` is a newly
    /// introduced divergence (a symbolic mirror) that drifts from
    /// `PyFrame.valuestackdepth` whenever the production walker handles an
    /// opcode that mutates the concrete stack (the walker records the
    /// residual_call but does not run `MIFrame::pop_value`'s `sym.valuestackdepth -= 1`).
    ///
    /// Returns `None` when:
    /// 1. `concrete_frame_addr == 0` (tests constructing a sym-only `MIFrame`)
    /// 2. `self.parent_frames` is non-empty (this MIFrame is an inline
    ///    callee — its `concrete_frame_addr` is the heap PyFrame snapshot
    ///    frozen at CALL entry and does not advance during inline body
    ///    tracing.  The live traced depth for an inline frame lives only in
    ///    `sym.valuestackdepth`, which the walker / trait dispatch update
    ///    via `push_typed_value` / `pop_value` as the inline body executes).
    ///
    /// Production tracer paths always seed `concrete_frame_addr` from the
    /// live `PyFrame` for the top frame.
    pub(crate) fn concrete_valuestackdepth(&self) -> Option<usize> {
        if !self.parent_frames.is_empty() {
            return None;
        }
        crate::state::concrete_stack_depth(self.concrete_frame_addr)
    }

    #[doc(hidden)]
    pub fn symbolic_registers_r(&self) -> &[OpRef] {
        &self.sym().registers_r
    }

    #[doc(hidden)]
    pub fn capture_value_type(&self, opref: OpRef) -> Type {
        self.value_type(opref)
    }

    pub(crate) fn ctx(&mut self) -> &mut TraceCtx {
        unsafe { &mut *self.ctx }
    }

    pub(crate) fn with_ctx<R>(&mut self, f: impl FnOnce(&mut Self, &mut TraceCtx) -> R) -> R {
        let ctx = self.ctx;
        unsafe { f(self, &mut *ctx) }
    }

    #[inline]
    pub(crate) fn sym(&self) -> &PyreSym {
        unsafe { &*self.sym }
    }

    #[inline]
    pub(crate) fn sym_mut(&mut self) -> &mut PyreSym {
        unsafe { &mut *self.sym }
    }

    /// #143 frame-advance gate: mark this trace as having performed a concrete
    /// heap mutation during tracing. Called exactly where the concrete
    /// mutation is a certainty: the BUILTIN-form list dispatch arms
    /// (append/pop/pop-at/reverse — the trait impl's `call_callable` executed
    /// the builtin concretely before dispatch) and the LIST_APPEND opcode hook
    /// (the eval loop's `execute_opcode_step` / inline-frame
    /// `concrete_execute_step` performs the write).
    ///
    /// Deliberately NOT marked: deferred stores (`STORE_SUBSCR` — the compiled
    /// loop performs the write exactly once, nbody/fannkuch), non-mutating
    /// calls, and the W_MethodObject method-form arms (`m = xs.pop; m(0)`) —
    /// the trait impl only executes `is_function` callables concretely, so no
    /// during-trace mutation happens on that path and marking it would
    /// wrongly advance past an iteration whose mutation only exists as
    /// deferred IR. (Those traces currently always abort before CloseLoop;
    /// unmarked-deferred stays correct if they ever close.)
    #[inline]
    pub(crate) fn dm143_mark_heap_mutated(&mut self) {
        self.sym_mut().dm143_heap_mutated = true;
    }

    pub(crate) fn frame(&self) -> OpRef {
        self.sym().frame
    }

    /// `pypy/module/pypyjit/interp_jit.py:67 reds = ['frame', 'ec']` requires
    /// every CALL_ASSEMBLER red-args list and JUMP-args list to carry ec.
    /// Normal trace setup seeds `sym.execution_context`; this recovery keeps
    /// adapter paths from passing OpRef::NONE as the ec red.
    pub(crate) fn ensure_execution_context(&mut self, ctx: &mut TraceCtx) -> OpRef {
        let ec = self.sym().execution_context;
        if !ec.is_none() {
            return ec;
        }
        let recovered = ctx.record_op_with_descr(
            majit_ir::OpCode::GetfieldGcR,
            &[self.frame()],
            crate::descr::pyframe_execution_context_descr(),
        );
        self.sym_mut().execution_context = recovered;
        recovered
    }

    #[inline]
    fn capture_pre_opcode_state(&mut self) {
        // SAFETY: `self.ctx` is initialized at MIFrame construction and
        // outlives this call (TraceCtx is pinned for the tracing
        // session). `&self.ctx` borrow is independent of the `&self.sym`
        // borrow we take below — both come from raw pointers stored on
        // self, not nested borrows.
        let ctx: &TraceCtx = unsafe { &*self.ctx };
        // portal-bridge keeps `s.valuestackdepth` at its initial seed
        // because residual-call paths bypass `push_typed_value` /
        // `pop_value` (`portal_bridge_vable_vsd` doc at line 2053-2074).
        // Consult metadata at `self.orgpc` so the shadow snapshot covers
        // the actual live extent at the current opcode, not the stale
        // symbolic counter.
        let portal_vsd = self.portal_bridge_vable_vsd(self.orgpc).map(|d| d as usize);
        // prefix_len fallback reads
        // `PyFrame.valuestackdepth` rather than the symbolic mirror.
        // `capture_pre_opcode_state` runs at the orgpc anchor where
        // PyFrame holds the pre-opcode state (the interpreter step has
        // not run yet) — exactly what the prefix snapshot needs.
        let concrete_vsd = self
            .concrete_valuestackdepth()
            .unwrap_or_else(|| self.sym().valuestackdepth);
        let s = self.sym();
        let owns_shadow = s.owns_virtualizable_shadow();
        let nlocals = s.nlocals;
        // pyjitpl.py:2954-2965 parity: snapshot the
        // `virtualizable_boxes[NUM_VABLE_SCALARS + i]` view of locals
        // and stack tail. The shadow is RPython's single source of truth
        // for `locals_cells_stack_w`; both halves are mirrored by
        // `push_typed_value` / `store_local_value` for every trace that
        // satisfies `owns_virtualizable_shadow()` (loop portal +
        // bridges with seeded `bridge_local_oprefs`). Non-owner traces
        // keep the semantic `registers_r` mirror snapshot.
        //
        // Prefix length: when reading from the shadow, use the full
        // `valuestackdepth` — the shadow covers the entire `nlocals +
        // co_stacksize` frame array regardless of `registers_r`
        // occupancy, and capping at `registers_r.len()` would silently
        // drop live shadow slots once `registers_r` lags behind the
        // operand stack.
        let prefix_len = portal_vsd.unwrap_or(concrete_vsd);
        let snapshot = if owns_shadow && prefix_len >= nlocals {
            let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
            let mut snapshot = Vec::with_capacity(prefix_len);
            for i in 0..nlocals {
                snapshot.push(
                    ctx.virtualizable_box_at(nvs + i)
                        .expect("capture_pre_opcode_state: missing virtualizable local box"),
                );
            }
            for d in 0..(prefix_len - nlocals) {
                snapshot.push(
                    ctx.virtualizable_box_at(nvs + nlocals + d)
                        .expect("capture_pre_opcode_state: missing virtualizable stack box"),
                );
            }
            snapshot
        } else {
            s.registers_r.clone()
        };
        self.pre_opcode_semantic_depth = Some(prefix_len);
        self.pre_opcode_registers_r = Some(snapshot);
    }

    #[inline]
    fn clear_pre_opcode_state(&mut self) {
        self.pre_opcode_registers_r = None;
        self.pre_opcode_semantic_depth = None;
    }

    /// Pre-opcode stack depth: snapshot-captured `pre_opcode_semantic_depth`
    /// when available, otherwise the concrete `PyFrame.valuestackdepth`
    /// (which holds the same pre-opcode state because the interpreter step
    /// for this opcode has not run yet); falls back to the symbolic
    /// `sym.valuestackdepth` only when `concrete_frame_addr == 0`
    /// (unit tests constructing sym-only `MIFrame`s).
    ///
    /// Replaces the `pre_opcode_depth_or(self.sym().valuestackdepth)`
    /// pattern.  The `pre_opcode_*` machinery exists precisely because
    /// pyre's `MIFrame::pop_value` / `push_typed_value` mutate
    /// `sym.valuestackdepth` mid-opcode and the guard/snapshot writers
    /// need the *pre-mutation* value.  Reading directly from PyFrame
    /// makes that pre-mutation guarantee structural rather than
    /// snapshot-bookkeeping-dependent.
    #[inline]
    fn pre_opcode_concrete_depth(&self) -> usize {
        self.pre_opcode_semantic_depth.unwrap_or_else(|| {
            self.concrete_valuestackdepth()
                .unwrap_or_else(|| self.sym().valuestackdepth)
        })
    }

    fn materialize_fail_arg_slot(
        &mut self,
        ctx: &mut TraceCtx,
        slot: OpRef,
        slot_type: Type,
        abs_idx: usize,
    ) -> OpRef {
        if !slot.is_none() {
            return slot;
        }
        let concrete_value = self.concrete_at(abs_idx).unwrap_or(PY_NULL);
        let typed_value = extract_concrete_typed_value(slot_type, concrete_value);
        fail_arg_opref_for_typed_value(ctx, typed_value)
    }

    /// `pyjitpl.py:177` `get_list_of_active_boxes` parity. Returns
    /// compact register boxes for live registers only.
    ///
    /// Both the tracer (here) and the blackhole bridge-resume decoder
    /// (`consume_one_section`, `resume.py:1381`) read the same
    /// `all_liveness` byte stream via `jitcode.get_live_vars_info(pc,
    /// op_live)` (`jitcode.py:82-93`) and iterate the per-bank register
    /// indices with `LivenessIterator` (`liveness.py:168-201`). One
    /// source, same order.
    fn get_list_of_active_boxes(
        &mut self,
        ctx: &mut TraceCtx,
        in_a_call: bool,
        after_residual_call: bool,
        top_frame_marker_call_pc: Option<usize>,
    ) -> Vec<OpRef> {
        // pyjitpl.py:194: in_a_call or after_residual_call → self.pc
        let live_pc = if in_a_call || after_residual_call {
            self.fallthrough_pc
        } else {
            self.orgpc
        };
        // resume.py:1045 consume_one_section invariant: every register
        // reported as live must be reachable via a valid OpRef. RPython
        // trivially satisfies this because every read populates
        // `registers_r[i]`. pyre's `registers_r` is the unified
        // abstract register file — locals occupy `[..nlocals]` and the
        // live stack tail occupies `[nlocals..nlocals+stack_only]`
        // (pyjitpl.py:70-78 MIFrame parity). A live register that the
        // trace has not yet produced (forward-live local across a
        // superinstruction edge, live stack slot resurrected after a
        // guard backtrack) keeps `OpRef::NONE`, poisoning the guard's
        // fail_args. Mirror RPython's invariant by forcing lazy init
        // for every live register via the same `_opimpl_getarrayitem
        // _vable` mirror read that LOAD_FAST uses (load_local_value
        // at trace_opcode.rs:660), BEFORE snapshotting registers_r
        // below. Source for the live indices is the same packed
        // `all_liveness` byte stream (`jitcode.get_live_vars_info(pc,
        // op_live)` at `jitcode.py:82-93`) that resume.py uses at
        // decode time — pyjitpl.py:218-225 `get_list_of_active_boxes`
        // analog, walking the full live register-file set.
        #[derive(Clone, Copy)]
        enum LiveBank {
            Int,
            Ref,
            Float,
        }
        impl LiveBank {
            #[allow(dead_code)]
            fn ty(self) -> Type {
                match self {
                    LiveBank::Int => Type::Int,
                    LiveBank::Ref => Type::Ref,
                    LiveBank::Float => Type::Float,
                }
            }
        }

        let jitcode_ptr_pre = self.sym().jitcode;
        // `pyjitpl.py:194-198`: a single `pc` drives both the result-box
        // clear and the liveness decode.  Resolve the resume jitcode pc once
        // here so the lazy-load preamble fills exactly the registers the
        // snapshot below reads, routed through the SAME `-live-` the snapshot
        // pc resolves to (`marker_aware_resume_pc` /
        // `marker_aware_parent_resume_pc`).  A try-block residual call resumes
        // at its OWN post-call `-live-`/catch via
        // `after_residual_call_resume_pc_for`:
        //   * in_a_call parent → the parent's stored CALL pc
        //     (`residual_call_pc`).
        //   * after_residual_call top frame → the CALL pc the caller folded
        //     into the snapshot pc (`Some(orgpc)` for the marker-routed
        //     GUARD_NOT_FORCED / GUARD_NO_EXCEPTION guards; `None` for the
        //     GUARD_EXCEPTION path, which carries the exception via the
        //     `jf_guard_exc` channel and resumes at a plain fallthrough pc).
        // No marker entry falls back to the fallthrough `-live-`.  Splitting
        // the two — preamble/boxes at the fallthrough `-live-`, snapshot at
        // the post-call `-live-` — would make the decoder consume a different
        // box count than the encoder wrote.
        let resume_jit_pc: Option<usize> = unsafe {
            let jc = &*jitcode_ptr_pre;
            if jc.payload.metadata.pc_map.is_empty() {
                None
            } else {
                let marker_call_pc = if in_a_call {
                    self.residual_call_pc
                } else if after_residual_call {
                    top_frame_marker_call_pc
                } else {
                    // Pre-call top-frame guard: resume at the plain `live_pc`
                    // `-live-`, matching `marker_aware_resume_pc`'s
                    // `wants_marker = after_residual_call && ...`.  Routing
                    // through the post-call marker here would make the box list
                    // shorter than the box count the decoder reads at the
                    // recorded (pre-call) snapshot position.
                    None
                };
                match marker_call_pc
                    .and_then(|call_pc| jc.payload.after_residual_call_resume_pc_for(call_pc))
                    .or_else(|| jc.payload.resume_jitcode_pc_for(live_pc))
                {
                    Some(jit_pc) => Some(jit_pc),
                    None => {
                        // This (parent) frame reports a `live_pc` the jitcode
                        // `pc_map` has no entry for — the cross-frame snapshot
                        // coordinate gap (#124/#130): an inlined callee +
                        // exception-resume shape whose parent resume pc was
                        // never recorded.  Building the guard from this frame
                        // would emit incorrect resume data, so request a trace
                        // abort and return no active boxes.  The recorded guard
                        // is thrown away with the aborted (pre-install) trace,
                        // so the empty list — already a valid return for the
                        // skeleton / short-liveness paths below — is harmless.
                        crate::state::request_trace_abort();
                        return Vec::new();
                    }
                }
            }
        };
        let live_regs_for_banks: Vec<(LiveBank, usize)> = unsafe {
            let jc = &*jitcode_ptr_pre;
            // Skeleton payload (no `pc_map` yet) → skip the lazy-load
            // preamble; the main path's skeleton-fallback branch
            // handles the same case.
            //
            // Portal-bridge (G.4.4-encoder.2/3): the portal jitcode
            // has no per-Python-PC `pc_map` because user opcodes are
            // dispatched by canonical portal arms at runtime.  The
            // RPython orthodox encoder (`pyjitpl.py:218-225`) reads
            // every live register unconditionally.  Pyre's portal-
            // bridge install seeds `metadata.stack_base` (G.3h) +
            // `depth_at_py_pc[pc]` (G.4.2) so the live ref slots at
            // this PC span `0..stack_base + depth` — locals + cells +
            // operand-stack tail.  The full range is covered; the
            // lazy-load preamble (lines 555+) routes each through
            // `load_local_value` whose vable-shadow read (line 1218)
            // works for any flat slot in `vb[NUM_VABLE_SCALARS..
            // NUM_VABLE_SCALARS + nlocals + ncells + stackdepth]`.
            // Portal-bridge has no regalloc so colors == slot indices
            // (identity); the `is_portal_bridge` guard in the Ref-bank
            // materialization loop below provides the same bypass.
            if jc.payload.metadata.pc_map.is_empty() {
                if jc.payload.is_portal_bridge() {
                    let stack_base = jc.payload.metadata.stack_base;
                    let depth = jc
                        .payload
                        .metadata
                        .depth_at_py_pc
                        .get(live_pc)
                        .copied()
                        .unwrap_or(0) as usize;
                    (0..stack_base + depth)
                        .map(|idx| (LiveBank::Ref, idx))
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                // RPython `pyjitpl.py:218-225` reads each liveness bank
                // from its matching register file. Pyre's unified semantic
                // stack can hold an OpRef before the kind bank has been
                // populated, so collect every listed bank/index and complete
                // the matching bank immediately before the direct snapshot.
                let jit_pc =
                    resume_jit_pc.expect("pc_map non-empty branch above ensures lookup hits");
                let op_live = crate::state::op_live();
                let off = jc.payload.jitcode.get_live_vars_info(jit_pc, op_live);
                let all_liveness = crate::state::liveness_info_snapshot();
                if off + 2 >= all_liveness.len() {
                    Vec::new()
                } else {
                    let length_i = all_liveness[off] as u32;
                    let length_r = all_liveness[off + 1] as u32;
                    let length_f = all_liveness[off + 2] as u32;
                    let mut cursor = off + 3;
                    let mut out: Vec<(LiveBank, usize)> =
                        Vec::with_capacity((length_i + length_r + length_f) as usize);
                    use majit_translate::liveness::LivenessIterator;
                    if length_i != 0 {
                        let mut it = LivenessIterator::new(cursor, length_i, &all_liveness);
                        while let Some(reg_idx) = it.next() {
                            out.push((LiveBank::Int, reg_idx as usize));
                        }
                        cursor = it.offset;
                    }
                    if length_r != 0 {
                        let mut it = LivenessIterator::new(cursor, length_r, &all_liveness);
                        while let Some(reg_idx) = it.next() {
                            out.push((LiveBank::Ref, reg_idx as usize));
                        }
                        cursor = it.offset;
                    }
                    if length_f != 0 {
                        let mut it = LivenessIterator::new(cursor, length_f, &all_liveness);
                        while let Some(reg_idx) = it.next() {
                            out.push((LiveBank::Float, reg_idx as usize));
                        }
                    }
                    out
                }
            }
        };
        // portal-bridge has stale `s.valuestackdepth` (residual-call paths
        // bypass push/pop — see `portal_bridge_vable_vsd` doc); consult
        // metadata at `live_pc` so the shadow gate's `live_max` reflects
        // the actual pyframe stack depth at the resume point. live_pc is
        // either `orgpc` (the standard get_list_of_active_boxes path) or
        // `fallthrough_pc` (in_a_call / after_residual_call), per RPython
        // pyjitpl.py:194-198 — pre_opcode_registers_r is captured at orgpc
        // and would mis-size live_max for the fallthrough_pc resume.
        let portal_live_vsd = self.portal_bridge_vable_vsd(live_pc).map(|d| d as usize);
        let (
            nlocals,
            valid_stack_only,
            jitcode_ptr,
            local_color_map,
            stack_slot_color_map,
            live_local_indices,
            is_portal_bridge,
        ) = {
            let s = self.sym();
            let (
                local_color_map,
                stack_slot_color_map,
                live_local_indices,
                is_portal_bridge,
                metadata_stack_depth,
            ) = if s.jitcode.is_null() {
                (Vec::new(), Vec::new(), Vec::new(), false, None)
            } else {
                unsafe {
                    let jc = &*s.jitcode;
                    let live_local_indices = if jc.payload.code_ptr.is_null() {
                        Vec::new()
                    } else {
                        let live_vars = crate::liveness::liveness_for(jc.payload.code_ptr);
                        (0..s.nlocals)
                            .filter(|&idx| live_vars.is_local_live(live_pc, idx))
                            .collect()
                    };
                    (
                        jc.payload.metadata.pyre_color_for_semantic_local.clone(),
                        jc.payload.metadata.stack_slot_color_map.clone(),
                        live_local_indices,
                        jc.payload.is_portal_bridge(),
                        jc.payload
                            .metadata
                            .depth_at_py_pc
                            .get(live_pc)
                            .copied()
                            .map(|d| d as usize),
                    )
                }
            };
            let valid_stack_only = if let Some(vsd) = portal_live_vsd {
                vsd.saturating_sub(s.nlocals)
            } else if self.pre_opcode_registers_r.is_some() {
                metadata_stack_depth.unwrap_or_else(|| s.valuestackdepth.saturating_sub(s.nlocals))
            } else {
                s.valuestackdepth.saturating_sub(s.nlocals)
            };
            (
                s.nlocals,
                valid_stack_only,
                s.jitcode,
                local_color_map,
                stack_slot_color_map,
                live_local_indices,
                is_portal_bridge,
            )
        };
        // SSA-authoritative live_r: Ref bank entries go
        // through the read_live / lazy-fill / materialize pipeline to
        // populate registers_r[color].  Int/Float banks already live in
        // their own register arrays (registers_i / registers_f) and the
        // clone at lines 1225-1227 captures them; materialization would
        // only corrupt those values by overwriting with a Ref-derived
        // fallback.  Skip non-Ref banks entirely.
        let mut bank_materializations: Vec<(LiveBank, usize, OpRef)> =
            Vec::with_capacity(live_regs_for_banks.len());
        for (bank, idx) in live_regs_for_banks {
            if !matches!(bank, LiveBank::Ref) {
                continue;
            }
            let color_idx = idx;
            // Derive semantic index for vable shadow / concrete_at.
            // A stack slot color may be coalesced with a local identity
            // color when the local is not live.  Mirror the decoder's
            // `semantic_ref_slot_for_reg_color`: consult the live stack
            // prefix first, and only fall back through the local color
            // map if no live stack slot owns this color.
            let Some(semantic_idx) = (if is_portal_bridge {
                Some(color_idx)
            } else {
                crate::state::semantic_ref_slot_for_reg_color(
                    nlocals,
                    valid_stack_only,
                    &local_color_map,
                    &stack_slot_color_map,
                    &live_local_indices,
                    color_idx,
                )
            }) else {
                continue;
            };
            {
                let s = self.sym_mut();
                if color_idx >= s.registers_r.len() {
                    s.registers_r.resize(color_idx + 1, OpRef::NONE);
                }
            }
            // pyjitpl.py:218-234 parity for the snapshot/fallback:
            // produce the color-indexed Ref bank, but source active
            // virtualizable frame slots from the semantic shadow. A
            // stack color may be coalesced with a dead local color;
            // reading `registers_r[color]` for the fallback would pick
            // up the stale local mirror for a live stack slot.
            let live_max = nlocals + valid_stack_only;
            let read_live = |this: &MIFrame, ctx: &TraceCtx| -> OpRef {
                let s = this.sym();
                if s.owns_virtualizable_shadow() && semantic_idx < live_max {
                    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                    return ctx
                        .virtualizable_box_at(nvs + semantic_idx)
                        .expect("get_list_of_active_boxes: missing vable frame box");
                }
                let val = s
                    .registers_r
                    .get(semantic_idx)
                    .copied()
                    .unwrap_or(OpRef::NONE);
                if val != OpRef::NONE {
                    return val;
                }
                OpRef::NONE
            };
            let live_value_pre = read_live(self, ctx);
            if live_value_pre == OpRef::NONE {
                if semantic_idx < nlocals {
                    let value = MIFrame::load_local_value(self, ctx, semantic_idx)
                        .expect("get_list_of_active_boxes: failed to lazy-load live local");
                    self.sym_mut().registers_r[color_idx] = value;
                } else {
                    // Stack lazy-fill: heap read at semantic index,
                    // store in both the semantic mirror (read_live) and
                    // the color bank (Ref-bank fail args).
                    let s = self.sym_mut();
                    if s.locals_cells_stack_array_ref == OpRef::NONE {
                        let frame_ref = s.frame;
                        s.locals_cells_stack_array_ref =
                            frame_locals_cells_stack_array(ctx, frame_ref);
                    }
                    let idx_const = ctx.const_int(semantic_idx as i64);
                    let arr = s.locals_cells_stack_array_ref;
                    let value = trace_array_getitem_value(ctx, arr, idx_const);
                    if semantic_idx >= s.registers_r.len() {
                        s.registers_r.resize(semantic_idx + 1, OpRef::NONE);
                    }
                    s.registers_r[semantic_idx] = value;
                    s.registers_r[color_idx] = value;
                }
            }
            let live_value = if live_value_pre == OpRef::NONE {
                read_live(self, ctx)
            } else {
                live_value_pre
            };
            let semantic_value = self
                .pre_opcode_registers_r
                .as_ref()
                // `capture_pre_opcode_state` stores a semantic frame
                // snapshot for both vable-owner and non-owner traces. A
                // live stack color may reuse a dead local color, so reading
                // the snapshot by color would capture the wrong local value.
                .and_then(|pre_r| pre_r.get(semantic_idx).copied())
                .filter(|value| !value.is_none())
                .unwrap_or(live_value);
            let bank_value =
                self.materialize_fail_arg_slot(ctx, semantic_value, Type::Ref, semantic_idx);
            bank_materializations.push((bank, idx, bank_value));
        }
        let (registers_i, registers_r_bank, registers_r_semantic, registers_f) = {
            let s = self.sym();
            // Unified abstract register file view.
            // When a guard is being captured mid-opcode, read from
            // `pre_opcode_registers_r` (the full snapshot of
            // `registers_r` at opcode start). Otherwise read the live
            // `registers_r`. Both variants share a single indexing rule
            // so `live_{i,r,f}_regs` indices (which live in the
            // stack_base=nlocals register space) can be resolved with
            // one lookup instead of the legacy
            // `idx < nlocals ? locals : stack` split.
            //
            // Dual-writes grow `registers_r` monotonically on
            // stack pushes; pop does not shrink it. Bound the view to
            // the valid locals + live stack_only range so stale slots
            // above the current (or pre-opcode) stack depth cannot
            // surface as active OpRefs. This matches the OLD
            // `stack_values.len()` bound on the
            // `stack_values[idx - nlocals]` read path.
            //
            // Portal-bridge (G.4.4-encoder.3): the trace's
            // `valuestackdepth` does not track user-bytecode stack
            // depth (the canonical portal jitcode owns the trace's
            // stack tracker, not user opcodes), so the per-CodeObject
            // bound `nlocals + valid_stack_only` undercounts.  The
            // metadata-derived `stack_base + depth_at_py_pc[pc]`
            // (G.3h + G.4.2) is the correct bound for portal-bridge
            // — symmetric with the encoder count side
            // (`state::frame_value_count_at` portal-bridge branch).
            let source_len = if let Some(ref pre_r) = self.pre_opcode_registers_r {
                pre_r.len()
            } else {
                s.registers_r.len()
            };
            let valid_len = if is_portal_bridge {
                let payload = unsafe { &(&*jitcode_ptr).payload };
                let stack_base = payload.metadata.stack_base;
                let depth = payload
                    .metadata
                    .depth_at_py_pc
                    .get(live_pc)
                    .copied()
                    .unwrap_or(0) as usize;
                (stack_base + depth).min(source_len)
            } else {
                (s.nlocals + valid_stack_only).min(source_len)
            };
            let mut registers_i = s.registers_i.clone();
            let mut registers_r_bank = s.registers_r.clone();
            let mut registers_f = s.registers_f.clone();
            for &(bank, reg_idx, value) in &bank_materializations {
                match bank {
                    LiveBank::Int => {
                        if reg_idx >= registers_i.len() {
                            registers_i.resize(reg_idx + 1, OpRef::NONE);
                        }
                        registers_i[reg_idx] = value;
                    }
                    LiveBank::Ref => {
                        if reg_idx >= registers_r_bank.len() {
                            registers_r_bank.resize(reg_idx + 1, OpRef::NONE);
                        }
                        registers_r_bank[reg_idx] = value;
                    }
                    LiveBank::Float => {
                        if reg_idx >= registers_f.len() {
                            registers_f.resize(reg_idx + 1, OpRef::NONE);
                        }
                        registers_f[reg_idx] = value;
                    }
                }
            }
            let mut registers_r_semantic: Vec<OpRef> =
                if let Some(ref pre_r) = self.pre_opcode_registers_r {
                    pre_r[..valid_len.min(pre_r.len())].to_vec()
                } else if s.owns_virtualizable_shadow() {
                    // Portal frames have the
                    // authoritative semantic-indexed shadow in
                    // `virtualizable_boxes` (`pyjitpl.py:1242
                    // _opimpl_setarrayitem_vable`).  When no opcode-start
                    // snapshot is available, source the encoder's
                    // semantic view directly from the vable shadow rather
                    // than the pyre-only `registers_r` semantic mirror.
                    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                    (0..valid_len)
                        .map(|idx| ctx.virtualizable_box_at(nvs + idx).unwrap_or(OpRef::NONE))
                        .collect()
                } else {
                    s.registers_r[..valid_len.min(s.registers_r.len())].to_vec()
                };
            if in_a_call {
                if let Some(result_idx) = self.pending_result_stack_idx {
                    let abs_idx = s.nlocals + result_idx;
                    match self.pending_result_type.unwrap_or(Type::Ref) {
                        Type::Int => {
                            if result_idx >= registers_i.len() {
                                registers_i.resize(result_idx + 1, OpRef::NONE);
                            }
                            registers_i[result_idx] = ctx.const_int(0);
                        }
                        Type::Ref => {
                            let null_ref = ctx.const_ref(pyre_object::PY_NULL as i64);
                            if abs_idx < registers_r_semantic.len() {
                                registers_r_semantic[abs_idx] = null_ref;
                            }
                            // PyPy uses `_result_argcode` plus the bytecode
                            // dst register to clear the typed result bank.
                            // Pyre's Python-frame path receives a semantic
                            // stack depth, so Ref must be translated through
                            // the per-jitcode stack color map before touching
                            // the bank used by packed liveness.
                            let color_idx_opt = if is_portal_bridge {
                                Some(abs_idx)
                            } else {
                                stack_slot_color_map
                                    .get(result_idx)
                                    .copied()
                                    .map(|c| c as usize)
                            };
                            if let Some(color_idx) = color_idx_opt {
                                if color_idx >= registers_r_bank.len() {
                                    registers_r_bank.resize(color_idx + 1, OpRef::NONE);
                                }
                                registers_r_bank[color_idx] = null_ref;
                            }
                        }
                        Type::Float => {
                            if result_idx >= registers_f.len() {
                                registers_f.resize(result_idx + 1, OpRef::NONE);
                            }
                            registers_f[result_idx] = ctx.const_float(0);
                        }
                        Type::Void => {}
                    }
                }
            }
            (
                registers_i,
                registers_r_bank,
                registers_r_semantic,
                registers_f,
            )
        };
        // pyjitpl.py:202-203: read the 2-byte offset from JitCode.code
        // (upstream uses `decode_offset(self.jitcode.code, pc + 1)`) and
        // then read the `[len_i][len_r][len_f]` header from the shared
        // all_liveness byte string. Pyre stores the packed bytes on its
        // MetaInterpStaticData JitCode entry; upstream stores them on
        // metainterp_sd.
        //
        // Skeleton payload (no `pc_map` yet) → fall back to the
        // pyre-jit-trace LiveVars analysis. With the call.py-parity
        // jitcode_for callback wired up, this branch only fires for
        // sentinel/null jitcodes (PyreSym::new_uninit) that never
        // reach final code emission.
        let jc = unsafe { &*jitcode_ptr };
        // Discriminate via the explicit 3-state predicates
        // (`pyjitcode.rs` module doc): PortalBridge runs the G.4.3
        // positional fallback; Skeleton panics; PerCodeObject (which
        // has a non-empty `pc_map`) falls through to the canonical
        // `pyjitpl.py:199-233` decode path below.
        if jc.payload.is_portal_bridge() {
            // Portal-bridge encoder (G.4.3 + G.4.4-encoder.2/3):
            // emit a positional Ref-typed box list of length
            // `stack_base + depth_at_py_pc[live_pc]`, covering locals
            // + cells + the live operand stack tail.
            // `metadata.stack_base = code.varnames.len() + ncells(code)`
            // (G.3h) is the absolute boundary in
            // `PyFrame.locals_cells_stack_w` between the "always live"
            // slots and the depth-dependent operand stack — the same
            // boundary `set_stack_at` (state.rs:2762) uses on the
            // decoder writeback side.
            //
            // Encoder/decoder symmetry (G.4.3 + G.4.4-encoder.3): both
            // `state::frame_value_count_at` and
            // `restore_guard_failure_values` source their counts from
            // `metadata.stack_base + depth_at_py_pc[pc]` so the
            // upstream `pyjitpl.py:177` / `resume.py:1017-1026`
            // packed-liveness invariant is preserved.  The
            // `live_reg_idxs` derivation above (line 504-516) emits
            // the same range so the lazy-load preamble (line 584-633)
            // populates `registers_r` from the vable shadow before
            // the snapshot — RPython orthodox always-box semantics
            // (`pyjitpl.py:218-225`).  Slots whose vable shadow is
            // NONE fall through to the heap-read path inside
            // `load_local_value` (line 1240-1265), so the encoder
            // emits a real OpRef for every live slot.
            //
            // Earlier the encoder fell back to
            // `OpRef::NONE` for slots beyond `registers_r.len()`
            // (relying on `consume_one_section` to overwrite before
            // BH deref); both divergences from `pyjitpl.py:177-234`
            // (single-bank read + always-box) are now closed by
            // routing the full live range through the same lazy-load
            // mechanism per-CodeObject mode uses.
            let stack_base = jc.payload.metadata.stack_base;
            let depth = jc
                .payload
                .metadata
                .depth_at_py_pc
                .get(live_pc)
                .copied()
                .unwrap_or(0) as usize;
            let target_count = stack_base + depth;
            let mut boxes = Vec::with_capacity(target_count);
            for reg in 0..target_count {
                boxes.push(
                    registers_r_semantic
                        .get(reg)
                        .copied()
                        .unwrap_or(OpRef::NONE),
                );
            }
            return boxes;
        }
        if jc.payload.is_skeleton() {
            // `CallControl.get_jitcode` drain fills pc_map before any
            // guard capture (pyjitpl.py:199 parity). Phase X-0 eliminated
            // the out-of-range-pc source. Phase X-1(a) migrated the
            // remaining guard/resume tests to the real compile path in
            // `pyre-jit`. Unconditional panic — any hit is a bug.
            panic!(
                "get_list_of_active_boxes: skeleton jitcode (pc_map empty) \
                 at live_pc={} — Phase X-0/X-1 removed all known triggers; \
                 further hits are bugs.",
                live_pc
            );
        }
        // `pyjitpl.py:199-233` parity: decode the `-live-` offset from
        // the jitcode byte stream via `jitcode.get_live_vars_info(pc,
        // op_live)` (`jitcode.py:82-93`), read the `[len_i][len_r]
        // [len_f]` header in `all_liveness`, then iterate per-bank
        // register indices with `LivenessIterator` (`liveness.py:168-
        // 201`). Register indices snapshot into `registers_r` in
        // int → ref → float bank order to match the encoder/decoder
        // contract (`all_liveness` byte layout).
        // Mirror RPython `pyjitpl.py:194-195 pc=self.pc`: an `in_a_call`
        // parent whose CALL sits in a try-block reads liveness at that call's
        // post-residual-call `-live-`/catch, so the encoded box count and
        // bank layout match the blackhole's marker-routed resume position
        // (`build_framestack_snapshot` folds the same marker into this
        // frame's snapshot pc).  The pyre split between a narrowed
        // fallthrough `-live-` and the un-narrowed post-call `-live-`
        // otherwise crosses the two markers — the post-call `-live-` keeps
        // the CALL result ref the next opcode pops, so the decoder reads one
        // more ref than the encoder wrote.  Without a catch marker, use the
        // plain fallthrough `-live-` via `pc_map`.
        let jit_pc = resume_jit_pc.unwrap_or_else(|| {
            panic!(
                "get_list_of_active_boxes: no pc_map entry for live_pc={} (pc_map.len={})",
                live_pc,
                jc.payload.metadata.pc_map.len()
            )
        });
        let op_live = crate::state::op_live();
        let off = jc.payload.jitcode.get_live_vars_info(jit_pc, op_live);
        let all_liveness = crate::state::liveness_info_snapshot();
        assert!(
            off + 2 < all_liveness.len(),
            "get_list_of_active_boxes: liveness offset {} + header 3 bytes exceeds all_liveness length {}",
            off,
            all_liveness.len()
        );
        let length_i = all_liveness[off] as u32;
        let length_r = all_liveness[off + 1] as u32;
        let length_f = all_liveness[off + 2] as u32;
        let mut cursor = off + 3;
        let mut boxes = Vec::with_capacity((length_i + length_r + length_f) as usize);
        // `pyjitpl.py:216-233` line-by-line parity: each live register
        // is read from its kind-specific bank via direct list indexing
        // (`self.registers_i[index]` etc).  Rust slice indexing
        // bounds-checks and panics on OOB, matching Python's
        // IndexError contract — a liveness-listed index out of bank
        // range is an encoder/codewriter invariant violation, not a
        // silent NONE.
        use majit_translate::liveness::LivenessIterator;
        if length_i != 0 {
            let mut it = LivenessIterator::new(cursor, length_i, &all_liveness);
            while let Some(reg_idx) = it.next() {
                boxes.push(registers_i[reg_idx as usize]);
            }
            cursor = it.offset;
        }
        if length_r != 0 {
            // PyPy parity: portal red args (`pypy/module/pypyjit/
            // interp_jit.py:67 reds = ['frame', 'ec']`) are JitCode
            // inputargs that appear in every `-live-` op's R-bank
            // (`liveness.py compute_liveness`). pyre's MIFrame stores
            // them on dedicated PyreSym fields (`sym.frame`,
            // `sym.execution_context`) rather than at color positions
            // in `sym.registers_r` — adapt by substituting at the
            // encoder boundary. After guard capture the wire-format
            // payload contains the OpRefs at the canonical portal
            // color positions; `_prepare_next_section` (resume.py:1381)
            // fills the BH bank from there, mirroring RPython exactly.
            let portal_frame_reg = jc.payload.metadata.portal_frame_reg as u32;
            let portal_ec_reg = jc.payload.metadata.portal_ec_reg as u32;
            let sym_frame = self.sym().frame;
            // [frame, ec] portal-reds contract: `sym.execution_context`
            // may be OpRef::NONE on adapter paths (CALL_ASSEMBLER bridge
            // attach, bridge-from-guard). `ensure_execution_context`
            // recovers it via GETFIELD_GC(frame, execution_context_descr)
            // when needed; otherwise returns the seeded value.
            let sym_ec = self.ensure_execution_context(ctx);
            let mut it = LivenessIterator::new(cursor, length_r, &all_liveness);
            while let Some(reg_idx) = it.next() {
                let opref = if reg_idx == portal_frame_reg {
                    sym_frame
                } else if reg_idx == portal_ec_reg {
                    sym_ec
                } else {
                    registers_r_bank[reg_idx as usize]
                };
                boxes.push(opref);
            }
            cursor = it.offset;
        }
        if length_f != 0 {
            let mut it = LivenessIterator::new(cursor, length_f, &all_liveness);
            while let Some(reg_idx) = it.next() {
                boxes.push(registers_f[reg_idx as usize]);
            }
        }
        boxes
    }

    /// RPython Box.type parity: build fail_arg_types matching compact
    /// active_boxes length. Each box carries its own immutable type.
    /// Header layout matches `virtualizable_gen.rs:33-35` (frame +
    /// `extra_reds` + `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`):
    /// `[frame:Ref, ec:Ref, last_instr:Int, pycode:Ref,
    ///   valuestackdepth:Int, debugdata:Ref, lastblock:Ref,
    ///   w_globals:Ref]` — line-by-line PyPy parity with
    /// `interp_jit.py:25-31` plus `interp_jit.py:67 reds = ['frame', 'ec']`.
    fn build_fail_arg_types_for_active_boxes(&self, active_boxes: &[OpRef]) -> Vec<Type> {
        let mut types = crate::virtualizable_gen::virt_live_value_types(0);
        for &opref in active_boxes {
            types.push(self.value_type(opref));
        }
        types
    }

    pub(crate) fn value_type(&self, value: OpRef) -> Type {
        if value.is_none() {
            return Type::Ref;
        }
        // history.py:220 ConstInt.type / 262 ConstPtr.type / 308
        // ResOperation.type parity: a Box's type is an intrinsic
        // property of the Box itself, not a property of the slot it
        // happens to occupy. `ctx.get_opref_type` resolves the type
        // from the OpRef's producing op (constant kind, recorded
        // result_type, or `Forwarded::Info(PtrInfo)` for virtualized
        // See the PtrInfo fallback at
        // optimizeopt/mod.rs:3995). Position-based scans of
        // `registers_r` / `virtualizable_boxes` were a pyre-only
        // adaptation that papered over earlier `get_opref_type` gaps;
        // those gaps are closed at the source now.
        let ctx_ref: &TraceCtx = unsafe { &*self.ctx };
        ctx_ref.get_opref_type(value).unwrap_or(Type::Ref)
    }

    /// RPython Box push: symbolic OpRef + concrete value together.
    ///
    /// Parity: the operand stack for virtualizable portal frames lives
    /// inside `locals_cells_stack_w` (virtualizable.py:86-98), a W_Root
    /// array, so every pushed slot is a Ref box. `value_type` is kept
    /// as a declared type hint but the stored OpRef is always boxed
    /// here. Callers that want to keep a raw Int/Float payload must
    /// unbox at the op site, not at push time.
    fn push_typed_value(
        &mut self,
        ctx: &mut TraceCtx,
        value: OpRef,
        value_type: Type,
        concrete: ConcreteValue,
    ) {
        let boxed = match value_type {
            Type::Int => wrapint(ctx, value),
            Type::Float => wrapfloat(ctx, value),
            _ => value,
        };
        let stack_idx = {
            let s = self.sym();
            s.valuestackdepth.saturating_sub(s.nlocals)
        };
        write_stack_slot(self.sym_mut(), ctx, stack_idx, boxed, concrete);
        let (owns_shadow, new_vsd) = {
            let s = self.sym_mut();
            s.valuestackdepth += 1;
            (s.owns_virtualizable_shadow(), s.valuestackdepth)
        };
        if owns_shadow {
            // `MAJIT_PROBE_BRIDGE` gated. See git
            // archaeology for context; logged for parity with the
            // pre-helper push path.
            if std::env::var_os("MAJIT_PROBE_BRIDGE").is_some() {
                let s = self.sym();
                let semantic_idx = s.nlocals + stack_idx;
                eprintln!(
                    "[probe-D][push_typed_value] valuestackdepth={} nlocals={} \
                     stack_idx={} semantic_idx={} flat_idx={} vable_boxes_len={:?}",
                    s.valuestackdepth,
                    s.nlocals,
                    stack_idx,
                    semantic_idx,
                    crate::virtualizable_gen::NUM_VABLE_SCALARS + semantic_idx,
                    ctx.virtualizable_boxes_len(),
                );
            }
            // pyjitpl.py:1188-1199 `_opimpl_setfield_vable` parity (Task
            // #114). Every `setarrayitem_vable` advances `valuestackdepth`
            // by one; upstream emits a following
            // `setfield_vable_i(virtualizable, vsd_descr, depth+1)`.
            let vsd_op = ctx.const_int(new_vsd as i64);
            self.sym_mut().vable_valuestackdepth = vsd_op;
            mirror_vable_static_to_boxes(
                ctx,
                "valuestackdepth",
                vsd_op,
                Value::Int(new_vsd as i64),
            );
        }
    }

    pub(crate) fn push_value(
        &mut self,
        _ctx: &mut TraceCtx,
        value: OpRef,
        concrete: ConcreteValue,
    ) {
        let value_type = self.value_type(value);
        self.push_typed_value(_ctx, value, value_type, concrete);
    }

    pub(crate) fn pop_value(&mut self, ctx: &mut TraceCtx) -> Result<OpRef, PyError> {
        let (stack_idx, semantic_idx) = {
            let s = self.sym();
            let stack_idx = s
                .valuestackdepth
                .checked_sub(s.nlocals + 1)
                .ok_or_else(|| pyre_interpreter::stack_underflow_error("trace opcode"))?;
            (stack_idx, s.nlocals + stack_idx)
        };
        let value = read_stack_slot(self.sym_mut(), ctx, stack_idx);
        let (is_active, owns_vable, new_vsd) = {
            let s = self.sym_mut();
            s.valuestackdepth -= 1;
            (
                s.is_active_vable_owner,
                s.owns_virtualizable_shadow(),
                s.valuestackdepth,
            )
        };
        // pyframe.py:411-417 `popvalue_maybe_none` parity: the popped slot
        // is cleared to None in `locals_cells_stack_w`. Upstream lowers
        // this via `setarrayitem_vable_r(locals_cells_stack_w, depth,
        // None)`, which `_opimpl_setarrayitem_vable` mirrors into
        // `virtualizable_boxes`. Mirror the clear into the shadow so
        // subsequent snapshots do not pick up a stale pushed OpRef above
        // the current stack depth.
        //
        // Gating on `is_active_vable_owner`
        // intentionally does NOT cover bridges here. Unlike the other
        // four mirror sites (push_typed_value / swap_values /
        // store_local_value / finishframe_exception, all flipped to
        // `owns_virtualizable_shadow()`), clearing the bridge shadow
        // to NULL on every pop triggers a severe perf regression on
        // fib_recursive (~50x slowdown: 0.88s → 47s; answer remains
        // correct).  Root cause: the pop-clear writes
        // `const_ref(PY_NULL)` into the vable shadow.  When a bridge
        // subsequently inherits that shadow state and its IR
        // references the cleared slot (e.g. a later `SetfieldGc` whose
        // base operand resolves to the shadow entry), the bridge
        // optimizer's `ConstPtrInfo._get_info` rejects the null
        // constant with `InvalidLoop("null constant base pointer")`
        // (optimizeopt/mod.rs:4021, info.py:720-721 parity).  Every
        // bridge attempt aborts the same way, so guard failures fall
        // through to the blackhole and immediately re-attempt
        // compilation — trace-abort storm.
        //
        // Upstream `pyframe.py:411 popvalue_maybe_none` clears without
        // this consequence because RPython's virtualizable layout does
        // NOT include the Python stack — the shadow has no stack-pop
        // story.  pyre's stack-in-vable TODO puts
        // the popped slots in a structure the bridge optimizer reads
        // as heap bases, so the None-clear pattern does not translate.
        // Resolving this requires either removing the stack from the
        // vable shadow or teaching the bridge
        // optimizer to recognise cleared-stack-slot sentinels
        // separately from real null heap bases.
        if is_active {
            let flat_idx = crate::virtualizable_gen::NUM_VABLE_SCALARS + semantic_idx;
            let null_opref = ctx.const_ref(pyre_object::PY_NULL as i64);
            let null_value = majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::PY_NULL as usize));
            ctx.set_virtualizable_entry_at(flat_idx, null_opref, null_value);
        }
        // pyjitpl.py:1188-1199 `_opimpl_setfield_vable` parity.
        // `pyframe.popvalue_maybe_none` decrements vsd as part
        // of the same sequence that clears the array slot; upstream emits a
        // `setfield_vable_i(virtualizable, vsd_descr, depth-1)` after the
        // setarrayitem_vable_r clear.  The bridge NULL-base issue
        // above is specific to writing a `const_ref(NULL)` into a
        // Ref-typed array slot; vsd is an Int scalar, so the bridge
        // optimizer is unaffected and the gate stays at
        // `owns_virtualizable_shadow()` to keep parity with
        // `push_typed_value`.
        if owns_vable {
            let vsd_op = ctx.const_int(new_vsd as i64);
            self.sym_mut().vable_valuestackdepth = vsd_op;
            mirror_vable_static_to_boxes(
                ctx,
                "valuestackdepth",
                vsd_op,
                Value::Int(new_vsd as i64),
            );
        }
        Ok(value)
    }

    pub(crate) fn peek_value(
        &mut self,
        ctx: &mut TraceCtx,
        depth: usize,
    ) -> Result<OpRef, PyError> {
        let stack_idx = {
            let s = self.sym();
            s.valuestackdepth
                .checked_sub(s.nlocals + depth + 1)
                .ok_or_else(|| pyre_interpreter::stack_underflow_error("trace peek"))?
        };
        Ok(read_stack_slot(self.sym_mut(), ctx, stack_idx))
    }

    fn push_call_replay_stack(
        &mut self,
        ctx: &mut TraceCtx,
        callable: OpRef,
        args: &[OpRef],
        call_pc: usize,
    ) {
        let null = ctx.const_ref(pyre_object::PY_NULL as i64);
        self.push_value(ctx, callable, ConcreteValue::Null);
        self.push_value(ctx, null, ConcreteValue::Ref(pyre_object::PY_NULL));
        for &arg in args {
            self.push_value(ctx, arg, ConcreteValue::Null);
        }
        self.sym_mut().pending_next_instr = Some(call_pc);
    }

    fn pop_call_replay_stack(
        &mut self,
        ctx: &mut TraceCtx,
        args_len: usize,
    ) -> Result<(), PyError> {
        for _ in 0..(2 + args_len) {
            let _ = self.pop_value(ctx)?;
        }
        self.sym_mut().pending_next_instr = None;
        Ok(())
    }

    /// Variant of `push_call_replay_stack` for the resolved-builtin call
    /// shape where the receiver already occupies `args[0]` (the
    /// `null_or_self` slot was non-null and the CALL handler folded it into
    /// the arg list). Replays the exact CALL operand stack
    /// `[callable, args...]` with no synthesised null sentinel.
    fn push_call_replay_stack_self_in_args(
        &mut self,
        ctx: &mut TraceCtx,
        callable: OpRef,
        callable_concrete: ConcreteValue,
        args: &[OpRef],
        call_pc: usize,
    ) {
        // The callable carries its real concrete (a Const unbound function
        // for resolved-builtin calls). A Const operand is numbered as
        // TAGCONST from its concrete value at guard-failure resume, so a
        // `ConcreteValue::Null` here would reconstruct a null callable.
        self.push_value(ctx, callable, callable_concrete);
        for &arg in args {
            self.push_value(ctx, arg, ConcreteValue::Null);
        }
        self.sym_mut().pending_next_instr = Some(call_pc);
    }

    fn pop_call_replay_stack_self_in_args(
        &mut self,
        ctx: &mut TraceCtx,
        args_len: usize,
    ) -> Result<(), PyError> {
        for _ in 0..(1 + args_len) {
            let _ = self.pop_value(ctx)?;
        }
        self.sym_mut().pending_next_instr = None;
        Ok(())
    }

    pub(crate) fn swap_values(&mut self, ctx: &mut TraceCtx, depth: usize) -> Result<(), PyError> {
        // Read the stack depth from the
        // concrete `PyFrame` at `concrete_frame_addr` rather than the
        // symbolic mirror `sym.valuestackdepth`.  This is the first reader
        // migration toward eliminating the `PyreSym.valuestackdepth`
        // mirror (a pyre-introduced divergence with no PyPy counterpart —
        // RPython's `MIFrame` holds only per-jitcode register banks, and
        // user-side stack state lives on `PyFrame` accessed via IR
        // getfield/setfield).  Falls back to the symbolic value when the
        // concrete frame is absent (test harnesses constructing sym-only
        // `MIFrame`s); production traces always seed `concrete_frame_addr`.
        let nlocals = self.sym().nlocals;
        let vsd = self
            .concrete_valuestackdepth()
            .unwrap_or_else(|| self.sym().valuestackdepth);
        let stack_only = vsd.saturating_sub(nlocals);
        if depth == 0 || stack_only < depth {
            return Err(PyError::type_error("stack underflow during trace swap"));
        }
        let (top_idx, other_idx) = (stack_only - 1, stack_only - depth);
        swap_stack_slots(self.sym_mut(), ctx, top_idx, other_idx);
        Ok(())
    }

    pub(crate) fn load_local_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
    ) -> Result<OpRef, PyError> {
        // pyjitpl.py:1231 `_opimpl_getarrayitem_vable` (standard path):
        //     return self.metainterp.virtualizable_boxes[index]
        //
        // When the standard virtualizable is active, read the current OpRef
        // straight from the virtualizable_boxes cache (seeded by
        // initialize_virtualizable at setup_tracing, mirrored by
        // store_local_value on every STORE_FAST). This is the RPython
        // orthodox read path: no extra IR op, shadow-state only.
        //
        // Pyre's flat layout puts locals at flat indices
        // `num_static_extra_boxes .. num_static + nlocals`, i.e.
        // `NUM_VABLE_SCALARS + idx`. The scalar static fields live before
        // the array items, the standard-vable identity
        // (`virtualizable_boxes[-1]`) after.
        let vable_entry = {
            let s = self.sym();
            if s.is_active_vable_owner && s.bridge_local_oprefs.is_none() {
                let flat_idx = crate::virtualizable_gen::NUM_VABLE_SCALARS + idx;
                ctx.virtualizable_box_at(flat_idx)
            } else {
                None
            }
        };
        if let Some(op) = vable_entry {
            let s = self.sym_mut();
            if idx >= s.registers_r.len() {
                return Err(PyError::type_error("local index out of range in trace"));
            }
            // Do NOT write registers_r[idx] = op here.  For active
            // virtualizable-owner traces, the vable shadow is the
            // authoritative source for locals, and stack colors can still
            // coalesce with local colors in the encoder's temporary bank.
            // Reintroducing a local mirror write here can overwrite the
            // value that guard capture is about to materialize for a stack
            // slot sharing the same color.
            return Ok(op);
        }
        let s = self.sym_mut();
        if idx >= s.registers_r.len() {
            return Err(PyError::type_error("local index out of range in trace"));
        }
        if s.registers_r[idx] == OpRef::NONE {
            if s.bridge_local_oprefs.is_some() {
                // Bridge trace: OpRef::NONE means this local is a constant
                // or virtual from resume data, not a missing vable slot.
                // Read from the concrete frame via the locals_cells_stack_w
                // array.  RPython `virtualizable.py:85-99 read_boxes` does the
                // array-field access in two steps:
                //
                //   for _, fieldname in unroll_array_fields:
                //       lst = getattr(virtualizable, fieldname)   # :94
                //       for i in range(len(lst)):
                //           boxes.append(wrap(cpu, lst[i], ...))  # :96
                //
                // Step 1 (`getattr`) yields the array pointer; step 2
                // (`lst[i]`) is the indexed read.  pyre currently emits
                // step 1 as `OpCode::GetfieldRawI` via
                // `state.rs:frame_locals_cells_stack_array`.  The
                // upstream-orthodox emission is `GETFIELD_GC_R` because
                // `pyframe_locals_cells_stack_descr` is field 0 of
                // `PYFRAME_DESCR_GROUP` with `field_type = Type::Ref`
                // on a `PYFRAME_GC_TYPE_ID`-typed PyFrame.  The
                // cranelift backend's GC-barrier coverage for the
                // PYFRAME_DESCR_GROUP read path is incomplete — a
                // direct swap to `GetfieldGcR` SIGABRTs in
                // fib_recursive — so the swap is gated on bringing
                // that barrier support up first.  Step 2 (`lst[i]`) is `GETARRAYITEM_GC_R`
                // indexed off the array.  `trace_array_getitem_value`
                // uses `pyobject_gcarray_descr` (`base_size =
                // FIXED_ARRAY_ITEMS_OFFSET`) and so requires the array
                // base, not the virtualizable (PyFrame*) pointer.
                // Emit `frame_locals_cells_stack_array` to materialise
                // the array OpRef before indexing.
                let frame_ref = s.frame;
                let array_ref = crate::state::frame_locals_cells_stack_array(ctx, frame_ref);
                let idx_const = ctx.const_int(idx as i64);
                s.registers_r[idx] = trace_array_getitem_value(ctx, array_ref, idx_const);
            } else {
                // Active vable owner whose registers_r[idx] is NONE cannot
                // exist: init_symbolic (state.rs:2618-2619) seeds
                // registers_r[idx] = OpRef::from_raw(base + idx) for every i in
                // 0..nlocals before any load_local_value runs. Reachability
                // audit (`MAJIT_PROBE_VABLE_FALLBACK`) confirmed
                // this empirically: 0 firings across debug unit tests
                // (debug_assert!(false)) and 0 firings across 28 release
                // benchmark runs (env-gated eprintln). The remaining
                // fallthrough is the non-vable-owner path —
                // `s.locals_cells_stack_array_ref` is the callee's own
                // locals_cells_stack_w array (seeded by Stage 1 at
                // inline_function_call).
                let idx_const = ctx.const_int(idx as i64);
                s.registers_r[idx] =
                    trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
            }
        }
        Ok(s.registers_r[idx])
    }

    pub(crate) fn store_local_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
        value: OpRef,
        concrete: ConcreteValue,
    ) -> Result<(), PyError> {
        // RPython `_opimpl_setarrayitem_vable` (pyjitpl.py:1242-1247)
        // writes the value's Ref box directly into
        // `virtualizable_boxes[flat_idx]` AND the frame's symbolic
        // tracking. `locals_cells_stack_w` is declared as a W_Root array
        // (virtualizable.py:86-98), so the stored box must always be
        // Ref. pyre's producers satisfy this contract BEFORE reaching
        // this store: `push_typed_value` (trace_opcode.rs:471-482)
        // wraps raw Int/Float with `wrapint` / `wrapfloat` when a value
        // lands on the operand stack, so `pop_value` — and thus every
        // STORE_FAST / STORE_FAST_LOAD_FAST / STORE_FAST_STORE_FAST
        // handler (pyopcode.rs:394-430) — hands us a Ref. The earlier
        // "box on store" safety net (d2e530f3b9) is therefore redundant
        // in production; asserting the invariant here keeps any future
        // non-push producer from slipping a raw Int/Float through
        // (RPython Box.type parity — the fix belongs at the producer,
        // not the consumer).
        debug_assert_eq!(
            self.value_type(value),
            Type::Ref,
            "store_local_value: expected Ref-typed box (locals_cells_stack_w \
             is W_Root), got {:?} for {:?}. Producer must emit \
             wrapint/wrapfloat before the value reaches the symbolic \
             stack / local slot.",
            self.value_type(value),
            value,
        );
        let (has_vable, frame_ref, nlocals) = {
            let s = self.sym_mut();
            if idx >= s.registers_r.len() {
                return Err(PyError::type_error("local index out of range in trace"));
            }
            // When `load_local_value`'s
            // vable-read predicate (`is_active_vable_owner` AND no
            // `bridge_local_oprefs`) fires, every reader of the local
            // OpRef sources from `virtualizable_boxes` and the
            // `registers_r[idx]` semantic-mirror write is dead — PyPy's
            // `_opimpl_setarrayitem_vable` (`pyjitpl.py:1242-1247`)
            // writes only the vable shadow.  Bridges with
            // `bridge_local_oprefs=Some(...)` still read from
            // `registers_r[idx]` (`load_local_value`'s non-vable arm at
            // line 1999), so they retain the mirror write; non-owner
            // frames also keep it.
            let vable_read_path = s.is_active_vable_owner && s.bridge_local_oprefs.is_none();
            if !vable_read_path {
                s.registers_r[idx] = value;
            }
            if idx >= s.symbolic_local_types.len() {
                s.symbolic_local_types.resize(idx + 1, Type::Ref);
            }
            // virtualizable.py:86-98 read_boxes() parity: every item of
            // locals_cells_stack_w is Ref.
            s.symbolic_local_types[idx] = Type::Ref;
            (s.owns_virtualizable_shadow(), s.frame, s.nlocals)
        };
        // RPython pyjitpl.py:1242-1247 `_opimpl_setarrayitem_vable` parity:
        //     self.metainterp.virtualizable_boxes[flat_idx] = valuebox
        //     self.metainterp.synchronize_virtualizable()
        //
        // `valuebox` carries OpRef + concrete as a single Box. Ref / Null
        // concrete updates both halves; Int / Float concrete has no real
        // W_Root heap pointer (pyre's lazy `wrapint` / `wrapfloat`), so
        // update only the OpRef half and leave the shadow concrete at the
        // PyFrame's existing valid W_Root. virtualizable.py:101
        // `write_boxes` must see real boxed W_Root values in every slot.
        if has_vable && idx < nlocals {
            let _ = frame_ref;
            // NUM_VABLE_SCALARS = 6 (vable static field count, excluding
            // both the frame-identity slot and any non-vable extra reds).
            // virtualizable_boxes layout is [scalars.., array_items..,
            // vable_ref], so local idx maps to `NUM_VABLE_SCALARS + idx`.
            let flat_idx = crate::virtualizable_gen::NUM_VABLE_SCALARS + idx;
            match concrete.to_ir_ref_value() {
                Some(v) => {
                    ctx.set_virtualizable_entry_at(flat_idx, value, v);
                }
                None => {
                    ctx.set_virtualizable_box_at(flat_idx, value);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn set_next_instr(&mut self, _ctx: &mut TraceCtx, target: usize) {
        self.sym_mut().pending_next_instr = Some(target);
    }

    pub(crate) fn fallthrough_pc(&self) -> usize {
        self.fallthrough_pc
    }

    /// Set pending_next_instr for trace advancement (step_root_frame).
    /// This is the NEXT bytecode PC — used by the MetaInterp to advance.
    pub(crate) fn prepare_fallthrough(&mut self) {
        self.sym_mut().pending_next_instr = Some(self.fallthrough_pc);
    }

    /// Set the original PC for the current opcode (RPython orgpc).
    /// All guards within this opcode will use orgpc as their resume PC.
    ///
    /// Pyre-only shadow refresh hook.  RPython's metainterp owns every opcode
    /// boundary so `metainterp.virtualizable_boxes` stays in lockstep with
    /// heap automatically.  Pyre splits dispatch between the walker (which
    /// mirrors via `vable_setfield → synchronize_virtualizable`) and
    /// `execute_opcode_step` (which mutates the heap PyFrame directly via
    /// `PyFrame::push` / `PyFrame::pop` etc.), so the shadow can lag the
    /// heap between opcodes.  The refresh must run BEFORE
    /// `capture_pre_opcode_state` reads from `virtualizable_boxes`
    /// (`trace_opcode.rs:1014`) — otherwise guard fail_args would snapshot a
    /// stale shadow whenever the prior opcode ran through trait dispatch.
    ///
    /// Inline-frame guard: when `self.parent_frames` is non-empty we are
    /// running `trace_code_step_inline` on a callee MIFrame.  The shared
    /// `TraceCtx.virtualizable_boxes` shadow still belongs to the portal
    /// (caller) frame; refreshing it from heap mid-inline would overwrite
    /// any caller-side updates that the walker pushed into the shadow
    /// before the inline call but has not yet written back to heap.
    ///
    /// When dispatch unification retires `execute_opcode_step`, this hook
    /// becomes a no-op (every mutation already lands in shadow) and can
    /// be removed.
    pub(crate) fn set_orgpc(&mut self, pc: usize) {
        self.orgpc = pc;
        // Refresh gating: only frames that own the vable shadow read
        // through it in `capture_pre_opcode_state` (line 1055) and in the
        // walker's `vable_getfield_*` arm bodies; non-owner frames snapshot
        // `s.registers_r` (line 1072) instead, so a stale shadow cannot
        // contaminate their guard fail_args.  Inline frames inherit the
        // portal's shadow and must not refresh — the portal's preceding
        // opcode boundary already ran the refresh and the inline body
        // may have pushed walker-side updates that have not yet
        // synchronized back to heap (refresh would clobber them).
        if self.parent_frames.is_empty() && self.sym().owns_virtualizable_shadow() {
            self.with_ctx(|_, ctx| ctx.refresh_virtualizable_shadow_from_heap());
        }
        self.publish_last_instr_to_vable(pc);
    }

    /// pyopcode.py:170-172 `dispatch_bytecode` parity:
    ///
    /// ```python
    /// while True:
    ///     self.last_instr = intmask(next_instr)   # explicit setattr at every
    ///                                             # dispatch top
    /// ```
    ///
    /// RPython's source-level setattr is picked up by the codewriter and
    /// emitted as a `setfield_vable_i(virtualizable, last_instr_descr, ...)`
    /// jitcode op; metainterp's `_opimpl_setfield_vable` (pyjitpl.py:1188-
    /// 1199) updates `metainterp.virtualizable_boxes[last_instr_index]` and
    /// `synchronize_virtualizable()`s the heap.  pyre's tracer is itself
    /// the dispatch loop, so the explicit publish happens here at the
    /// orgpc anchor — every `set_orgpc(pc)` call is the analogue of one
    /// `self.last_instr = next_instr` setattr.
    ///
    /// Semantic mismatch (TODO):
    /// pyre's `PyFrame.last_instr` carries "PC of the last completed
    /// opcode" (= `orgpc - 1`), whereas RPython's carries "PC of the
    /// opcode now starting" (= `orgpc`).  Currently publishes
    /// `orgpc - 1` to keep every existing reader (`flush_to_frame_for_guard`
    /// re-seed, `get_last_lineno`, `fget_f_lasti`, blackhole resume,
    /// guard fail_args) seeing the same value they always did.  The parity fix
    /// flips the semantic to RPython's "current opcode PC" and removes
    /// the `-1` adjustment from this publish + the four other vable-side
    /// sites (`flush_to_frame`, `flush_to_frame_for_guard`,
    /// `close_loop_args_at` target override, `build_pending_inline_frame`
    /// after-call return-point).
    fn publish_last_instr_to_vable(&mut self, pc: usize) {
        if !self.sym().owns_virtualizable_shadow() {
            return;
        }
        let last_instr_value = pc as i64 - 1;
        self.with_ctx(|frame, ctx| {
            let last_instr_op = ctx.const_int(last_instr_value);
            frame.sym_mut().vable_last_instr = last_instr_op;
            mirror_vable_static_to_boxes(
                ctx,
                "last_instr",
                last_instr_op,
                Value::Int(last_instr_value),
            );
        });
    }

    /// Update virtualizable last_instr and valuestackdepth.
    /// RPython parity: always use orgpc (opcode start PC) as the semantic
    /// next instruction, so the heap frame stores `last_instr = orgpc - 1`.
    /// The trace loop advancement uses pending_next_instr separately
    /// (in metainterp.rs step_*_frame).
    pub(crate) fn flush_to_frame(&mut self, ctx: &mut TraceCtx) {
        let resume_pc = self.orgpc;
        let frame_addr = self.concrete_frame_addr;
        // virtualizable.py:86-93 read_boxes reads statics from the LIVE
        // virtualizable.  The root MIFrame's concrete frame is the
        // trace-stepping snapshot (`snapshot_for_tracing`), whose
        // `debugdata` / `lastblock` are owned clones freed when tracing
        // ends — a const captured from the snapshot dangles in the
        // compiled trace's resume data, and the guard-failure vable
        // write (`write_from_resume_data_partial`) then stamps the
        // dangling pointer into the live frame.  Read the pointer-valued
        // statics from the live virtualizable instead; `pycode` is
        // copied by the snapshot so either source gives the same value.
        // `last_instr` / `valuestackdepth` are plain values that evolve
        // with the trace, so they stay snapshot-sourced below.
        let statics_addr = {
            let live = self.sym().live_vable_frame_addr;
            if live != 0 && self.sym().owns_virtualizable_shadow() {
                live
            } else {
                frame_addr
            }
        };
        let (code_ptr, debugdata, lastblock) = if statics_addr != 0 {
            unsafe {
                (
                    *((statics_addr + PYFRAME_PYCODE_OFFSET) as *const usize),
                    *((statics_addr + PYFRAME_DEBUGDATA_OFFSET) as *const usize),
                    *((statics_addr + PYFRAME_LASTBLOCK_OFFSET) as *const usize),
                )
            }
        } else {
            (0, 0, 0)
        };
        let ns_ptr = self.sym().concrete_namespace as i64;
        // G.4.3a: portal-bridge frames trace eval_loop_jit, so each user
        // opcode is a residual call to execute_opcode_step whose
        // valuestackdepth side-effects do NOT advance `sym.valuestackdepth`.
        // The stale symbolic value would encode `vable_valuestackdepth =
        // stack_base`, leaving the resumed PyFrame with vsd ≤ stack_base
        // and crashing the next pop. Recompute from the per-PC user-side
        // depth metadata derived in `install_portal_for` (G.4.2).
        //
        // When the portal-bridge metadata is absent,
        // read from the concrete `PyFrame.valuestackdepth` rather than the
        // stale `sym.valuestackdepth`.  `resume_pc == self.orgpc` (the
        // start PC of the current opcode) so PyFrame holds the correct
        // pre-opcode value (the interpreter step for this opcode has not
        // run yet).  Falls back to the symbolic value only when
        // `concrete_frame_addr == 0` (test-only sym-only MIFrames).
        let vsd = self.portal_bridge_vable_vsd(resume_pc).unwrap_or_else(|| {
            self.concrete_valuestackdepth()
                .unwrap_or_else(|| self.sym().valuestackdepth) as i64
        });
        // virtualizable.py:86-93 read_boxes: ALL static fields from the heap.
        let last_instr_value = resume_pc as i64 - 1;
        let last_instr_op = ctx.const_int(last_instr_value);
        let pycode_op = ctx.const_ref(code_ptr as i64);
        let vsd_op = ctx.const_int(vsd);
        let debugdata_op = ctx.const_ref(debugdata as i64);
        let lastblock_op = ctx.const_ref(lastblock as i64);
        let w_globals_op = ctx.const_ref(ns_ptr);
        let owns = {
            let s = self.sym_mut();
            s.vable_last_instr = last_instr_op;
            s.vable_pycode = pycode_op;
            s.vable_valuestackdepth = vsd_op;
            s.vable_debugdata = debugdata_op;
            s.vable_lastblock = lastblock_op;
            s.vable_w_globals = w_globals_op;
            s.owns_virtualizable_shadow()
        };
        // pyjitpl.py:1188-1199 `_opimpl_setfield_vable` parity:
        // mirror the heap-read seed into the canonical
        // `metainterp.virtualizable_boxes` shadow so subsequent readers
        // (snapshot, JUMP-arg dedup) see the same identity that
        // `s.vable_*` carries.
        if owns {
            mirror_vable_static_to_boxes(
                ctx,
                "last_instr",
                last_instr_op,
                Value::Int(last_instr_value),
            );
            mirror_vable_static_to_boxes(ctx, "pycode", pycode_op, Value::Ref(GcRef(code_ptr)));
            mirror_vable_static_to_boxes(ctx, "valuestackdepth", vsd_op, Value::Int(vsd));
            mirror_vable_static_to_boxes(
                ctx,
                "debugdata",
                debugdata_op,
                Value::Ref(GcRef(debugdata)),
            );
            mirror_vable_static_to_boxes(
                ctx,
                "lastblock",
                lastblock_op,
                Value::Ref(GcRef(lastblock)),
            );
            mirror_vable_static_to_boxes(
                ctx,
                "w_globals",
                w_globals_op,
                Value::Ref(GcRef(ns_ptr as usize)),
            );
        }
    }

    /// G.4.3a: For portal-bridge frames, derive the absolute
    /// valuestackdepth from `metadata.stack_base + depth_at_py_pc[pc]`
    /// (both populated by `install_portal_for` G.3h + G.4.2).  Returns
    /// `None` for non-portal-bridge or null-jitcode states so the caller
    /// can fall back to the stale `sym.valuestackdepth` /
    /// `pre_opcode_semantic_depth` heuristic that the per-CodeObject
    /// path relies on.
    ///
    /// Why this is needed: portal-bridge tracing records each user opcode
    /// as a residual call to `execute_opcode_step`.  The symbolic
    /// `sym.valuestackdepth` only advances through `push_typed_value` /
    /// `pop_value` (`trace_opcode.rs:808/872`), neither of which fires
    /// for residual-call paths.  As a result the symbolic value stays at
    /// its initial seed (`= stack_base`) for the lifetime of the trace,
    /// and the encoded `vable_valuestackdepth` is too low.  Restoring
    /// that value into PyFrame.valuestackdepth crashes the next pop with
    /// `assertion failed: self.valuestackdepth > self.stack_base()`
    /// (`pyframe.rs:862`).  The metadata-driven computation matches the
    /// runtime PyFrame state for the same `(jitcode, py_pc)` pair, just
    /// like the per-CodeObject path's stack-effect walk does inside
    /// `pre_opcode_registers_r`.
    fn portal_bridge_vable_vsd(&self, pc: usize) -> Option<i64> {
        let s = self.sym();
        if s.jitcode.is_null() {
            return None;
        }
        let payload = unsafe { &(*s.jitcode).payload };
        if !payload.is_portal_bridge() {
            return None;
        }
        let depth = payload
            .metadata
            .depth_at_py_pc
            .get(pc)
            .copied()
            .unwrap_or(0) as usize;
        Some((payload.metadata.stack_base + depth) as i64)
    }

    /// capture_resumedata(resumepc=orgpc) parity: flush vable fields for guards.
    ///
    /// When a pre-opcode snapshot is present, sets vable_last_instr = orgpc - 1
    /// and vable_valuestackdepth = the snapshot depth. The guard's fail_args
    /// then carry the pre-opcode stack state so the blackhole interpreter
    /// can re-execute the opcode from orgpc.
    ///
    /// Note: `record_branch_guard` does NOT call this — branch guards
    /// build their own fail_args with post-pop state and other_target PC
    /// (see the comment there for why).
    fn flush_to_frame_for_guard(&mut self, ctx: &mut TraceCtx) {
        // RPython capture_resumedata(resumepc=orgpc) parity:
        // Always use orgpc (opcode start PC) as the resume PC.
        let resume_pc = self.orgpc;
        let vsd = self
            .portal_bridge_vable_vsd(resume_pc)
            .unwrap_or_else(|| self.pre_opcode_concrete_depth() as i64);
        // pyjitpl.py:2586-2602 `capture_resumedata` parity: RPython reads
        // `metainterp.virtualizable_boxes` without mutating it. The two
        // fields that advance per-opcode (`last_instr`, `valuestackdepth`)
        // need a guard-time-correct override here because pyre's tracer
        // is itself the dispatch loop: at guard time the active opcode is
        // the one at `orgpc`, so the snapshot must encode the pre-opcode
        // state (`last_instr = orgpc - 1`, `valuestackdepth = pre-opcode
        // depth via pre_opcode_registers_r / portal_bridge_vable_vsd`).
        // The other four scalars (`pycode`, `debugdata`, `lastblock`,
        // `w_globals`) keep the inputarg OpRefs `init_vable_indices`
        // seeded at trace start because pyre-jit-trace never enters
        // their mutators under CPython 3.14 bytecode: `pycode` /
        // `w_globals` are set only by `pyframe.rs::frame_reinit`;
        // `debugdata` only by `getorcreate_debug_data` on debug paths;
        // `lastblock` only by `pyopcode.py:1268 SETUP_*/POP_BLOCK`,
        // none of which CPython 3.14 emits.  This matches RPython's
        // "boxes carry vable inputargs" model — see
        // `mirror_vable_static_to_boxes` doc for the convergence path
        // when those handlers are re-introduced.
        let last_instr_value = resume_pc as i64 - 1;
        let last_instr_op = ctx.const_int(last_instr_value);
        let vsd_op = ctx.const_int(vsd);
        let s = self.sym_mut();
        s.vable_last_instr = last_instr_op;
        s.vable_valuestackdepth = vsd_op;
        // The shared `ctx.virtualizable_boxes` shadow is intentionally
        // not mirrored here — see `mirror_vable_static_to_boxes` doc
        // for the convention.  The two stores have distinct roles:
        // `s.vable_*` is the snapshot reader's view (carries pre-opcode
        // overrides set here, save/restored by `record_branch_guard`),
        // `ctx.virtualizable_boxes` is the JUMP/JIT-time view (consumed
        // by `close_loop_args_at`'s JUMP-arg derivation).
    }

    /// pyjitpl.py:3317-3335 vable_and_vrefs_before_residual_call.
    ///
    /// RPython structure:
    ///
    /// ```text
    /// def vable_and_vrefs_before_residual_call(self):
    ///     vrefinfo = self.staticdata.virtualref_info
    ///     for i in range(1, len(self.virtualref_boxes), 2):
    ///         vrefbox = self.virtualref_boxes[i]
    ///         vref = vrefbox.getref_base()
    ///         vrefinfo.tracing_before_residual_call(vref)
    ///     #
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         virtualizable_box = self.virtualizable_boxes[-1]
    ///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///         vinfo.tracing_before_residual_call(virtualizable)
    ///         force_token = self.history.record0(rop.FORCE_TOKEN, ...)
    ///         self.history.record2(rop.SETFIELD_GC, virtualizable_box,
    ///                              force_token, None,
    ///                              descr=vinfo.vable_token_descr)
    /// ```
    ///
    /// Key points:
    ///   1. vref token marking is unconditional (no vinfo check).
    ///   2. virtualizable processing only runs when vinfo is not None.
    ///   3. No call to `gen_store_back_in_vable` — that helper is only
    ///      invoked from `opimpl_hint_force_virtualizable` (pyjitpl.py:1071).
    fn vable_and_vrefs_before_residual_call(&mut self, ctx: &mut TraceCtx) {
        // pyjitpl.py:3319-3322: virtualref token marking (ALWAYS runs,
        // even without virtualizable info).
        self.vrefs_before_residual_call();

        // pyjitpl.py:3326: vinfo = self.jitdriver_sd.virtualizable_info
        // pyjitpl.py:3327: if vinfo is not None:
        //
        // majit's pyre port uses `standard_virtualizable_box()` as the
        // vinfo proxy — it returns `Some(box)` iff the jitdriver has a
        // standard virtualizable registered for the current frame. RPython
        // checks the per-jitdriver `vinfo` first, then derefs the box; the
        // pyre-side null check on `concrete_vable_ptr` is the defensive
        // analogue of `unwrap_virtualizable_box(virtualizable_box)`.
        let Some(vable_ref) = ctx.standard_virtualizable_box() else {
            return;
        };
        let obj_ptr = self.sym().concrete_vable_ptr;
        if obj_ptr.is_null() {
            return;
        }
        // pyjitpl.py:3329-3330:
        //   virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
        //   vinfo.tracing_before_residual_call(virtualizable)
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        unsafe {
            info.tracing_before_residual_call(obj_ptr);
        }
        // pyjitpl.py:3332-3335:
        //   force_token = self.history.record0(rop.FORCE_TOKEN,
        //                                      lltype.nullptr(llmemory.GCREF.TO))
        //   self.history.record2(rop.SETFIELD_GC, virtualizable_box,
        //                        force_token, None,
        //                        descr=vinfo.vable_token_descr)
        let force_token = ctx.force_token();
        ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
    }

    /// pyjitpl.py:3349-3366 vable_after_residual_call.
    ///
    /// Only checks virtualizable (not vrefs — those are checked
    /// separately by vrefs_after_residual_call at the call site).
    /// If virtualizable escaped, reloads fields and aborts tracing
    /// (SwitchToBlackhole parity).
    fn vable_after_residual_call(&mut self) -> Result<(), PyError> {
        // pyjitpl.py:3350-3351: if vinfo is not None:
        // RPython parity: use the same gate as before_residual_call
        // (ctx.standard_virtualizable_box). If before didn't set
        // TOKEN_TRACING_RESCALL (no vbox), after must not check it.
        let has_vbox = self.with_ctx(|_, ctx| ctx.standard_virtualizable_box().is_some());
        if !has_vbox {
            return Ok(());
        }
        let obj_ptr = self.sym().concrete_vable_ptr;
        if obj_ptr.is_null() {
            return Ok(());
        }
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let vable_forced = unsafe { info.tracing_after_residual_call(obj_ptr) };
        if vable_forced {
            // pyjitpl.py:3356: self.load_fields_from_virtualizable()
            self.load_fields_from_virtualizable();
            // pyjitpl.py:3365-3366:
            //   raise SwitchToBlackhole(Counters.ABORT_ESCAPE,
            //                           raising_exception=True)
            return Err(PyError::runtime_error(
                "ABORT_ESCAPE: virtualizable escaped during residual call",
            ));
        }
        Ok(())
    }

    /// pyjitpl.py:3452-3463 load_fields_from_virtualizable.
    ///
    /// Force a reload of the virtualizable fields into the local
    /// boxes (called only in escaping cases, just before abort).
    fn load_fields_from_virtualizable(&mut self) {
        let obj_ptr = self.sym().concrete_vable_ptr;
        if obj_ptr.is_null() {
            return;
        }
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        // pyjitpl.py:3460-3462: self.virtualizable_boxes = vinfo.read_boxes(
        //     self.cpu, virtualizable, 0)
        // Re-read all virtualizable fields from the heap object.
        let lengths = unsafe { info.read_array_lengths_from_heap(obj_ptr as *const u8) };
        let (static_boxes, array_boxes) =
            unsafe { info.read_all_boxes(obj_ptr as *const u8, &lengths) };
        // Store back into PyreSym's concrete state so the blackhole
        // interpreter sees the up-to-date values.
        let sym = self.sym_mut();
        // Static fields: update concrete_locals for the virtualizable fields.
        for (i, &val) in static_boxes.iter().enumerate() {
            if i < sym.concrete_locals.len() {
                sym.concrete_locals[i] = ConcreteValue::Int(val);
            }
        }
        // Array fields: update concrete locals/stack from array boxes.
        if let Some(arr) = array_boxes.first() {
            let nlocals = sym.nlocals;
            for (i, &val) in arr.iter().enumerate() {
                if i < nlocals && i < sym.concrete_locals.len() {
                    sym.concrete_locals[i] = ConcreteValue::Ref(val as PyObjectRef);
                } else {
                    let stack_idx = i.saturating_sub(nlocals);
                    if stack_idx < sym.concrete_stack.len() {
                        sym.concrete_stack[stack_idx] = ConcreteValue::Ref(val as PyObjectRef);
                    }
                }
            }
        }
    }

    /// pyjitpl.py:3317-3337 parity: before residual call, set all
    /// active virtualref tokens to TOKEN_TRACING_RESCALL.
    fn vrefs_before_residual_call(&self) {
        let vref_boxes = &self.sym().virtualref_boxes;
        if vref_boxes.is_empty() {
            return;
        }
        // pyjitpl.py:3339: for each pair, call tracing_before on the
        // ODD slot (vrefbox = second element), not the virtual (first).
        let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
        // virtualref_boxes = [(virt_sym, virt_ptr), (vref_sym, vref_ptr), ...]
        for pair in vref_boxes.chunks(2) {
            if let Some(&(_vref_sym, vref_ptr)) = pair.get(1) {
                if vref_ptr != 0 {
                    unsafe {
                        vref_info.tracing_before_residual_call(vref_ptr as *mut u8);
                    }
                }
            }
        }
    }

    /// pyjitpl.py:3337-3347 vrefs_after_residual_call parity:
    /// after residual call, check if any virtualref was forced.
    /// If forced, call stop_tracking_virtualref(i) to record
    /// VIRTUAL_REF_FINISH and replace odd slot with CONST_NULL.
    fn vrefs_after_residual_call(&mut self, ctx: &mut TraceCtx) {
        let sym = unsafe { &mut *self.sym };
        if sym.virtualref_boxes.is_empty() {
            return;
        }
        let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
        let len = sym.virtualref_boxes.len();
        let mut i = 0;
        while i < len {
            let (_, vref_ptr) = sym.virtualref_boxes[i + 1];
            if vref_ptr != 0 {
                let forced = unsafe { vref_info.tracing_after_residual_call(vref_ptr as *mut u8) };
                if forced {
                    Self::stop_tracking_virtualref(sym, ctx, i);
                }
            }
            i += 2;
        }
    }

    /// pyjitpl.py:3371-3378 stop_tracking_virtualref parity.
    ///
    /// Record VIRTUAL_REF_FINISH(vrefbox, virtualbox) and replace
    /// the odd slot with ConstPtr(NULL).
    fn stop_tracking_virtualref(sym: &mut PyreSym, ctx: &mut TraceCtx, i: usize) {
        let virt_opref = sym.virtualref_boxes[i].0;
        let (vref_opref, _) = sym.virtualref_boxes[i + 1];
        // pyjitpl.py:3376: record VIRTUAL_REF_FINISH(vrefbox, virtualbox)
        let _ = ctx.record_op(OpCode::VirtualRefFinish, &[vref_opref, virt_opref]);
        // pyjitpl.py:3378: self.virtualref_boxes[i+1] = CONST_NULL
        // history.py:361: CONST_NULL = ConstPtr(ConstPtr.value)
        let null_opref = ctx.const_null();
        sym.virtualref_boxes[i + 1] = (null_opref, 0);
    }

    /// Loop-carried values must follow the typed live-state contract used by
    /// PyreMeta::slot_types / restore_values().
    ///
    /// In pyre's typed INT/REF/FLOAT model, integer locals cross a loop JUMP
    /// as raw Int values, not freshly boxed W_Int objects.
    fn materialize_loop_carried_value(
        &mut self,
        ctx: &mut TraceCtx,
        value: OpRef,
        slot_type: Type,
    ) -> OpRef {
        match slot_type {
            Type::Int => match self.value_type(value) {
                Type::Int => value,
                Type::Ref => {
                    // Convert boxed W_Int back to its raw payload so the loop
                    // header sees the typed INT stream expected by restore_values().
                    self.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, value))
                }
                _ => value,
            },
            Type::Ref => match self.value_type(value) {
                Type::Int => {
                    // Virtualizable slots are Ref — re-box raw Int for the
                    // loop header which expects boxed W_IntObject.
                    let int_type_addr = &INT_TYPE as *const _ as i64;
                    crate::generated::trace_box_int(
                        ctx,
                        value,
                        w_int_size_descr(),
                        ob_type_descr(),
                        int_intval_descr(),
                        int_type_addr,
                    )
                }
                Type::Float => {
                    let float_type_addr = &FLOAT_TYPE as *const _ as i64;
                    crate::trace_box_float(
                        ctx,
                        value,
                        w_float_size_descr(),
                        ob_type_descr(),
                        float_floatval_descr(),
                        float_type_addr,
                    )
                }
                _ => value,
            },
            _ => value,
        }
    }

    pub(crate) fn close_loop_args(&mut self, ctx: &mut TraceCtx) -> Vec<OpRef> {
        self.close_loop_args_at(ctx, None)
    }

    /// Pure-read shape predictor for `close_loop_args_at` output.
    ///
    /// Returns the LENGTH that `close_loop_args_at` would produce at
    /// the current sym/ctx state, without mutating either.  Used by
    /// the merge-point seed sites (`trace.rs::trace_bytecode`,
    /// `TraceCtx::new`, `TraceCtx::with_green_key`) that need to
    /// allocate `original_boxes` of the same shape future
    /// `close_loop_args` calls will produce, so
    /// `pyjitpl.py:2996 assert len(original_boxes) == len(live_arg_boxes)`
    /// can fire (see memory
    /// `merge_point_shape_assert_prerequisite_2026_05_03.md`).
    ///
    /// Shape derivation matches `close_loop_args_at`:
    /// `1 (frame) + extra_reds (ec) + 6 (vable scalars) + target_array_capacity`
    /// where the vable scalars are
    /// `[next_instr, code, stack_depth, debugdata, lastblock, namespace]`
    /// and `target_array_capacity` is either the virtualizable array
    /// lengths sum (when known) or the fallback `nlocals + stack_only`.
    pub(crate) fn live_args_shape_at(&self, ctx: &TraceCtx) -> usize {
        let extra_reds = crate::virtualizable_gen::NUM_EXTRA_REDS;
        let nlocals = self.sym().nlocals;
        // Pure-read of stack depth comes from
        // the concrete `PyFrame` (no symbolic mirror).  RPython's
        // pyjitpl.py:2957-2965 `live_arg_boxes` shape derives directly
        // from PyFrame's `locals_cells_stack_w` length + `valuestackdepth`
        // — there is no symbolic mirror to consult.
        let vsd = self
            .concrete_valuestackdepth()
            .unwrap_or_else(|| self.sym().valuestackdepth);
        let stack_only = vsd.saturating_sub(nlocals);
        let target_array_capacity = ctx
            .virtualizable_array_lengths()
            .map(|lengths| lengths.iter().copied().sum::<usize>())
            .filter(|&len| len >= nlocals)
            .unwrap_or(nlocals + stack_only);
        // 1 (frame) + extra_reds + 6 (vable_scalars) + target_array_capacity
        7 + extra_reds + target_array_capacity
    }

    /// TODO: bundles `pyjitpl.py:2957-2965` `live_arg_boxes`
    /// construction (`greenboxes + redboxes + virtualizable_boxes`,
    /// `pop()` the trailing token) with the `vable_last_instr` pin
    /// (`pyjitpl.py:2973`). RPython performs both inline within
    /// `reached_loop_header`; pyre extracts them because pyre's "args"
    /// are `OpRef`s pulled from the unified `registers_r` register
    /// file, and the merge `target_pc` must be threaded through
    /// explicitly (RPython has it implicitly in `redboxes`).
    pub(crate) fn close_loop_args_at(
        &mut self,
        ctx: &mut TraceCtx,
        target_pc: Option<usize>,
    ) -> Vec<OpRef> {
        // pyjitpl.py:2954-2965 reached_loop_header: virtualizable_boxes
        // (read from locals_cells_stack_w[*] by virtualizable.py:86-98
        // read_boxes) are carried into the JUMP unchanged, including
        // stack slots. Do NOT truncate to nlocals here.
        //
        // Read the user-side
        // `valuestackdepth` from the concrete `PyFrame` to match
        // `live_args_shape_at`'s reader.  Both helpers share the same
        // shape derivation; reading from different sources lets the two
        // diverge whenever the symbolic mirror drifts from PyFrame.
        // RPython's pyjitpl.py:2957-2965 derives `live_arg_boxes` from
        // PyFrame's `locals_cells_stack_w` length + `valuestackdepth`
        // — no symbolic mirror in the loop.
        let concrete_nlocals = self.sym().nlocals;
        let concrete_vsd = self
            .concrete_valuestackdepth()
            .unwrap_or_else(|| self.sym().valuestackdepth)
            .max(concrete_nlocals);
        {
            let s = self.sym_mut();
            s.nlocals = concrete_nlocals;
            s.valuestackdepth = concrete_vsd;
            let stack_only = s.valuestackdepth.saturating_sub(s.nlocals);
            // virtualizable.py:44 + interp_jit.py:25-31: locals_cells_stack_w[*]
            // is a W_Root array → every item is declared Ref. The loop-carried
            // types passed to the JUMP / merge point MUST be Ref for every
            // array slot; tracker-observed Int/Float types are internal to
            // unboxing lowering and must not leak into the inputarg contract.
            //
            // The type-map reset is deferred until AFTER the materialize loop
            // below. Resetting here would poison `self.value_type(value)` used
            // by `materialize_loop_carried_value` — an Int-typed OpRef on the
            // symbolic stack (e.g. a GetarrayitemRawI result LOAD_FASTed onto
            // the stack) would appear Ref to the materializer and skip the
            // Int→Ref boxing, handing the MIFrame merge-point a raw Int value
            // in a Ref-typed slot. Cut_trace_from then installs that raw Int
            // at a cut-inputarg position whose declared type is Ref, and the
            // downstream unroll pass produces JUMP args that pass an Int into
            // a Ref slot at runtime → SIGSEGV (str x24, [x12, #0x10] with x12
            // carrying the unboxed int payload).
            //
            // `registers_r` is the unified abstract register file;
            // reserve `[nlocals..nlocals+stack_only)` for the stack so
            // merge-point JUMP args have a stable slice.
            let min_regs_len = concrete_nlocals + stack_only;
            if s.registers_r.len() < min_regs_len {
                s.registers_r.resize(min_regs_len, OpRef::NONE);
            }
        }
        self.flush_to_frame(ctx);
        // pyjitpl.py:2973 reached_loop_header: a merge-point resume enters
        // the target loop at `pc`, so last_instr must be `pc - 1` so the
        // interpreter's `next_instr() = last_instr + 1` returns the merge
        // point. flush_to_frame already stored `orgpc - 1`; override with
        // the merge target.
        //
        // Propagation gap #1: propagate the override
        // into the virtualizable_boxes shadow so the writeback below
        // emits the merge-target PC, not the orgpc placed by
        // flush_to_frame. virtualizable_boxes[0] = vable_last_instr per
        // virtualizable_gen.rs:37-44 inputargs ordering.
        if let Some(pc) = target_pc {
            let last_instr_value = pc as i64 - 1;
            let opref = ctx.const_int(last_instr_value);
            let owns = {
                let s = self.sym_mut();
                s.vable_last_instr = opref;
                s.owns_virtualizable_shadow()
            };
            if owns {
                mirror_vable_static_to_boxes(
                    ctx,
                    "last_instr",
                    opref,
                    Value::Int(last_instr_value),
                );
            }
        }
        // No vable heap-writeback before the closing JUMP: the
        // virtualizable stays virtual across the loop/bridge edge. The
        // live vable scalars/array items reach the target LABEL through
        // the JUMP-arg derivation below; loop-invariant fields fold into
        // the resume-data payload of each guard. A guard failure rebuilds
        // the heap frame from that payload (`consume_vref_and_vable_boxes`
        // → `write_boxes`), and a forcing residual call materializes it
        // via `synchronize_virtualizable` — neither needs a pre-written
        // heap frame at the JUMP boundary.
        // An active virtualizable owner must have its array base seeded by
        // `init_symbolic` / `become_active_vable_owner` before the JUMP-arg
        // derivation below reads `nlocals`/`vable_array_base`.  `nlocals` is
        // NOT a usable proxy for "init ran": module-scope (`<module>`) frames
        // are vable owners (M1 portal gate) yet have `co_nlocals == 0`, the
        // same value as the struct default — names go through globals, not
        // fast locals.  `target_array_capacity` below handles `nlocals == 0`
        // via the `valuestackdepth` saturating-sub, so the seeded base is the
        // real precondition.
        debug_assert!(
            !self.sym().is_active_vable_owner || self.sym().vable_array_base.is_some(),
            "an active vable owner must have a seeded vable_array_base before close_loop_args_at"
        );
        // RPython close_loop_args parity: JUMP args must match the target
        // label's types (inputarg_types). materialize_loop_carried_value
        // boxes values to match (e.g. Int → Ref for virtualizable locals).
        //
        // For bridge traces, ctx.inputarg_types() returns the bridge's
        // guard fail_arg types, NOT the root loop's label types. The JUMP
        // targets the root loop label, so resolve the root loop's LABEL/
        // inputargs types via `front_target_inputarg_types` (peeled-entry
        // LABEL when unrolled, root TreeLoop.inputargs otherwise — see
        // `MetaInterp::front_target_inputarg_types` doc).
        let inputarg_types = {
            let (driver, _) = crate::driver::driver_pair();
            if driver.is_bridge_tracing() {
                if let Some(gk) = driver.current_trace_green_key() {
                    driver
                        .front_target_inputarg_types(gk)
                        .unwrap_or_else(|| ctx.inputarg_types())
                } else {
                    ctx.inputarg_types()
                }
            } else {
                ctx.inputarg_types()
            }
        };
        let num_scalars = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        // `extra_reds` reflects the canonical ec/red layout (NUM_EXTRA_REDS).
        // Drives the conditional ec push at args[1] and the dedup-side
        // OpRef ↔ virtualizable_box mapping below.
        let extra_reds = crate::virtualizable_gen::NUM_EXTRA_REDS;
        // pyjitpl.py:2954-2965 reached_loop_header parity: once the
        // descriptor-driven virtualizable path is active, JUMP args must carry
        // the full virtualizable array capacity. compile.rs later expands the
        // loop entry from the same heap lengths; emitting only the live stack
        // window here leaves too few source args for that expansion.
        let target_array_capacity = ctx
            .virtualizable_array_lengths()
            .map(|lengths| lengths.iter().copied().sum::<usize>())
            .filter(|&len| len >= self.sym().nlocals)
            .unwrap_or_else(|| {
                self.sym().nlocals
                    + self
                        .sym()
                        .valuestackdepth
                        .saturating_sub(self.sym().nlocals)
            });
        // portal-bridge keeps `s.valuestackdepth` at its initial seed
        // (see `portal_bridge_vable_vsd` doc); consult metadata at the
        // current pc so `stack_only` reflects the actual JUMP-source
        // stack depth instead of the stale symbolic counter.
        let portal_vsd = self.portal_bridge_vable_vsd(self.orgpc).map(|d| d as usize);
        // [frame, ec] portal-reds contract: recover ec before the sym()
        // snapshot below so JUMP args never carry OpRef::NONE in the ec
        // slot on adapter / bridge-from-guard paths.
        let recovered_ec = self.ensure_execution_context(ctx);
        // When the portal-bridge metadata is absent,
        // the stack depth fallback reads from `PyFrame.valuestackdepth`
        // (via `concrete_valuestackdepth()`) rather than the symbolic
        // mirror.  `close_loop_args_at` runs at the orgpc anchor where
        // PyFrame still holds the pre-opcode state.
        let concrete_vsd = self
            .concrete_valuestackdepth()
            .unwrap_or_else(|| self.sym().valuestackdepth);
        let (
            frame,
            execution_context,
            next_instr,
            code,
            stack_depth,
            debugdata,
            lastblock,
            namespace,
            nlocals,
            locals,
            stack,
            _local_types,
            _stack_types,
        ) = {
            let s = self.sym();
            let nlocals = s.nlocals;
            let stack_only = portal_vsd.unwrap_or(concrete_vsd).saturating_sub(s.nlocals);
            // virtualizable.py:86-98 `read_boxes` + pyjitpl.py:2954-2965
            // `reached_loop_header`: `virtualizable_boxes` length is the
            // target vable array capacity (`nlocals + ncells + co_stacksize`),
            // not the live Python stack depth. JUMP args carry that full
            // capacity so every target LABEL slot has a matching source.
            // Slots beyond the live prefix are left as `OpRef::NONE` and
            // filled by `materialize_fail_arg_slot` below, which reads
            // `concrete_value_at` and falls back to `PY_NULL` for dead
            // capacity slots — mirroring RPython's null-padded
            // virtualizable_boxes tail.
            let target_stack_capacity = target_array_capacity.saturating_sub(nlocals);
            let mut stack_types_vec =
                s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())].to_vec();
            stack_types_vec.resize(target_stack_capacity, Type::Ref);
            // pyjitpl.py:2954-2965 `reached_loop_header` parity: read
            // both locals and stack values from the virtualizable shadow
            // (`virtualizable_boxes[NUM_VABLE_SCALARS + i]`). The shadow
            // is RPython's single source of truth for the
            // `locals_cells_stack_w` view; `push_typed_value` /
            // `store_local_value` mirror every write into it for traces
            // that satisfy `owns_virtualizable_shadow()` (the loop
            // portal AND every bridge that seeded its own
            // `bridge_local_oprefs`). Non-owner traces (rare —
            // inline-callee scaffolding before the inline path takes
            // over) keep the legacy semantic registers_r read.
            //
            // Shadow path bounds: virtualizable_boxes is sized to
            // `target_array_capacity` (NUM_VABLE_SCALARS + nlocals +
            // co_stacksize) at vable init, so `stack_only.min(
            // target_stack_capacity)` is the correct live-prefix
            // length; the legacy `reg_len.saturating_sub(locals_len)`
            // cap silently dropped OpRefs once the operand stack
            // overgrew the registers_r slice (RPython
            // `reached_loop_header` carries the full
            // `virtualizable_boxes[:-1]` regardless of register-file
            // occupancy).  The non-shadow registers_r read keeps the
            // reg_len cap because reading past `registers_r.len()`
            // panics there.
            let (locals_vec, mut stack_vec) = if s.owns_virtualizable_shadow() {
                let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                let shadow_stack_len = stack_only.min(target_stack_capacity);
                let locals_vec: Vec<OpRef> = (0..nlocals)
                    .map(|i| {
                        ctx.virtualizable_box_at(nvs + i)
                            .expect("close_loop_args_at: missing virtualizable local box")
                    })
                    .collect();
                let stack_vec: Vec<OpRef> = (0..shadow_stack_len)
                    .map(|d| {
                        ctx.virtualizable_box_at(nvs + nlocals + d)
                            .expect("close_loop_args_at: missing virtualizable stack box")
                    })
                    .collect();
                (locals_vec, stack_vec)
            } else {
                let read_color =
                    |color: usize| s.registers_r.get(color).copied().unwrap_or(OpRef::NONE);
                let locals_vec: Vec<OpRef> = (0..nlocals).map(|i| read_color(i)).collect();
                let live_stack_len = stack_only.min(target_stack_capacity);
                let stack_vec: Vec<OpRef> = (0..live_stack_len)
                    .map(|d| read_color(nlocals + d))
                    .collect();
                (locals_vec, stack_vec)
            };
            stack_vec.resize(target_stack_capacity, OpRef::NONE);
            (
                s.frame,
                recovered_ec,
                s.vable_last_instr,
                s.vable_pycode,
                s.vable_valuestackdepth,
                s.vable_debugdata,
                s.vable_lastblock,
                s.vable_w_globals,
                nlocals,
                locals_vec,
                stack_vec,
                s.symbolic_local_types.clone(),
                stack_types_vec,
            )
        };
        let mut args = vec![frame];
        // NUM_EXTRA_REDS == 1 (crate const-assert): `reds = ['frame', 'ec']`.
        args.push(execution_context);
        args.extend_from_slice(&[
            next_instr,
            code,
            stack_depth,
            debugdata,
            lastblock,
            namespace,
        ]);
        for (idx, value) in locals.into_iter().enumerate() {
            let target_type = inputarg_types
                .get(num_scalars + idx)
                .copied()
                .unwrap_or(Type::Ref);
            // Materialize NONE slots from concrete frame before boxing.
            // RPython's live_arg_boxes never contains holes at loop closure
            // because MIFrame.run_one_step always updates all live registers.
            let value = self.materialize_fail_arg_slot(ctx, value, target_type, idx);
            args.push(self.materialize_loop_carried_value(ctx, value, target_type));
        }
        // Live value-stack window: slots at index >= live_stack_len are dead
        // capacity (Python index >= valuestackdepth). `interpreter/pyframe.py`
        // `popvalue_maybe_none` nulls a popped slot, so `read_boxes` reports
        // None for every dead stack slot and `virtualizable_boxes` carries a
        // null tail; the target loop LABEL's stack tail is therefore all-null
        // and folds away. pyre's bridge value-stack clear is gated off
        // (`is_active_vable_owner` excludes bridges to avoid the null-base
        // `InvalidLoop` abort), so the concrete frame still holds the stale
        // popped pointers (e.g. a caught exception) in those slots. Reading
        // them through `materialize_fail_arg_slot` would put live pointers in
        // the JUMP tail where the loop label expects null, blocking
        // `optimize_bridge` retarget, and would also disagree with resume,
        // which reconstructs the dead tail as null. Force the null here: these
        // slots reach only the terminal JUMP args, never an in-trace field
        // base, so no `get_const_info_mut` null-base abort.
        let live_stack_len = portal_vsd.unwrap_or(concrete_vsd).saturating_sub(nlocals);
        for (stack_idx, value) in stack.into_iter().enumerate() {
            let target_type = inputarg_types
                .get(num_scalars + nlocals + stack_idx)
                .copied()
                .unwrap_or(Type::Ref);
            let value = if stack_idx >= live_stack_len {
                let typed_null = extract_concrete_typed_value(target_type, PY_NULL);
                fail_arg_opref_for_typed_value(ctx, typed_null)
            } else {
                self.materialize_fail_arg_slot(ctx, value, target_type, nlocals + stack_idx)
            };
            args.push(self.materialize_loop_carried_value(ctx, value, target_type));
        }
        // virtualizable.py:44 parity (delayed): now that all materialize_loop_
        // carried_value calls have consulted each OpRef's actual type, flip the
        // symbolic type maps to the post-loop invariant where every array slot
        // is Ref. Downstream consumers (reached_loop_header's live_types, the
        // merge-point snapshot it stores) observe the Ref contract while the
        // box/unbox decisions above still see the pre-loop truth.
        {
            let s = self.sym_mut();
            let stack_only = s.valuestackdepth.saturating_sub(s.nlocals);
            s.symbolic_local_types = vec![Type::Ref; concrete_nlocals];
            s.symbolic_stack_types = vec![Type::Ref; stack_only];
        }
        // pyjitpl.py:2934-2965 remove_consts_and_duplicates:
        //     def remove_consts_and_duplicates(self, boxes, endindex, duplicates):
        //         for i in range(endindex):
        //             box = boxes[i]
        //             if isinstance(box, Const) or box in duplicates:
        //                 boxes[i] = self.history.record_same_as(box)
        //             else:
        //                 duplicates[box] = None
        //
        //     def reached_loop_header(self, greenboxes, redboxes):
        //         duplicates = {}
        //         self.remove_consts_and_duplicates(redboxes, len(redboxes),
        //                                           duplicates)
        //         live_arg_boxes = greenboxes + redboxes
        //         if self.jitdriver_sd.virtualizable_info is not None:
        //             self.remove_consts_and_duplicates(
        //                 self.virtualizable_boxes,
        //                 len(self.virtualizable_boxes)-1,
        //                 duplicates)
        //             live_arg_boxes += self.virtualizable_boxes
        //             live_arg_boxes.pop()
        //
        // RPython dedups ALL of redboxes (1 = vable_box = frame) AND
        // ALL of virtualizable_boxes[:-1] (static_fields + array_items),
        // sharing one `duplicates` dict across both calls. In pyre's flat
        // layout `args = [frame, ni, code, vsd, ns, locals..., stack...]`,
        // that corresponds to every index 0..args.len(). Previously pyre
        // skipped the 7 scalar header slots (frame + 6 static fields),
        // which is a line-by-line divergence from RPython.
        // Track slots that the dedup actually mutated so we can mirror the
        // `put_back_list_of_boxes3` mutation below (pyjitpl.py:1578 writes
        // the deduped redboxes back to the frame's registers; RPython's
        // `remove_consts_and_duplicates` additionally mutates
        // `self.virtualizable_boxes` in place so subsequent reads see the
        // SameAs-wrapped identities).
        let mut dedup_changed: Vec<(usize, OpRef)> = Vec::new();
        {
            use std::collections::HashSet;
            let mut duplicates: HashSet<OpRef> = HashSet::new();
            for i in 0..args.len() {
                let opref = args[i];
                if opref.is_constant() || !duplicates.insert(opref) {
                    // pyjitpl.py:2934-2965 `record_same_as(box)` uses the
                    // `box.type` intrinsic to pick `same_as_i/r/f` — the
                    // SameAs op's result type matches the input box, NEVER
                    // the slot's declared type. When `args[i]` is a constant
                    // whose Value type differs from the slot's declared
                    // `inputarg_types[i]` (e.g. an Int constant placeholder
                    // routed into a Ref-typed vable header slot), wrapping
                    // it as `same_as_for_type(slot_type)` produces a
                    // cross-type SameAs whose `make_equal_to` absorb in
                    // `optimizer.rs::propagate_from_pass_range` violates the
                    // Box.type invariant in `OptContext::replace_op`.
                    //
                    // Match RPython by deriving the SameAs op from the
                    // OpRef's actual type via `ctx.get_opref_type`, falling
                    // back to the slot type only when the OpRef has no
                    // recoverable type (which would be a separate bug).
                    let tp = ctx
                        .get_opref_type(opref)
                        .or_else(|| inputarg_types.get(i).copied())
                        .unwrap_or(majit_ir::Type::Ref);
                    let same_as_op = majit_ir::OpCode::same_as_for_type(tp);
                    let new_opref = ctx.record_op(same_as_op, &[opref]);
                    args[i] = new_opref;
                    dedup_changed.push((i, new_opref));
                }
            }
        }
        // pyjitpl.py:2961-2963 in-place mutation of self.virtualizable_boxes:
        //     self.remove_consts_and_duplicates(
        //         self.virtualizable_boxes,
        //         len(self.virtualizable_boxes)-1,
        //         duplicates)
        //
        // RPython's `remove_consts_and_duplicates` writes the SameAs results
        // back into `self.virtualizable_boxes[i]` IN PLACE for `i` in
        // `range(len-1)`. The trailing element (`virtualizable_boxes[-1]`,
        // the standard vable identity itself = pyre's frame OpRef) is
        // intentionally skipped. The mutated `self.virtualizable_boxes`
        // then feeds the GUARD_FUTURE_CONDITION snapshot below.
        //
        // pyre's `args` Vec layout is `[frame, ni, code, vsd, ns,
        // locals..., stack...]` where `args[0]` is the trailing
        // virtualizable identity (mapped to `vb[len-1]`) and `args[1..]`
        // is `vb[0..len-1]`. The line-by-line mirror here mutates
        // `ctx.virtualizable_boxes[i-1]` for every dedup'd `args[i]`
        // with `i >= 1`, leaving `vb[len-1]` (the trailing identity)
        // untouched.
        //
        // Note: pyjitpl.py:1578 `put_back_list_of_boxes3` writes the
        // dedup'd `redboxes` back to the FRAME's `registers_i/r/f`
        // arrays. RPython only runs `put_back_list_of_boxes3` from the
        // `opimpl_jit_merge_point` failed-to-close path (i.e. when
        // `reached_loop_header` returns normally instead of raising
        // SwitchToBlackhole). pyre's `close_loop_args_at` is the
        // SUCCESS path (the trace is closing), so the put_back has no
        // matching call site here — it would belong on the path where
        // pyre fails to close at a merge point and continues tracing,
        // which pyre's tracer does not currently expose.
        for &(idx, new_opref) in &dedup_changed {
            if idx <= extra_reds {
                // args[0] = frame = ctx.virtualizable_boxes[len-1].
                // Any extra reds that follow it are not part of
                // `virtualizable_boxes`, so only the virtualizable payload
                // starting after `[frame, extra_reds...]` is mirrored back.
                continue;
            }
            let vb_idx = idx - (1 + extra_reds);
            ctx.set_virtualizable_box_at(vb_idx, new_opref);
        }
        // pyjitpl.py:1578 put_back_list_of_boxes3: write dedup'd values back
        // to frame symbolic state so subsequent tracing sees the SameAs-wrapped
        // identities. RPython runs this on the "continue tracing" path after
        // reached_loop_header returns without closing. Harmless on the "close
        // loop" path since the frame won't be reused.
        {
            // `num_scalars` (NUM_SCALAR_INPUTARGS) already counts extra_reds
            // (frame + extra_reds + vable static fields).
            let total_scalar_prefix = num_scalars;
            let s = self.sym_mut();
            for &(idx, new_opref) in &dedup_changed {
                if idx < total_scalar_prefix {
                    match idx {
                        0 => s.frame = new_opref,
                        // NUM_EXTRA_REDS == 1 (crate const-assert).
                        1 => s.execution_context = new_opref,
                        _ => match idx - extra_reds {
                            1 => s.vable_last_instr = new_opref,
                            2 => s.vable_pycode = new_opref,
                            3 => s.vable_valuestackdepth = new_opref,
                            4 => s.vable_debugdata = new_opref,
                            5 => s.vable_lastblock = new_opref,
                            6 => s.vable_w_globals = new_opref,
                            _ => {}
                        },
                    }
                } else {
                    let local_idx = idx - total_scalar_prefix;
                    // `registers_r` is the unified abstract register
                    // file; locals + stack tail share the same addr
                    // space, so the dedup'd rename writes to the single
                    // slot regardless of whether the dedup refers to a
                    // local or a stack entry.
                    if local_idx < s.registers_r.len() {
                        s.registers_r[local_idx] = new_opref;
                    }
                }
            }
        }
        // pyjitpl.py:2967-2969: generate a dummy GUARD_FUTURE_CONDITION
        // just before the JUMP so that unroll can use it when it's
        // creating artificial guards (patchguardop). record_guard calls
        // capture_resumedata which captures the full framestack +
        // virtualizable_boxes + virtualref_boxes.
        //
        // RPython only emits GUARD_FUTURE_CONDITION here. GUARD_NOT_INVALIDATED
        // is *not* unconditionally emitted before JUMP — pyjitpl.py:1086-1089
        // emits it only inside `opimpl_record_quasi_immutable_field`, after a
        // quasi-immut field read sets `heapcache.need_guard_not_invalidated`.
        // The pyre frontend does the same via `flush_guard_not_invalidated`,
        // so an unconditional emit here would (a) leak resume data for traces
        // that have no quasi-immut dep at all, and (b) leave a runtime guard
        // whose flag is decoupled from any watcher, which can spuriously
        // exit a hot inner loop with no chance of re-tracing.
        //
        // RPython parity: orgpc must be the loop header TARGET, not the
        // JUMP_BACKWARD's PC. The patchguardop from this GuardFutureCondition
        // provides the resume_position for all peeled body virtual state guards.
        // If orgpc is wrong, all those guards resume at the wrong PC.
        if let Some(pc) = target_pc {
            self.orgpc = pc;
        }
        self.generate_guard(ctx, majit_ir::OpCode::GuardFutureCondition, &[]);
        // pyjitpl.py:2971 assert len(self.virtualref_boxes) == 0,
        //     "missing virtual_ref_finish()?"
        // Reached loop header must not have dangling virtualrefs — they
        // should have been finished by prior vrefs_after_residual_call /
        // stop_tracking_virtualref. pyre's equivalent is sym.virtualref_boxes.
        debug_assert!(
            self.sym().virtualref_boxes.is_empty(),
            "missing virtual_ref_finish()? close_loop_args_at reached with \
             virtualref_boxes={:?}",
            self.sym().virtualref_boxes.len()
        );
        // Verify `live_args_shape_at` formula matches actual output.
        // If this fires, the helper's shape derivation is stale relative
        // to `close_loop_args_at`'s args layout — update both in lockstep.
        debug_assert_eq!(
            args.len(),
            self.live_args_shape_at(ctx),
            "live_args_shape_at must predict close_loop_args_at output length",
        );
        // virtualstate.py:39-67 — populate the `Box.value` stamp
        // so the optimizer can route `cpu.cls_of_box(runtime_box)` /
        // `runtime_box.getref_base()` through the materialised BoxRef's
        // per-type mixin slot. Writes go into `ctx.opref_concrete`; the
        // optimizer stamps them onto BoxRefs before virtualstate matching.
        //
        // virtualstate.py:646-648 requires `runtime_boxes` to be fully
        // parallel with `boxes`. Pyre attempts to populate every slot
        // but skips type-mismatched and Null (untracked) entries:
        //   args[0]                                ↔ frame (raw ptr)
        //   args[1..1+extra_reds]                  ↔ ec (raw ptr)
        //   args[1+extra_reds..num_scalars]        ↔ vable scalars (shadow)
        //   args[num_scalars + i]                  ↔ locals slot i
        //   args[num_scalars + nlocals + j]        ↔ stack slot nlocals+j
        // `num_scalars` already counts `extra_reds` (per dedup loop above).
        {
            let header_off = num_scalars;
            // Mirror args' stack range (line 2891): portal-bridge keeps
            // `s.valuestackdepth` at the initial seed (residual-call
            // paths bypass `push_typed_value`), so the args layout uses
            // `portal_vsd` metadata when present.  The runtime-value
            // walk must use the same range or the locals/stack slot
            // indices misalign relative to `args[header_off + ..]`.
            let stack_only = portal_vsd
                .unwrap_or(self.sym().valuestackdepth)
                .saturating_sub(self.sym().nlocals);
            let collect_kind =
                |opref: OpRef, cv: crate::state::ConcreteValue| -> Option<majit_ir::Value> {
                    let tp = opref.ty()?;
                    match (tp, cv) {
                        (Type::Int, crate::state::ConcreteValue::Int(v)) => {
                            Some(majit_ir::Value::Int(v))
                        }
                        (Type::Float, crate::state::ConcreteValue::Float(v)) => {
                            Some(majit_ir::Value::Float(v))
                        }
                        (Type::Ref, crate::state::ConcreteValue::Ref(obj)) => {
                            Some(majit_ir::Value::Ref(majit_ir::GcRef(obj as usize)))
                        }
                        // ConcreteValue::Null is the "untracked" sentinel
                        // (state.rs:1286); real frame nulls are preserved as
                        // ConcreteValue::Ref(PY_NULL). Do not stamp Null as
                        // a typed null ref — it means "no runtime value
                        // recorded for this slot".
                        (_, crate::state::ConcreteValue::Null) => None,
                        // Type mismatch: pyre's locals/stack OpRefs are
                        // Type::Ref (Python values are PyObject*), but
                        // ConcreteValue auto-decodes unboxed int/float from
                        // the live pyobj header. RPython's typed
                        // InputArgInt/Ref/Float boxes prevent this
                        // structurally. Skip stamp to preserve main's
                        // "no value" baseline rather than injecting a
                        // cross-typed value.
                        _ => None,
                    }
                };
            let record = |ctx: &mut TraceCtx, opref: OpRef, value: majit_ir::Value| {
                if opref != OpRef::NONE && !opref.is_constant() {
                    ctx.set_opref_concrete(opref, value);
                }
            };
            // args[0] frame
            let frame_addr = self.concrete_frame_addr;
            if frame_addr != 0 {
                if let Some(&opref) = args.first() {
                    record(
                        ctx,
                        opref,
                        majit_ir::Value::Ref(majit_ir::GcRef(frame_addr)),
                    );
                }
            }
            // args[1..1+extra_reds] ec — NUM_EXTRA_REDS == 1.
            let ec_ptr = self.sym().concrete_execution_context as usize;
            if ec_ptr != 0 {
                if let Some(&opref) = args.get(1) {
                    record(ctx, opref, majit_ir::Value::Ref(majit_ir::GcRef(ec_ptr)));
                }
            }
            // args[1+extra_reds..num_scalars] vable static fields — the
            // shadow `ctx.virtualizable_entry_at(i)` tracks the JIT's
            // current belief about each vable scalar, kept in sync with
            // `s.vable_*` OpRefs across setfield_vable updates.
            let vable_start = 1 + extra_reds;
            let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
            for i in 0..nvs {
                let slot_idx = vable_start + i;
                let Some(&opref) = args.get(slot_idx) else {
                    break;
                };
                if let Some((shadow_opref, value)) = ctx.virtualizable_entry_at(i) {
                    if shadow_opref == opref {
                        record(ctx, opref, value);
                    }
                }
            }
            for i in 0..nlocals {
                let slot_idx = header_off + i;
                if let Some(&opref) = args.get(slot_idx) {
                    if let Some(v) = collect_kind(opref, self.sym().concrete_value_at(i)) {
                        record(ctx, opref, v);
                    }
                }
            }
            for j in 0..stack_only {
                let slot_idx = header_off + nlocals + j;
                if let Some(&opref) = args.get(slot_idx) {
                    if let Some(v) = collect_kind(opref, self.sym().concrete_value_at(nlocals + j))
                    {
                        record(ctx, opref, v);
                    }
                }
            }
        }
        args
    }

    /// pyjitpl.py:2586 capture_resumedata: build fail_args for CURRENT
    /// top frame. Returns the scalar header plus active_boxes —
    /// `[frame, (ec)?, last_instr, pycode, valuestackdepth, debugdata,
    /// lastblock, w_globals, active_boxes...]` — matching
    /// `interp_jit.py:25-31 PyFrame._virtualizable_` /
    /// `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS` line-by-line.
    /// `NUM_EXTRA_REDS` controls whether the ec slot
    /// (interp_jit.py:67 `reds = ['frame', 'ec']`) is present between
    /// frame and the vable static fields. Dormant under
    /// NUM_EXTRA_REDS=0 (skips ec push, preserves pre-ec 7-scalar
    /// layout). virtualizable.py:86 read_boxes: all static fields in
    /// order.
    pub(crate) fn current_fail_args(&mut self, ctx: &mut TraceCtx) -> Vec<OpRef> {
        self.flush_to_frame_for_guard(ctx);
        let active_boxes = self.get_list_of_active_boxes(ctx, false, false, None);
        // [frame, ec] portal-reds contract. Recover ec before snapshotting
        // sym fields so guard fail_args never carry OpRef::NONE in the ec
        // slot (adapter/bridge-from-guard paths).
        let ec = self.ensure_execution_context(ctx);
        let s = self.sym();
        let mut fa =
            Vec::with_capacity(crate::virtualizable_gen::NUM_SCALAR_INPUTARGS + active_boxes.len());
        fa.push(s.frame);
        // NUM_EXTRA_REDS == 1 (crate const-assert in `lib.rs`).
        // `interp_jit.py:67 reds = ['frame', 'ec']`.
        fa.push(ec);
        fa.extend_from_slice(&[
            s.vable_last_instr,
            s.vable_pycode,
            s.vable_valuestackdepth,
            s.vable_debugdata,
            s.vable_lastblock,
            s.vable_w_globals,
        ]);
        fa.extend_from_slice(&active_boxes);
        fa
    }

    /// pyjitpl.py:1087 parity: after a field read that might have set the
    /// needs_guard_not_invalidated flag (quasi-immutable field), emit the
    /// guard with full snapshot via record_guard.
    pub(crate) fn flush_guard_not_invalidated(&mut self, ctx: &mut TraceCtx) {
        if let Some(saved_orgpc) = ctx.pending_guard_not_invalidated_pc() {
            ctx.set_pending_guard_not_invalidated(None);
            // pyjitpl.py:1087 parity: use the field read's orgpc so the
            // snapshot captures the correct liveness state.
            let current_orgpc = self.orgpc;
            self.orgpc = saved_orgpc;
            self.generate_guard(ctx, OpCode::GuardNotInvalidated, &[]);
            self.orgpc = current_orgpc;
        }
    }

    /// PyPy generate_guard + capture_resumedata: uses current_fail_args
    /// which encodes the full framestack for multi-frame resume.
    /// pyjitpl.py:3222 store_token_in_vable():
    ///   force_token = self.history.record0(rop.FORCE_TOKEN, ...)
    ///   self.history.record2(rop.SETFIELD_GC, vbox, force_token, ...)
    ///   self.generate_guard(rop.GUARD_NOT_FORCED_2)
    pub(crate) fn store_token_in_vable(&mut self, ctx: &mut TraceCtx) {
        if ctx.store_token_in_vable_setfield() {
            self.generate_guard(ctx, OpCode::GuardNotForced2, &[]);
        }
    }

    pub(crate) fn generate_guard(&mut self, ctx: &mut TraceCtx, opcode: OpCode, args: &[OpRef]) {
        // pyjitpl.py:2558-2560 generate_guard parity:
        //     if isinstance(box, Const):    # no need for a guard
        //         return
        // The first arg of every data guard (GUARD_CLASS, GUARD_TRUE,
        // GUARD_NONNULL, GUARD_VALUE, ...) is the box being checked.
        // Control-flow guards (GUARD_NOT_FORCED, GUARD_NO_OVERFLOW,
        // GUARD_NOT_INVALIDATED, ...) call generate_guard with `args=&[]`,
        // so `args.first()` is None and the check is skipped — matching
        // RPython where `box=None` for those guards.
        if let Some(&first) = args.first() {
            if first.is_constant() {
                return;
            }
        }
        // pyjitpl.py:1087 parity: flush pending guard_not_invalidated
        // before recording any new guard (the quasi-immut guard should be
        // emitted with its own snapshot before the current guard).
        if opcode != OpCode::GuardNotInvalidated {
            self.flush_guard_not_invalidated(ctx);
        }
        // pyjitpl.py:2575-2578: determine after_residual_call from guard opcode.
        // opencoder.py:767: when true, all boxes in top frame are live
        // (liveness filter disabled for residual call guards).
        let after_residual_call = matches!(
            opcode,
            OpCode::GuardException
                | OpCode::GuardNoException
                | OpCode::GuardNotForced
                | OpCode::GuardAlwaysFails
        );
        if after_residual_call {
            // pyjitpl.py:2586-2602: residual-call guards snapshot the state
            // AFTER the call, using the auto-advanced pc and post-call
            // register file. The opcode-start snapshot is only for
            // re-executing the current opcode from orgpc.
            self.clear_pre_opcode_state();
        }
        // opencoder.py:819 capture_resumedata(framestack) parity:
        // Encode the full framestack [callee (top), caller (parent)] into
        // a multi-frame snapshot. The callee's pc is set to resumepc
        // (orgpc), while the caller keeps its original pc (return_point).
        // pyjitpl.py:2597 passes full framestack + vable/vref boxes.
        if !self.parent_frames.is_empty() {
            // pyjitpl.py:2593-2596: top frame pc = resumepc (orgpc)
            self.flush_to_frame_for_guard(ctx);
            // pyjitpl.py:177: active boxes = registers only (no header).
            let callee_active_boxes =
                self.get_list_of_active_boxes(ctx, false, after_residual_call, Some(self.orgpc));
            // RPython Box.type parity: snapshot types match the full
            // (un-filtered) active_boxes — constants are part of the
            // snapshot via TAGCONST.
            let callee_snapshot_types_full =
                self.build_fail_arg_types_for_active_boxes(&callee_active_boxes);

            // snapshot.pc must match the liveness PC used for active boxes
            // (get_list_of_active_boxes uses fallthrough_pc when
            // after_residual_call), and folds in the after-residual-call
            // marker so an inlined frame's try-block residual call resumes
            // at its OWN catch_exception — the single-frame path
            // (capture_resumedata) applies the same fold.
            let callee_live_pc = self.marker_aware_resume_pc(self.orgpc, after_residual_call);
            // opencoder.py:819-834: snapshot uses active boxes (not fail_args).
            let snapshot = self.build_framestack_snapshot(
                ctx,
                callee_live_pc,
                &callee_active_boxes,
                &callee_snapshot_types_full,
            );
            // Snapshot is the source of truth — the
            // optimizer's `store_final_boxes_in_guard`
            // (`optimizeopt/mod.rs:3200`) overwrites `op.fail_args` from
            // the snapshot via `op.store_final_boxes(liveboxes)`
            // (mod.rs:3392), so the inline `fail_args` copy that the
            // legacy `record_guard_typed_with_fail_args` path used to
            // write was redundant.  Mirrors RPython
            // `pyjitpl.MetaInterp.generate_guard` (pyjitpl.py:2558-2602)
            // which records the guard with no inline fail_args and lets
            // `capture_resumedata` + `_number_boxes` populate them from
            // the snapshot chain.
            //
            // The state-fields header `[s.frame, s.vable_*]` that fed
            // the inline fail_args is now dead and removed; the
            // snapshot's `vable_boxes` already encodes the same
            // information via `build_virtualizable_boxes`. Parent-frame
            // types still flow into the recorded guard via
            // `extend_types_with_parents`.
            let types = self.extend_types_with_parents(ctx, callee_snapshot_types_full);
            let snapshot_id = ctx.capture_resumedata(snapshot);

            ctx.record_guard_typed(opcode, args, types);
            ctx.set_last_guard_resume_position(snapshot_id);
            return;
        }

        // pyjitpl.py:2586-2596 capture_resumedata(resumepc) parity:
        // Normal guards: resumepc = orgpc (re-execute the opcode from start).
        // after_residual_call guards (GUARD_NOT_FORCED, GUARD_NO_EXCEPTION):
        //   RPython generate_guard passes resumepc=-1, and capture_resumedata
        //   skips the "frame.pc = resumepc" assignment — frame.pc stays at
        //   the auto-advanced next instruction (pyre fallthrough_pc equivalent).
        //   This ensures the liveness PC, header ni, and blackhole resume PC
        //   all point to the instruction AFTER the call, not the call itself.
        let resume_pc = if after_residual_call {
            self.fallthrough_pc
        } else {
            self.orgpc
        };
        self.generate_guard_core(ctx, opcode, args, resume_pc, after_residual_call);
    }

    /// Core guard recording with explicit resume PC.
    ///
    /// pyjitpl.py:2558-2584 generate_guard parity: record guard op,
    /// then call capture_resumedata.
    fn generate_guard_core(
        &mut self,
        ctx: &mut TraceCtx,
        opcode: OpCode,
        args: &[OpRef],
        resume_pc: usize,
        after_residual_call: bool,
    ) {
        self.flush_to_frame_for_guard(ctx);
        let active_boxes =
            self.get_list_of_active_boxes(ctx, false, after_residual_call, Some(self.orgpc));
        let snapshot_full_types = self.build_fail_arg_types_for_active_boxes(&active_boxes);
        let fail_arg_types = snapshot_full_types.clone();

        // Snapshot is the source of truth — the
        // optimizer's `store_final_boxes_in_guard`
        // (`optimizeopt/mod.rs:3200`) overwrites `op.fail_args` from the
        // snapshot built below via `op.store_final_boxes(liveboxes)`
        // (mod.rs:3392), so the inline `fail_args` copy that the legacy
        // `record_guard_typed_with_fail_args` path used to write was
        // redundant.  Mirrors RPython
        // `pyjitpl.MetaInterp.generate_guard` (pyjitpl.py:2558-2602)
        // which records the guard with no inline fail_args and lets
        // `capture_resumedata` + `_number_boxes` populate them from the
        // snapshot chain.
        ctx.record_guard_typed(opcode, args, fail_arg_types);

        // pyjitpl.py:2579: self.capture_resumedata(resumepc, after_residual_call)
        self.capture_resumedata(
            ctx,
            resume_pc,
            after_residual_call,
            &active_boxes,
            &snapshot_full_types,
        );
        // pyjitpl.py:2581: self.staticdata.profiler.count_ops(opnum, Counters.GUARDS).
        // Atomic fetch_add through the shared `Arc<MetaInterpStaticData>`
        // — `&self` access is enough because `JitProfiler::count_ops`
        // bumps an `AtomicUsize`.
        ctx.profiler()
            .count_ops(opcode, majit_metainterp::counters::GUARDS);
    }

    /// pyjitpl.py:2586-2602 capture_resumedata parity.
    ///
    /// Temporarily sets frame.pc = resumepc, captures the full framestack
    /// ([self as top/callee] + self.parent_frames as parents) plus
    /// virtualizable_boxes + virtualref_boxes into a snapshot, then
    /// restores frame.pc.  Matches opencoder.py:819-832
    /// `capture_resumedata(framestack, virtualizable_boxes,
    /// virtualref_boxes, after_residual_call=False)` which walks the full
    /// `self.framestack` to build one SnapshotFrame per live frame.
    /// Resolve a top-frame snapshot pc, folding in the after-residual-call
    /// marker when the residual call at `call_pc` sits inside a try-block.
    ///
    /// `call_pc` is the CALL opcode pc (the frame's pc before it was
    /// advanced to the resume target).  Used by BOTH the single-frame
    /// `capture_resumedata` and the multi-frame guard path so a residual
    /// call resumes at its OWN catch_exception even when the calling frame
    /// was inlined (`pyjitpl.py:2582 capture_resumedata` applies the
    /// `after_residual_call` resume consistently to the top frame).
    ///
    /// The marker bit (1 << 14) steals the top bit of the i16 pc word
    /// `resumecode::append_int` allows.  A *marked* pc ORs the bit onto a
    /// value gated `< 1 << 14`; an *unmarked* pc (no try-block catch, or
    /// the non-after_residual_call path) must independently leave bit 14
    /// free, or `decode_resume_pc` mis-reads it as marked.  A function with
    /// >= 16384 trace-bytecode units exceeds this and cannot be encoded
    /// under the bit-14 scheme — request a trace abort rather than corrupt
    /// decode silently at resume time (resumedata.rs:48-62).
    /// `abort_unencodable_resume_pc` sets the flag `metainterp::interpret`
    /// polls each step, so an unencodable pc (an oversized function, or a
    /// corrupted cross-frame coordinate, #124/#130) falls back to the
    /// interpreter rather than crashing.
    fn marker_aware_resume_pc(&self, call_pc: usize, after_residual_call: bool) -> usize {
        let flag = majit_ir::resumedata::AFTER_RESIDUAL_CALL_PC_FLAG as usize;
        let wants_marker = after_residual_call && {
            let jitcode_index = unsafe { (*self.sym().jitcode).index } as i32;
            crate::state::pyjitcode_for_jitcode_index(jitcode_index)
                .and_then(|pj| pj.after_residual_call_resume_pc_for(call_pc))
                .is_some()
        };
        if wants_marker {
            // A residual call in a try-block must resume at its OWN catch,
            // routed by the bit-14 marker.  If `call_pc` cannot fit under the
            // marker, downgrading to an unmarked pc would silently mis-resume
            // (decode routes through `pc_map` re-execution instead of
            // `after_residual_call_resume_pc_for`), so request a trace abort
            // here too → interpreter fallback.
            if call_pc >= flag {
                return crate::state::abort_unencodable_resume_pc(call_pc);
            }
            majit_ir::resumedata::encode_after_residual_call_pc(call_pc as i32) as usize
        } else {
            let raw = if after_residual_call {
                self.fallthrough_pc
            } else {
                call_pc
            };
            if raw >= flag {
                return crate::state::abort_unencodable_resume_pc(raw);
            }
            raw
        }
    }

    /// Parent-frame analogue of [`Self::marker_aware_resume_pc`].  A parent
    /// (`in_a_call`) frame whose CALL sits in a try-block must resume at that
    /// call's OWN `catch_exception` when a guard deopts inside the callee and
    /// the exception unwinds to a handler here (`pyjitpl.py:2601-2602`;
    /// `blackhole.py:396-410 handle_exception_in_frame`).  Fold the bit-14
    /// marker onto the CALL pc so the decoder routes through
    /// `after_residual_call_resume_pc` (its catch) rather than `pc_map` (the
    /// next opcode).  Without a catch marker, keep the plain `return_point_pc`
    /// (fallthrough) — the exception then propagates out of this frame.  The
    /// liveness encoded for this frame (`get_list_of_active_boxes` →
    /// `materialize_parent_snapshot_state`) is keyed on the same marker, so
    /// encode and decode read the one `-live-`.
    fn marker_aware_parent_resume_pc(
        parent_jitcode_index: u32,
        call_pc: Option<usize>,
        return_point_pc: usize,
    ) -> usize {
        let flag = majit_ir::resumedata::AFTER_RESIDUAL_CALL_PC_FLAG as usize;
        let marked_call_pc = call_pc.filter(|&cp| {
            crate::state::pyjitcode_for_jitcode_index(parent_jitcode_index as i32)
                .and_then(|pj| pj.after_residual_call_resume_pc_for(cp))
                .is_some()
        });
        if let Some(cp) = marked_call_pc {
            if cp >= flag {
                return crate::state::abort_unencodable_resume_pc(cp);
            }
            majit_ir::resumedata::encode_after_residual_call_pc(cp as i32) as usize
        } else {
            if return_point_pc >= flag {
                return crate::state::abort_unencodable_resume_pc(return_point_pc);
            }
            return_point_pc
        }
    }

    fn capture_resumedata(
        &mut self,
        ctx: &mut TraceCtx,
        resume_pc: usize,
        after_residual_call: bool,
        active_boxes: &[OpRef],
        snapshot_full_types: &[Type],
    ) {
        // pyjitpl.py:2594-2596: saved_pc = frame.pc; frame.pc = resumepc
        let saved_orgpc = self.orgpc;
        let saved_ni = self.sym().vable_last_instr;
        let saved_vsd = self.sym().vable_valuestackdepth;
        self.orgpc = resume_pc;

        // The snapshot's frame.pc must match the liveness PC used by
        // get_list_of_active_boxes.  For after-residual-call guards the
        // resume target depends on whether the residual call sits inside
        // a try-block: only then did the jitcode emit a post-call
        // `-live-`/`catch_exception` (`after_residual_call_resume_pc` has
        // an entry keyed by the CALL pc).
        //
        //   * try-block call: resume at the call's OWN catch.  Store the
        //     CALL pc (`saved_orgpc`) with the marker bit folded in so the
        //     decoder routes it through `after_residual_call_resume_pc`
        //     rather than `pc_map` (which would re-execute the call).
        //   * non-try call: no catch to land on, so keep the next-opcode
        //     resume (`fallthrough_pc`) — the exception then propagates
        //     out of the frame via `handle_exception_in_frame`, exactly
        //     as before this fix.
        //
        // RPython always keeps `frame.pc` at the post-call `-live-`
        // (`pyjitpl.py:2610-2624`) and lets `handle_exception_in_frame`
        // decide catch-vs-propagate; pyre only emits that marker for
        // try-block calls, so the non-try case falls back here.
        let snapshot_live_pc = self.marker_aware_resume_pc(saved_orgpc, after_residual_call);

        // pyjitpl.py:2597-2600: history.trace.capture_resumedata(
        //     self.framestack, virtualizable_boxes, self.virtualref_boxes,
        //     after_residual_call)
        let snapshot = self.build_framestack_snapshot(
            ctx,
            snapshot_live_pc,
            active_boxes,
            snapshot_full_types,
        );
        let snapshot_id = ctx.capture_resumedata(snapshot);
        ctx.set_last_guard_resume_position(snapshot_id);

        // pyjitpl.py:2602: frame.pc = saved_pc (restore)
        self.orgpc = saved_orgpc;
        let s = self.sym_mut();
        s.vable_last_instr = saved_ni;
        s.vable_valuestackdepth = saved_vsd;
    }

    /// Extend a single-frame callee's `fail_arg_types` with the type
    /// contribution of every active parent frame.  Mirrors the
    /// `pyjitpl.py:2597` generate_guard collection: parent boxes ride
    /// the snapshot (built by `build_framestack_snapshot`), and the
    /// optimizer's `store_final_boxes_in_guard` (`optimizeopt/mod.rs:
    /// 3200`) overwrites `op.fail_args` from the snapshot via
    /// `op.store_final_boxes(liveboxes)` (mod.rs:3392), so an inline
    /// `fail_args` collection is redundant — only the per-frame
    /// `parent_types` need to flow into `record_guard_typed`.
    ///
    /// `build_framestack_snapshot` strips the `[s.frame, s.vable_*..]`
    /// scalar inputarg header from each parent frame's snapshot boxes
    /// (`opencoder.py:806` — parent frames contribute only active
    /// boxes; the virtualizable section is emitted once for the whole
    /// snapshot via `list_of_boxes_virtualizable`).  Mirror that here
    /// so the static type vector matches the static box shape per
    /// frame, otherwise the inline `fail_arg_types` stretched parent
    /// types over the snapshot's header-less box positions before the
    /// optimizer's snapshot-derived overwrite hid the divergence.
    fn extend_types_with_parents(&mut self, ctx: &mut TraceCtx, mut types: Vec<Type>) -> Vec<Type> {
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        for parent in self.parent_frames.clone() {
            let (parent_types_full, _jitcode_index) =
                self.materialize_parent_frame_state(ctx, parent);
            if parent_types_full.len() > n {
                types.extend_from_slice(&parent_types_full[n..]);
            }
        }
        types
    }

    /// Append `self.parent_frames` onto an innermost-first `lead` frame
    /// list and reverse the whole vector to the outermost-first order
    /// required by `recorder.rs:54` ("Frames in the snapshot, outermost
    /// first") and `resume.rs:252`.  Extracted so both the ordinary
    /// `build_framestack_snapshot` (lead = `[top]`) and the issue #143
    /// synthetic-callee path (lead = `[helper, self-as-parent]`) share one
    /// parent-collection + reverse seam.
    fn build_framestack_frames(
        &mut self,
        ctx: &mut TraceCtx,
        mut lead: Vec<majit_metainterp::recorder::SnapshotFrame>,
    ) -> Vec<majit_metainterp::recorder::SnapshotFrame> {
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        // opencoder.py:806: parent frames keep their original pc.
        // Snapshot boxes = active boxes only (skip scalar inputarg header).
        for parent in self.parent_frames.clone() {
            let (parent_types_full, parent_jitcode_index, parent_active) =
                self.materialize_parent_snapshot_state(ctx, parent);
            let parent_types: &[Type] = if parent_types_full.len() > n {
                &parent_types_full[n..]
            } else {
                &[]
            };
            // A parent whose CALL is in a try-block resumes at that call's
            // own catch (bit-14 marker on the CALL pc); otherwise the raw
            // return_point_pc, which must leave bit 14 free or decode would
            // mis-read it as marked (resumedata.rs:48-62).  The marker gate
            // matches the catch-vs-fallthrough liveness chosen for
            // `parent_active` in `materialize_parent_snapshot_state`.
            let parent_pc = Self::marker_aware_parent_resume_pc(
                parent_jitcode_index,
                parent.call_pc,
                parent.resume_pc,
            );
            lead.push(majit_metainterp::recorder::SnapshotFrame {
                jitcode_index: parent_jitcode_index,
                pc: parent_pc as u32,
                boxes: Self::fail_args_to_snapshot_boxes_typed(&parent_active, parent_types, ctx),
            });
        }
        // opencoder.py:217 `SnapshotIterator.__init__` calls
        // `self.framestack.reverse()` so the numbering loop at
        // `resume.py:249-253` iterates outermost-first.  pyre builds the
        // frame list innermost-first (`[top, immediate_caller, ...,
        // outermost]`); reverse so `Snapshot.frames[0]` is outermost,
        // matching `recorder.rs:54` / `resume.rs:252`.
        lead.reverse();
        lead
    }

    /// Build the full framestack `Snapshot` — top (callee) frame
    /// followed by every parent frame in `self.parent_frames` — plus
    /// virtualizable and virtualref boxes.  Mirrors opencoder.py:819-832
    /// `capture_resumedata(framestack, virtualizable_boxes,
    /// virtualref_boxes, ...)`.
    ///
    /// The caller is responsible for swapping `self.orgpc` if the
    /// snapshot pc differs from the current orgpc (RPython
    /// pyjitpl.py:2594-2602 MIFrame.capture_resumedata does the same
    /// before calling `history.trace.capture_resumedata`), and for
    /// computing `top_active_boxes` / `top_snapshot_types_full` under
    /// the liveness that applies to the swapped pc.
    fn build_framestack_snapshot(
        &mut self,
        ctx: &mut TraceCtx,
        top_pc: usize,
        top_active_boxes: &[OpRef],
        top_snapshot_types_full: &[Type],
    ) -> majit_metainterp::recorder::Snapshot {
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let top_snapshot_types = &top_snapshot_types_full[n..];
        let top_jitcode_index = unsafe { (*self.sym().jitcode).index } as u32;
        let top_frame = majit_metainterp::recorder::SnapshotFrame {
            jitcode_index: top_jitcode_index,
            pc: top_pc as u32,
            boxes: Self::fail_args_to_snapshot_boxes_typed(
                top_active_boxes,
                top_snapshot_types,
                ctx,
            ),
        };
        let frames = self.build_framestack_frames(ctx, vec![top_frame]);
        let vable_boxes = self.list_of_boxes_virtualizable(ctx);
        let vref_boxes = Self::build_virtualref_boxes(self.sym(), ctx);
        // PHASE 1.4 candidate D probe: detect snapshot-time divergence
        // between vable_boxes (heap mirror) and registers_r (machine
        // register source). Both should be populated by store_local_value's
        // dual-write. Any divergence here means a code path updated one
        // shadow without the other — most likely load_local_value's lazy
        // fallback (trace_opcode.rs:1041-1063) which writes registers_r
        // but does NOT call set_virtualizable_box_at. See
        // memory/phase_1_4_cand_a_landed_raise_catch_diagnostic_2026_04_26.md.
        if std::env::var("PYRE_PROBE_SNAPSHOT").ok().as_deref() == Some("1") {
            let num_static = ctx
                .virtualizable_info()
                .map(|info| info.num_static_extra_boxes)
                .unwrap_or(0);
            let nlocals = self.sym().nlocals;
            let registers_r_src: Vec<OpRef> = if let Some(ref pre_r) = self.pre_opcode_registers_r {
                pre_r[..pre_r.len().min(nlocals)].to_vec()
            } else {
                let s = self.sym();
                s.registers_r[..s.registers_r.len().min(nlocals)].to_vec()
            };
            let mut diverge = 0usize;
            for i in 0..registers_r_src.len() {
                let reg_op = registers_r_src[i];
                let vable_op = ctx
                    .virtualizable_box_at(num_static + i)
                    .unwrap_or(OpRef::NONE);
                if !reg_op.is_none() && reg_op != vable_op {
                    eprintln!(
                        "[PROBE-D] vable/reg divergence top_pc={} local={} reg_opref={:?} vable_opref={:?}",
                        top_pc, i, reg_op, vable_op
                    );
                    diverge += 1;
                }
            }
            eprintln!(
                "[PROBE-D] ENTER top_pc={} nlocals={} reg_len={} num_static={} diverge_count={}",
                top_pc,
                nlocals,
                registers_r_src.len(),
                num_static,
                diverge
            );
        }
        majit_metainterp::recorder::Snapshot {
            frames,
            vable_boxes,
            vref_boxes,
        }
    }

    /// Issue #143 core: synthesize a framestack whose innermost frame is a
    /// Rust runtime-helper jitcode (the folded `list.append`/`pop` body),
    /// not a Python frame.  Because `PyreMetaInterp::interpret`
    /// (metainterp.rs:88) can only decode a `W_CodeObject` jitcode, the
    /// helper callee can never be *traced*; instead the resize `GuardTrue`
    /// captures a snapshot in which the helper is the top/callee frame and
    /// the current Python frame is demoted to its immediate caller.  On
    /// guard failure the blackhole resumes into the helper at `helper_pc`
    /// (an identity-`pc_map` byte offset registered by
    /// `register_runtime_helper_jitcode`, M-core1), runs the generic
    /// append+realloc, returns its result into the caller's pending slot,
    /// and the caller resumes at the post-CALL `fallthrough_pc` — instead
    /// of re-executing the method-shape `CALL` and rebuilding a NULL
    /// callable.
    ///
    /// `boxes` is the hand-built `[list, value]` carried into the helper.
    /// It has NO scalar-inputarg header (the helper has no virtualizable
    /// Python frame), so its types are computed 1:1 and passed UNSLICED.
    /// `result_stack_idx` / `result_type` are the caller's pending call
    /// result slot (relative to `nlocals`) and kind; they MUST be `Some`
    /// in production so the demoted caller's `get_list_of_active_boxes`
    /// (`in_a_call`, trace_opcode.rs:1538) nulls the undefined result slot
    /// rather than capturing a stale box.  Returns frames outermost-first
    /// (`[...outer parents, self, helper]`).
    #[allow(dead_code)]
    fn build_synthetic_callee_frames(
        &mut self,
        ctx: &mut TraceCtx,
        helper_jitcode_index: u32,
        helper_pc: usize,
        boxes: &[OpRef],
        result_stack_idx: Option<usize>,
        result_type: Option<Type>,
    ) -> Vec<majit_metainterp::recorder::SnapshotFrame> {
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;

        // Synthetic Rust-helper callee (new innermost/top frame). Its boxes
        // are hand-built with NO 8-entry scalar-inputarg header, so
        // `helper_types` matches `boxes` 1:1 and is passed UNSLICED. Each
        // box keeps its intrinsic kind via `value_type` (list -> Ref, an
        // unboxed int value -> Int) so resume `_number_boxes` tags it
        // correctly.
        let helper_types: Vec<Type> = boxes.iter().map(|&op| self.value_type(op)).collect();
        let helper_frame = majit_metainterp::recorder::SnapshotFrame {
            jitcode_index: helper_jitcode_index,
            pc: helper_pc as u32,
            boxes: Self::fail_args_to_snapshot_boxes_typed(boxes, &helper_types, ctx),
        };

        // Demote `self` (the Python frame doing the append) to the helper's
        // immediate caller, resuming at the post-CALL fallthrough. Routing
        // self through the SAME parent path as real parents is borrow-safe:
        // `materialize_parent_snapshot_state` reads nothing from `self`
        // (trace_opcode.rs:3925) — it rebuilds a fresh MIFrame from the raw
        // `sym` pointer — so at most one `&mut PyreSym` is ever live, even
        // when `parent.sym == self.sym`. `self.sym` / `concrete_frame_addr`
        // / `fallthrough_pc` are Copy and read into the struct literal
        // before the `&mut self` call. The pending result slot is carried
        // so the caller's undefined call result is nulled at snapshot time.
        let self_as_parent = ResumeFrameState {
            sym: self.sym,
            concrete_frame_addr: self.concrete_frame_addr,
            resume_pc: self.fallthrough_pc,
            pending_result_stack_idx: result_stack_idx,
            pending_result_type: result_type,
            // Helper caller resumes at the post-CALL fallthrough, not a
            // try-block catch, so it carries no marker call pc.
            call_pc: None,
        };
        let (self_types_full, self_jitcode_index, self_active) =
            self.materialize_parent_snapshot_state(ctx, self_as_parent);
        let self_types: &[Type] = if self_types_full.len() > n {
            &self_types_full[n..]
        } else {
            &[]
        };
        let self_frame = majit_metainterp::recorder::SnapshotFrame {
            jitcode_index: self_jitcode_index,
            pc: self.fallthrough_pc as u32,
            boxes: Self::fail_args_to_snapshot_boxes_typed(&self_active, self_types, ctx),
        };

        // Innermost-first lead = [helper(top), self(first parent)]; append
        // self.parent_frames and reverse -> outermost-first
        // [...outer parents, self, helper] (synthetic callee = frames[len-1]).
        self.build_framestack_frames(ctx, vec![helper_frame, self_frame])
    }

    /// Full M-core2 `Snapshot` for the issue #143 resize guard: the
    /// synthetic-callee framestack from `build_synthetic_callee_frames`
    /// plus the one-shot virtualizable + virtualref boxes, both sourced
    /// from the REAL Python frame's `sym` (the helper has none) exactly as
    /// `build_framestack_snapshot` does.
    ///
    /// Like `build_framestack_snapshot`, this does NOT swap `self.orgpc`
    /// or save/restore the vable scalars; the M-core3 caller must do that
    /// around this call (mirroring `capture_resumedata`,
    /// trace_opcode.rs:3742-3771) since `list_of_boxes_virtualizable`
    /// derives the vable `last_instr`/`valuestackdepth` from `self.orgpc`.
    #[allow(dead_code)]
    fn build_framestack_snapshot_with_synthetic_callee(
        &mut self,
        ctx: &mut TraceCtx,
        helper_jitcode_index: u32,
        helper_pc: usize,
        boxes: &[OpRef],
        result_stack_idx: Option<usize>,
        result_type: Option<Type>,
    ) -> majit_metainterp::recorder::Snapshot {
        let frames = self.build_synthetic_callee_frames(
            ctx,
            helper_jitcode_index,
            helper_pc,
            boxes,
            result_stack_idx,
            result_type,
        );
        let vable_boxes = self.list_of_boxes_virtualizable(ctx);
        let vref_boxes = Self::build_virtualref_boxes(self.sym(), ctx);
        majit_metainterp::recorder::Snapshot {
            frames,
            vable_boxes,
            vref_boxes,
        }
    }

    /// Issue #143 M-core3: emit the `list.append`/`list.pop` resize
    /// `GuardTrue` whose resume captures a synthetic-callee framestack —
    /// the runtime-helper jitcode (folded `jit_list_append`) as the
    /// innermost/top frame, the current Python frame demoted to its
    /// immediate caller resuming at the post-CALL `fallthrough_pc`.
    ///
    /// On guard failure the blackhole runs the helper (a residual
    /// `jit_list_append(list, value)` that performs the generic
    /// append+realloc and returns `w_none()`), threads the result into the
    /// caller's pending slot via `setup_return_value_r`, and the Python
    /// frame continues *after* the method-shape `CALL` — never
    /// re-executing it and rebuilding a NULL callable (the issue #143 bug).
    ///
    /// Mirrors `generate_guard` / `generate_guard_core` (const-skip →
    /// flush_guard_not_invalidated → record_guard_typed → capture) but
    /// swaps the snapshot builder for
    /// `build_framestack_snapshot_with_synthetic_callee` and the resume pc
    /// for `fallthrough_pc`.  The orgpc swap + vable scalar save/restore
    /// around the build mirrors `capture_resumedata` (trace_opcode.rs:
    /// 3742-3771): `list_of_boxes_virtualizable` re-derives
    /// `last_instr`/`valuestackdepth` from `self.orgpc`
    /// (trace_opcode.rs:4256-4264), so it must read `fallthrough_pc` to
    /// encode the demoted caller's post-CALL state.
    ///
    /// `list` / `value` are the helper boxes (`[list, value]`).  The gate
    /// at the call site (`guard_append_without_resize`) routes here only
    /// when `value` is a `Ref` (a real `PyObjectRef`), matching the
    /// helper's `live live_r=[0,1]` declaration — an unboxed `Int` value
    /// would mis-number on resume.
    pub(crate) fn generate_guard_with_synthetic_callee(
        &mut self,
        ctx: &mut TraceCtx,
        opcode: OpCode,
        args: &[OpRef],
        list: OpRef,
        value: OpRef,
    ) {
        // pyjitpl.py:2558-2560: a const guard operand needs no guard.
        if let Some(&first) = args.first() {
            if first.is_constant() {
                return;
            }
        }
        // pyjitpl.py:1087: flush a pending guard_not_invalidated first.
        if opcode != OpCode::GuardNotInvalidated {
            self.flush_guard_not_invalidated(ctx);
        }

        let helper_index = crate::state::ensure_list_append_resize_helper_index() as u32;
        // helper resumes at byte 0: `live` → `residual_call jit_list_append`
        // → `ref_return w_none()` (identity pc_map, M-core1).
        let helper_pc = 0usize;
        let boxes = [list, value];
        let result_type = Type::Ref;

        // The CALL handler already popped the callable + args, so the
        // append result (None) lands at the current top of self's value
        // stack: `result_stack_idx` is that slot relative to nlocals. The
        // demoted-caller snapshot nulls it (get_list_of_active_boxes
        // in_a_call, trace_opcode.rs:1538) so no stale box is captured
        // where `setup_return_value_r` will write the helper's return.
        let result_stack_idx = {
            let s = self.sym();
            s.valuestackdepth.saturating_sub(s.nlocals)
        };

        // fail_arg_types in snapshot box order [helper(1:1, no header),
        // self(sliced), ...parents(sliced)] — mirrors generate_guard's
        // extend_types_with_parents.  These are an overwritten fallback:
        // store_final_boxes_in_guard (optimizeopt/mod.rs:3200) repopulates
        // op.fail_args from the snapshot below.
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let helper_types: Vec<Type> = boxes.iter().map(|&op| self.value_type(op)).collect();
        let mut types = helper_types;
        let self_as_parent = crate::state::ResumeFrameState {
            sym: self.sym,
            concrete_frame_addr: self.concrete_frame_addr,
            resume_pc: self.fallthrough_pc,
            pending_result_stack_idx: Some(result_stack_idx),
            pending_result_type: Some(result_type),
            // Helper caller resumes at the post-CALL fallthrough, not a
            // try-block catch, so it carries no marker call pc.
            call_pc: None,
        };
        let (self_types_full, _) = self.materialize_parent_frame_state(ctx, self_as_parent);
        if self_types_full.len() > n {
            types.extend_from_slice(&self_types_full[n..]);
        }
        let types = self.extend_types_with_parents(ctx, types);
        ctx.record_guard_typed(opcode, args, types);

        // capture_resumedata parity (trace_opcode.rs:3742-3771): swap orgpc
        // to the post-CALL fallthrough so the demoted caller's vable
        // scalars encode the resume state, then restore.
        let saved_orgpc = self.orgpc;
        let saved_ni = self.sym().vable_last_instr;
        let saved_vsd = self.sym().vable_valuestackdepth;
        self.orgpc = self.fallthrough_pc;

        let snapshot = self.build_framestack_snapshot_with_synthetic_callee(
            ctx,
            helper_index,
            helper_pc,
            &boxes,
            Some(result_stack_idx),
            Some(result_type),
        );
        let snapshot_id = ctx.capture_resumedata(snapshot);
        ctx.set_last_guard_resume_position(snapshot_id);

        self.orgpc = saved_orgpc;
        let s = self.sym_mut();
        s.vable_last_instr = saved_ni;
        s.vable_valuestackdepth = saved_vsd;

        // pyjitpl.py:2581 profiler.count_ops parity.
        ctx.profiler()
            .count_ops(opcode, majit_metainterp::counters::GUARDS);
    }

    fn materialize_parent_frame_state(
        &mut self,
        ctx: &mut TraceCtx,
        parent: ResumeFrameState,
    ) -> (Vec<Type>, u32) {
        let (full_types, jitcode_index, _active_boxes) =
            self.materialize_parent_snapshot_state(ctx, parent);
        (full_types, jitcode_index)
    }

    fn materialize_parent_snapshot_state(
        &mut self,
        ctx: &mut TraceCtx,
        parent: ResumeFrameState,
    ) -> (Vec<Type>, u32, Vec<OpRef>) {
        // pyjitpl.py:2586 capture_resumedata parity: parent frames
        // contribute only their per-frame regular boxes (locals + stack)
        // to the snapshot.  The virtualizable scalars are emitted once
        // via `list_of_boxes_virtualizable` for the whole snapshot, not
        // per-frame.  Earlier pyre revisions called
        // `parent_frame.flush_to_frame_for_guard(ctx)` here and built a
        // `[s.frame, s.vable_*..., active_boxes]` fail_args vec for the
        // legacy `record_guard_typed_with_fail_args` path; both are dead
        // since the snapshot reader (`build_framestack_snapshot`) reads
        // active boxes directly and `store_final_boxes_in_guard`
        // overwrites `op.fail_args` from the snapshot
        // (`optimizeopt/mod.rs:3200`).  Keeping the parent flush around
        // mutates parent's `s.vable_*` mid-callee snapshot — the
        // structural blocker called out in
        // `vable_shadow_split_brain_load_bearing_2026_04_28` — so it is
        // dropped together with the dead fail_args build.
        let parent_sym = unsafe { &mut *parent.sym };
        let mut parent_frame = MIFrame::from_sym(
            ctx,
            parent_sym,
            parent.concrete_frame_addr,
            parent.resume_pc,
            parent.resume_pc,
        );
        parent_frame.pending_result_stack_idx = parent.pending_result_stack_idx;
        parent_frame.pending_result_type = parent.pending_result_type;
        // When the parent's CALL sits in a try-block, read this frame's
        // liveness at the call's post-residual-call `-live-`/catch (mirroring
        // `pyjitpl.py:194-195 pc=self.pc`) so the encoded box count matches
        // the blackhole's marker-routed resume position (`build_framestack_
        // snapshot` folds the same marker into this parent's snapshot pc).
        parent_frame.residual_call_pc = parent.call_pc;
        let active_boxes = parent_frame.get_list_of_active_boxes(ctx, true, false, None);
        let full_types = parent_frame.build_fail_arg_types_for_active_boxes(&active_boxes);
        let jitcode_index = unsafe { (*parent_frame.sym().jitcode).index } as u32;
        (full_types, jitcode_index, active_boxes)
    }

    /// virtualizable.py:139 _get_virtualizable_field_boxes parity:
    /// [static_fields..., array_items..., virtualizable_ptr].
    /// pyjitpl.py:2586: self.virtualizable_boxes → vable_array.
    /// opencoder.py:603 _encode parity: encode OpRef as SnapshotTagged.
    /// Constant-pool OpRefs → Const(value, type) from pool.
    /// NONE → Const(0, Ref). Regular → Box.
    fn opref_to_snapshot_tagged(
        opref: OpRef,
        ctx: &majit_metainterp::TraceCtx,
    ) -> majit_metainterp::recorder::SnapshotTagged {
        Self::opref_to_snapshot_tagged_for_slot(opref, ctx, None)
    }

    /// virtualizable.py:86-98 `read_boxes(cpu, virtualizable, startindex)` parity:
    /// each slot is wrapped via `wrap(cpu, value, startindex + i)` where the
    /// lltype (ARRAYITEMTYPE or static field `FIELDTYPE`) is declared and
    /// determines the resulting Const's INT/REF/FLOAT kind. pyre stores
    /// constants in a unified pool whose stored `const_type` may disagree
    /// with the slot's declared type (e.g. pointer constants opened via
    /// `const_int`), so snapshot encoding must prefer the slot's declared
    /// type when it is known — otherwise `_gettagged` → `getconst(val, tp)`
    /// picks TAGINT for a Ref-typed slot and the resume reader decodes a
    /// raw i64 where a PyObjectRef is expected.
    fn opref_to_snapshot_tagged_for_slot(
        opref: OpRef,
        ctx: &majit_metainterp::TraceCtx,
        declared_type: Option<majit_ir::Type>,
    ) -> majit_metainterp::recorder::SnapshotTagged {
        if opref.is_none() {
            majit_metainterp::recorder::SnapshotTagged::Const(
                0,
                declared_type.unwrap_or(majit_ir::Type::Ref),
            )
        } else if ctx.constant_value(opref).is_some() {
            let val = ctx.constant_value(opref).unwrap_or(0);
            // resume.py:157-183 `getconst(const)` dispatches on `const.type`.
            // Prefer the pool's actual const type over `declared_type`:
            // Box.type is immutable, so an Int-typed constant (e.g. an
            // intbounds-promoted local) must stay Int even when the slot
            // layout declares Ref. Retyping it here would seed the bridge
            // optimizer's const_pool with `Value::Ref(GcRef(small_int))`
            // and later trip the getintbound forwarding assertion. Fall
            // back to `declared_type` only when the pool has no type for
            // this OpRef (e.g. raw-pointer constants seeded without a
            // const_type entry).
            let tp = ctx
                .const_type(opref)
                .or(declared_type)
                .unwrap_or(majit_ir::Type::Int);
            majit_metainterp::recorder::SnapshotTagged::Const(val, tp)
        } else {
            // resume.py:211,214: box.type lives on the Box itself; the
            // typed `OpRef` carries the matching variant tag and the
            // explicit `tp` is the lockstep authority for any
            // transitional `Untyped` opref (resoperation.py:719/727/739).
            let tp = ctx
                .get_opref_type(opref)
                .unwrap_or_else(|| panic!("missing snapshot box type for {:?}", opref));
            majit_metainterp::recorder::SnapshotTagged::Box(opref, tp)
        }
    }

    /// RPython pyjitpl.py:2586 virtualizable_boxes parity.
    ///
    /// RPython creates SEPARATE Box objects for virtualizable_boxes via
    /// read_boxes()/wrap() — these are distinct from frame register boxes.
    /// _number_boxes dedup uses object identity, so vable and frame get
    /// independent TAGBOX indices → deadframe stores both.
    ///
    /// pyre uses the SAME OpRefs for both → _number_boxes dedup merges them
    /// → vable and frame sections share TAGBOX indices. Recovery uses frame
    /// sections with liveness-based mapping (restore_guard_failure_values),
    /// matching RPython's consume_boxes(position_info) architecture.
    ///
    /// Fresh identity approaches (VABLE_FRESH_BIT, VABLE_KEY_OFFSET)
    /// expand num_boxes → larger fail_args → deadframe/exit layout mismatch.
    /// Fix requires backend exit block recompilation after numbering,
    /// or trace-time SameAs emission for fresh vable OpRefs.
    fn list_of_boxes_virtualizable(
        &self,
        ctx: &mut majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        let sym = self.sym();
        // opencoder.py:718-726 _list_of_boxes_virtualizable parity:
        // RPython format: [virtualizable_ptr, static_fields..., array_items...]
        // (virtualizable_ptr moved from end to front).
        // virtualizable.py:86/139 read_boxes / load_list_of_boxes:
        // Memory order: [static_field_0, ..., array_items..., vable_ptr]
        // read_boxes creates fresh Box objects for each field via wrap().
        // opencoder.py:722 _list_of_boxes_virtualizable: reorders
        //   vable_ptr from end to front → snapshot = [vable_ptr, fields..., items...]
        let stack_only = sym.valuestackdepth.saturating_sub(sym.nlocals);
        let mut boxes = Vec::new();
        // opencoder.py:722: virtualizable_ptr FIRST.
        // The virtualizable frame pointer is always a GCREF.
        boxes.push(Self::opref_to_snapshot_tagged_for_slot(
            sym.frame,
            ctx,
            Some(majit_ir::Type::Ref),
        ));
        // Static fields in declared order (virtualizable.py:90-93).
        // virtualizable.py:131-133 wraps each value with its declared
        // `FIELDTYPE`; pyre mirrors that by consulting
        // `VirtualizableInfo::static_fields[i].field_type`.
        //
        // opencoder.py:718-726 `_list_of_boxes_virtualizable(boxes)`
        // parity: read from `ctx.virtualizable_boxes` (the canonical
        // analog of RPython's `metainterp.virtualizable_boxes`) for
        // the four invariant scalars (`pycode`, `debugdata`,
        // `lastblock`, `w_globals`), and recompute the two
        // per-opcode-advancing scalars (`last_instr`, `valuestackdepth`)
        // from `self.orgpc` / `pre_opcode_registers_r` /
        // `portal_bridge_vable_vsd(orgpc)` so the snapshot encodes
        // the pre-opcode state at `resume_pc` (the PROBE-VABLE-DIV
        // diagnostic confirmed slot 0 / slot 2 are the
        // only divergence sources between the shared shadow and
        // `s.vable_*` — slots 1/3/4/5 always agree because their
        // mutators are unreachable under CPython 3.14 bytecode).
        //
        // The slot-0 inline override re-derives `resume_pc - 1`
        // because `flush_to_frame_for_guard` swaps `self.orgpc` to
        // `resume_pc` (capture_resumedata at line 2773), and writes
        // `s.vable_last_instr = const_int(resume_pc - 1)` without
        // mirroring to `ctx.virtualizable_boxes[0]`.  Reading
        // `ctx.virtualizable_boxes[0]` directly would pick up the
        // value `publish_last_instr_to_vable` wrote at the original
        // (pre-swap) orgpc, which is one off when `resume_pc !=
        // orgpc` (most branch guards).  The read-time recompute
        // matches `flush_to_frame_for_guard` so the snapshot stays
        // self-consistent.
        //
        // The slot-2 inline override re-derives the pre-opcode
        // valuestackdepth via `portal_bridge_vable_vsd(resume_pc)
        // .unwrap_or(pre_opcode_semantic_depth / s.valuestackdepth)`
        // — same source `flush_to_frame_for_guard` uses to set
        // `s.vable_valuestackdepth`.
        //
        // Test-fixture fallback: `TraceCtx::for_test_types` callers
        // construct a ctx without registering `VirtualizableInfo` and
        // without seeding `ctx.virtualizable_boxes`.  In that mode
        // the `_with_compiled_trace_jitcode` fixtures expect
        // `sym.vable_field_oprefs()` to drive the snapshot — fall
        // back when the shared shadow is unseeded.
        let (vable_static_types, vsd_field_index, ni_field_index, num_static): (
            Vec<majit_ir::Type>,
            Option<usize>,
            Option<usize>,
            usize,
        ) = match ctx.virtualizable_info() {
            Some(info) => (
                info.static_fields.iter().map(|f| f.field_type).collect(),
                info.static_field_index_by_name("valuestackdepth"),
                info.static_field_index_by_name("last_instr"),
                info.num_static_extra_boxes,
            ),
            None => (Vec::new(), None, None, 0),
        };
        let pre_opcode_vsd: Option<i64> = if vsd_field_index.is_some() {
            let resume_pc = self.orgpc;
            Some(
                self.portal_bridge_vable_vsd(resume_pc)
                    .unwrap_or_else(|| self.pre_opcode_concrete_depth() as i64),
            )
        } else {
            None
        };
        let pre_opcode_last_instr: Option<i64> = ni_field_index.map(|_| self.orgpc as i64 - 1);
        if num_static > 0 && ctx.has_virtualizable_boxes() {
            for idx in 0..num_static {
                let declared = vable_static_types.get(idx).copied();
                let opref = if Some(idx) == ni_field_index {
                    let li = pre_opcode_last_instr
                        .expect("pre_opcode_last_instr seeded when ni_field_index is Some");
                    ctx.const_int(li)
                } else if Some(idx) == vsd_field_index {
                    let vsd =
                        pre_opcode_vsd.expect("pre_opcode_vsd seeded when vsd_field_index is Some");
                    ctx.const_int(vsd)
                } else {
                    ctx.virtualizable_box_at(idx).unwrap_or(OpRef::NONE)
                };
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    opref, ctx, declared,
                ));
            }
        } else {
            // Test fallback: ctx has no vinfo and no seeded shadow.
            let fallback_types: Vec<majit_ir::Type> = if !vable_static_types.is_empty() {
                vable_static_types.clone()
            } else {
                vec![majit_ir::Type::Ref; sym.vable_field_oprefs().len()]
            };
            for (idx, opref) in sym.vable_field_oprefs().iter().enumerate() {
                let declared = fallback_types.get(idx).copied();
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    *opref, ctx, declared,
                ));
            }
        }
        // Array items: locals + stack (virtualizable.py:86 read_boxes).
        let _ = stack_only;
        let symbolic_stack_len = if self.pre_opcode_registers_r.is_some() {
            self.pre_opcode_concrete_depth().saturating_sub(sym.nlocals)
        } else {
            sym.registers_r.len().saturating_sub(sym.nlocals)
        };
        let concrete_frame_ptr = if !sym.concrete_vable_ptr.is_null() {
            sym.concrete_vable_ptr as usize
        } else {
            self.concrete_frame_addr
        };
        let concrete_frame = if concrete_frame_ptr != 0 {
            Some(unsafe { &*(concrete_frame_ptr as *const pyre_interpreter::pyframe::PyFrame) })
        } else {
            None
        };
        // virtualizable.py:86 read_boxes parity: encoder must emit one
        // box per slot in the heap-side `locals_cells_stack_w` array
        // because the decoder reads `vinfo.get_total_size(virtualizable)`
        // (= static_fields + heap array length) on the runtime PyFrame.
        // Using the symbolic current stack depth here was off by
        // (max_stackdepth - current_stack_depth) and produced
        // `vable_size - 1 != vinfo.get_total_size` panics whenever a
        // bridge tried to consume the snapshot at a state where the
        // physical frame had been allocated with stack room beyond the
        // current symbolic depth. Read the physical frame length and
        // pad missing slots with the live concrete value (or NULL).
        let physical_array_len = ctx
            .virtualizable_array_lengths()
            .and_then(|lengths| lengths.first().copied())
            .or_else(|| concrete_frame.map(|f| f.locals_w().len()))
            .unwrap_or_else(|| {
                if !sym.jitcode.is_null() {
                    let code = unsafe { &*(*sym.jitcode).raw_code() };
                    code.varnames.len()
                        + pyre_interpreter::pyframe::ncells(code)
                        + code.max_stackdepth as usize
                } else {
                    let current_vsd = self.pre_opcode_concrete_depth();
                    let stack_depth = current_vsd
                        .saturating_sub(sym.nlocals)
                        .min(symbolic_stack_len);
                    sym.nlocals + stack_depth
                }
            });
        let full_array_len = physical_array_len;
        // virtualizable.py:135-137 `lst[j] = reader.load_next_value_of_type(
        // ARRAYITEMTYPE)` — every array slot is the array's declared item
        // type (GCREF for pyre's `locals_cells_stack_w`), regardless of
        // what the optimizer chose for the OpRef's own kind. Enforce this
        // at encoding time so a `LOAD_CONST 0` whose OpRef is Int-typed
        // still lands in the snapshot as a Ref constant.
        let array_item_type = ctx
            .virtualizable_info()
            .and_then(|info| info.array_fields.first().map(|a| a.item_type))
            .unwrap_or(majit_ir::Type::Ref);
        // virtualizable.py:86 read_boxes / opencoder.py:718 _list_of_boxes_virtualizable
        // parity: RPython snapshot reads directly from `self.virtualizable_boxes`,
        // which is the single source of truth mirrored by every
        // `_opimpl_setarrayitem_vable` via `synchronize_virtualizable`.
        // pyre's tracer mirrors EVERY write into `locals_cells_stack_w`
        // (locals via `store_local_value`, stack via `push_typed_value`
        // + `pop_value` + `swap_values` + `finishframe_exception`) into
        // `virtualizable_boxes`, so the shadow is the only source we
        // read here.
        let num_static = ctx
            .virtualizable_info()
            .map(|info| info.num_static_extra_boxes)
            .unwrap_or(0);
        for i in 0..full_array_len {
            // virtualizable_boxes layout:
            //   [field0, ..., fieldN, arr[0..M], vable_ref]
            // so array slot `i` lives at `num_static + i`. The trailing
            // `vable_ref` is at `virtualizable_boxes[-1]` and NEVER covers
            // an array slot — skip it via `.get()`.
            let opref = ctx
                .virtualizable_box_at(num_static + i)
                .unwrap_or(OpRef::NONE);
            if !opref.is_none() {
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    opref,
                    ctx,
                    Some(array_item_type),
                ));
            } else {
                // TODO: legacy `register == PyFrame
                // slot` conflation lets STORE_FAST
                // write an unboxed int into `locals_cells_stack_w[i]`
                // when the trace IR optimizer promoted that local's
                // OpRef to Int. Reading those raw bits back here and
                // const-seeding them as `Const(val, Ref)` mistypes
                // the value in the bridge optimizer's const_pool —
                // verified root cause of the LoadFastLoadFast vable
                // conversion regression.
                //
                // Emit a NULL sentinel; the bridge resume will
                // re-fetch the actual slot value via
                // `vable_getarrayitem_r` against the live frame.
                let _ = concrete_frame;
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    OpRef::NONE,
                    ctx,
                    Some(array_item_type),
                ));
            }
        }
        boxes
    }

    /// pyjitpl.py:2597 virtualref_boxes parity.
    /// pyjitpl.py:2597 virtualref_boxes parity.
    /// Returns pairs of (jit_virtual, real_vref) as SnapshotTagged.
    fn build_virtualref_boxes(
        sym: &PyreSym,
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        sym.virtualref_boxes
            .iter()
            .map(|&(opref, _concrete)| Self::opref_to_snapshot_tagged(opref, ctx))
            .collect()
    }

    /// RPython pyjitpl.py:177 get_list_of_active_boxes parity:
    #[allow(dead_code)]
    fn fail_args_to_snapshot_boxes(
        fail_args: &[OpRef],
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        fail_args
            .iter()
            .map(|&opref| Self::opref_to_snapshot_tagged(opref, ctx))
            .collect()
    }

    /// snapshot boxes from active_boxes = [locals, stack].
    /// RPython: each Box carries type ('r'/'i'/'f') — pyre passes types
    /// explicitly so _number_boxes can detect virtual vs int correctly.
    fn fail_args_to_snapshot_boxes_typed(
        active_boxes: &[OpRef],
        types: &[majit_ir::Type],
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        active_boxes
            .iter()
            .enumerate()
            .map(|(i, &opref)| {
                if opref.is_none() {
                    majit_metainterp::recorder::SnapshotTagged::Const(0, majit_ir::Type::Ref)
                } else if ctx.constant_value(opref).is_some() {
                    let val = ctx.constant_value(opref).unwrap_or(0);
                    // resume.py:157-183 `getconst(const)` dispatches on
                    // `const.type`; pyre's plain `const_int(v)` has an
                    // intrinsic INT type (see `opref_to_snapshot_tagged`).
                    let tp = ctx.const_type(opref).unwrap_or(majit_ir::Type::Int);
                    majit_metainterp::recorder::SnapshotTagged::Const(val, tp)
                } else {
                    let tp = types.get(i).copied().unwrap_or_else(|| {
                        panic!("missing fail-arg box type at index {} for {:?}", i, opref)
                    });
                    majit_metainterp::recorder::SnapshotTagged::Box(opref, tp)
                }
            })
            .collect()
    }

    /// pyjitpl.py:1916-1927 implement_guard_value parity.
    /// executor.py:544-551 constant_from_op(box): dispatches on box.type.
    pub(crate) fn implement_guard_value(
        &mut self,
        ctx: &mut TraceCtx,
        value: OpRef,
        expected: i64,
    ) {
        let expected_ref = match self.value_type(value) {
            majit_ir::Type::Ref => ctx.const_ref(expected),
            _ => ctx.const_int(expected),
        };
        self.generate_guard(ctx, OpCode::GuardValue, &[value, expected_ref]);
        // pyjitpl.py:3512: replace_box
        ctx.heap_cache_mut().replace_box(value, expected_ref);
    }

    pub(crate) fn guard_nonnull(&mut self, ctx: &mut TraceCtx, value: OpRef) {
        // pyjitpl.py:3517-3520 `_establish_nullity` cache hit:
        //     if heapcache.is_nullity_known(box):
        //         self.metainterp.staticdata.profiler.count_ops(rop.GUARD_NONNULL, Counters.HEAPCACHED_OPS)
        //         return value
        if ctx
            .heap_cache()
            .is_nullity_known(value, |op| ctx.const_value(op))
            == Some(true)
        {
            ctx.profiler().count_ops(
                OpCode::GuardNonnull,
                majit_metainterp::counters::HEAPCACHED_OPS,
            );
            return;
        }
        if ctx.heap_cache().is_class_known(value) {
            return; // class known implies nonnull (pyre-only short-circuit, no RPython count)
        }
        self.generate_guard(ctx, OpCode::GuardNonnull, &[value]);
        ctx.heap_cache_mut().nullity_now_known(value, true);
    }

    pub(crate) fn guard_range_iter(&mut self, ctx: &mut TraceCtx, obj: OpRef) {
        self.guard_class(ctx, obj, &RANGE_ITER_TYPE as *const PyType);
    }

    pub(crate) fn record_for_iter_guard(
        &mut self,
        ctx: &mut TraceCtx,
        next: OpRef,
        continues: bool,
    ) {
        // RPython range/xrange iter traces carry the next value as a raw int,
        // not an optional boxed object. Only ref-typed iterators need the
        // optional-value guard here.
        if self.value_type(next) != Type::Ref {
            return;
        }
        let opcode = if continues {
            OpCode::GuardNonnull
        } else {
            OpCode::GuardIsnull
        };
        self.generate_guard(ctx, opcode, &[next]);
        // heapcache: track nullity after for-iter guard
        ctx.heap_cache_mut().nullity_now_known(next, continues);
    }

    pub(crate) fn record_branch_guard(
        &mut self,
        ctx: &mut TraceCtx,
        _branch_value: OpRef,
        truth: OpRef,
        concrete_truth: bool,
        other_target: usize,
    ) {
        let opcode = if concrete_truth {
            OpCode::GuardTrue
        } else {
            OpCode::GuardFalse
        };

        // pyjitpl.py:510-520 goto_if_not(box, target, orgpc):
        //   self.metainterp.generate_guard(opnum, box, resumepc=orgpc)
        // blackhole.py:864 bhimpl_goto_if_not(a, target, pc) re-pops the
        // truth from a register and re-decides which arm to take.
        //
        // RPython's goto_if_not is a compound jitcode opcode produced by
        // jtransform.optimize_goto_if_not (jtransform.py:196): COMPARE_OP +
        // GOTO_IF_NOT are fused into a single op that takes the comparison
        // operands directly and branches. The trace's "register" for the
        // truth and the jitcode's "register" agree on type (int).
        //
        // Pyre emits the same fused op at tracer-dispatch level via
        // try_fused_compare_goto_if_not (pyjitpl.py:541-556 parity): the
        // fused dispatch pops COMPARE's operands, emits IntLt/FloatLt,
        // and calls into this function with the raw Int truth — the
        // symbolic stack carries no comparison value at the guard site.
        // Resuming at other_target (the runtime branch destination)
        // matches the fused jitcode's semantics: the blackhole skips the
        // separate POP_JUMP_IF_FALSE entirely and re-enters the
        // interpreter at the not-taken branch. Inline-frame branch
        // guards use the same resume_pc; parent frames keep their own
        // return_point pc.
        let resume_pc = other_target;

        // pyjitpl.py:2593-2602: saved_pc = frame.pc; frame.pc = resumepc;
        // capture_resumedata(); frame.pc = saved_pc
        // Save ALL state BEFORE flush (generate_guard_core parity).
        let saved_orgpc = self.orgpc;
        let saved_ni = self.sym().vable_last_instr;
        let saved_vsd = self.sym().vable_valuestackdepth;
        self.orgpc = resume_pc;

        // Branch guards resume at `other_target` (the runtime jump destination,
        // not the POP_JUMP_IF_* opcode's orgpc). At `other_target` the Python
        // interpreter has already popped the comparison truth, so the snapshot
        // must reflect the POST-POP register file. `pre_opcode_registers_r`
        // was captured at the START of POP_JUMP_IF
        // (PRE-POP) and still carry the Int truth — using them here would
        // emit a `Box(truth_opref, Int)` at a Ref-declared
        // `locals_cells_stack_w` slot and corrupt the PyFrame when the
        // guard resumes (the decoder writes a raw i64 where a PyObjectRef
        // is expected).
        //
        // pyjitpl.py:2586-2602 capture_resumedata mutates `frame.pc` to
        // `resumepc` for the duration of the capture and restores it on
        // exit; RPython has no pre-opcode snapshot because its jitcode is
        // register-machine and liveness is per-PC. Pyre's pre_opcode_*
        // fields are a deviation that will be removed once per-PC
        // liveness is ported (codewriter/liveness.py). Until then, clear
        // them here so flush_to_frame_for_guard, get_list_of_active_boxes,
        // and list_of_boxes_virtualizable all read the current (post-pop)
        // `registers_r` / valuestackdepth. The next trace_code_step
        // re-captures the snapshot, so no restore is needed.
        self.clear_pre_opcode_state();

        self.flush_to_frame_for_guard(ctx);
        // pyjitpl.py:177: get_list_of_active_boxes uses frame.pc for liveness
        let callee_active_boxes = self.get_list_of_active_boxes(ctx, false, false, None);
        let callee_snapshot_types_full =
            self.build_fail_arg_types_for_active_boxes(&callee_active_boxes);

        // opencoder.py:819 capture_resumedata(framestack) parity:
        // Encode the full framestack [callee (top) with resume_pc=other_target,
        // caller(s) with return_point pc]. Branch guards in inline callees
        // use the same other_target adaptation — the callee frame's resume
        // pc points past POP_JUMP_IF_* (not at it), while parent frames
        // keep their original return_point pc.
        //
        // A branch guard never carries the after-residual-call marker, so its
        // plain top-frame resume pc must leave bit 14 free or `decode_resume_pc`
        // mis-reads it as marked (resumedata.rs:48-62) — fail loudly otherwise.
        assert!(
            resume_pc < majit_ir::resumedata::AFTER_RESIDUAL_CALL_PC_FLAG as usize,
            "branch-guard resume pc {resume_pc} >= AFTER_RESIDUAL_CALL_PC_FLAG; \
             function too large for bit-14 resume encoding"
        );
        let snapshot = self.build_framestack_snapshot(
            ctx,
            resume_pc,
            &callee_active_boxes,
            &callee_snapshot_types_full,
        );
        // Snapshot is the source of truth — the
        // optimizer's `store_final_boxes_in_guard`
        // (`optimizeopt/mod.rs:3200`) overwrites `op.fail_args` from
        // the snapshot via `op.store_final_boxes(liveboxes)`
        // (mod.rs:3392), so the inline `fail_args` copy that the legacy
        // `record_guard_typed_with_fail_args` path used to write was
        // redundant.  Mirrors RPython
        // `pyjitpl.MetaInterp.generate_guard` (pyjitpl.py:2558-2602)
        // which records the guard with no inline fail_args and lets
        // `capture_resumedata` + `_number_boxes` populate them from the
        // snapshot chain.
        let types = self.extend_types_with_parents(ctx, callee_snapshot_types_full);
        let snapshot_id = ctx.capture_resumedata(snapshot);

        ctx.record_guard_typed(opcode, &[truth], types);
        ctx.set_last_guard_resume_position(snapshot_id);

        // pyjitpl.py:2602: frame.pc = saved_pc (restore all, generate_guard_core parity)
        self.orgpc = saved_orgpc;
        let s = self.sym_mut();
        s.vable_last_instr = saved_ni;
        s.vable_valuestackdepth = saved_vsd;
    }

    /// pyjitpl.py:541-556 opimpl_goto_if_not_<op>_<type> parity.
    ///
    /// Fused COMPARE_OP + POP_JUMP_IF_FALSE/TRUE dispatch. Mirrors the
    /// RPython jitcode-level `goto_if_not_int_<op>` fused opcode: emits a
    /// single IntLt/FloatLt followed by GUARD_TRUE/GUARD_FALSE without ever
    /// pushing the comparison truth as a stack value.
    ///
    /// Returns Ok(Some(action)) when the fused path consumed both the
    /// COMPARE_OP and the following PopJumpIf*. Returns Ok(None) when
    /// fuseability fails (next instruction is not PopJumpIf*, or the
    /// operands are not a compatible int/float pair) — caller falls back
    /// to non-fused per-opcode dispatch.
    pub(crate) fn try_fused_compare_goto_if_not(
        &mut self,
        code: &CodeObject,
        compare_pc: usize,
        op: ComparisonOperator,
    ) -> Result<Option<TraceAction>, PyError> {
        // Peek next non-trivia instruction for PopJumpIf*.
        let branch_pc = crate::metainterp::semantic_fallthrough_pc(code, compare_pc);
        let Some((branch_instr, branch_op_arg)) = decode_instruction_at(code, branch_pc) else {
            return Ok(None);
        };
        let (jump_if_true, delta) = match branch_instr {
            Instruction::PopJumpIfFalse { delta } => (false, delta.get(branch_op_arg).as_usize()),
            Instruction::PopJumpIfTrue { delta } => (true, delta.get(branch_op_arg).as_usize()),
            _ => return Ok(None),
        };

        // Fuseability gate: peek concrete values without popping. Only
        // commit to fusion when the codegen int/float fast path will
        // succeed (matches codegen.rs:1056 is_int×is_int and
        // codegen.rs:1103-1109 float-pair conditions). Inspect the
        // ConcreteValue variants directly so the gate does not allocate
        // a throwaway w_int box just to read its type.
        let (a_concrete, b_concrete) = {
            let s = self.sym();
            let top_idx = s
                .valuestackdepth
                .checked_sub(s.nlocals + 1)
                .ok_or_else(|| PyError::type_error("stack underflow in fused compare"))?;
            let a_idx = top_idx
                .checked_sub(1)
                .ok_or_else(|| PyError::type_error("stack underflow in fused compare"))?;
            (
                s.concrete_stack
                    .get(a_idx)
                    .copied()
                    .unwrap_or(ConcreteValue::Null),
                s.concrete_stack
                    .get(top_idx)
                    .copied()
                    .unwrap_or(ConcreteValue::Null),
            )
        };
        let (lhs_is_int, lhs_is_float) = classify_concrete(a_concrete);
        let (rhs_is_int, rhs_is_float) = classify_concrete(b_concrete);
        let ints_ok = lhs_is_int && rhs_is_int;
        let floats_ok = (lhs_is_float || lhs_is_int)
            && (rhs_is_float || rhs_is_int)
            && (lhs_is_float || rhs_is_float);
        if !ints_ok && !floats_ok {
            return Ok(None);
        }

        // Materialize the full concrete values only after the gate
        // succeeds; compare_value_direct + objspace_compare_{ints,floats}
        // both need them in PyObjectRef form to call into baseobjspace.
        let lhs_obj = a_concrete.to_pyobj();
        let rhs_obj = b_concrete.to_pyobj();
        if lhs_obj.is_null() || rhs_obj.is_null() {
            return Ok(None);
        }

        // Compute branch destinations (matches opcode_pop_jump_if at
        // pyopcode.rs:556). `fallthrough` is MIFrame's semantic_fallthrough_pc
        // past trivia; `target` is jump_target_forward (skip_caches + delta).
        let branch_fallthrough = crate::metainterp::semantic_fallthrough_pc(code, branch_pc);
        let branch_target =
            pyre_interpreter::jump_target_forward(&code.instructions, branch_pc + 1, delta);

        // Compute concrete branch direction locally. Matches baseobjspace
        // ::compare int/float pair semantics so the fused path does not
        // depend on any tracer-side cache written during trace emission.
        let concrete_truth = if ints_ok {
            unsafe { crate::state::objspace_compare_ints(lhs_obj, rhs_obj, op) }
        } else {
            unsafe { crate::state::objspace_compare_floats(lhs_obj, rhs_obj, op) }
        };

        // Commit: pop b, pop a (matches opcode_compare_op at pyopcode.rs:530).
        let b_ref = self.with_ctx(|this, ctx| this.pop_value(ctx))?;
        let a_ref = self.with_ctx(|this, ctx| this.pop_value(ctx))?;

        // Emit IntLt/FloatLt via compare_value_direct (generated int/float
        // fast path). `next_instruction_consumes_comparison_truth()` is
        // true here (the fuseability gate already saw PopJumpIf* as the
        // next non-trivia instruction), so the returned OpRef is the raw
        // Int truth, NOT a boxed bool.
        let truth = self.compare_value_direct(a_ref, b_ref, op, lhs_obj, rhs_obj)?;

        // Decide branch direction (opcode_pop_jump_if at pyopcode.rs:565-572).
        let should_jump = concrete_truth == jump_if_true;
        let other_target = if should_jump {
            branch_fallthrough
        } else {
            branch_target
        };
        let next_target = if should_jump {
            branch_target
        } else {
            branch_fallthrough
        };

        // Emit GUARD_TRUE/GUARD_FALSE. record_branch_guard swaps orgpc to
        // resume_pc (other_target), suppresses pre_opcode_*, and builds
        // the framestack snapshot before restoring state.
        self.with_ctx(|this, ctx| {
            MIFrame::record_branch_guard(this, ctx, truth, truth, concrete_truth, other_target);
            Ok::<(), PyError>(())
        })?;

        // Advance past the branch; the outer dispatch loop picks this up.
        self.sym_mut().pending_next_instr = Some(next_target);
        Ok(Some(TraceAction::Continue))
    }

    /// RPython registers[idx] parity: read concrete value from Box arrays.
    fn concrete_at(&self, abs_idx: usize) -> Option<PyObjectRef> {
        let v = self.sym().concrete_value_at(abs_idx);
        if !v.is_null() {
            return Some(v.to_pyobj());
        }
        None
    }

    fn existing_ref_for_concrete(
        &self,
        ctx: &TraceCtx,
        concrete_obj: PyObjectRef,
    ) -> Option<OpRef> {
        if concrete_obj.is_null() {
            return None;
        }
        let s = self.sym();
        let total_slots = s.nlocals + s.concrete_stack.len();
        let owns_shadow = s.owns_virtualizable_shadow();
        let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
        for abs_idx in 0..total_slots {
            if s.concrete_value_at(abs_idx).to_pyobj() == concrete_obj {
                // Portal frames carry the
                // current OpRef on `virtualizable_boxes` (`pyjitpl.py:
                // 1230 _opimpl_getarrayitem_vable`).  The legacy
                // `registers_r[abs_idx]` semantic mirror returns the
                // init-symbolic seed once `store_local_value` retires
                // the mirror write, so consult vable first
                // for portal frames.
                let opref = if owns_shadow {
                    ctx.virtualizable_box_at(nvs + abs_idx).unwrap_or_else(|| {
                        s.registers_r.get(abs_idx).copied().unwrap_or(OpRef::NONE)
                    })
                } else {
                    *s.registers_r.get(abs_idx)?
                };
                if opref != OpRef::NONE && self.value_type(opref) == Type::Ref {
                    return Some(opref);
                }
            }
        }
        None
    }

    #[allow(dead_code)]
    fn guard_int_object_value(&mut self, ctx: &mut TraceCtx, int_obj: OpRef, expected: i64) {
        self.guard_class(ctx, int_obj, &INT_TYPE as *const PyType);
        let actual_value = opimpl_getfield_gc_i(ctx, int_obj, int_intval_descr());
        self.implement_guard_value(ctx, actual_value, expected);
    }

    #[allow(dead_code)]
    pub(crate) fn guard_int_like_value(&mut self, ctx: &mut TraceCtx, value: OpRef, expected: i64) {
        if self.value_type(value) == Type::Int {
            self.implement_guard_value(ctx, value, expected);
        } else {
            self.guard_int_object_value(ctx, value, expected);
        }
    }

    /// pyjitpl.py:1518 opimpl_guard_class
    pub(crate) fn guard_class(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        expected_type: *const PyType,
    ) {
        // heapcache.py: skip guard if class already known for this object
        if ctx.heap_cache().is_class_known(obj) {
            return;
        }
        // pyjitpl.py:2558-2560 generate_guard parity:
        //     if isinstance(box, Const):    # no need for a guard
        //         return
        // The concrete value (and therefore its class) is known at trace
        // time, so the runtime type check is guaranteed to pass. RPython
        // also short-circuits before capture_resumedata, so no snapshot is
        // attached for the skipped guard. heapcache.class_now_known is
        // still called below — pyjitpl.py:1523 opimpl_guard_class invokes
        // it unconditionally after generate_guard.
        if obj.is_constant() {
            // pyjitpl.py:1087 parity: a pending GUARD_NOT_INVALIDATED from a
            // preceding quasi-immut field read must still be flushed even
            // when the type guard is skipped, otherwise the watcher and the
            // trace's quasi-immut dependency would be silently dropped.
            self.flush_guard_not_invalidated(ctx);
            ctx.heap_cache_mut()
                .class_now_known(obj, expected_type as usize as i64);
            return;
        }
        let expected_type_const = ctx.const_int(expected_type as usize as i64);
        self.generate_guard(ctx, OpCode::GuardNonnullClass, &[obj, expected_type_const]);
        // heapcache.py:470-473: class_now_known sets class + nullity.
        ctx.heap_cache_mut()
            .class_now_known(obj, expected_type as usize as i64);
    }

    pub(crate) fn trace_guarded_int_payload(
        &mut self,
        ctx: &mut TraceCtx,
        int_obj: OpRef,
    ) -> OpRef {
        if self.value_type(int_obj) == Type::Int {
            return int_obj;
        }
        self.guard_class(ctx, int_obj, &INT_TYPE as *const PyType);
        opimpl_getfield_gc_i(ctx, int_obj, int_intval_descr())
    }

    pub(crate) fn guard_len_gt_index(&mut self, ctx: &mut TraceCtx, len: OpRef, index: usize) {
        let index = ctx.const_int(index as i64);
        let in_bounds = ctx.record_op(OpCode::IntGt, &[len, index]);
        self.generate_guard(ctx, OpCode::GuardTrue, &[in_bounds]);
    }

    #[allow(dead_code)]
    pub(crate) fn guard_len_eq(&mut self, ctx: &mut TraceCtx, len: OpRef, expected: usize) {
        self.implement_guard_value(ctx, len, expected as i64);
    }

    pub(crate) fn guard_list_strategy(&mut self, ctx: &mut TraceCtx, obj: OpRef, expected: i64) {
        let strategy = opimpl_getfield_gc_i(ctx, obj, list_strategy_descr());
        self.implement_guard_value(ctx, strategy, expected);
    }

    /// pyjitpl.py:841-852 opimpl_check_resizable_neg_index delegate.
    pub(crate) fn trace_dynamic_list_index(
        &mut self,
        ctx: &mut TraceCtx,
        key: OpRef,
        len: OpRef,
        concrete_key: i64,
    ) -> OpRef {
        crate::generated_dynamic_list_index(self, ctx, key, len, concrete_key)
    }

    /// Trace-visible tuple construction, matching
    /// `objspace/std/objspace.py:515 fixedview` consumers and
    /// `tupleobject.py` / `specialisedtupleobject.py` producers. This
    /// replaces the older opaque `jit_build_tuple_N` helper for concrete
    /// tuple shapes so OptVirtualize can remove a tuple that is built only
    /// to be immediately unpacked.
    pub(crate) fn trace_build_tuple_value(
        &mut self,
        items: &[OpRef],
        concrete_items: &[PyObjectRef],
    ) -> Result<OpRef, PyError> {
        // Build the trace-visible specialised tuple (NewWithVtable +
        // inline `value0`/`value1` SetfieldGc) so OptVirtualize can
        // virtualize the build→unpack pair and elide the allocation. The
        // paired `w_class` guard on the int/float items is a header-field
        // read that OptVirtualize resolves via `FieldDescr::is_w_class`
        // (virtualize.rs), so the virtual loop-carried items no longer
        // trip the `make_equal_to` Box.type invariant.
        if concrete_items.iter().any(|item| item.is_null()) {
            // STRUCTURAL ADAPTATION: PyPy's `space.newtuple()` always
            // reaches `wraptuple()` with concrete W_Root instances, so
            // `makespecialisedtuple2()` can choose `Cls_ii` / `Cls_ff` /
            // `Cls_oo` for arity 2. Pyre's trace-side FrontendOp may
            // lack that concrete side channel after earlier trace-only
            // operations. In that case, keep the older helper path rather
            // than guessing a canonical W_TupleObject and regressing
            // specialised tuple parity on escaped two-tuples.
            return self.trace_build_tuple(items);
        }

        self.with_ctx(|this, ctx| unsafe {
            if items.len() == 2 && concrete_items.len() >= 2 {
                let lhs = concrete_items[0];
                let rhs = concrete_items[1];
                if pyre_object::is_plain_int1(lhs) && pyre_object::is_plain_int1(rhs) {
                    // `listobject.py:2390 is_plain_int1` rejects app-level
                    // int subclasses via `type(w) is W_IntObject/W_LongObject`.
                    // Pyre stores that subclass identity in `PyObject.w_class`
                    // while `ob_type` stays at the payload layout, so the
                    // unbox guard (which only checks `ob_type`) is not
                    // enough — emit a paired `w_class` guard so a later
                    // int subclass with the same payload layout side-exits
                    // instead of replaying the Cls_ii fast path and
                    // re-wrapping its payload as a plain int.
                    let int_typeobj = get_instantiate(&INT_TYPE);
                    trace_guard_exact_w_class(this, ctx, items[0], int_typeobj);
                    trace_guard_exact_w_class(this, ctx, items[1], int_typeobj);
                    let raw0 = trace_plain_int_payload(this, ctx, items[0], lhs);
                    let raw1 = trace_plain_int_payload(this, ctx, items[1], rhs);
                    let tuple = ctx.record_op_with_descr(
                        OpCode::NewWithVtable,
                        &[],
                        crate::descr::specialised_tuple_ii_size_descr(),
                    );
                    ctx.heap_cache_mut().new_object(tuple);
                    trace_set_tuple_w_class(
                        ctx,
                        tuple,
                        crate::descr::specialised_tuple_ii_w_class_descr(),
                    );
                    ctx.record_op_with_descr(
                        OpCode::SetfieldGc,
                        &[tuple, raw0],
                        crate::descr::specialised_tuple_ii_value0_descr(),
                    );
                    ctx.heapcache_setfield_cached(
                        tuple,
                        crate::descr::specialised_tuple_ii_value0_descr().index(),
                        raw0,
                    );
                    ctx.record_op_with_descr(
                        OpCode::SetfieldGc,
                        &[tuple, raw1],
                        crate::descr::specialised_tuple_ii_value1_descr(),
                    );
                    ctx.heapcache_setfield_cached(
                        tuple,
                        crate::descr::specialised_tuple_ii_value1_descr().index(),
                        raw1,
                    );
                    return Ok(tuple);
                }

                if pyre_object::is_plain_float_strict(lhs)
                    && pyre_object::is_plain_float_strict(rhs)
                {
                    // `specialisedtupleobject.py:176` requires
                    // `type(w) is W_FloatObject` strict identity. The
                    // unbox guard only checks `ob_type == FLOAT_TYPE`,
                    // so emit a paired `w_class` guard against the
                    // canonical float class — a later float subclass
                    // with the same payload layout must side-exit
                    // instead of replaying the Cls_ff fast path and
                    // re-wrapping its payload as a plain float.
                    let float_typeobj = get_instantiate(&FLOAT_TYPE);
                    trace_guard_exact_w_class(this, ctx, items[0], float_typeobj);
                    trace_guard_exact_w_class(this, ctx, items[1], float_typeobj);
                    let raw0 = if this.value_type(items[0]) == Type::Float {
                        items[0]
                    } else {
                        crate::state::trace_unbox_float_with_resume(
                            this,
                            items[0],
                            &FLOAT_TYPE as *const _ as i64,
                        )
                    };
                    let raw1 = if this.value_type(items[1]) == Type::Float {
                        items[1]
                    } else {
                        crate::state::trace_unbox_float_with_resume(
                            this,
                            items[1],
                            &FLOAT_TYPE as *const _ as i64,
                        )
                    };
                    let tuple = ctx.record_op_with_descr(
                        OpCode::NewWithVtable,
                        &[],
                        crate::descr::specialised_tuple_ff_size_descr(),
                    );
                    ctx.heap_cache_mut().new_object(tuple);
                    trace_set_tuple_w_class(
                        ctx,
                        tuple,
                        crate::descr::specialised_tuple_ff_w_class_descr(),
                    );
                    ctx.record_op_with_descr(
                        OpCode::SetfieldGc,
                        &[tuple, raw0],
                        crate::descr::specialised_tuple_ff_value0_descr(),
                    );
                    ctx.heapcache_setfield_cached(
                        tuple,
                        crate::descr::specialised_tuple_ff_value0_descr().index(),
                        raw0,
                    );
                    ctx.record_op_with_descr(
                        OpCode::SetfieldGc,
                        &[tuple, raw1],
                        crate::descr::specialised_tuple_ff_value1_descr(),
                    );
                    ctx.heapcache_setfield_cached(
                        tuple,
                        crate::descr::specialised_tuple_ff_value1_descr().index(),
                        raw1,
                    );
                    return Ok(tuple);
                }

                let tuple = ctx.record_op_with_descr(
                    OpCode::NewWithVtable,
                    &[],
                    crate::descr::specialised_tuple_oo_size_descr(),
                );
                ctx.heap_cache_mut().new_object(tuple);
                trace_set_tuple_w_class(
                    ctx,
                    tuple,
                    crate::descr::specialised_tuple_oo_w_class_descr(),
                );
                ctx.record_op_with_descr(
                    OpCode::SetfieldGc,
                    &[tuple, items[0]],
                    crate::descr::specialised_tuple_oo_value0_descr(),
                );
                ctx.heapcache_setfield_cached(
                    tuple,
                    crate::descr::specialised_tuple_oo_value0_descr().index(),
                    items[0],
                );
                ctx.record_op_with_descr(
                    OpCode::SetfieldGc,
                    &[tuple, items[1]],
                    crate::descr::specialised_tuple_oo_value1_descr(),
                );
                ctx.heapcache_setfield_cached(
                    tuple,
                    crate::descr::specialised_tuple_oo_value1_descr().index(),
                    items[1],
                );
                return Ok(tuple);
            }

            let len = ctx.const_int(items.len() as i64);
            let array_descr = crate::state::pyobject_gcarray_descr();
            let items_block = ctx.record_op_with_descr(OpCode::NewArrayClear, &[len], array_descr);
            ctx.heap_cache_mut().new_array(items_block, len, true);
            for (idx, &item) in items.iter().enumerate() {
                let idx = ctx.const_int(idx as i64);
                crate::state::trace_items_block_setitem_value(ctx, items_block, idx, item);
            }

            let tuple = ctx.record_op_with_descr(
                OpCode::NewWithVtable,
                &[],
                crate::descr::w_tuple_size_descr(),
            );
            ctx.heap_cache_mut().new_object(tuple);
            trace_set_tuple_w_class(ctx, tuple, crate::descr::tuple_w_class_descr());
            let wrappeditems_descr = crate::descr::tuple_wrappeditems_descr();
            ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[tuple, items_block],
                wrappeditems_descr.clone(),
            );
            ctx.heapcache_setfield_cached(tuple, wrappeditems_descr.index(), items_block);
            Ok(tuple)
        })
    }

    /// Unpack a known-length tuple. `W_TupleObject` follows
    /// `tupleobject.py:376-390`: `wrappeditems` is a GcArray and the
    /// length is read via `arraylen_gc(items_block, pyobject_gcarray_descr)`.
    ///
    /// Arity-2 specialised tuple variants follow
    /// `specialisedtupleobject.py`: after `guard_class` their immutable
    /// inline `value0` / `value1` fields are loaded directly. This keeps
    /// `UNPACK_SEQUENCE` structurally aligned with the tuple getitem path
    /// and avoids tracing a canonical `W_TupleObject` guard for `Cls_ii`.
    fn trace_unpack_known_tuple(
        &mut self,
        ctx: &mut TraceCtx,
        seq: OpRef,
        count: usize,
        concrete_seq: PyObjectRef,
        items_descr: DescrRef,
    ) -> Vec<OpRef> {
        let ob_type = unsafe { (*concrete_seq).ob_type };
        let spec_ii = &SPECIALISED_TUPLE_II_TYPE as *const PyType;
        let spec_ff = &SPECIALISED_TUPLE_FF_TYPE as *const PyType;
        let spec_oo = &SPECIALISED_TUPLE_OO_TYPE as *const PyType;

        if std::ptr::eq(ob_type, spec_ii) {
            debug_assert_eq!(count, 2);
            self.guard_class(ctx, seq, spec_ii);
            let value0 = ctx.record_op_with_descr(
                OpCode::GetfieldGcPureI,
                &[seq],
                crate::descr::specialised_tuple_ii_value0_descr(),
            );
            let value1 = ctx.record_op_with_descr(
                OpCode::GetfieldGcPureI,
                &[seq],
                crate::descr::specialised_tuple_ii_value1_descr(),
            );
            return vec![
                crate::state::wrapint(ctx, value0),
                crate::state::wrapint(ctx, value1),
            ];
        }

        if std::ptr::eq(ob_type, spec_ff) {
            debug_assert_eq!(count, 2);
            self.guard_class(ctx, seq, spec_ff);
            let value0 = ctx.record_op_with_descr(
                OpCode::GetfieldGcPureF,
                &[seq],
                crate::descr::specialised_tuple_ff_value0_descr(),
            );
            let value1 = ctx.record_op_with_descr(
                OpCode::GetfieldGcPureF,
                &[seq],
                crate::descr::specialised_tuple_ff_value1_descr(),
            );
            return vec![
                crate::state::wrapfloat(ctx, value0),
                crate::state::wrapfloat(ctx, value1),
            ];
        }

        if std::ptr::eq(ob_type, spec_oo) {
            debug_assert_eq!(count, 2);
            self.guard_class(ctx, seq, spec_oo);
            let value0 = ctx.record_op_with_descr(
                OpCode::GetfieldGcPureR,
                &[seq],
                crate::descr::specialised_tuple_oo_value0_descr(),
            );
            let value1 = ctx.record_op_with_descr(
                OpCode::GetfieldGcPureR,
                &[seq],
                crate::descr::specialised_tuple_oo_value1_descr(),
            );
            return vec![value0, value1];
        }

        self.guard_class(ctx, seq, &TUPLE_TYPE as *const PyType);

        let items_block = crate::state::opimpl_getfield_gc_r(ctx, seq, items_descr);
        let len = crate::state::opimpl_arraylen_gc(
            ctx,
            items_block,
            crate::state::pyobject_gcarray_descr(),
        );
        self.implement_guard_value(ctx, len, count as i64);

        (0..count)
            .map(|idx| {
                let idx = ctx.const_int(idx as i64);
                crate::state::trace_items_block_getitem_value(ctx, items_block, idx)
            })
            .collect()
    }

    /// Unpack a known-length list under the Object strategy. List
    /// keeps the inline `length` field (rlist.py:116 `("length",
    /// Signed)`) so the length comes via `getfield_gc_i(length_descr)`,
    /// not `arraylen_gc`. Items live in the same `Ptr(GcArray(
    /// OBJECTPTR))` shape as tuples, read via `getarrayitem_gc_r`
    /// against `pyobject_gcarray_descr`.
    fn trace_unpack_known_list(
        &mut self,
        ctx: &mut TraceCtx,
        seq: OpRef,
        count: usize,
        length_descr: DescrRef,
        items_descr: DescrRef,
    ) -> Vec<OpRef> {
        self.guard_class(ctx, seq, &LIST_TYPE as *const PyType);

        let len = opimpl_getfield_gc_i(ctx, seq, length_descr);
        self.implement_guard_value(ctx, len, count as i64);

        let items_block = crate::state::opimpl_getfield_gc_r(ctx, seq, items_descr);
        (0..count)
            .map(|idx| {
                let idx = ctx.const_int(idx as i64);
                crate::state::trace_items_block_getitem_value(ctx, items_block, idx)
            })
            .collect()
    }

    pub(crate) fn unpack_sequence_value(
        &mut self,
        seq: OpRef,
        count: usize,
        concrete_seq: PyObjectRef,
    ) -> Result<Vec<FrontendOp>, PyError> {
        if concrete_seq.is_null() {
            let oprefs = TraceHelperAccess::trace_unpack_sequence(self, seq, count)?;
            return Ok(oprefs.into_iter().map(FrontendOp::void).collect());
        }

        // Extract concrete items from the sequence for RPython Box parity.
        let concrete_items: Vec<PyObjectRef> = unsafe {
            if is_tuple(concrete_seq) {
                (0..count)
                    .filter_map(|i| w_tuple_getitem(concrete_seq, i as i64))
                    .collect()
            } else if is_list(concrete_seq) && w_list_uses_object_storage(concrete_seq) {
                (0..count)
                    .filter_map(|i| w_list_getitem(concrete_seq, i as i64))
                    .collect()
            } else {
                Vec::new()
            }
        };

        let oprefs = self.with_ctx(|this, ctx| unsafe {
            if is_tuple(concrete_seq) && w_tuple_len(concrete_seq) == count {
                return Ok(this.trace_unpack_known_tuple(
                    ctx,
                    seq,
                    count,
                    concrete_seq,
                    crate::descr::tuple_wrappeditems_descr(),
                ));
            }
            if is_list(concrete_seq)
                && w_list_uses_object_storage(concrete_seq)
                && w_list_len(concrete_seq) == count
            {
                return Ok(this.trace_unpack_known_list(
                    ctx,
                    seq,
                    count,
                    crate::descr::list_length_descr(),
                    crate::descr::list_items_descr(),
                ));
            }
            TraceHelperAccess::trace_unpack_sequence(this, seq, count)
        })?;

        Ok(oprefs
            .into_iter()
            .enumerate()
            .map(|(i, opref)| {
                let cv = concrete_items
                    .get(i)
                    .copied()
                    .map(ConcreteValue::from_pyobj)
                    .unwrap_or(ConcreteValue::Null);
                FrontendOp::new(opref, cv)
            })
            .collect())
    }

    pub(crate) fn binary_subscr_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        concrete_obj: PyObjectRef,
        concrete_key: PyObjectRef,
    ) -> Result<FrontendOp, PyError> {
        // Concrete subscr result for FrontendOp Box tracking.
        let subscr_concrete = if !concrete_obj.is_null() && !concrete_key.is_null() {
            if let Ok(result) = pyre_interpreter::baseobjspace::getitem(concrete_obj, concrete_key)
            {
                ConcreteValue::from_pyobj(result)
            } else {
                ConcreteValue::Null
            }
        } else {
            ConcreteValue::Null
        };

        // jtransform do_fixed_list_getitem / do_resizable_list_getitem dispatch.
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_binary_subscr_value(
                this,
                ctx,
                a,
                b,
                concrete_obj,
                concrete_key,
            ))
        })?;
        if let Some(opref) = gen_result {
            return Ok(FrontendOp::new(opref, subscr_concrete));
        }
        let opref = self.trace_binary_value(a, b, BinaryOperator::Subscr)?;
        Ok(FrontendOp::new(opref, subscr_concrete))
    }

    pub(crate) fn binary_int_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: BinaryOperator,
        concrete_lhs: PyObjectRef,
        concrete_rhs: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // guard_class + getfield_gc_i + int_OP_ovf + guard_no_overflow
        // + new_with_vtable + setfield_gc).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_binary_int_value(
                this,
                ctx,
                a,
                b,
                op,
                concrete_lhs,
                concrete_rhs,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_binary_value(a, b, op)
    }

    pub(crate) fn binary_float_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: BinaryOperator,
        concrete_lhs: PyObjectRef,
        concrete_rhs: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // guard_class + getfield_gc_f/cast_int_to_float + float_OP
        // + new_with_vtable + setfield_gc).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_binary_float_value(
                this,
                ctx,
                a,
                b,
                op,
                concrete_lhs,
                concrete_rhs,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_binary_value(a, b, op)
    }

    pub(crate) fn compare_value_direct(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: ComparisonOperator,
        concrete_lhs: PyObjectRef,
        concrete_rhs: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // guard_class + getfield_gc_i/f + int_LT/float_LT, with
        // goto_if_not fusion truth caching).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_compare_value_direct(
                this,
                ctx,
                a,
                b,
                op,
                concrete_lhs,
                concrete_rhs,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_compare_value(a, b, op)
    }

    pub(crate) fn store_subscr_value(
        &mut self,
        obj: OpRef,
        key: OpRef,
        value: OpRef,
        concrete_obj: PyObjectRef,
        concrete_key: PyObjectRef,
        concrete_value: PyObjectRef,
    ) -> Result<(), PyError> {
        // STORE_SUBSCR is deferred: the emitted IR performs the heap write in
        // the compiled loop (exactly once), and no concrete write happens
        // during trace. So it must NOT mark the loop as heap-mutated — a
        // pure-STORE_SUBSCR loop (nbody / fannkuch) is not advanced, keeping
        // its single deferred write. Only the concrete-during-trace method
        // mutations (append / pop / reverse) drive `dm143_advance_live_locals`.
        self.store_subscr_value_emit(obj, key, value, concrete_obj, concrete_key, concrete_value)?;
        Ok(())
    }

    fn store_subscr_value_emit(
        &mut self,
        obj: OpRef,
        key: OpRef,
        value: OpRef,
        concrete_obj: PyObjectRef,
        concrete_key: PyObjectRef,
        concrete_value: PyObjectRef,
    ) -> Result<(), PyError> {
        if let Some((raw_start, raw_stop, start, stop, step_is_none)) =
            const_step_one_slice_bounds(concrete_obj, concrete_key, concrete_value)
        {
            let specialized_same_len = unsafe {
                let obj_len = w_list_len(concrete_obj);
                let value_len = w_list_len(concrete_value);
                let slice_len = (stop - start) as usize;
                if concrete_obj == concrete_value {
                    None
                } else if value_len == slice_len {
                    match (
                        concrete_list_strategy_id(concrete_obj),
                        concrete_list_strategy_id(concrete_value),
                    ) {
                        (Some(obj_sid), Some(value_sid)) if obj_sid == value_sid => {
                            Some((obj_sid, obj_len, value_len))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            };
            if let Some((strategy_id, obj_len, value_len)) = specialized_same_len {
                self.with_ctx(|this, ctx| {
                    this.guard_class(
                        ctx,
                        key,
                        &pyre_object::sliceobject::SLICE_TYPE as *const _
                            as *const pyre_object::PyType,
                    );
                    let start_box =
                        crate::state::opimpl_getfield_gc_r(ctx, key, slice_w_start_descr());
                    this.guard_int_object_value(ctx, start_box, raw_start);
                    let stop_box =
                        crate::state::opimpl_getfield_gc_r(ctx, key, slice_w_stop_descr());
                    this.guard_int_object_value(ctx, stop_box, raw_stop);
                    let step_box =
                        crate::state::opimpl_getfield_gc_r(ctx, key, slice_w_step_descr());
                    if step_is_none {
                        this.implement_guard_value(ctx, step_box, pyre_object::w_none() as i64);
                    } else {
                        this.guard_int_object_value(ctx, step_box, 1);
                    }
                    crate::generated_list_setslice_same_len_by_strategy(
                        this,
                        obj,
                        value,
                        raw_start,
                        raw_stop,
                        start,
                        stop,
                        strategy_id,
                        obj_len,
                        value_len,
                    );
                    Ok::<_, PyError>(())
                })?;
                return Ok(());
            }
        }
        // jtransform do_resizable_list_setitem dispatch.
        let handled: bool = self.with_ctx(|this, _ctx| {
            Ok::<_, PyError>(crate::generated_store_subscr_value(
                this,
                obj,
                key,
                value,
                concrete_obj,
                concrete_key,
                concrete_value,
            ))
        })?;
        if handled {
            return Ok(());
        }
        self.trace_store_subscr(obj, key, value)
    }

    pub(crate) fn list_append_value(
        &mut self,
        list: OpRef,
        value: OpRef,
        concrete_list: PyObjectRef,
        concrete_value: PyObjectRef,
    ) -> Result<(), PyError> {
        if concrete_list.is_null() {
            return self.trace_list_append(list, value);
        }

        unsafe {
            if is_list(concrete_list) && w_list_can_append_without_realloc(concrete_list) {
                // listobject.rs:234: typed strategies (int/float) de-specialize
                // to object strategy if the appended value doesn't match.
                // Only enter typed fast path when concrete value type matches.
                //
                // `IntegerListStrategy.is_correct_type`
                // (`listobject.py:1957-1958`) accepts both `W_IntObject` and
                // a fits_int `W_LongObject`. Pyre threads the choice through
                // `unbox_long`: false → `trace_unbox_int_with_resume(INT_TYPE)`,
                // true → `trace_unbox_long_with_resume(LONG_TYPE)` (which
                // emits the fits_int residual GUARD_TRUE before extraction).
                let strategy = if w_list_uses_object_storage(concrete_list) {
                    Some((0i64, false))
                } else if w_list_uses_int_storage(concrete_list)
                    && pyre_object::is_plain_int1(concrete_value)
                {
                    let unbox_long = pyre_object::pyobject::is_long(concrete_value);
                    Some((1i64, unbox_long))
                } else if w_list_uses_float_storage(concrete_list) && is_float(concrete_value) {
                    Some((2i64, false))
                } else {
                    None
                };
                if let Some((sid, unbox_long)) = strategy {
                    let is_inline = w_list_is_inline_storage(concrete_list);
                    return self.with_ctx(|this, ctx| {
                        crate::generated_list_append_by_strategy(
                            this, ctx, list, value, sid, is_inline, unbox_long,
                        );
                        Ok(())
                    });
                }
            }
        }

        self.trace_list_append(list, value)
    }

    /// Mirror of `list_append_value` for `lst.pop()`. Caller has already
    /// verified `concrete_len > 0`; this function picks a strategy fast
    /// path or falls back to generic call dispatch.
    ///
    /// `callable` is the bound `W_MethodObject` OpRef: fallback paths pass
    /// it to `trace_call_callable` so the residual emits `jit_call_callable_0`
    /// on the *method*, not on the receiver (calling the list itself would be
    /// a TypeError).
    ///
    /// Currently unused — paired with the `list.pop` tracing abort at
    /// `trace_call_callable` (`trace_opcode.rs:5141`); becomes the trace-
    /// time replacement for that abort once guard-failure blackhole
    /// resume is complete.  Kept in sync with `list_append_value` /
    /// `list_reverse_value` so the wire-up is a one-line edit.
    #[allow(dead_code)]
    pub(crate) fn list_pop_value(
        &mut self,
        callable: OpRef,
        list: OpRef,
        concrete_list: PyObjectRef,
        concrete_len: usize,
        // Generic-call replay shape when the fast path bails: `[]` for the
        // bound-method form `xs.pop()` (`callable` is the bound method) and
        // `[receiver]` for the builtin form `list.pop(xs)` (`callable` is the
        // unbound builtin, so an empty list must record `list.pop(xs)` to
        // raise IndexError rather than `list.pop()`'s arity TypeError).
        fallback_args: &[OpRef],
    ) -> Result<OpRef, PyError> {
        if concrete_list.is_null() || concrete_len == 0 {
            // Caller's guard ensures concrete_len > 0; defensive fallback.
            return self.trace_call_callable(callable, fallback_args);
        }
        unsafe {
            if is_list(concrete_list) {
                let strategy_id = if w_list_uses_object_storage(concrete_list) {
                    Some(0i64)
                } else if w_list_uses_int_storage(concrete_list) {
                    Some(1i64)
                } else if w_list_uses_float_storage(concrete_list) {
                    Some(2i64)
                } else {
                    None
                };
                if let Some(sid) = strategy_id {
                    let is_inline = w_list_is_inline_storage(concrete_list);
                    return Ok(self.with_ctx(|this, ctx| {
                        crate::generated_list_pop_by_strategy(this, ctx, list, sid, is_inline)
                    }));
                }
            }
        }
        self.trace_call_callable(callable, fallback_args)
    }

    /// Direct trace path for `list.reverse()`.
    ///
    /// PyPy reaches `listobject.py`'s strategy `reverse` through the normal
    /// rtyper/list helper path. Pyre does not have that full oopspec lowering
    /// yet, so mirror the existing append/pop trace-time specialization and
    /// emit a narrow helper call on the proven builtin list receiver.
    pub(crate) fn list_reverse_value(
        &mut self,
        callable: OpRef,
        list: OpRef,
        concrete_list: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        if concrete_list.is_null() || unsafe { !is_list(concrete_list) } {
            return self.trace_call_callable(callable, &[]);
        }
        let result = self.with_ctx(|this, ctx| {
            this.guard_class(ctx, list, &LIST_TYPE as *const PyType);
            crate::helpers::emit_trace_call_void_typed(
                ctx,
                pyre_object::listobject::jit_list_reverse as *const (),
                &[list],
                &[Type::Ref],
            );
            ctx.const_ref(pyre_object::w_none() as i64)
        });
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    /// Direct trace path for `list.pop(index)` (front / indexed pop).
    ///
    /// `ll_pop_zero` (`oopspec = 'list.pop(l, 0)'`) / `ll_pop_nonneg`
    /// (`'list.pop(l, index)'`). The O(n) element shift is performed by the
    /// `jit_list_pop_at` residual (an escaped, non-virtual list lowers the
    /// oopspec to a residual call upstream too), but the residual is opaque:
    /// `default_effect_info` only invalidates the list's cached length, so
    /// the compiled loop would track a stale length and a later `append`
    /// would write past the live elements (issue #143, the dominant len
    /// error). Mirror the already-correct end-pop
    /// (`generated_list_pop_by_strategy`): overlay an inline
    /// `SetfieldGc(new_len)` so the post-pop length is a known,
    /// resume-reconstructible value. The inline write is idempotent with the
    /// residual's own decrement (both leave `length == len - 1`).
    pub(crate) fn list_pop_at_value(
        &mut self,
        callable: OpRef,
        list: OpRef,
        index: OpRef,
        concrete_list: PyObjectRef,
        concrete_index: PyObjectRef,
        // Operand-stack shape to replay when the fast path bails to the
        // generic residual: `[index]` for the bound-method form `xs.pop(i)`
        // (`callable` is the bound method) and `[receiver, index]` for the
        // builtin form `list.pop(xs, i)` (`callable` is the unbound builtin,
        // so the receiver must be re-supplied for the wrong-exception cases to
        // match the concrete call).
        fallback_args: &[OpRef],
    ) -> Result<OpRef, PyError> {
        // Pick the strategy-specific length field so the inline length update
        // mirrors `generated_list_pop_by_strategy`. A null / non-list / mixed
        // receiver has no known length field, so fall back to the generic
        // residual on the method object.
        let strategy = unsafe {
            if concrete_list.is_null() || !is_list(concrete_list) {
                None
            } else if w_list_uses_object_storage(concrete_list) {
                Some((0i64, crate::descr::list_length_descr()))
            } else if w_list_uses_int_storage(concrete_list) {
                Some((1i64, crate::descr::list_int_items_len_descr()))
            } else if w_list_uses_float_storage(concrete_list) {
                Some((2i64, crate::descr::list_float_items_len_descr()))
            } else {
                None
            }
        };
        let Some((strategy_id, len_descr)) = strategy else {
            return self.trace_call_callable(callable, fallback_args);
        };
        // The `jit_list_pop_at` residual panics on an out-of-range index
        // instead of raising IndexError, so the fast path may only be taken
        // with a trace-time in-range proof plus matching runtime guards
        // (emitted by `trace_dynamic_list_index` below). A non-W_IntObject
        // index or an out-of-range concrete index (IndexError at runtime)
        // falls back to the generic call.
        let concrete_key = unsafe {
            if pyre_object::pyobject::py_type_check(concrete_index, &INT_TYPE) {
                Some(pyre_object::w_int_get_value(concrete_index))
            } else {
                None
            }
        };
        let Some(concrete_key) = concrete_key else {
            return self.trace_call_callable(callable, fallback_args);
        };
        let concrete_len = unsafe { w_list_len(concrete_list) } as i64;
        let normalized_key = if concrete_key < 0 {
            concrete_key + concrete_len
        } else {
            concrete_key
        };
        if normalized_key < 0 || normalized_key >= concrete_len {
            return self.trace_call_callable(callable, fallback_args);
        }
        let len_descr_idx = len_descr.index();
        let result = self.with_ctx(|this, ctx| {
            this.guard_class(ctx, list, &LIST_TYPE as *const PyType);
            this.guard_list_strategy(ctx, list, strategy_id);
            // Pre-pop length (cached); `new_len = len - 1` is the
            // reconstructible post-pop length.
            let len = opimpl_getfield_gc_i(ctx, list, len_descr.clone());
            // `descr_pop` shape: unbox the index, normalize a negative one
            // against `len`, and guard `0 <= idx < len`, so the residual's
            // out-of-range panic is unreachable from compiled code.
            let idx_int = this.trace_dynamic_list_index(ctx, index, len, concrete_key);
            // The residual does the real heap mutation (read + O(n) shift +
            // length decrement + tail clear) in compiled code.
            let res = crate::helpers::emit_trace_call_ref_typed(
                ctx,
                pyre_object::listobject::jit_list_pop_at as *const (),
                &[list, idx_int],
                &[Type::Ref, Type::Int],
            );
            // Load-bearing #143 fix: publish the post-pop length as a known
            // value the optimizer/heapcache can carry forward and resume can
            // replay (SetfieldGc lands in pending_fields).
            let one = ctx.const_int(1);
            let new_len = ctx.record_op(OpCode::IntSub, &[len, one]);
            let new_len_value = ctx
                .heapcache_getfield_cached(list, len_descr_idx)
                .and_then(|b| match ctx.box_value(b)? {
                    Value::Int(n) => Some(n),
                    _ => None,
                })
                .map(|n| Value::Int(n.wrapping_sub(1)))
                .unwrap_or(Value::Void);
            if !matches!(new_len_value, Value::Void) {
                ctx.set_opref_concrete(new_len, new_len_value);
            }
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, new_len], len_descr.clone());
            ctx.heapcache_setfield_cached(list, len_descr_idx, new_len);
            res
        });
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    pub(crate) fn concrete_iter_continues(
        &self,
        concrete_iter: PyObjectRef,
    ) -> Result<bool, PyError> {
        range_iter_continues(concrete_iter)
    }

    pub(crate) fn trace_known_builtin_call(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
    ) -> Result<OpRef, PyError> {
        let result = self.with_ctx(|this, ctx| {
            let boxed_args = box_args_for_python_helper(this, ctx, args);
            crate::helpers::emit_trace_call_known_builtin(ctx, callable, &boxed_args)
        })?;
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    pub(crate) fn direct_len_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // guard_class + getfield(length) for str/dict/list/tuple).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_direct_len_value(
                this,
                ctx,
                value,
                concrete_value,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_known_builtin_call(callable, &[value])
    }

    fn direct_abs_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_direct_abs_value(
                this,
                ctx,
                value,
                concrete_value,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_known_builtin_call(callable, &[value])
    }

    fn direct_type_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_direct_type_value(
                this,
                ctx,
                value,
                concrete_value,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_known_builtin_call(callable, &[value])
    }

    fn direct_isinstance_value(
        &mut self,
        callable: OpRef,
        obj: OpRef,
        type_name: OpRef,
        concrete_obj: PyObjectRef,
        concrete_type_name: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_direct_isinstance_value(
                this,
                ctx,
                obj,
                type_name,
                concrete_obj,
                concrete_type_name,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_known_builtin_call(callable, &[obj, type_name])
    }

    fn direct_minmax_value(
        &mut self,
        callable: OpRef,
        a: OpRef,
        b: OpRef,
        choose_max: bool,
        concrete_a: PyObjectRef,
        concrete_b: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_direct_minmax_value(
                this, ctx, a, b, choose_max, concrete_a, concrete_b,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_known_builtin_call(callable, &[a, b])
    }

    pub(crate) fn call_callable_value(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
        concrete_callable: PyObjectRef,
        concrete_args: &[PyObjectRef],
    ) -> Result<OpRef, PyError> {
        if concrete_callable.is_null() {
            debug_assert!(
                false,
                "concrete_callable should always be available during tracing"
            );
            return self.trace_call_callable(callable, args);
        }

        // TODO. In RPython the `lst.append(i)` /
        // `lst.pop()` lowering goes through rtyper helpers in
        // `rpython/rtyper/rlist.py`: `ll_append` (rlist.py:588) is
        // documented "no oopspec — inlined by the JIT", and the pop
        // path picks `ll_pop_nonneg` / `ll_pop_zero` (which carry
        // `oopspec = 'list.pop(l, index)'` / `'list.pop(l, 0)'`) over
        // `ll_pop_default` based on the index sign. By the time the
        // metainterp sees the trace, those helpers have already been
        // inlined or oopspec'd into direct array operations. Pyre's
        // codewriter is `partial` (per `majit/COMPATIBILITY_MATRIX.md`)
        // and does not run that pass, so the analogous specialization
        // happens here at trace recording time instead. Convergence
        // path: port the rtyper/codewriter inlining + oopspec
        // recognition and remove this arm.
        //
        // `baseobjspace::getattr_str` returns a fresh `W_MethodObject` per
        // iteration, so the receiver is pushed by `load_method` as
        // `null_value` (load_method:6334) and call sees
        // `concrete_callable = W_MethodObject`, with the receiver missing
        // from `args`. Recover the receiver via `GetfieldGcR(callable,
        // w_self)` after guarding the method object's class. The function
        // pointer inside the method object IS stable across iterations
        // (unlike the freshly-allocated method-object pointer itself), so
        // a `guard_value` on the `w_function` slot pins the specialization
        // without invalidating on the next iteration's fresh allocation.
        if unsafe { is_method(concrete_callable) } {
            let inner_func = unsafe { w_method_get_func(concrete_callable) };
            let inner_self = unsafe { w_method_get_self(concrete_callable) };
            // `is_builtin_code` gate plus canonical-method identity check:
            // a list subclass overriding `append` or `pop` with a Python
            // `def` would still produce `is_function(inner_func) == true`
            // and may reuse the same name, which must not silently rewire
            // the call to builtin list IR.
            // PyPy's `_Method` dispatch (`baseobjspace.py:1252` →
            // `function.py:566`) just unwraps and calls `w_function`
            // generically, so the user override runs. Restrict the spec
            // arm to builtin-coded functions; Python overrides fall
            // through to `trace_call_callable` below.
            if !inner_func.is_null()
                && !inner_self.is_null()
                && unsafe { is_function(inner_func) }
                && unsafe {
                    is_builtin_code(
                        pyre_interpreter::getcode(inner_func) as pyre_object::PyObjectRef
                    )
                }
                && unsafe { is_list(inner_self) }
            {
                let canonical_list_method = |name: &str| {
                    let list_type = pyre_interpreter::typedef::gettypeobject(&LIST_TYPE);
                    unsafe { pyre_interpreter::lookup_in_type(list_type, name) }
                };
                let recover_self = |this: &mut Self| {
                    // Pin the callable identity before any specialization:
                    // the receiver opref alone does not tie the trace to the
                    // builtin method — the bound method (or the local it was
                    // stored in) can be rebound between loop entries while
                    // the receiver still passes its class/strategy guards.
                    // The method object is freshly allocated per iteration
                    // but its `w_function` slot is stable, so guard on that.
                    this.with_ctx(|this, ctx| {
                        this.guard_class(ctx, callable, &METHOD_TYPE as *const PyType);
                        let func_ref = ctx.record_op_with_descr(
                            majit_ir::OpCode::GetfieldGcR,
                            &[callable],
                            crate::descr::method_w_function_descr(),
                        );
                        this.implement_guard_value(ctx, func_ref, inner_func as i64);
                    });
                    if let Some(existing) =
                        this.with_ctx(|this, ctx| this.existing_ref_for_concrete(ctx, inner_self))
                    {
                        return existing;
                    }
                    this.with_ctx(|this, ctx| {
                        ctx.record_op_with_descr(
                            majit_ir::OpCode::GetfieldGcR,
                            &[callable],
                            crate::descr::method_w_self_descr(),
                        )
                    })
                };
                if args.len() == 1 && canonical_list_method("append") == Some(inner_func) {
                    // Issue #143: the folded fast path's resize guard routes
                    // through a synthetic-callee snapshot
                    // (`generate_guard_with_synthetic_callee`) so a
                    // realloc-driven failure resumes into the `jit_list_append`
                    // helper jitcode and the Python frame continues *after* the
                    // CALL. No CALL re-execution, so the prior
                    // `push_call_replay_stack` workaround (which reconstructed a
                    // NULL method callable on resume) is removed.
                    let self_ref = recover_self(self);
                    self.list_append_value(self_ref, args[0], inner_self, concrete_args[0])?;
                    let none_ref =
                        self.with_ctx(|_this, ctx| ctx.const_ref(pyre_object::w_none() as i64));
                    return Ok(none_ref);
                }
                if args.len() == 1 && canonical_list_method("pop") == Some(inner_func) {
                    // Indexed pop `xs.pop(i)`: recorded inline by
                    // `list_pop_at_value` (residual shift + reconstructible
                    // length overlay). Called directly like the append arm — no
                    // replay scaffolding.
                    let self_ref = recover_self(self);
                    return self.list_pop_at_value(
                        callable,
                        self_ref,
                        args[0],
                        inner_self,
                        concrete_args[0],
                        // Bound method: the receiver is inside `callable`, so
                        // the generic shape is `callable(index)`.
                        &[args[0]],
                    );
                }
                if args.len() == 0 && canonical_list_method("pop") == Some(inner_func) {
                    let call_pc = self.fallthrough_pc.saturating_sub(1);
                    self.with_ctx(|this, ctx| {
                        this.push_call_replay_stack(ctx, callable, args, call_pc)
                    });
                    let self_ref = recover_self(self);
                    let concrete_len = unsafe { w_list_len(inner_self) };
                    let res =
                        self.list_pop_value(callable, self_ref, inner_self, concrete_len, &[]);
                    self.with_ctx(|this, ctx| this.pop_call_replay_stack(ctx, args.len()))?;
                    return res;
                }
                if args.len() == 0 && canonical_list_method("reverse") == Some(inner_func) {
                    let self_ref = recover_self(self);
                    return self.list_reverse_value(callable, self_ref, inner_self);
                }
            }
            if !inner_func.is_null()
                && !inner_self.is_null()
                && unsafe { is_function(inner_func) }
                && unsafe {
                    !is_builtin_code(
                        pyre_interpreter::getcode(inner_func) as pyre_object::PyObjectRef
                    )
                }
            {
                // Do not remove this abort until bound-method replay and
                // guard-failure blackhole resume are aligned. Letting
                // user-defined bound methods fall through to generic tracing
                // currently corrupts `synth/class_attrs_methods` output on
                // both dynasm and cranelift.
                return Err(trace_abort_error(
                    "abort tracing user-defined bound method call",
                ));
            }
        }

        unsafe {
            let is_builtin = is_function(concrete_callable)
                && is_builtin_code(
                    pyre_interpreter::getcode(concrete_callable) as pyre_object::PyObjectRef
                );
            if is_builtin {
                let builtin_name = pyre_interpreter::function_get_name(concrete_callable);
                let canonical_list_method = |name: &str| {
                    let list_type = pyre_interpreter::typedef::gettypeobject(&LIST_TYPE);
                    pyre_interpreter::lookup_in_type(list_type, name)
                };
                if args.len() == 2
                    && canonical_list_method("append") == Some(concrete_callable)
                    && !concrete_args.first().copied().unwrap_or(PY_NULL).is_null()
                    && is_list(concrete_args[0])
                {
                    // Issue #143: the folded fast path's resize guard routes
                    // through a synthetic-callee snapshot
                    // (`generate_guard_with_synthetic_callee`), so a
                    // realloc-driven failure resumes into the `jit_list_append`
                    // helper jitcode and the Python frame continues *after* the
                    // CALL. No CALL re-execution, so the prior
                    // `push_call_replay_stack` workaround — which reconstructed a
                    // NULL callable on resume and inflated `valuestackdepth`,
                    // skewing the synthetic guard's `result_stack_idx` — is
                    // removed.
                    //
                    // Builtin-form arms mark the heap mutation here: the trait
                    // impl's `call_callable` already executed the builtin
                    // concretely before dispatch, so the mutation is a
                    // certainty. The W_Method method-form arms above do NOT
                    // mark — no concrete execution happens on that path.
                    self.dm143_mark_heap_mutated();
                    // Pin the callable identity (as the reverse arm below):
                    // the receiver's class/strategy guards alone do not keep
                    // the trace from running the builtin append after the
                    // callable has been rebound.
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    self.list_append_value(args[0], args[1], concrete_args[0], concrete_args[1])?;
                    let none_ref =
                        self.with_ctx(|_this, ctx| ctx.const_ref(pyre_object::w_none() as i64));
                    return Ok(none_ref);
                }
                if args.len() == 2
                    && canonical_list_method("pop") == Some(concrete_callable)
                    && !concrete_args.first().copied().unwrap_or(PY_NULL).is_null()
                    && is_list(concrete_args[0])
                {
                    // Builtin-form indexed pop `list.pop(xs, i)`: recorded
                    // inline by `list_pop_at_value` (residual shift +
                    // reconstructible length overlay), called directly like the
                    // append arm — no replay scaffolding.
                    self.dm143_mark_heap_mutated();
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self.list_pop_at_value(
                        callable,
                        args[0],
                        args[1],
                        concrete_args[0],
                        concrete_args[1],
                        // Unbound builtin: the receiver is the explicit first
                        // arg, so the generic shape is `list.pop(xs, i)`.
                        &[args[0], args[1]],
                    );
                }
                if args.len() == 1
                    && canonical_list_method("pop") == Some(concrete_callable)
                    && !concrete_args.first().copied().unwrap_or(PY_NULL).is_null()
                    && is_list(concrete_args[0])
                {
                    let concrete_len = w_list_len(concrete_args[0]);
                    // Mark the mutation only once the concrete pop is known to
                    // have removed an element: `call_callable` already ran the
                    // builtin concretely, but on an empty list it raised
                    // without mutating, so `dm143_advance_live_locals` must not
                    // advance across that non-mutating iteration.
                    if concrete_len > 0 {
                        self.dm143_mark_heap_mutated();
                    }
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    let call_pc = self.fallthrough_pc.saturating_sub(1);
                    self.with_ctx(|this, ctx| {
                        this.push_call_replay_stack_self_in_args(
                            ctx,
                            callable,
                            ConcreteValue::Ref(concrete_callable),
                            args,
                            call_pc,
                        )
                    });
                    let res = self.list_pop_value(
                        callable,
                        args[0],
                        concrete_args[0],
                        concrete_len,
                        // Unbound builtin: re-supply the receiver so an empty
                        // list records `list.pop(xs)` (IndexError), not
                        // `list.pop()` (arity TypeError).
                        &[args[0]],
                    );
                    self.with_ctx(|this, ctx| {
                        this.pop_call_replay_stack_self_in_args(ctx, args.len())
                    })?;
                    return res;
                }
                if args.len() == 1
                    && canonical_list_method("reverse") == Some(concrete_callable)
                    && !concrete_args.first().copied().unwrap_or(PY_NULL).is_null()
                    && is_list(concrete_args[0])
                {
                    self.dm143_mark_heap_mutated();
                    let call_pc = self.fallthrough_pc.saturating_sub(1);
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64);
                        this.push_call_replay_stack_self_in_args(
                            ctx,
                            callable,
                            ConcreteValue::Ref(concrete_callable),
                            args,
                            call_pc,
                        );
                    });
                    let res = self.list_reverse_value(callable, args[0], concrete_args[0]);
                    self.with_ctx(|this, ctx| {
                        this.pop_call_replay_stack_self_in_args(ctx, args.len())
                    })?;
                    return res;
                }
                if args.len() == 1 {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    if builtin_name == "type" {
                        return self.direct_type_value(callable, args[0], c_arg0);
                    }
                    if builtin_name == "len" {
                        return self.direct_len_value(callable, args[0], c_arg0);
                    }
                    if builtin_name == "abs" {
                        return self.direct_abs_value(callable, args[0], c_arg0);
                    }
                } else if args.len() == 2 && builtin_name == "isinstance" {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    let c_arg1 = concrete_args.get(1).copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self
                        .direct_isinstance_value(callable, args[0], args[1], c_arg0, c_arg1);
                } else if args.len() == 2 && builtin_name == "min" {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    let c_arg1 = concrete_args.get(1).copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self
                        .direct_minmax_value(callable, args[0], args[1], false, c_arg0, c_arg1);
                } else if args.len() == 2 && builtin_name == "max" {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    let c_arg1 = concrete_args.get(1).copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self
                        .direct_minmax_value(callable, args[0], args[1], true, c_arg0, c_arg1);
                }
                let result = self.with_ctx(|this, ctx| {
                    this.implement_guard_value(ctx, callable, concrete_callable as i64);
                    let boxed_args = box_args_for_python_helper(this, ctx, args);
                    crate::helpers::emit_trace_call_known_builtin(ctx, callable, &boxed_args)
                })?;
                self.trace_record_no_exception_guard();
                return Ok(result);
            }
            if is_function(concrete_callable) {
                let w_callee_code = pyre_interpreter::getcode(concrete_callable);
                let callee_key = crate::driver::make_green_key(w_callee_code, 0);
                // pyjitpl.py:1396-1401 element-wise greenkey: pyre's
                // tuple is `(code_ptr, pc)` so structural equality cannot
                // be fooled by hash collisions the way the derived u64
                // `callee_key` can (driver.rs:12 make_green_key).
                let callee_raw: (usize, usize) = (w_callee_code as usize, 0);
                let caller_raw: (usize, usize) = ((*self.sym().jitcode).code as usize, 0);
                let callee_code =
                    &*(pyre_interpreter::w_code_get_ptr(w_callee_code as pyre_object::PyObjectRef)
                        as *const CodeObject);
                let (driver, _) = crate::driver::driver_pair();
                let nargs = args.len();

                // RPython pyjitpl.py: do_residual_or_indirect_call() follows
                // direct jitcode calls via perform_call() before falling back
                // to residual helpers.  Mirror that for ordinary direct calls:
                // if we know the callee body and it is a small acyclic helper,
                // trace through it directly instead of waiting for
                // should_inline() to bless a helper-boundary inline.
                let is_self_recursive = callee_raw == caller_raw;
                let inline_decision = driver.should_inline(callee_key, callee_raw);
                let inline_framestack_active = !self.parent_frames.is_empty();
                let callee_inline_eligible = driver
                    .meta_interp()
                    .warm_state_ref()
                    .can_inline_callable(callee_key);
                let max_unroll_recursion =
                    driver.meta_interp().warm_state_ref().max_unroll_recursion() as usize;
                let recursive_depth = self.with_ctx(|_, ctx| ctx.recursive_depth(callee_raw));
                let concrete_arg0 = if nargs == 1 {
                    concrete_args.first().copied()
                } else {
                    None
                };
                // pyjitpl.py:1388-1402 element-wise greenkey walk:
                //   count = 0
                //   for f in self.metainterp.framestack:
                //       if f.jitcode is not portal_code: continue
                //       gk = f.greenkey
                //       if gk is None: continue
                //       for i in range(len(gk)):
                //           if not gk[i].same_constant(greenboxes[i]): break
                //       else: count += 1
                //
                // Pyre's greenkey is `(code_ptr, target_pc)`; matching the
                // tuple implies `f.jitcode is portal_code`. With
                // `is_recursive=True` on the single PyPyJitDriver every
                // framestack entry is a portal frame, so the upstream
                // filter is automatically satisfied and the count equals
                // `ctx.recursive_depth(callee_key)`: only already-inlined
                // portal frames count, matching PyPy's root frame whose
                // `greenkey` is None.
                let recursive_count = recursive_depth;
                let recursion_exceeded =
                    callee_inline_eligible && recursive_count >= max_unroll_recursion;

                // pyjitpl.py:1413 warmrunnerstate.dont_trace_here(greenboxes).
                // Upstream calls this unconditionally when the recursion
                // limit is hit on an inlinable callee; pyre's warm-state
                // equivalent sets the DONT_TRACE_HERE jitcell flag via
                // `disable_noninlinable_function` (majit-trace
                // warmstate.rs:931). Fires for self-recursion *and*
                // mutual recursion because upstream's count walk does
                // not distinguish them.
                if recursion_exceeded {
                    driver
                        .meta_interp_mut()
                        .warm_state_mut()
                        .disable_noninlinable_function(callee_key);
                }

                // pyjitpl.py:1376-1423 _opimpl_recursive_call: compute
                // assembler_call boolean.  While recursive depth stays under
                // `max_unroll_recursion`, InlineDecision::Inline takes the
                // perform_call path below; CALL_ASSEMBLER is reserved for
                // the same fall-through cases as PyPy.
                let assembler_call = if is_self_recursive
                    && inline_decision == majit_metainterp::InlineDecision::Inline
                    && !recursion_exceeded
                {
                    driver
                        .get_loop_token_number(callee_key)
                        .or_else(|| driver.get_pending_token_number(callee_key))
                        .is_some()
                } else {
                    // pyjitpl.py:1417 `assembler_call = True` after the
                    // inlining path falls through (either dont_trace_here
                    // fired or can_inline_callable was False). The
                    // `CallAssembler` variant covers both subcases.
                    recursion_exceeded
                        || inline_decision == majit_metainterp::InlineDecision::CallAssembler
                };

                // pyjitpl.py:1414-1416 / pyjitpl.py:2174-2186 parity:
                // perform_call when callee is inlinable and the recursion
                // count is below `max_unroll_recursion`.  PyPy does not
                // special-case self-recursive calls here; the framestack
                // depth check is the gate.  RPython places no `nargs` or
                // callee-loop constraint on the inline decision; the
                // resulting trace length is bounded by `trace_limit`
                // (rlib/jit.py:592) and `max_unroll_recursion`.
                let can_trace_through = callee_inline_eligible && !recursion_exceeded;

                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][direct-call] key={} nargs={} inline_eligible={} self_recursive={} recursive_depth={} max_unroll={} assembler_call={} can_trace_through={}",
                        callee_key,
                        nargs,
                        callee_inline_eligible,
                        is_self_recursive,
                        recursive_depth,
                        max_unroll_recursion,
                        assembler_call,
                        can_trace_through,
                    );
                }

                if can_trace_through {
                    // pyjitpl.py:1414-1416 perform_call parity.
                    match self.build_pending_inline_frame(
                        callable,
                        args,
                        concrete_callable,
                        callee_key,
                        concrete_args,
                    ) {
                        Ok(pending) => {
                            self.pending_inline_frame = Some(pending);
                            return self
                                .with_ctx(|_, ctx| Ok(ctx.const_ref(pyre_object::PY_NULL as i64)));
                        }
                        Err(err) => {
                            if majit_metainterp::majit_log_enabled() {
                                eprintln!(
                                    "[jit][perform-call] build_pending failed key={} err={}, residual path",
                                    callee_key, err
                                );
                            }
                            // Fall through to residual helper path
                        }
                    }
                }

                // pyjitpl.py:1422 do_recursive_call / do_residual_call:
                // assembler_call is already computed above. The Inline
                // path passes it to inline_function_call which gates
                // CALL_ASSEMBLER vs CALL_MAY_FORCE on the boolean.
                // Non-Inline paths (pending_token, CallAssembler match)
                // handle their own CALL_ASSEMBLER emission.
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][call-check] is_self={} cache_safe={} inline_active={} callee_key={}",
                        is_self_recursive,
                        (crate::callbacks::get().recursive_force_cache_safe)(concrete_callable),
                        inline_framestack_active,
                        callee_key
                    );
                }

                if inline_decision == majit_metainterp::InlineDecision::Inline {
                    if let Some(frame_helper) = (crate::callbacks::get().callee_frame_helper)(nargs)
                    {
                        return self.inline_function_call(
                            callable,
                            args,
                            concrete_callable,
                            callee_key,
                            callee_raw,
                            frame_helper,
                            concrete_args,
                            assembler_call,
                        );
                    }
                }

                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][call-dispatch] callee_key={} pending_token={:?} loop_token={:?} is_self={}",
                        callee_key,
                        driver.get_pending_token_number(callee_key),
                        driver.get_loop_token_number(callee_key),
                        is_self_recursive
                    );
                }
                if let Some(token_number) = driver.get_pending_token_number(callee_key) {
                    if nargs == 1 || (crate::callbacks::get().callee_frame_helper)(nargs).is_some()
                    {
                        let call_pc = self.fallthrough_pc.saturating_sub(1);
                        return self.with_ctx(|this, ctx| {
                            if !is_self_recursive {
                                this.implement_guard_value(ctx, callable, concrete_callable as i64);
                            }
                            let self_recursive_raw_arg = if is_self_recursive
                                && nargs == 1
                                && matches!(concrete_arg0, Some(arg) if is_int(arg))
                            {
                                Some(this.trace_guarded_int_payload(ctx, args[0]))
                            } else {
                                None
                            };
                            let (callee_frame, drop_callee_frame) =
                                emit_call_assembler_callee_frame(
                                    this,
                                    ctx,
                                    callable,
                                    args,
                                    concrete_callable,
                                    w_callee_code,
                                    callee_code,
                                    is_self_recursive,
                                    self_recursive_raw_arg,
                                )?;
                            // pyjitpl.py:2017: do_residual_call step 1
                            this.vable_and_vrefs_before_residual_call(ctx);
                            let ec = this.ensure_execution_context(ctx);
                            let ca_result = ctx.call_assembler_red_only_ref(
                                token_number,
                                &[callee_frame, ec],
                                &[Type::Ref, Type::Ref],
                            );
                            // pyjitpl.py:2080-2081 direct_assembler_call:
                            // record KEEPALIVE on callee virtualizable so
                            // it survives until the result is consumed.
                            ctx.record_op(OpCode::Keepalive, &[callee_frame]);
                            if drop_callee_frame {
                                // Only the opaque arena-helper path needs the
                                // explicit drop. Trace-visible PyFrames are
                                // GC-owned and must not go through arena.put.
                                ctx.call_void(
                                    crate::callbacks::get().jit_drop_callee_frame,
                                    &[callee_frame],
                                );
                            }
                            // pyjitpl.py:2049
                            this.vrefs_after_residual_call(ctx);
                            // pyjitpl.py:2078
                            this.vable_after_residual_call()?;
                            // pyjitpl.py:2079
                            this.push_call_replay_stack(ctx, callable, args, call_pc);
                            this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                            this.generate_guard(ctx, OpCode::GuardNoException, &[]);
                            ctx.heap_cache_mut().invalidate_caches_for_escaped();
                            this.pop_call_replay_stack(ctx, args.len())?;
                            let result = if inline_framestack_active {
                                ca_result // already Ref — no unbox+rebox needed
                            } else {
                                // Caller unboxes: guard_class + getfield_gc_i
                                this.trace_guarded_int_payload(ctx, ca_result)
                            };
                            Ok(result)
                        });
                    }
                }

                match inline_decision {
                    majit_metainterp::InlineDecision::CallAssembler => {
                        // Trace-through: inline callee body instead of CallAssembler.
                        // Guards use parent_frames to avoid OpRef::NONE in fail_args.
                        // RPython's _opimpl_recursive_call (pyjitpl.py:1376-
                        // 1423) does not apply `nargs` or callee-loop gates
                        // here; the assembler-call fall-through gates only
                        // on `can_inline_callable` + recursion count.
                        if callee_inline_eligible && !is_self_recursive {
                            match self.build_pending_inline_frame(
                                callable,
                                args,
                                concrete_callable,
                                callee_key,
                                concrete_args,
                            ) {
                                Ok(pending) => {
                                    self.pending_inline_frame = Some(pending);
                                    return self.with_ctx(|_, ctx| {
                                        Ok(ctx.const_ref(pyre_object::PY_NULL as i64))
                                    });
                                }
                                Err(err) => {
                                    if majit_metainterp::majit_log_enabled() {
                                        eprintln!(
                                            "[jit][perform-call] call-assembler inline failed key={} err={}",
                                            callee_key, err
                                        );
                                    }
                                }
                            }
                        }
                        // pyjitpl.py:1417 `assembler_call = True` route:
                        // `should_inline_core` already accepts both compiled
                        // and pending tokens (`callee_compiled` predicate),
                        // but the emit path here requires the callee to be
                        // fully compiled — `compiled_loops[callee_key]` must
                        // resolve, and the descr we build threads through
                        // `make_call_assembler_descr_by_number`'s number
                        // factory which then keys back to compiled_loops.
                        // Including the pending-token slot here regresses
                        // against main: when the callee is mid-compilation
                        // (pending only), `get_compiled_meta` returns None
                        // and the downstream consumers fail.  RPython's
                        // `get_assembler_token` (warmstate.py:714) handles
                        // the pending case by synthesising a
                        // `compile_tmp_callback` token; that path is not
                        // yet ported.  Until it is, mirror
                        // main and gate strictly on compiled presence.
                        let Some(token_number) = driver.get_loop_token_number(callee_key) else {
                            let call_pc = self.fallthrough_pc.saturating_sub(1);
                            return self.with_ctx(|this, ctx| {
                                this.implement_guard_value(ctx, callable, concrete_callable as i64);
                                let result = crate::helpers::emit_trace_call_known_function(
                                    ctx,
                                    this.frame(),
                                    callable,
                                    args,
                                )?;
                                this.push_call_replay_stack(ctx, callable, args, call_pc);
                                this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                                this.generate_guard(ctx, OpCode::GuardNoException, &[]);
                                ctx.heap_cache_mut().invalidate_caches_for_escaped();
                                this.pop_call_replay_stack(ctx, args.len())?;
                                Ok(result)
                            });
                        };

                        {
                            let call_pc = self.fallthrough_pc.saturating_sub(1);
                            return self.with_ctx(|this, ctx| {
                                // Self-recursive: no callable guard needed (same function).
                                // Non-self-recursive: guard on callable value.
                                if !is_self_recursive {
                                    this.implement_guard_value(
                                        ctx,
                                        callable,
                                        concrete_callable as i64,
                                    );
                                }
                                let self_recursive_raw_arg = if is_self_recursive
                                    && args.len() == 1
                                    && matches!(concrete_arg0, Some(arg) if is_int(arg))
                                {
                                    Some(this.trace_guarded_int_payload(ctx, args[0]))
                                } else {
                                    None
                                };
                                let (callee_frame, drop_callee_frame) =
                                    emit_call_assembler_callee_frame(
                                        this,
                                        ctx,
                                        callable,
                                        args,
                                        concrete_callable,
                                        w_callee_code,
                                        callee_code,
                                        is_self_recursive,
                                        self_recursive_raw_arg,
                                    )?;

                                // pyjitpl.py:2017: do_residual_call step 1
                                this.vable_and_vrefs_before_residual_call(ctx);
                                let ec = this.ensure_execution_context(ctx);
                                let ca_result = ctx.call_assembler_red_only_ref(
                                    token_number,
                                    &[callee_frame, ec],
                                    &[Type::Ref, Type::Ref],
                                );
                                ctx.record_op(OpCode::Keepalive, &[callee_frame]);
                                if drop_callee_frame {
                                    ctx.call_void(
                                        crate::callbacks::get().jit_drop_callee_frame,
                                        &[callee_frame],
                                    );
                                }
                                // pyjitpl.py:2049
                                this.vrefs_after_residual_call(ctx);
                                // pyjitpl.py:2078
                                this.vable_after_residual_call()?;
                                // pyjitpl.py:2079
                                this.push_call_replay_stack(ctx, callable, args, call_pc);
                                this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                                this.generate_guard(ctx, OpCode::GuardNoException, &[]);
                                ctx.heap_cache_mut().invalidate_caches_for_escaped();
                                this.pop_call_replay_stack(ctx, args.len())?;
                                let result = if inline_framestack_active {
                                    ca_result // already Ref
                                } else {
                                    // Caller unboxes: guard_class + getfield_gc_i
                                    this.trace_guarded_int_payload(ctx, ca_result)
                                };
                                Ok(result)
                            });
                        }
                    }
                    // Inline is handled at the top of the dispatch (line 2763).
                    // If we reach here, either frame_helper was unavailable
                    // or inline_decision changed; fall to residual below.
                    majit_metainterp::InlineDecision::Inline
                    | majit_metainterp::InlineDecision::ResidualCall => {}
                }

                let call_pc = self.fallthrough_pc.saturating_sub(1);
                return self.with_ctx(|this, ctx| {
                    this.implement_guard_value(ctx, callable, concrete_callable as i64);
                    let boxed_args = box_args_for_python_helper(this, ctx, args);
                    let result = crate::helpers::emit_trace_call_known_function(
                        ctx,
                        this.frame(),
                        callable,
                        &boxed_args,
                    )?;
                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                    this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                    this.generate_guard(ctx, OpCode::GuardNoException, &[]);
                    ctx.heap_cache_mut().invalidate_caches_for_escaped();
                    this.pop_call_replay_stack(ctx, args.len())?;
                    Ok(result)
                });
            }
        }

        if let Some(new_op) =
            self.try_trace_exception_new(callable, args, concrete_callable, concrete_args)?
        {
            return Ok(new_op);
        }

        self.trace_call_callable(callable, args)
    }

    /// `Type(args)` for a *canonical* builtin exception class: emit the
    /// allocation as traced `NewWithVtable` + `SetfieldGc` (kind /
    /// w_class / args_w) so the optimizer can virtualize it when the
    /// exception never escapes, instead of the opaque residual
    /// `jit_call_callable_N` constructor call
    /// (`helpers::emit_trace_call_callable`).
    ///
    /// The GC rewrite pass lowers the `NewWithVtable` to a nursery
    /// allocation carrying the `W_EXCEPTION` type id + per-kind vtable
    /// (`rewrite.rs gen_malloc_nursery` / `gen_initialize_tid` /
    /// `gen_initialize_vtable`), so the result is a fully GC-managed
    /// `W_ExceptionObject` identical to the runtime `malloc_typed` +
    /// `exc_new_wrapper` + `descr_init` path.  `args_w` is built inline
    /// (`emit_exception_args_list_inline`) when `w_list_new` would pick
    /// the Object strategy, so the args list virtualizes too; Empty /
    /// Integer / Float strategies fall back to a residual list.
    ///
    /// Restricted to a callable that is exactly
    /// `lookup_exc_class_for_kind(kind)`: user subclasses (whose ctor may
    /// run a Python `__init__`, and whose `w_class` differs from the
    /// per-kind builtin class) fall through to the generic call path.
    fn try_trace_exception_new(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
        concrete_callable: PyObjectRef,
        concrete_args: &[PyObjectRef],
    ) -> Result<Option<OpRef>, PyError> {
        let is_exc_class = unsafe {
            pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(concrete_callable)
        };
        // Concrete positional args only — the residual `args_w` list must
        // match the runtime `descr_init` list exactly.
        if !is_exc_class || concrete_args.iter().any(|a| a.is_null()) {
            return Ok(None);
        }
        // Reject user subclasses before the probe construction below:
        // their Python `__init__` would run concretely an extra time per
        // trace attempt (a user-visible side effect on top of the real
        // execution).  Canonical per-kind classes have a pure Rust
        // `descr_init`, so probing them is unobservable.
        if !pyre_object::excobject::is_canonical_exc_class(concrete_callable) {
            return Ok(None);
        }
        // Build the exception concretely on the plain eval loop (no tracer
        // re-entry) to read its kind and confirm a flat builtin instance.
        // The instance is trace-time only and is discarded after the read.
        let exc = {
            let _plain_guard = pyre_interpreter::call::force_plain_eval();
            pyre_interpreter::call::call_function_impl_result(concrete_callable, concrete_args)
        };
        let Ok(exc) = exc else { return Ok(None) };
        let kind = unsafe {
            if !pyre_object::is_exception(exc) {
                return Ok(None);
            }
            pyre_object::excobject::w_exception_get_kind(exc)
        };
        // Only the canonical per-kind builtin class maps to the flat
        // NewWithVtable layout; a user subclass resolves to its builtin
        // parent here and is rejected.
        if pyre_object::excobject::lookup_exc_class_for_kind(kind) != concrete_callable {
            return Ok(None);
        }
        // The inline constructor reproduces only kind / w_class / args_w.
        // Kinds whose descr_init stores extra fields (OSError errno /
        // strerror / filename; the Unicode errors' object / start / end /
        // reason / encoding — and OSError's args_w rewrite) cannot be
        // rebuilt from those three alone, so defer them to the full
        // runtime constructor via the residual call path.
        if !kind.has_trivial_args_constructor() {
            return Ok(None);
        }
        // Pin the callable identity so the trace-time kind / vtable stay
        // valid across iterations.
        self.with_ctx(|this, ctx| {
            this.implement_guard_value(ctx, callable, concrete_callable as i64);
        });
        // Build `args_w` inline when `w_list_new` would pick the Object
        // strategy (non-empty, mixed/non-numeric) — then the whole list
        // (W_ListObject + ItemsBlock) virtualizes alongside the exception.
        // Empty / Integer / Float strategies fall back to the residual
        // `trace_build_list`, which builds the matching typed storage.
        let args_list = if pyre_object::listobject::list_strategy_for(concrete_args)
            == pyre_object::listobject::ListStrategy::Object
        {
            self.with_ctx(|_this, ctx| crate::helpers::emit_exception_args_list_inline(ctx, args))
        } else {
            TraceHelperAccess::trace_build_list(self, args)?
        };
        let new_op = self.with_ctx(|_this, ctx| {
            crate::helpers::emit_exception_new_inline(ctx, kind, callable, args_list)
        });
        // E1: record the fresh instance keyed by the New op so a following
        // RAISE_VARARGS can recover the concrete exception (the type-call
        // result carries concrete=Null) and take the instance fast path —
        // skip the residual publish + GUARD_EXCEPTION so the exception
        // stays virtualizable.  Trace-time only; the runtime value is the
        // per-iteration New op.
        self.sym_mut().trace_built_exc.insert(new_op, exc);
        Ok(Some(new_op))
    }

    fn build_pending_inline_frame(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
        concrete_callable: PyObjectRef,
        callee_key: u64,
        passed_concrete_args: &[PyObjectRef],
    ) -> Result<PendingInlineFrame, PyError> {
        use pyre_interpreter::pyframe::PyFrame;

        self.with_ctx(|this, ctx| {
            this.implement_guard_value(ctx, callable, concrete_callable as i64);
        });

        for (_idx, arg) in passed_concrete_args.iter().copied().enumerate() {
            if arg.is_null() {
                return Err(PyError::type_error(
                    "pending inline frame lost concrete arg",
                ));
            }
        }

        let caller_code = unsafe { (*self.sym().jitcode).code };
        let caller_exec_ctx = self.sym().concrete_execution_context;
        let caller_namespace_ptr = self.sym().concrete_namespace;
        let w_code = unsafe { pyre_interpreter::getcode(concrete_callable) };
        let globals = unsafe { function_get_globals(concrete_callable) };
        let callee_globals_obj = unsafe { function_get_globals_obj(concrete_callable) };
        let closure = unsafe { pyre_interpreter::function_get_closure(concrete_callable) };
        // pyjitpl.py:1396-1401 element-wise greenkey — `(code_ptr, 0)`
        // tuple equality is lossless vs the derived u64 hash.
        let is_self_recursive = caller_code as usize == w_code as usize;
        let concrete_args = fill_positional_defaults_for_trace_call(
            concrete_callable,
            unsafe {
                &*pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef).cast::<CodeObject>()
            },
            passed_concrete_args,
        );
        let concrete_args = concrete_args.as_ref();
        let mut callee_frame = PyFrame::try_new_for_call_with_closure_and_globals_obj(
            w_code,
            concrete_args,
            globals,
            callee_globals_obj,
            caller_exec_ctx,
            closure,
        )?;
        callee_frame.fix_array_ptrs();

        let callee_code = unsafe { &*pyre_interpreter::pyframe_get_pycode(&callee_frame) };
        // pyframe.py:111: nlocals + ncellvars + nfreevars = stack base
        let callee_nlocals =
            callee_code.varnames.len() + pyre_interpreter::pyframe::ncells(callee_code);
        let caller_namespace = caller_namespace_ptr;
        let can_skip_traced_callee_frame = !is_self_recursive
            && callee_globals_obj == caller_namespace
            && concrete_args.len() == args.len()
            && callee_nlocals == args.len();

        let (callee_sym, drop_frame_opref) = if can_skip_traced_callee_frame {
            let frame = self.frame();
            let mut sym = PyreSym::new_uninit(frame);
            sym.nlocals = callee_nlocals;
            sym.valuestackdepth = callee_nlocals;
            sym.registers_r = args.to_vec();
            sym.symbolic_local_types = args.iter().map(|&arg| self.value_type(arg)).collect();
            sym.symbolic_stack_types = Vec::new();
            // MIFrame Box tracking: set concrete metadata for callee
            sym.concrete_locals = concrete_args
                .iter()
                .map(|&a| ConcreteValue::from_pyobj(a))
                .collect();
            sym.concrete_locals
                .resize(callee_nlocals, ConcreteValue::Null);
            sym.concrete_stack = Vec::new();
            sym.jitcode = jitcode_for(w_code);
            sym.concrete_namespace = callee_globals_obj;
            sym.concrete_execution_context = self.sym().concrete_execution_context;
            let (
                vable_last_instr,
                vable_pycode,
                vable_valuestackdepth,
                vable_debugdata,
                vable_lastblock,
                vable_w_globals,
            ) = self.with_ctx(|_, ctx| {
                // pyjitpl.py:74-90 MIFrame.setup parity for the per-kind banks
                // (including pyjitpl.py:97-119 copy_constants).
                sym.setup_kind_register_banks(ctx);
                let null = ctx.const_ref(pyre_object::PY_NULL as i64);
                (
                    ctx.const_int(0),
                    ctx.const_ref(w_code as i64),
                    ctx.const_int(callee_nlocals as i64),
                    null, // debugdata = None
                    null, // lastblock = None
                    // pyframe.py:128 self.w_globals is the dict OBJECT; the
                    // vable slot is PYFRAME_W_GLOBALS_OBJ_OFFSET, so seed the
                    // W_DictObject sibling, not the raw DictStorage*.
                    ctx.const_ref(callee_globals_obj as i64),
                )
            });
            sym.vable_last_instr = vable_last_instr;
            sym.vable_pycode = vable_pycode;
            sym.vable_valuestackdepth = vable_valuestackdepth;
            sym.vable_debugdata = vable_debugdata;
            sym.vable_lastblock = vable_lastblock;
            sym.vable_w_globals = vable_w_globals;
            (sym, None)
        } else {
            let default_oprefs: Vec<OpRef> = if concrete_args.len() > args.len() {
                self.with_ctx(|_, ctx| {
                    Ok::<_, PyError>(
                        concrete_args[args.len()..]
                            .iter()
                            .map(|&default| ctx.const_ref(default as i64))
                            .collect(),
                    )
                })?
            } else {
                Vec::new()
            };
            let mut frame_args = args.to_vec();
            frame_args.extend_from_slice(&default_oprefs);
            if !default_oprefs.is_empty() {
                let expected_defaults = unsafe { function_get_defaults(concrete_callable) };
                self.with_ctx(|this, ctx| {
                    let defaults = ctx.call_ref_typed_with_effect(
                        trace_function_get_defaults as *const (),
                        &[callable],
                        &[Type::Ref],
                        CANNOT_RAISE_NO_HEAP_EFFECT_INFO.clone(),
                    );
                    this.implement_guard_value(ctx, defaults, expected_defaults as i64);
                    Ok::<_, PyError>(())
                })?;
            }
            // Create symbolic OpRef for callee frame in trace
            let callee_frame_opref = self.with_ctx(|this, ctx| {
                if frame_args.len() == 1 {
                    let (helper, helper_arg_types) = one_arg_callee_frame_helper(
                        this.value_type(frame_args[0]),
                        is_self_recursive,
                    );
                    if is_self_recursive {
                        ctx.call_ref_typed_with_effect(
                            helper,
                            &[this.frame(), frame_args[0]],
                            &helper_arg_types,
                            default_effect_info(),
                        )
                    } else {
                        ctx.call_ref_typed_with_effect(
                            helper,
                            &[this.frame(), callable, frame_args[0]],
                            &helper_arg_types,
                            default_effect_info(),
                        )
                    }
                } else if let Some(frame_helper) =
                    (crate::callbacks::get().callee_frame_helper)(frame_args.len())
                {
                    let mut helper_args = vec![this.frame(), callable];
                    helper_args.extend_from_slice(&frame_args);
                    let helper_arg_types = frame_callable_arg_types(frame_args.len());
                    ctx.call_ref_typed_with_effect(
                        frame_helper,
                        &helper_args,
                        &helper_arg_types,
                        default_effect_info(),
                    )
                } else {
                    panic!("no frame helper for {} args", frame_args.len());
                }
            });
            // PyPy `recursive_call_*` emits GUARD_NO_EXCEPTION right after
            // the callee-frame build helper (pyjitpl.py:2106).
            self.trace_record_no_exception_guard();

            let mut sym = PyreSym::new_uninit(callee_frame_opref);
            sym.nlocals = callee_nlocals;
            sym.valuestackdepth = sym.nlocals;
            sym.registers_r = Vec::with_capacity(sym.nlocals);
            sym.symbolic_local_types = Vec::with_capacity(sym.nlocals);
            for i in 0..sym.nlocals {
                if i < frame_args.len() {
                    sym.registers_r.push(frame_args[i]);
                    sym.symbolic_local_types
                        .push(self.value_type(frame_args[i]));
                } else {
                    sym.registers_r.push(OpRef::NONE);
                    sym.symbolic_local_types.push(Type::Ref);
                }
            }
            sym.symbolic_stack_types = Vec::new();
            // MIFrame Box tracking: set concrete metadata for callee
            sym.concrete_locals = concrete_args
                .iter()
                .map(|&a| ConcreteValue::from_pyobj(a))
                .collect();
            sym.concrete_locals
                .resize(callee_nlocals, ConcreteValue::Null);
            sym.concrete_stack = Vec::new();
            sym.jitcode = jitcode_for(w_code);
            // pyjitpl.py:74-90 MIFrame.setup parity for the per-kind banks
            // (including pyjitpl.py:97-119 copy_constants).
            //
            // pyjitpl.py:1230 `_opimpl_getarrayitem_vable`: every MIFrame
            // accesses the metainterp-scope `virtualizable_boxes` cache
            // (already provided by `TraceCtx::virtualizable_boxes` /
            // `ctx.virtualizable_box_at`) only when its own bytecode
            // emits a vable opcode — i.e. when this MIFrame is the
            // active vable owner. For inlined callee frames, vable_*
            // opcodes are not emitted; locals are accessed via
            // `metainterp.framestack[i].registers_r` (in pyre this is
            // `sym.registers_r`) or, for slots that have not yet been
            // populated symbolically (cell / nested-scope slots that
            // arrive pre-populated on the heap), via the callee's own
            // `locals_cells_stack_w` array.  Seed
            // `locals_cells_stack_array_ref` to that array OpRef so the
            // heap fall-through in `load_local_value` /
            // `push_typed_value` / `pop_value` reads from the callee's
            // locals_cells_stack_w (parity with init_symbolic's non-vable
            // branch in state.rs:2540-2543), not from the unset OpRef::NONE
            // that emitted invalid `getarrayitem(NONE, idx)` IR before.
            self.with_ctx(|_, ctx| {
                sym.setup_kind_register_banks(ctx);
                sym.locals_cells_stack_array_ref =
                    frame_locals_cells_stack_array(ctx, callee_frame_opref);
            });
            sym.concrete_namespace = callee_globals_obj;
            sym.concrete_execution_context = self.sym().concrete_execution_context;
            (sym, Some(callee_frame_opref))
        };

        // pyjitpl.py:2601-2602: parent frame keeps its original pc
        // (return_point_pc = CALL fallthrough). The callee blackhole
        // handles the call; the caller continues AFTER the call returns.
        // Stack is post-dispatch (args consumed, no result yet).
        // last_instr = return_point_pc - 1 so the caller's next_instr()
        // returns the fallthrough PC when the blackhole restores the frame.
        //
        // Propagation gap #2: mirror the parent's
        // sym.vable_last_instr + sym.vable_valuestackdepth into the
        // virtualizable_boxes shadow when this sym owns it. Without
        // propagation, the heap still carries the caller's pre-call PC
        // and pre-call vsd, and once the writeback gate flips on, the
        // patched-parent loop's GETFIELD preamble re-loads the stale
        // values. virtualizable_boxes[0] = vable_last_instr,
        // [2] = vable_valuestackdepth per virtualizable_gen.rs:37-44.
        let return_point_pc = self.fallthrough_pc;
        self.with_ctx(|this, ctx| {
            let last_instr_value = return_point_pc as i64 - 1;
            let ni = ctx.const_int(last_instr_value);
            let vsd_value = this.sym().valuestackdepth as i64;
            let vsd = ctx.const_int(vsd_value);
            let owns = {
                let s = this.sym_mut();
                s.vable_last_instr = ni;
                s.vable_valuestackdepth = vsd;
                s.owns_virtualizable_shadow()
            };
            if owns {
                mirror_vable_static_to_boxes(ctx, "last_instr", ni, Value::Int(last_instr_value));
                mirror_vable_static_to_boxes(ctx, "valuestackdepth", vsd, Value::Int(vsd_value));
            }
        });
        // opencoder.py:819-834 parity: accumulate full parent chain.
        // Current frame becomes the newest parent; existing parents follow.
        let mut parent_frames = vec![ResumeFrameState {
            sym: self.sym,
            concrete_frame_addr: self.concrete_frame_addr,
            resume_pc: return_point_pc,
            // The caller's CALL pc — its post-call `-live-`/`catch_exception`
            // (keyed by this pc in `after_residual_call_resume_pc`) is where
            // the blackhole must resume this frame if a guard deopts inside
            // the callee and the exception unwinds to a handler here
            // (`pyjitpl.py:2601-2602`).  `orgpc` is the CALL opcode start.
            call_pc: Some(self.orgpc),
            // `metainterp::push_inline_frame` overwrites this with the
            // caller's just-computed result stack idx (`pyjitpl.py:181-193`
            // parity).
            pending_result_stack_idx: None,
            pending_result_type: None,
        }];
        parent_frames.extend(self.parent_frames.iter().cloned());
        Ok(PendingInlineFrame {
            sym: callee_sym,
            concrete_frame: callee_frame,
            drop_frame_opref,
            green_key: callee_key,
            // Raw greenkey pair for element-wise recursion comparison
            // (pyjitpl.py:1396-1401). target_pc is 0 for function
            // entries — matches `make_green_key(w_code, 0)`. `w_code`
            // and `callee_key` are the same greenkey inputs that built
            // `callee_key` near the top of this function.
            green_key_raw: (w_code as usize, 0),
            parent_frames,
            nargs: args.len(),
            caller_result_stack_idx: None,
            caller_result_type: Some(Type::Ref),
            replay_callable: callable,
            replay_args: args.to_vec(),
        })
    }

    /// pyjitpl.py:1425-1432 do_recursive_call(assembler_call=True) +
    /// pyjitpl.py:3613-3635 direct_assembler_call, invoked on the PARENT
    /// frame after the inline callee was popped at its loop back-edge
    /// (opimpl_jit_merge_point portal_call_depth>0, pyjitpl.py:1579-1602).
    ///
    /// Records CALL_ASSEMBLER into the callee loop's compiled token with
    /// `[callee_frame, ec]` red args (interp_jit.py:67 reds), bracketed by
    /// the same vable/vref + GuardNotForced / GuardNoException sequence as
    /// the residual-call emission (do_residual_call, pyjitpl.py:1995-2083).
    /// The guard resume is shaped by `push_call_replay_stack` with the
    /// original CALL's callable/args so a failure re-executes the call in
    /// the interpreter — the same capture the residual path uses.
    ///
    /// Returns the CALL_ASSEMBLER result OpRef (Ref-typed: the callee's
    /// boxed return value).
    pub(crate) fn do_recursive_call_assembler(
        &mut self,
        token_number: u64,
        callee_frame: OpRef,
        replay_callable: OpRef,
        replay_args: &[OpRef],
        call_pc: usize,
    ) -> Result<OpRef, PyError> {
        self.with_ctx(|this, ctx| {
            // pyjitpl.py:2017: do_residual_call step 1
            this.vable_and_vrefs_before_residual_call(ctx);
            let ec = this.ensure_execution_context(ctx);
            let ca_result = ctx.call_assembler_red_only_ref(
                token_number,
                &[callee_frame, ec],
                &[Type::Ref, Type::Ref],
            );
            // pyjitpl.py:3625-3631 direct_assembler_call: keep the callee
            // virtualizable alive past the CALL_ASSEMBLER.
            ctx.record_op(OpCode::Keepalive, &[callee_frame]);
            // pyjitpl.py:2049
            this.vrefs_after_residual_call(ctx);
            // pyjitpl.py:2078
            this.vable_after_residual_call()?;
            // pyjitpl.py:2079
            this.push_call_replay_stack(ctx, replay_callable, replay_args, call_pc);
            this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
            this.generate_guard(ctx, OpCode::GuardNoException, &[]);
            ctx.heap_cache_mut().invalidate_caches_for_escaped();
            this.pop_call_replay_stack(ctx, replay_args.len())?;
            Ok(ca_result)
        })
    }

    /// pyjitpl.py:1425-1432 do_recursive_call +
    /// pyjitpl.py:1995-2083 do_residual_call parity.
    /// When assembler_call=true, emits CALL_ASSEMBLER via token lookup.
    /// Otherwise emits CALL_MAY_FORCE (normal residual call).
    fn inline_function_call(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
        concrete_callable: PyObjectRef,
        callee_key: u64,
        callee_raw: (usize, usize),
        frame_helper: *const (),
        passed_concrete_args: &[PyObjectRef],
        assembler_call: bool,
    ) -> Result<OpRef, PyError> {
        let (driver, _) = crate::driver::driver_pair();
        let concrete_arg0 = if args.len() == 1 {
            passed_concrete_args.first().copied()
        } else {
            None
        };
        // Save CALL instruction PC so GuardNotForced can resume the CALL.
        let call_pc = self.fallthrough_pc.saturating_sub(1);

        let result = self.with_ctx(|this, ctx| {
            this.implement_guard_value(ctx, callable, concrete_callable as i64);

            if args.len() == 1 {
                let result = if matches!(concrete_arg0, Some(arg) if unsafe { is_int(arg) }) {
                    let raw_arg = this.trace_guarded_int_payload(ctx, args[0]);
                    // pyjitpl.py:1396-1401 element-wise greenkey.
                    let _ = callee_key;
                    let caller_raw: (usize, usize) =
                        (unsafe { (*this.sym().jitcode).code } as usize, 0);
                    let is_self_recursive = callee_raw == caller_raw;
                    let needs_positional_defaults = is_self_recursive
                        && unsafe {
                            let w_callee_code = pyre_interpreter::getcode(concrete_callable);
                            let callee_code =
                                &*pyre_interpreter::w_code_get_ptr(w_callee_code as PyObjectRef)
                                    .cast::<CodeObject>();
                            positional_defaults_to_load(concrete_callable, callee_code, args.len())
                                .is_some()
                        };
                    // RPython parity: an opaque helper-boundary Python CALL
                    // still produces a boxed object result.  Even if the
                    // callee itself can finish with a raw int, the helper
                    // boxes at the boundary and the trace records a Ref.
                    let force_fn = if is_self_recursive
                        && !needs_positional_defaults
                        && (crate::callbacks::get().recursive_force_cache_safe)(concrete_callable)
                    {
                        crate::callbacks::get().jit_force_self_recursive_call_argraw_boxed_1
                    } else {
                        crate::callbacks::get().jit_force_recursive_call_argraw_boxed_1
                    };
                    // pyjitpl.py:2053-2055: direct_assembler_call only when
                    // assembler_call=True (computed in _opimpl_recursive_call).
                    // Token lookup happens AFTER the decision, not before.
                    let ca_token = if assembler_call {
                        driver
                            .get_loop_token_number(callee_key)
                            .or_else(|| driver.get_pending_token_number(callee_key))
                    } else {
                        None
                    };
                    let raw_result = if let Some(token_number) = ca_token {
                        let w_callee_code = unsafe { pyre_interpreter::getcode(concrete_callable) };
                        let callee_code = unsafe {
                            &*(pyre_interpreter::w_code_get_ptr(w_callee_code as PyObjectRef)
                                as *const CodeObject)
                        };
                        let self_recursive_raw_arg = if is_self_recursive {
                            Some(raw_arg)
                        } else {
                            None
                        };
                        let (callee_frame, drop_callee_frame) = emit_call_assembler_callee_frame(
                            this,
                            ctx,
                            callable,
                            args,
                            concrete_callable,
                            w_callee_code,
                            callee_code,
                            is_self_recursive,
                            self_recursive_raw_arg,
                        )?;
                        // pyjitpl.py:2017: do_residual_call step 1
                        this.vable_and_vrefs_before_residual_call(ctx);
                        let ec = this.ensure_execution_context(ctx);
                        let ca_result = ctx.call_assembler_red_only_ref(
                            token_number,
                            &[callee_frame, ec],
                            &[Type::Ref, Type::Ref],
                        );
                        ctx.record_op(OpCode::Keepalive, &[callee_frame]);
                        if drop_callee_frame {
                            ctx.call_void(
                                crate::callbacks::get().jit_drop_callee_frame,
                                &[callee_frame],
                            );
                        }
                        ca_result
                    } else if force_fn
                        == crate::callbacks::get().jit_force_self_recursive_call_argraw_boxed_1
                    {
                        // pyjitpl.py:2017: do_residual_call step 1
                        this.vable_and_vrefs_before_residual_call(ctx);
                        ctx.call_may_force_ref_typed(
                            force_fn,
                            &[this.frame(), raw_arg],
                            &[Type::Ref, Type::Int],
                        )
                    } else {
                        // pyjitpl.py:2017: do_residual_call step 1
                        this.vable_and_vrefs_before_residual_call(ctx);
                        ctx.call_may_force_ref_typed(
                            force_fn,
                            &[this.frame(), callable, raw_arg],
                            &[Type::Ref, Type::Ref, Type::Int],
                        )
                    };
                    // pyjitpl.py:2049: vrefs_after_residual_call
                    this.vrefs_after_residual_call(ctx);
                    // pyjitpl.py:2078: vable_after_residual_call
                    this.vable_after_residual_call()?;
                    // pyjitpl.py:2079: GUARD_NOT_FORCED — emitted on every
                    // residual-call path (CA and CALL_MAY_FORCE alike).
                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                    this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                    this.generate_guard(ctx, OpCode::GuardNoException, &[]);
                    ctx.heap_cache_mut().invalidate_caches_for_escaped();
                    this.pop_call_replay_stack(ctx, args.len())?;
                    // CA-path-only: unbox boxed result via guard_class + getfield_gc_i.
                    let result = if ca_token.is_some() {
                        this.trace_guarded_int_payload(ctx, raw_result)
                    } else {
                        raw_result
                    };
                    result
                } else {
                    let force_fn = crate::callbacks::get().jit_force_recursive_call_1;
                    // pyjitpl.py:2017: do_residual_call step 1
                    this.vable_and_vrefs_before_residual_call(ctx);
                    let result = ctx.call_may_force_ref_typed(
                        force_fn,
                        &[this.frame(), callable, args[0]],
                        &[Type::Ref, Type::Ref, Type::Ref],
                    );
                    // pyjitpl.py:2049: vrefs_after_residual_call
                    this.vrefs_after_residual_call(ctx);
                    // pyjitpl.py:2078: vable_after_residual_call
                    this.vable_after_residual_call()?;
                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                    this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                    this.generate_guard(ctx, OpCode::GuardNoException, &[]);
                    ctx.heap_cache_mut().invalidate_caches_for_escaped();
                    this.pop_call_replay_stack(ctx, args.len())?;
                    result
                };
                Ok(result)
            } else {
                let mut helper_args = vec![this.frame(), callable];
                helper_args.extend_from_slice(args);
                let helper_arg_types = frame_callable_arg_types(args.len());
                let callee_frame = ctx.call_ref_typed_with_effect(
                    frame_helper,
                    &helper_args,
                    &helper_arg_types,
                    default_effect_info(),
                );
                let force_fn = crate::callbacks::get().jit_force_callee_frame;
                // pyjitpl.py:2017: do_residual_call step 1
                this.vable_and_vrefs_before_residual_call(ctx);
                let result = ctx.call_may_force_ref_typed(force_fn, &[callee_frame], &[Type::Ref]);
                // pyjitpl.py:2049: vrefs_after_residual_call
                this.vrefs_after_residual_call(ctx);
                // pyjitpl.py:2078: vable_after_residual_call
                this.vable_after_residual_call()?;
                this.push_call_replay_stack(ctx, callable, args, call_pc);
                this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                this.generate_guard(ctx, OpCode::GuardNoException, &[]);
                ctx.heap_cache_mut().invalidate_caches_for_escaped();
                this.pop_call_replay_stack(ctx, args.len())?;
                ctx.call_void(
                    crate::callbacks::get().jit_drop_callee_frame,
                    &[callee_frame],
                );
                Ok(result)
            }
        });

        result
    }

    pub(crate) fn iter_next_value(
        &mut self,
        iter: OpRef,
        concrete_iter: PyObjectRef,
    ) -> Result<FrontendOp, PyError> {
        let concrete_continues = range_iter_continues(concrete_iter)?;
        let concrete_step =
            unsafe { (*(concrete_iter as *const pyre_object::rangeobject::W_RangeIterator)).step };
        let concrete_current = unsafe {
            (*(concrete_iter as *const pyre_object::rangeobject::W_RangeIterator)).current
        };

        // Delegate to auto-generated function (RPython jitcode parity:
        // getfield(current/stop/step) → step sign guard → continues guard
        // → int_add_ovf → guard_no_overflow → setfield).
        let gen_result: Option<(OpRef, i64)> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_iter_next_value(
                this,
                ctx,
                iter,
                concrete_continues,
                concrete_step,
                concrete_current,
            ))
        })?;
        if let Some((opref, cv)) = gen_result {
            return Ok(FrontendOp::new(opref, ConcreteValue::Int(cv)));
        }
        let opref = self.trace_iter_next_value(iter)?;
        Ok(FrontendOp::void(opref))
    }

    /// TODO: pyre's tracer holds raw `PyObjectRef`
    /// (the runtime W_Root pointer) instead of typed `FrontendOp`
    /// banks. RPython's `pyjitpl.py:opimpl_goto_if_*` reads
    /// `box.getint()` / `RefFrontendOp._resref` directly, dispatching
    /// on `box.type == 'i'` / `'r'`. This wrapper collapses that
    /// type-dispatched read into a single `objspace_truth_value` probe
    /// gated by a null guard, since pyre's tracer keeps the W_Root
    /// pointer rather than a typed-Box hierarchy.
    pub(crate) fn concrete_branch_truth_for_value(
        &mut self,
        _value: OpRef,
        concrete_val: PyObjectRef,
    ) -> Result<bool, PyError> {
        if concrete_val.is_null() {
            return Err(PyError::type_error(
                "missing concrete branch value during trace",
            ));
        }
        objspace_truth_value(concrete_val)
    }

    #[allow(dead_code)]
    pub(crate) fn concrete_branch_truth(&mut self) -> Result<bool, PyError> {
        self.concrete_branch_truth_for_value(OpRef::NONE, PY_NULL)
    }

    pub(crate) fn truth_value_direct(
        &mut self,
        value: OpRef,
        concrete_val: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // type-specialized is_true via guard_class + getfield → int_ne).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_truth_value_direct(
                this,
                ctx,
                value,
                concrete_val,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_truth_value(value)
    }

    pub(crate) fn unary_int_value(
        &mut self,
        value: OpRef,
        opcode: OpCode,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // guard_class + getfield_gc_i + INT_NEG/INT_INVERT).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_unary_int_value(
                this,
                ctx,
                value,
                opcode,
                concrete_value,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        match opcode {
            OpCode::IntNeg => self.trace_unary_negative_value(value),
            OpCode::IntInvert => self.trace_unary_invert_value(value),
            _ => unreachable!("unexpected unary opcode"),
        }
    }

    pub(crate) fn into_trace_action(
        &mut self,
        result: Result<pyre_interpreter::StepResult<FrontendOp>, PyError>,
    ) -> TraceAction {
        trace_step_result_to_action(self, result)
    }

    /// RPython parity: `pyjitpl.py:1892 MetaInterp._interpret` dispatches
    /// each opcode through `staticdata.opcode_implementations[opnum]`,
    /// which is the jitcode-bytecode interpreter (the only dispatch path
    /// upstream).  Pyre's transient deviation is the dual trait/walker
    /// dispatch pair: `pyre_interpreter::execute_opcode_step` runs the
    /// trait-driven Python-opcode interpreter while
    /// `jitcode_dispatch::dispatch_via_miframe` walks the codewriter-
    /// emitted jitcode arm.  The trait path retires opcode-by-
    /// opcode; this helper is the per-opcode walker entry that root-frame
    /// production dispatch (`trace_code_step`) calls for allow-listed
    /// instructions.  Inline dispatch (`trace_code_step_inline`) remains
    /// on the trait path until walker snapshot capture can represent the
    /// full parent-frame chain.
    ///
    /// PyPy `_opimpl_*` direct-record entry point for opcodes that emit
    /// IR and produce concrete effects directly from the tracer, bypassing
    /// the auto-gen arm-jitcode walk.  Mirrors `MetaInterp.interpret`'s
    /// upstream dispatch shape where `pyjitpl.py:1346+ _opimpl_*` methods
    /// invoke `metainterp.history.record*` and concrete-execute helpers
    /// inline rather than walking an intermediate jitcode representation.
    ///
    /// Returns `Ok(Some(step_result))` when the opcode was handled here.
    /// `Ok(None)` falls through to the arm-jitcode walker in
    /// [`MIFrame::dispatch_via_walker_for_opcode`].  `Err(_)` propagates a
    /// dispatch-time exception (e.g. `MIFrame::reraise` raising with
    /// `reraise_lasti` populated).
    ///
    /// Each direct-dispatch handler must:
    ///   1. Update the symbolic stack via `SharedOpcodeHandler::pop_value`
    ///      / `peek_at` so subsequent walker opcodes see consistent vsd.
    ///   2. Record any IR for the opcode (via a trait method like
    ///      `MIFrame::store_subscr_value` or a `_opimpl_*`-style direct
    ///      `record_op_*` call against `WalkContext`).
    ///   3. Produce concrete heap effects when the upstream `_opimpl_*`
    ///      counterpart does (e.g. `bh_execute_store_subscr`-style helper
    ///      invocation).
    ///
    /// This entry point is the natural home for future `_opimpl_*` ports
    /// — adding a new opcode here is the structural equivalent of writing
    /// a new `pyjitpl.py:_opimpl_<name>` method.
    fn try_walker_direct_opcode_dispatch(
        &mut self,
        instruction: &Instruction,
        op_arg: pyre_interpreter::OpArg,
        code: &CodeObject,
    ) -> Result<Option<pyre_interpreter::StepResult<FrontendOp>>, PyError> {
        // LOAD_CONST walker activation via trait-path delegation.
        //
        // The auto-gen arm jitcode for `LoadConst` residualises
        // `opcode_load_const(frame, &ConstantData)` — the `&ConstantData`
        // operand is oparg-derived (resolved from `consti` against the
        // code's constant pool), but the per-opcode arm entry
        // (`dispatch_via_miframe_at_opcode_entry`) seeds only `r0 = frame`,
        // leaving that operand register unbound — the walk aborts with
        // `ResidualCallArgUnbound { arg_index: 2 }`.  (A const-specialising
        // JIT wants the constant BAKED into the IR, not threaded as a
        // runtime arg, so seeding the register is also the wrong shape.)
        //
        // Resolve the constant here and delegate to the existing
        // `OpcodeStepExecutor::load_const` (the same method
        // `execute_load_const` calls on the trait leg), which emits the
        // type-specialised IR (`ConstRef` / `int_constant` / `str_constant`
        // / …) and pushes via `push_value` — keeping `sym.valuestackdepth`
        // / vable shadow / concrete mirror coherent.  Same delegation
        // pattern as the StoreSubscr / PushNull hooks below; bypasses
        // `apply_walker_stack_effect` because `push_value` handles vsd.
        if let Instruction::LoadConst { consti } = instruction {
            use pyre_interpreter::OpcodeStepExecutor;
            let const_idx = consti.get(op_arg);
            OpcodeStepExecutor::load_const(self, &code.constants[const_idx])?;
            return Ok(Some(pyre_interpreter::StepResult::Continue));
        }

        // LOAD_SMALL_INT walker activation — the const-push sibling of
        // LoadConst (the small int lives in the oparg itself, not the
        // constant pool).  Delegate to `OpcodeStepExecutor::load_small_int`
        // (the same method `execute_load_small_int` calls), which emits an
        // int ConstRef and pushes via `push_value` (advancing vsd).
        if let Instruction::LoadSmallInt { i } = instruction {
            use pyre_interpreter::OpcodeStepExecutor;
            OpcodeStepExecutor::load_small_int(self, i64::from(i.get(op_arg)))?;
            return Ok(Some(pyre_interpreter::StepResult::Continue));
        }

        // LOAD_FAST / LOAD_FAST_BORROW walker activation via trait-path
        // delegation.  Same oparg-payload shape as LoadConst: the auto-gen
        // arm reads the `var_num` local index from a register the r0-only
        // arm entry leaves unbound.  Resolve the local index + name
        // (mirroring `execute_load_fast`, which serves both opcodes) and
        // delegate to the existing `OpcodeStepExecutor::load_fast_checked`,
        // whose vable read + `push_value` emits the specialised IR and
        // advances vsd.  LoadFastBorrow shares the handler — the
        // borrow-vs-own distinction is a runtime refcount concern the
        // symbolic IR does not model.
        if let Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } =
            instruction
        {
            use pyre_interpreter::OpcodeStepExecutor;
            let idx = pyre_interpreter::load_fast_var_num_to_index(*var_num, op_arg);
            let name = if idx < pyre_interpreter::code_varnames_len(code) {
                code.varnames[idx].as_ref()
            } else {
                "<cell>"
            };
            OpcodeStepExecutor::load_fast_checked(self, idx, name)?;
            return Ok(Some(pyre_interpreter::StepResult::Continue));
        }

        // STORE_FAST walker activation via trait-path delegation.
        //
        // Same oparg-payload shape as LoadFast (the var_num local index is
        // read from an arm register the r0-only entry leaves unbound).
        // Delegate to `OpcodeStepExecutor::store_fast`, which pops the TOS
        // (via `pop_value`, advancing vsd) and emits the `setarrayitem_
        // vable_r` local write — a VABLE array write (the JIT-modeled,
        // resume-safe kind), distinct from a globals-dict write.  The
        // updated local rides the vable shadow to the live PyFrame through
        // `synchronize_virtualizable`.  No `code` needed (store_fast takes
        // only the index); bypasses `apply_walker_stack_effect` because
        // `pop_value` handles vsd.
        if let Instruction::StoreFast { var_num } = instruction {
            use pyre_interpreter::OpcodeStepExecutor;
            OpcodeStepExecutor::store_fast(self, var_num.get(op_arg).as_usize())?;
            return Ok(Some(pyre_interpreter::StepResult::Continue));
        }

        // BINARY_OP walker activation via trait-path delegation.
        //
        // Same oparg-payload shape (the operator tag is read from an arm
        // register the r0-only entry leaves unbound).  Delegate to
        // `OpcodeStepExecutor::binary_op`, which pops the two operands (via
        // `pop_value`, advancing vsd), emits the type-specialised
        // arithmetic IR (`int_add_ovf` &c. behind class guards), and pushes
        // the result.  Unlike the load/store hooks above this is MAY-RAISE
        // (TypeError on mismatched operands): `binary_op` returns
        // `Err(PyError)`, which propagates through
        // `dispatch_via_walker_for_opcode` to `trace_code_step` exactly as
        // an `execute_opcode_step` error would on the trait leg, so the
        // recorder's exception handling is identical on either dispatch
        // leg.  Bypasses `apply_walker_stack_effect` because `pop_value` /
        // `push_value` handle vsd.
        if let Instruction::BinaryOp { op } = instruction {
            use pyre_interpreter::OpcodeStepExecutor;
            OpcodeStepExecutor::binary_op(self, op.get(op_arg))?;
            return Ok(Some(pyre_interpreter::StepResult::Continue));
        }

        // STORE_SUBSCR walker activation via trait-path delegation.
        //
        // The auto-gen arm jitcode for `StoreSubscr` is
        // `int_copy, residual_call_r_r(bh_execute_store_subscr, frame),
        // live, ref_return`.  `try_execute_residual_call_via_executor`
        // matches the `CallR` shape and concrete-executes
        // `bh_execute_store_subscr(frame)`, which casts the arg to
        // `*mut PyFrame` and pops 3 values from
        // `PyFrame.locals_cells_stack_w`.  Walker `LOAD_FAST` /
        // `LOAD_CONST` / etc. populate only MIFrame's symbolic +
        // concrete-shadow stacks — not the concrete `PyFrame`'s slots
        // (`setfield_vable_i(vsd)` advances depth without writing the
        // slot) — so the pop reads NULL and the helper raises every
        // iteration, surfacing as `DispatchOutcome::SubRaise`.
        //
        // Trait dispatch documents `MIFrame::store_subscr` as
        // "trace-only, no concrete mutation" (compiled trace handles
        // real mutations later).  Mirror that here by short-circuiting
        // the arm walk: pop 3 via `SharedOpcodeHandler::pop_value`
        // (updates symbolic + concrete-shadow stacks + vsd shadow)
        // and delegate to the existing `MIFrame::store_subscr_value`
        // which records the same specialized IR shape
        // (`guard_class + SETARRAYITEM_GC` family via
        // `generated_store_subscr_value`, or `Call(jit_setitem, ...)`
        // fallback via `trace_store_subscr`).  Bypasses the
        // `apply_walker_stack_effect` post-walk step because
        // `pop_value` already handles vsd.
        //
        // This delegation reuses the trait method intentionally; the
        // 141 cutover can later replace it with a pure-walker variant
        // once the walker grows symbolic-stack-aware specialization
        // primitives.
        if matches!(instruction, Instruction::StoreSubscr) {
            use pyre_interpreter::SharedOpcodeHandler;
            let _ = op_arg;
            let key = SharedOpcodeHandler::pop_value(self)?;
            let obj = SharedOpcodeHandler::pop_value(self)?;
            let value = SharedOpcodeHandler::pop_value(self)?;
            self.store_subscr_value(
                obj.opref,
                key.opref,
                value.opref,
                obj.concrete.to_pyobj(),
                key.concrete.to_pyobj(),
                value.concrete.to_pyobj(),
            )?;
            return Ok(Some(pyre_interpreter::StepResult::Continue));
        }

        // PUSH_NULL walker activation via direct symbolic push.
        //
        // The auto-gen arm jitcode for `PushNull` is `int_copy,
        // residual_call_r_r(opcode_push_null, frame), live, ref_return` —
        // a residual wrapper whose helper is NOT in the runtime fnaddr
        // registry, so `try_execute_residual_call_via_executor` rejects
        // its placeholder funcptr (47-bit gate) and the arm walk records
        // the call WITHOUT executing it.  The concrete `PyFrame.
        // valuestackdepth` then never advances, and the post-walk
        // `resync_sym_vsd_from_concrete_frame` clobbers
        // `sym.valuestackdepth` with the stale frame value — losing every
        // preceding trait-leg push and underflowing the next CALL
        // (`LOAD_NAME f; PUSH_NULL; LOAD_NAME s; CALL` at module scope,
        // `f = g; f(s)` local-callable calls in function scope).
        //
        // Short-circuit the arm walk with the symbolic effect directly:
        // PUSH_NULL pushes a constant NULL stack slot (the `_null_or_self`
        // operand `opcode_call` pops and discards), so no IR op is needed —
        // a `const_ref(PY_NULL)` through the trait push machinery keeps
        // `sym.valuestackdepth` / vable shadow / concrete mirror coherent.
        // Same delegation pattern as the StoreSubscr hook above; bypasses
        // `apply_walker_stack_effect` because `push_value` handles vsd.
        if matches!(instruction, Instruction::PushNull) {
            use pyre_interpreter::SharedOpcodeHandler;
            let _ = op_arg;
            let null_opref = self.with_ctx(|_this, ctx| {
                Ok::<_, PyError>(ctx.const_ref(pyre_object::PY_NULL as i64))
            })?;
            SharedOpcodeHandler::push_value(
                self,
                FrontendOp::new(null_opref, ConcreteValue::Ref(pyre_object::PY_NULL)),
            )?;
            return Ok(Some(pyre_interpreter::StepResult::Continue));
        }

        // `pyopcode.py:1348-1376 RERAISE` parity for the walker leg.
        //
        // Production walker excluded `Reraise` because the trait path
        // returns `Err(PyError)` with `reraise_lasti` populated for
        // `oparg > 0`, while a walker hook returning `StepResult::Continue`
        // (`Ok`) loses that channel and `trace_code_step` aborts on
        // `oparg > 0 && reraise_lasti < 0`.  Delegate to the existing
        // `MIFrame::reraise` impl (which mirrors `pyopcode.py:1357-1376`
        // verbatim, reading the lasti from `concrete_stack[stack_idx]`
        // and seeding `err.reraise_lasti`) and propagate its `Err` so
        // `trace_code_step`'s `step_result.err().map(|e| e.reraise_lasti)`
        // extraction works the same on either dispatch leg.
        if let Instruction::Reraise { depth } = instruction {
            use pyre_interpreter::OpcodeStepExecutor;
            OpcodeStepExecutor::reraise(self, depth.get(op_arg))?;
            return Ok(Some(pyre_interpreter::StepResult::Continue));
        }

        Ok(None)
    }

    /// `is_top_level=false` so the arm's `ref_return/r` terminator
    /// surfaces as `DispatchOutcome::SubReturn` rather than emitting a
    /// `Finish` (the outer Python frame's `Finish` is owned by the
    /// trace_opcode dispatch's higher-level Return/Yield/CloseLoop
    /// handling, not the per-opcode arm).
    pub(crate) fn dispatch_via_walker_for_opcode(
        &mut self,
        instruction: &Instruction,
        op_arg: pyre_interpreter::OpArg,
        code: &CodeObject,
    ) -> Result<pyre_interpreter::StepResult<FrontendOp>, PyError> {
        // PyPy `_opimpl_*` direct-record entry point — opcodes that bypass
        // the auto-gen arm-jitcode walk and emit IR / produce concrete
        // effects directly, matching `MetaInterp.interpret`'s upstream
        // dispatch shape (`pyjitpl.py:1346+ _opimpl_*`).
        //
        // Returns `Some(step_result)` when the opcode was handled here.
        // The arm-jitcode walker below runs only when this returns `None`.
        if let Some(step_result) =
            self.try_walker_direct_opcode_dispatch(instruction, op_arg, code)?
        {
            return Ok(step_result);
        }

        let jitcode = match crate::jitcode_runtime::jitcode_for_instruction(instruction) {
            Some(jc) => jc,
            None => {
                // The production-walker allow-list (`production_walker_handles`)
                // expanded ahead of `jitcode_for_instruction` registering an
                // arm for `instruction`; degrade to a trace abort so the
                // outer tracer can fall back instead of crashing.
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][abort-reason] dispatch_via_walker_for_opcode \
                         no_codewriter_arm instr={:?}",
                        instruction,
                    );
                }
                return Err(trace_abort_error(
                    "production walker allow-listed instruction has no codewriter arm",
                ));
            }
        };

        let (done_void, done_int, done_ref, done_float, exit_exc_ref) =
            {
                let sd = self.ctx().metainterp_sd();
                let void = sd.done_with_this_frame_descr_void.clone().expect(
                    "done_with_this_frame_descr_void must be wired before production walker",
                );
                let int = sd.done_with_this_frame_descr_int.clone().expect(
                    "done_with_this_frame_descr_int must be wired before production walker",
                );
                let ref_ = sd.done_with_this_frame_descr_ref.clone().expect(
                    "done_with_this_frame_descr_ref must be wired before production walker",
                );
                let float = sd.done_with_this_frame_descr_float.clone().expect(
                    "done_with_this_frame_descr_float must be wired before production walker",
                );
                let exc = sd.exit_frame_with_exception_descr_ref.clone().expect(
                    "exit_frame_with_exception_descr_ref must be wired before production walker",
                );
                (void, int, ref_, float, exc)
            };

        let sub_jitcode_lookup = |idx: usize| -> Option<crate::jitcode_dispatch::SubJitCodeBody> {
            let all = crate::jitcode_runtime::all_jitcodes();
            all.get(idx)
                .map(|jc| crate::jitcode_dispatch::SubJitCodeBody {
                    code: jc.code.as_slice(),
                    num_regs_r: jc.num_regs_r() as usize,
                    num_regs_i: jc.num_regs_i() as usize,
                    num_regs_f: jc.num_regs_f() as usize,
                    constants_i: jc.constants_i.as_slice(),
                    constants_r: jc.constants_r.as_slice(),
                    constants_f: jc.constants_f.as_slice(),
                })
        };

        // Per-opcode arm entry allocates fresh
        // per-jitcode register banks and seeds r0 = sym.frame.  This
        // matches RPython MIFrame.setup (`pyjitpl.py:74-91`) and breaks
        // the prior alias of sym.registers_r (semantic frame mirror)
        // into the walker's r0..rN — the structural blocker documented
        // in `project_issue73_phase5_design.md`.  `jitcode_for_instruction`
        // returns an Arc<JitCode> over the registry's `'static`
        // collection; the leak below upgrades the borrow to `'static`
        // for the duration of this call.
        let entry_jitcode: &'static majit_translate::jitcode::JitCode =
            unsafe { &*(jitcode.as_ref() as *const majit_translate::jitcode::JitCode) };
        let walk_result = crate::jitcode_dispatch::dispatch_via_miframe_at_opcode_entry(
            self,
            entry_jitcode,
            crate::jitcode_runtime::all_descr_refs(),
            &sub_jitcode_lookup,
            done_ref,
            done_int,
            done_float,
            done_void,
            exit_exc_ref,
        );

        use crate::jitcode_dispatch::DispatchOutcome;
        match walk_result {
            Ok((DispatchOutcome::SubReturn { .. }, _)) => {
                // The walker arm records IR for the
                // opcode but does NOT run the trait dispatch's
                // `MIFrame::pop_value` / `push_typed_value` paths that
                // mutate `sym.valuestackdepth`.  Apply the opcode's net
                // stack effect here so the symbolic mirror stays
                // consistent with the IR the walker just emitted, matching
                // how the trait dispatch advances `sym.valuestackdepth` via
                // its own push/pop trait methods.
                apply_walker_stack_effect(self, instruction);
                Ok(pyre_interpreter::StepResult::Continue)
            }
            Ok((outcome, _)) => {
                // Allow-list expansion let an opcode through whose arm
                // returns something other than `SubReturn` (e.g. an arm
                // that emits `Finish` or `Goto`); the walker leg cannot
                // unify with `trace_code_step`'s top-level control flow
                // for those, so abort the trace and let `trace_code_step`
                // fall back to the trait dispatch.
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][abort-reason] dispatch_via_walker_for_opcode \
                         unexpected_outcome instr={:?} outcome={:?}",
                        instruction, outcome,
                    );
                }
                Err(trace_abort_error(
                    "production walker received unexpected dispatch outcome",
                ))
            }
            Err(e) => {
                // Walker dispatch raised a structured error (e.g.
                // `GotoIfNotValueNotConcrete`, register-bank size
                // mismatch).  Surface as a trace abort so production
                // recovers rather than crashing the process.
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][abort-reason] dispatch_via_walker_for_opcode \
                         walker_error instr={:?} err={:?}",
                        instruction, e,
                    );
                }
                Err(trace_abort_error("production walker dispatch failed"))
            }
        }
    }

    /// True for the method-form `LOAD_ATTR` of a foldable builtin list
    /// method (`append` / `pop` / `reverse`) on a concrete list receiver
    /// at TOS.  The dispatch gate routes exactly this shape OFF the
    /// walker leg so `MIFrame::load_method` can resolve it to a
    /// class-guarded Const unbound function instead of the walker arm's
    /// residual getattr (which materialises a fresh bound method every
    /// iteration and resolves to a null callable on blackhole CALL
    /// re-execution).
    fn is_foldable_list_method_load_attr(
        &self,
        instruction: &Instruction,
        op_arg: pyre_interpreter::OpArg,
        code: &CodeObject,
    ) -> bool {
        let Instruction::LoadAttr { namei } = instruction else {
            return false;
        };
        let attr = namei.get(op_arg);
        if !attr.is_method() {
            return false;
        }
        let Some(name) = code.names.get(attr.name_idx() as usize) else {
            return false;
        };
        if !matches!(name.as_ref(), "append" | "pop" | "reverse") {
            return false;
        }
        let s = self.sym();
        let Some(stack_idx) = s.valuestackdepth.checked_sub(s.nlocals + 1) else {
            return false;
        };
        matches!(
            s.concrete_stack.get(stack_idx),
            Some(ConcreteValue::Ref(obj)) if !obj.is_null() && unsafe { is_list(*obj) }
        )
    }

    pub fn trace_code_step(&mut self, code: &CodeObject, pc: usize) -> TraceAction {
        if pc >= code.instructions.len() {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step pc_oob pc={} code_len={}",
                    pc,
                    code.instructions.len()
                );
            }
            return TraceAction::Abort;
        }

        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step decode_failed pc={}",
                    pc
                );
            }
            return TraceAction::Abort;
        };

        // Resolve each opcode to its per-arm jitcode
        // through the shared `MetaInterpStaticData.jitcodes` table
        // (RPython `CallControl.jitcodes ≡ metainterp_sd.jitcodes`,
        // warmspot.py:281-282). Env-gated log-only probe — no execution —
        // confirming the build→runtime bridge stays in sync.
        // `unresolved` logs the single-variant miss so a dispatch
        // gap surfaces immediately.
        if std::env::var_os("PYRE_JIT_JITCODE_PROBE").is_some() {
            match crate::jitcode_runtime::jitcode_for_instruction(&instruction) {
                Some(jc) => {
                    // Walk the jitcode byte stream using the insns argcode
                    // protocol (blackhole.py:105-232 `_get_method.handler`).
                    // `complete` flags whether the decoder reached
                    // `code.len()` without hitting an unknown opcode — a
                    // structural check the probe can surface per-opcode.
                    let mut opnames = Vec::new();
                    let mut last_next = 0usize;
                    for op in crate::jitcode_runtime::decoded_ops(&jc.code) {
                        opnames.push(op.opname);
                        last_next = op.next_pc;
                    }
                    let complete = last_next == jc.code.len();
                    eprintln!(
                        "[jit][probe] pc={pc} instr={} arm_jitcode={} code_len={} ops={} complete={} trace={:?}",
                        crate::jitcode_runtime::instruction_variant_name(&instruction),
                        jc.name,
                        jc.code.len(),
                        opnames.len(),
                        complete,
                        opnames,
                    );
                }
                None => eprintln!(
                    "[jit][probe] pc={pc} instr={} unresolved",
                    crate::jitcode_runtime::instruction_variant_name(&instruction),
                ),
            }
        }

        self.set_orgpc(pc);
        self.prepare_fallthrough();
        // RPython pyjitpl.py captures resumedata at each guard site, not at
        // every opcode boundary. Pyre still needs an opcode-start snapshot
        // for stack-machine opcodes that can mutate stack/register state
        // before emitting a guard; keep it off for instructions that cannot
        // reach a guard at all.
        let needs_pre_opcode_snapshot =
            crate::state::instruction_needs_pre_opcode_snapshot(instruction);
        if needs_pre_opcode_snapshot {
            self.capture_pre_opcode_state();
        }
        // pyjitpl.py:541-556 opimpl_goto_if_not_<op>_<type> parity: when
        // the CompareOp is followed by PopJumpIf* and operands form a
        // compatible int/float pair, run the fused dispatch path and
        // skip the separate PopJumpIf* dispatch entirely.
        let fused_result = if let Instruction::CompareOp { opname } = instruction {
            let comp_op = opname.get(op_arg);
            match self.try_fused_compare_goto_if_not(code, pc, comp_op) {
                Ok(Some(action)) => Some(Ok(action)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        } else {
            None
        };
        let step_result = match fused_result {
            Some(Ok(action)) => {
                // Fused path consumed COMPARE + BRANCH. Clear snapshot
                // and return the action immediately (matches the
                // post-dispatch clear that normally follows
                // execute_opcode_step below).
                if needs_pre_opcode_snapshot {
                    self.clear_pre_opcode_state();
                }
                return action;
            }
            Some(Err(e)) => Err(e),
            None => {
                // Inline-frame gate (PopTop activation):
                // `walker_capture_snapshot_for_last_guard` only captures a
                // single Python frame (the outer pyjitcode coordinates,
                // jitcode_dispatch.rs:3568) because pyre's blackhole cannot
                // resume into helper/sub jitcodes.  For inline-traced sub
                // frames (`self.parent_frames` non-empty), the snapshot
                // must carry the full framestack chain to make the resume
                // point well-defined; walker dispatch in that context drops
                // the parent frames, so deopt would re-enter at the wrong
                // PyFrame.  Until the walker grows multi-frame snapshot
                // capture (analogous to `capture_snapshot_for_last_guard_
                // multi_frame_with_vable_vref`), fall back to trait
                // dispatch in inline frames.
                let in_inline_frame = !self.parent_frames.is_empty();
                let foldable_list_load_attr =
                    self.is_foldable_list_method_load_attr(&instruction, op_arg, code);
                if production_walker_handles(&instruction)
                    && !in_inline_frame
                    && !foldable_list_load_attr
                {
                    self.dispatch_via_walker_for_opcode(&instruction, op_arg, code)
                } else {
                    if flip_probe_enabled() {
                        let reason = if !production_walker_handles(&instruction) {
                            "not-in-allowlist"
                        } else if in_inline_frame {
                            "in-inline-frame"
                        } else {
                            "foldable-list-load-attr"
                        };
                        flip_probe_record(&instruction, reason);
                    }
                    let shadow_outcome =
                        crate::shadow_walker::shadow_validate_pre(self, &instruction, op_arg);
                    let result = execute_opcode_step(self, code, instruction, op_arg, pc + 1);
                    if result.is_ok() {
                        if let Some(outcome) = shadow_outcome {
                            crate::shadow_walker::shadow_validate_post(self, outcome);
                        }
                    }
                    result
                }
            }
        };
        // Clear pre-opcode snapshot immediately after opcode execution.
        // RPython: capture_resumedata is called at each guard; there is
        // no per-opcode snapshot lifecycle. Clearing here ensures the
        // snapshot doesn't leak into subsequent opcodes.
        if needs_pre_opcode_snapshot {
            self.clear_pre_opcode_state();
        }
        // RPython execute_ll_raised parity: generic exc=True ops store the
        // exception in last_exc_value before handle_possible_exception.
        // Valid raise/reraise paths seed `last_exc_box` directly; malformed
        // bytecode-level raises still surface as ordinary interpreter errors
        // and use the generic raised-exception path below.
        if step_result.as_ref().err().is_some_and(is_trace_abort_error) {
            return TraceAction::Abort;
        }
        if let Err(ref err) = step_result {
            if !matches!(
                instruction,
                Instruction::Reraise { .. } | Instruction::RaiseVarargs { .. }
            ) || self.sym().last_exc_box == OpRef::NONE
            {
                let exc_obj = err.to_exc_object();
                let s = self.sym_mut();
                if s.last_exc_value.is_null() {
                    s.last_exc_value = exc_obj;
                }
            }
        }
        if let Instruction::Reraise { depth } = instruction {
            // `pyopcode.py:1361` RERAISE extracts the saved lasti from the
            // stack and routes through RaiseWithExplicitTraceback, which
            // becomes `err.reraise_lasti` on our PyError.  When `oparg != 0`
            // and the trace could not fold a const-int from the symbolic
            // slot (`reraise()` returned -1), bail to the interpreter — the
            // trace cannot represent a dynamic lasti at handler-entry push
            // time as a constant int box.
            let oparg_val = depth.get(op_arg);
            let reraise_lasti = step_result
                .as_ref()
                .err()
                .map(|e| e.reraise_lasti)
                .unwrap_or(-1);
            if oparg_val != 0 && reraise_lasti < 0 {
                return TraceAction::Abort;
            }
            if self.sym().last_exc_box != OpRef::NONE {
                return self.handle_reraise(code, pc, reraise_lasti);
            }
            // `last_exc_box == NONE` for RERAISE means the trace entered an
            // except* handler without observing the prior PUSH_EXC_INFO —
            // anomalous.  If the RERAISE carried a saved lasti, the
            // generic handle_possible_exception fallback would drop it
            // (its inner finishframe_exception call passes -1).  Abort so
            // the interpreter handles the case faithfully.
            if oparg_val != 0 {
                return TraceAction::Abort;
            }
            if step_result.is_err() {
                return self.handle_possible_exception(code, pc);
            }
        }
        if let Instruction::RaiseVarargs { argc } = instruction {
            if self.sym().last_exc_box != OpRef::NONE {
                return self.handle_raise_varargs(code, pc, argc.get(op_arg) as usize);
            }
            if step_result.is_err() {
                return self.handle_possible_exception(code, pc);
            }
        }
        // RPython pyjitpl.py:1956-1957 execute_varargs: exc=True ops
        // always call handle_possible_exception, which internally decides
        // GUARD_EXCEPTION vs GUARD_NO_EXCEPTION.
        if instruction_may_raise(instruction) {
            let action = self.handle_possible_exception(code, pc);
            if !matches!(action, TraceAction::Continue) {
                return action;
            }
            // RPython: handle_possible_exception consumes the exception.
            // If it returned Continue, the exception was handled (e.g.,
            // GUARD_EXCEPTION emitted) and tracing continues normally.
            // Treat handled exceptions as successful opcode completion.
            if step_result.is_err() {
                return TraceAction::Continue;
            }
        }
        match step_result {
            Err(ref e) => {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][abort-reason] trace_code_step err at pc={} instr={:?} err={:?}",
                        pc, instruction, e
                    );
                }
                TraceAction::Abort
            }
            other => self.into_trace_action(other),
        }
    }

    /// RPython pyjitpl.py:3380 handle_possible_exception.
    ///
    /// Called after every may-raise opcode. Checks last_exc_value to decide:
    /// - exception raised → GUARD_EXCEPTION + finishframe_exception
    /// - no exception → GUARD_NO_EXCEPTION
    pub(crate) fn handle_possible_exception(
        &mut self,
        code: &CodeObject,
        pc: usize,
    ) -> TraceAction {
        if !self.sym().last_exc_value.is_null() {
            let exc_obj = self.sym().last_exc_value;

            // pyjitpl.py:3382-3384: ALWAYS emit GUARD_EXCEPTION first,
            // regardless of class_of_last_exc_is_const.
            let exc_type_ptr = unsafe {
                (*(exc_obj as *const pyre_object::excobject::W_ExceptionObject))
                    .ob_header
                    .ob_type as i64
            };

            let guard_op = self.with_ctx(|this, ctx| {
                // pyjitpl.py:2575-2578: after_residual_call=true for
                // GuardException — all boxes in top frame are live.
                let after_residual_call = true;
                let resume_pc = this.fallthrough_pc;
                let saved_orgpc = this.orgpc;
                this.orgpc = resume_pc;
                this.clear_pre_opcode_state();

                this.flush_to_frame_for_guard(ctx);
                // GUARD_EXCEPTION resumes via the `jf_guard_exc` channel at a
                // plain fallthrough pc (no bit-14 marker), so read the boxes at
                // the fallthrough `-live-` to match the snapshot pc below.
                let active_boxes =
                    this.get_list_of_active_boxes(ctx, false, after_residual_call, None);
                let fail_arg_types = this.build_fail_arg_types_for_active_boxes(&active_boxes);

                // capture_resumedata parity: full framestack snapshot.
                // pyjitpl.py:2597: capture_resumedata(self.framestack, ...)
                //
                // The post-call exception is carried to blackhole resume via
                // `jf_guard_exc` (the guard_exc channel), not the bit-14 resume
                // marker, so the top-frame resume pc stays a plain `fallthrough_pc`
                // and must leave bit 14 free or `decode_resume_pc` mis-reads it
                // as marked (resumedata.rs:48-62) — fail loudly otherwise.
                assert!(
                    resume_pc < majit_ir::resumedata::AFTER_RESIDUAL_CALL_PC_FLAG as usize,
                    "exception-guard resume pc {resume_pc} >= AFTER_RESIDUAL_CALL_PC_FLAG; \
                     function too large for bit-14 resume encoding"
                );
                let snapshot =
                    this.build_framestack_snapshot(ctx, resume_pc, &active_boxes, &fail_arg_types);
                // Snapshot is the source of truth —
                // the optimizer's `store_final_boxes_in_guard`
                // (`optimizeopt/mod.rs:3200`) overwrites `op.fail_args`
                // from the snapshot via
                // `op.store_final_boxes(liveboxes)` (mod.rs:3392), so
                // the inline `record_guard_typed_with_fail_args` copy
                // was redundant.  Mirrors RPython
                // `pyjitpl.MetaInterp.generate_guard`
                // (pyjitpl.py:2558-2602) which records the guard with
                // no inline fail_args and lets `capture_resumedata` +
                // `_number_boxes` populate them from the snapshot
                // chain.
                let all_types = this.extend_types_with_parents(ctx, fail_arg_types);
                let snapshot_id = ctx.capture_resumedata(snapshot);

                let exc_type_const = ctx.const_int(exc_type_ptr);
                let op = ctx.record_guard_typed(
                    majit_ir::OpCode::GuardException,
                    &[exc_type_const],
                    all_types,
                );
                ctx.set_last_guard_resume_position(snapshot_id);

                this.orgpc = saved_orgpc;
                op
            });

            // pyjitpl.py:3385-3392:
            //   val = cast_opaque_ptr(GCREF, self.last_exc_value)
            //   if self.class_of_last_exc_is_const:
            //       self.last_exc_box = ConstPtr(val)
            //   else:
            //       self.last_exc_box = op
            //       op.setref_base(val)
            //   self.class_of_last_exc_is_const = True
            if self.sym().class_of_last_exc_is_const {
                let exc_box = self.with_ctx(|_this, ctx| ctx.const_ref(exc_obj as i64));
                self.sym_mut().last_exc_box = exc_box;
            } else {
                self.sym_mut().last_exc_box = guard_op;
            }
            self.sym_mut().class_of_last_exc_is_const = true;

            // pyopcode.py: generic raise paths carry no saved lasti; the
            // RERAISE-issuing site is the only producer of reraise_lasti.
            self.finishframe_exception(code, pc, -1)
        } else {
            // Per-caller GUARD_NO_EXCEPTION is emitted inline at each
            // can-raise CALL_* site (pyjitpl.py:2082 do_residual_call), so
            // no fallback emit is needed when no exception was observed.
            TraceAction::Continue
        }
    }

    /// RPython pyjitpl.py:1701 opimpl_reraise parity.
    ///
    /// Unlike generic exc=True ops, RERAISE does not go through
    /// GUARD_EXCEPTION. It resumes directly from the already-known
    /// `last_exc_value` and unwinds to the enclosing handler.
    ///
    /// `reraise_lasti` mirrors PyPy `pyopcode.py:122 handle_operation_error`'s
    /// like-named parameter: when non-negative it is the original raise-site
    /// offset the RERAISE bytecode extracted from the stack via
    /// `pyopcode.py:1361 self.space.int_w(self.peekvalue(oparg))`.
    fn handle_reraise(&mut self, code: &CodeObject, pc: usize, reraise_lasti: i32) -> TraceAction {
        let s = self.sym();
        if s.last_exc_value.is_null() || s.last_exc_box == OpRef::NONE {
            return TraceAction::Abort;
        }
        self.finishframe_exception(code, pc, reraise_lasti)
    }

    /// RPython pyjitpl.py:1688 opimpl_raise parity.
    ///
    /// Explicit raise shares the unwind path with reraise, but seeds
    /// `last_exc_box` from the just-popped exception box instead of
    /// going through GUARD_EXCEPTION.
    fn handle_raise_varargs(&mut self, code: &CodeObject, pc: usize, argc: usize) -> TraceAction {
        if argc == 0 {
            // RAISE_VARARGS 0 is the bare `raise` re-raise.  It carries no
            // saved lasti — only the explicit `RERAISE N` opcode does.
            return self.handle_reraise(code, pc, -1);
        }
        {
            let s = self.sym();
            if s.last_exc_value.is_null() || s.last_exc_box == OpRef::NONE {
                return TraceAction::Abort;
            }
        }
        self.finishframe_exception(code, pc, -1)
    }

    /// PyPy `RAISE_VARARGS` materialization paired with RPython
    /// pyjitpl.py:1690-1696 `opimpl_raise` bookkeeping.
    ///
    /// `GuardClass(exc_box, cls_const)` reads `ob_header.ob_type` at
    /// `cpu.vtable_offset = OB_TYPE_OFFSET = 0`.  Pyre allocates each
    /// `W_ExceptionObject` with `ob_type` pointing at the per-`ExcKind`
    /// `PyType` static (`EXC_VALUE_ERROR_TYPE`, `EXC_OVERFLOW_ERROR_TYPE`,
    /// …; `excobject.rs::exc_kind_to_pytype`), so this guard
    /// discriminates the actual subclass.  Matches RPython's
    /// `OBJECT.typeptr = specific class` (`rclass.py:167-174`) and
    /// `opimpl_raise`'s `cls_of_box(exc)` shape (`pyjitpl.py:1687-1693`).
    fn seed_raised_exception(&mut self, exc_box: OpRef, concrete_exc: PyObjectRef) {
        if !concrete_exc.is_null() {
            let exc_class_ptr = unsafe {
                (*(concrete_exc as *const pyre_object::excobject::W_ExceptionObject))
                    .ob_header
                    .ob_type
            };
            self.with_ctx(|this, ctx| {
                if !ctx.heap_cache().is_class_known(exc_box) {
                    let cls_const = ctx.const_int(exc_class_ptr as usize as i64);
                    this.generate_guard(ctx, OpCode::GuardClass, &[exc_box, cls_const]);
                    ctx.heap_cache_mut()
                        .class_now_known(exc_box, exc_class_ptr as usize as i64);
                }
            });
        }
        let s = self.sym_mut();
        s.last_exc_value = concrete_exc;
        s.last_exc_box = exc_box;
        s.class_of_last_exc_is_const = true;
    }

    /// RPython pyjitpl.py:2506 finishframe_exception (single-frame).
    ///
    /// Checks current frame for an exception handler.
    /// If found: unwind stack to handler depth, push exception, continue.
    /// If not found: return Abort (metainterp handles multi-frame unwind).
    ///
    /// `reraise_lasti` is PyPy `pyopcode.py:122 handle_operation_error`'s
    /// like-named parameter — non-negative when this dispatch was driven by
    /// a `RERAISE N` opcode that saved the original raise-site offset.  Used
    /// (a) for the synthesized `lasti` push at handler entry per
    /// `pyopcode.py:165-170`, and (b) to restore `frame.last_instr` on the
    /// no-handler propagation path per `pyopcode.py:181-184`.
    fn finishframe_exception(
        &mut self,
        code: &CodeObject,
        pc: usize,
        reraise_lasti: i32,
    ) -> TraceAction {
        let exc_opref = self.sym().last_exc_box;
        let exc_obj = self.sym().last_exc_value;
        let concrete_frame_addr = self.concrete_frame_addr;

        // pyjitpl.py:2510-2520: scan for catch_exception handler
        // (Python 3.11+ exception table replaces RPython's op_catch_exception).
        // `lookup_exceptiontable` takes byte offsets; pyre tracks `pc` as
        // a code-unit index, so multiply/divide by 2 at the boundary.
        if let Some((target_bytes, depth, lasti)) =
            pyre_interpreter::exception_table::lookup_exceptiontable(
                &code.exceptiontable,
                (pc * 2) as u32,
            )
        {
            let handler_pc = target_bytes as usize / 2;
            let handler_depth = depth as usize;
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][finishframe_exception] pc={} handler={} depth={}",
                    pc, handler_pc, handler_depth
                );
            }

            // pyjitpl.py:2506 finishframe_exception: unwind stack to handler,
            // pyjitpl.py:2517: frame.pc = target; raise ChangeFrame
            let ncells = code.cellvars.len() + code.freevars.len();
            let nlocals = self.sym().nlocals;
            let target_stack_len = ncells + handler_depth;
            let (has_vable, old_vsd, post_truncate_vsd) = {
                let s = self.sym_mut();
                let old_vsd = s.valuestackdepth;
                s.symbolic_stack_types.truncate(target_stack_len);
                s.concrete_stack.truncate(target_stack_len);
                s.valuestackdepth = nlocals + target_stack_len;
                if s.registers_r.len() > nlocals + target_stack_len {
                    s.registers_r.truncate(nlocals + target_stack_len);
                }
                (s.owns_virtualizable_shadow(), old_vsd, s.valuestackdepth)
            };
            // opencoder.py:718 `_list_of_boxes_virtualizable` parity: the
            // snapshot reads only `virtualizable_boxes`, so the unwind
            // must also mirror onto the shadow. Clear truncated slots to
            // PY_NULL (pyframe.py:411 popvalue_maybe_none) BEFORE pushing
            // lasti / exc so the subsequent `write_stack_slot` shadow set
            // for exc is the last write.
            let static_offset = crate::virtualizable_gen::NUM_VABLE_SCALARS;
            if has_vable && old_vsd > post_truncate_vsd {
                let null_opref =
                    self.with_ctx(|_this, ctx| ctx.const_ref(pyre_object::PY_NULL as i64));
                let null_value =
                    majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::PY_NULL as usize));
                self.with_ctx(|_this, ctx| {
                    for reg_idx in post_truncate_vsd..old_vsd {
                        let flat_idx = static_offset + reg_idx;
                        ctx.set_virtualizable_entry_at(flat_idx, null_opref, null_value);
                    }
                });
            }
            let lasti_obj = if lasti {
                // pyopcode.py:165-170 lasti push:
                //   if reraise_lasti >= 0:
                //       lasti_value = reraise_lasti
                //   else:
                //       lasti_value = intmask(self.last_instr)
                //   self.pushvalue(self.space.newint(lasti_value))
                //
                // Python 3.11 exception-table adaptation: `push_lasti`
                // pushes a real W_Int object onto `locals_cells_stack_w`.
                // Mirror it through the same stack/vable helper as every
                // other W_Root push so guard/resume snapshots see the
                // object in `virtualizable_boxes`, not a PY_NULL lazy-fill
                // placeholder.
                let lasti_value: i64 = if reraise_lasti >= 0 {
                    reraise_lasti as i64
                } else {
                    pc as i64
                };
                let lasti_obj = pyre_object::w_int_new(lasti_value);
                let lasti_opref = self.with_ctx(|_this, ctx| ctx.const_ref(lasti_obj as i64));
                self.with_ctx(|this, ctx| {
                    let s = this.sym_mut();
                    let stack_idx = s.valuestackdepth - s.nlocals;
                    write_stack_slot(
                        s,
                        ctx,
                        stack_idx,
                        lasti_opref,
                        ConcreteValue::Ref(lasti_obj),
                    );
                    this.sym_mut().valuestackdepth += 1;
                });
                Some(lasti_obj)
            } else {
                None
            };
            self.with_ctx(|this, ctx| {
                let s = this.sym_mut();
                let stack_idx = s.valuestackdepth - s.nlocals;
                write_stack_slot(s, ctx, stack_idx, exc_opref, ConcreteValue::Ref(exc_obj));
                this.sym_mut().valuestackdepth += 1;
            });
            // Sync concrete frame.
            let frame = unsafe { &mut *(concrete_frame_addr as *mut pyre_interpreter::PyFrame) };
            let target_depth = frame.nlocals() + frame.ncells() + handler_depth;
            while frame.valuestackdepth > target_depth {
                frame.pop();
            }
            if let Some(lasti_obj) = lasti_obj {
                frame.push(lasti_obj);
            }
            frame.push(exc_obj);
            // pyjitpl.py:2518: frame.pc = target; raise ChangeFrame
            self.sym_mut().pending_next_instr = Some(handler_pc);
            TraceAction::Continue
        } else {
            // pyopcode.py:175-184 no-handler propagation:
            //   if reraise_lasti >= 0:
            //       self.last_instr = reraise_lasti
            //   self.frame_finished_execution = True
            //
            // The trace bails to the interpreter (Abort).  The interpreter's
            // `handle_exception` will continue propagation through parent
            // frames, so the RERAISE-issuing frame's `last_instr` must be
            // restored here for `f_lineno` to report the original raise
            // site rather than the RERAISE site, and `frame_finished_execution`
            // must mirror PyPy so anything inspecting the dead frame (clear,
            // generator close, traceback walkers) sees the same flag state.
            {
                let frame =
                    unsafe { &mut *(concrete_frame_addr as *mut pyre_interpreter::PyFrame) };
                if reraise_lasti >= 0 {
                    frame.last_instr = reraise_lasti as isize;
                }
                frame.frame_finished_execution = true;
            }
            // No handler in this frame — return Abort so metainterp's
            // multi-frame finishframe_exception can pop this frame and
            // try the parent (pyjitpl.py:2520 self.popframe() loop).
            // Root frame with no handler → metainterp emits FINISH
            // (pyjitpl.py:2532 compile_exit_frame_with_exception).
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[jit][finishframe_exception] no handler pc={}", pc);
            }
            TraceAction::Abort
        }
    }

    pub fn trace_code_step_inline(
        &mut self,
        code: &CodeObject,
        pc: usize,
    ) -> InlineTraceStepAction {
        if pc >= code.instructions.len() {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step_inline pc_oob pc={} code_len={}",
                    pc,
                    code.instructions.len()
                );
            }
            return InlineTraceStepAction::Trace(TraceAction::Abort);
        }

        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step_inline decode_failed pc={}",
                    pc
                );
            }
            return InlineTraceStepAction::Trace(TraceAction::Abort);
        };

        self.set_orgpc(pc);
        self.prepare_fallthrough();
        // Keep inline-frame guard capture aligned with the root-frame path:
        // only opcodes that can actually reach a guard carry an opcode-start
        // snapshot, and specific guard paths may still suppress it.
        let needs_pre_opcode_snapshot =
            crate::state::instruction_needs_pre_opcode_snapshot(instruction);
        if needs_pre_opcode_snapshot {
            self.capture_pre_opcode_state();
        }
        // pyjitpl.py:541-556 fused COMPARE_OP + PopJumpIf* parity (see
        // trace_code_step's counterpart for the full rationale).
        let fused_result = if let Instruction::CompareOp { opname } = instruction {
            let comp_op = opname.get(op_arg);
            match self.try_fused_compare_goto_if_not(code, pc, comp_op) {
                Ok(Some(action)) => Some(Ok(action)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        } else {
            None
        };
        let step_result = match fused_result {
            Some(Ok(action)) => {
                // Symmetry with trace_code_step: if pre_opcode_* were ever
                // captured for the inline frame (none do today, but future
                // work may add it), clear them on the fused-return path the
                // same way the non-inline path does at the post-dispatch
                // clear below.
                if needs_pre_opcode_snapshot {
                    self.clear_pre_opcode_state();
                }
                return InlineTraceStepAction::Trace(action);
            }
            Some(Err(e)) => Err(e),
            None => {
                // Inline-frame gate: `trace_code_step_inline` always
                // steps in an inline-traced sub frame, so
                // `walker_capture_snapshot_for_last_guard`'s single-frame
                // capture would drop the parent chain (see comment in
                // `trace_code_step` for full rationale).  Walker dispatch
                // remains disabled here until multi-frame walker
                // snapshots land; route every opcode through trait
                // dispatch so the snapshot encoder sees a fully formed
                // framestack at every guard.  Shadow validation is gated
                // the same way: `shadow_validate_pre` runs the symbolic
                // walker, which captures a single-frame snapshot under
                // `MAJIT_SHADOW_WALKER=1` and would diverge from the
                // trait dispatcher's multi-frame view in inline frames.
                if flip_probe_enabled() {
                    let reason = if self.parent_frames.is_empty() {
                        "toplevel-inline-step"
                    } else {
                        "inline-frame"
                    };
                    flip_probe_record(&instruction, reason);
                }
                let shadow_outcome = if self.parent_frames.is_empty() {
                    crate::shadow_walker::shadow_validate_pre(self, &instruction, op_arg)
                } else {
                    None
                };
                let result = execute_opcode_step(self, code, instruction, op_arg, pc + 1);
                if result.is_ok() {
                    if let Some(outcome) = shadow_outcome {
                        crate::shadow_walker::shadow_validate_post(self, outcome);
                    }
                }
                result
            }
        };
        if needs_pre_opcode_snapshot {
            self.clear_pre_opcode_state();
        }
        // RPython execute_ll_raised parity: generic exc=True ops store the
        // exception in last_exc_value. Valid raise/reraise paths seed
        // `last_exc_box` directly; malformed bytecode-level raises still
        // use the generic raised-exception path below.
        if step_result.as_ref().err().is_some_and(is_trace_abort_error) {
            return InlineTraceStepAction::Trace(TraceAction::Abort);
        }
        if let Err(ref err) = step_result {
            if !matches!(
                instruction,
                Instruction::Reraise { .. } | Instruction::RaiseVarargs { .. }
            ) || self.sym().last_exc_box == OpRef::NONE
            {
                let exc_obj = err.to_exc_object();
                let s = self.sym_mut();
                if s.last_exc_value.is_null() {
                    s.last_exc_value = exc_obj;
                }
            }
        }
        if let Instruction::Reraise { depth } = instruction {
            // Mirror the root-frame dispatcher: thread `err.reraise_lasti`
            // into `handle_reraise`, abort if `oparg != 0 && lasti < 0`
            // (trace cannot fold a const-int from the symbolic stack — fall
            // back to interpreter via TraceAction::Abort).
            let oparg_val = depth.get(op_arg);
            let reraise_lasti = step_result
                .as_ref()
                .err()
                .map(|e| e.reraise_lasti)
                .unwrap_or(-1);
            if oparg_val != 0 && reraise_lasti < 0 {
                // Wipe the trace's tracked exception state so the
                // metainterp's Abort handler skips its in-trace
                // `finishframe_exception` (which would otherwise dispatch
                // a handler-entry push with `reraise_lasti=-1`,
                // synthesizing the WRONG lasti and mutating
                // `owned_concrete_frame` in a way the interpreter would
                // then resume with).  `executor.reraise()` only mutated
                // the symbolic stack and emitted IR — both abandoned on
                // abort.  The concrete PyFrame is untouched, so the
                // interpreter re-runs RERAISE freshly via its own
                // `peekvalue(oparg)` path and gets the correct saved
                // lasti from runtime state.
                let s = self.sym_mut();
                s.last_exc_value = pyre_object::PY_NULL;
                s.last_exc_box = OpRef::NONE;
                return InlineTraceStepAction::Trace(TraceAction::Abort);
            }
            if self.sym().last_exc_box != OpRef::NONE {
                return InlineTraceStepAction::Trace(self.handle_reraise(code, pc, reraise_lasti));
            }
            // Same rationale as root dispatcher: abort RERAISE-with-saved-
            // lasti when the trace lacks a tracked exception box (generic
            // fallback would drop the lasti).
            if oparg_val != 0 {
                // last_exc_box is already NONE here; keep last_exc_value
                // clear for symmetry with the const-fold-abort branch
                // above so metainterp's `has_exc` check observes no
                // tracked exception either way.
                self.sym_mut().last_exc_value = pyre_object::PY_NULL;
                return InlineTraceStepAction::Trace(TraceAction::Abort);
            }
            if step_result.is_err() {
                return InlineTraceStepAction::Trace(self.handle_possible_exception(code, pc));
            }
        }
        if let Instruction::RaiseVarargs { argc } = instruction {
            if self.sym().last_exc_box != OpRef::NONE {
                return InlineTraceStepAction::Trace(self.handle_raise_varargs(
                    code,
                    pc,
                    argc.get(op_arg) as usize,
                ));
            }
            if step_result.is_err() {
                return InlineTraceStepAction::Trace(self.handle_possible_exception(code, pc));
            }
        }
        if instruction_may_raise(instruction) {
            let exc_action = self.handle_possible_exception(code, pc);
            if !matches!(exc_action, TraceAction::Continue) {
                return InlineTraceStepAction::Trace(exc_action);
            }
            if step_result.is_err() {
                return InlineTraceStepAction::Trace(TraceAction::Continue);
            }
        }
        let action = match step_result {
            Ok(pyre_interpreter::StepResult::Return(value)) => {
                // pyjitpl.py:2489-2502 compile_done_with_this_frame parity:
                // result_type is always REF. ensure_boxed_for_ca re-boxes
                // unboxed Int/Float values (NewWithVtable + SetfieldGc).
                let v = value.opref;
                let finish_value =
                    self.with_ctx(|this, ctx| crate::state::ensure_boxed_for_ca(ctx, this, v));
                TraceAction::Finish {
                    finish_args: vec![finish_value],
                    finish_arg_types: vec![Type::Ref],
                    exit_with_exception: false,
                }
            }
            Err(_) => TraceAction::Abort,
            other => self.into_trace_action(other),
        };
        match action {
            TraceAction::Continue => {
                if let Some(mut pending) = self.pending_inline_frame.take() {
                    let sym = self.sym();
                    let result_idx = sym
                        .valuestackdepth
                        .saturating_sub(sym.nlocals)
                        .checked_sub(1)
                        .expect("pending inline frame missing caller result slot");
                    pending.caller_result_stack_idx = Some(result_idx);
                    InlineTraceStepAction::PushFrame(pending)
                } else {
                    InlineTraceStepAction::Trace(TraceAction::Continue)
                }
            }
            other => InlineTraceStepAction::Trace(other),
        }
    }
}

/// Returns (is_int, is_float) for the fused-dispatch fuseability gate.
/// Mirrors the concrete-type classification codegen.rs uses to pick
/// between the int and float fast paths in generated_compare_value_direct,
/// but reads the ConcreteValue variant directly so the check does not
/// have to allocate an intermediate w_int/w_float box.
/// Trace-side mirror of `pyre_interpreter::eval::check_exc_match_against`
/// (`pyre/pyre-interpreter/src/eval.rs:81-130`).  Kept structurally
/// identical so the recorded boolean matches the interpreter's
/// concrete computation — including tuple-of-types, builtin-function
/// alias, str-named exception kinds, and `is_type` + MRO fallback.
///
/// SAFETY: `exc_value` and `exc_type` must be non-null PyObjectRefs
/// owned by the running interpreter for the duration of this call.
unsafe fn trace_check_exc_match_against(
    exc_value: pyre_object::PyObjectRef,
    exc_type: pyre_object::PyObjectRef,
) -> bool {
    // pyopcode.py:1040 — bool-returning mirror of
    // `pyre_interpreter::eval::check_exc_match_against`. The validity
    // gate at pyopcode.py:1034-1039 lives in
    // `pyre_interpreter::eval::validate_check_exc_match_class` and runs
    // BEFORE this helper in the BC handler, matching the interpreter's
    // split. PyPy's `@jit.unroll_safe cmp_exc_match` inlines both into
    // the trace and emits a guard for the `raise oefmt(...)` arm;
    // pyre's residual-call ABI keeps `bool` so the raise/guard split
    // lives on the caller side.
    let Some(w_exc_class) = pyre_interpreter::typedef::r#type(exc_value) else {
        return false;
    };
    pyre_interpreter::baseobjspace::exception_match(w_exc_class, exc_type)
}

/// Production-walker allow-list for the walker cutover.
///
/// RPython parity: `pyjitpl.py:1892 MetaInterp._interpret` dispatches
/// every opcode through the single jitcode-bytecode path; there is no
/// dual trait/walker split upstream.  This predicate names the Python
/// instructions for which pyre has already retired the root-frame trait
/// dispatch — for those, `trace_code_step` routes directly through
/// `MIFrame::dispatch_via_walker_for_opcode` and the root-frame trait
/// path is dead code at runtime.  `trace_code_step_inline` intentionally
/// stays on the trait path until walker snapshots can encode the full
/// parent-frame chain.  The trait impl will be removed once every opcode
/// is in this set and the inline-frame snapshot gap is closed.
///
/// The pre-jtransform unit-variant fold
/// (`translator/rtyper/unit_variant_fold.rs::fold_unit_variant_ctors`)
/// unblocked the Nop family by rewriting zero-arg
/// `SyntheticTransparentCtor` calls (`StepResult::Continue`,
/// `LoopResult::Done`, …) to `OpKind::ConstRef(prebuilt_instance)` on
/// `model::FunctionGraph` before `Transformer::transform` runs.  Arm
/// bytes then decode to the real no-op shape (`ref_return/r …`) without
/// residual-call wrappers, so the Nop batch — plus every entry below —
/// is safe to dispatch via walker today.
///
/// Subsequent batches grow this set until it covers every Python
/// opcode.  The remaining wall is execution-side, not recording-side:
/// `eval_loop_jit` skips `execute_opcode_step` for walker-handled
/// opcodes, which is correct only when the arm body's heap effects ride
/// `vable_setfield` / `setarrayitem_vable_r`, or when walker dispatch
/// concrete-executes any non-elidable `residual_call` through
/// `try_execute_residual_call_via_executor`.
/// The trait infra is deleted only after this predicate covers every
/// opcode.
///
pub fn production_walker_handles(instruction: &Instruction) -> bool {
    // The unit-variant fold rewrites `StepResult::Continue` &c. to
    // `ConstRef(prebuilt_instance)`; the assembler lowers `ConstRef`
    // through `ref_copy/r>r` (`assembler.rs::emit_const_r`), so arms
    // decode without residual-call wrappers around unit variants.
    //
    // BuildTuple / UnpackSequence are intentionally NOT in this set: the
    // trait-dispatch handlers (`trace_build_tuple_value`,
    // `unpack_sequence_value`) carry the specialised arity-2 tuple
    // build/unpack fast paths, and the walker's jitcode dispatch cannot
    // yet express them (it aborts with `InlineCallArityMismatch` when the
    // tuple flows through an inlined call). Keep them on the trait path
    // until the walker can encode the specialised tuple layout.
    //
    // StoreFastStoreFast is also excluded: its codewriter arm lowers to a
    // chain of `residual_call` ops whose funcptr `constants_i` entries are
    // unresolved placeholders (not patched by `patch_constants_i_fnaddrs`,
    // which only rewrites build→runtime fnaddr pairs). The walker reads one
    // of those placeholders as the call target and branches to it (`blr`),
    // faulting on an unmapped address. Route it through the trait handler
    // (`opcode_store_fast_store_fast`) until the arm's helper fnaddrs are
    // resolved.
    matches!(
        instruction,
        Instruction::Nop
            // LoadConst handled by the dispatch_via_walker_for_opcode entry
            // hook: resolves the constant from `consti` + the code pool and
            // delegates to `OpcodeStepExecutor::load_const`, whose
            // `push_value` advances vsd.  The auto-gen arm residualises
            // `opcode_load_const(frame, &ConstantData)` with the
            // oparg-derived `&ConstantData` operand left unbound by the
            // r0-only arm entry (ResidualCallArgUnbound).
            | Instruction::LoadConst { .. }
            // LoadSmallInt handled by the entry hook (const-push sibling of
            // LoadConst; small int in the oparg → load_small_int → int
            // ConstRef + push_value).
            | Instruction::LoadSmallInt { .. }
            // LoadFast / LoadFastBorrow handled by the
            // dispatch_via_walker_for_opcode entry hook: resolves the
            // var_num local index + name and delegates to
            // OpcodeStepExecutor::load_fast_checked (push_value advances
            // vsd).  The auto-gen arm reads the var_num oparg payload from
            // a register the r0-only arm entry leaves unbound.
            | Instruction::LoadFast { .. }
            | Instruction::LoadFastBorrow { .. }
            // StoreFast handled by the dispatch_via_walker_for_opcode entry
            // hook: delegates to OpcodeStepExecutor::store_fast (pop_value
            // advances vsd, setarrayitem_vable_r writes the local).
            // Distinct from StoreFastStoreFast, which stays trait-routed
            // (#405 arm-entry var_nums seeding).
            | Instruction::StoreFast { .. }
            // BinaryOp handled by the dispatch_via_walker_for_opcode entry
            // hook: delegates to OpcodeStepExecutor::binary_op (pop_value
            // ×2 + specialised arithmetic IR + push_value).  May-raise; the
            // Err propagates to trace_code_step like the trait leg.
            | Instruction::BinaryOp { .. }
            | Instruction::ExtendedArg
            | Instruction::Resume { .. }
            | Instruction::Cache
            | Instruction::NotTaken
            | Instruction::ExitInitCheck
            | Instruction::EndFor
            | Instruction::UnaryNot
            | Instruction::UnaryInvert
            | Instruction::GetIter
            | Instruction::MatchMapping
            | Instruction::MatchSequence
            | Instruction::SetupAnnotations
            | Instruction::FormatSimple
            | Instruction::FormatWithSpec
            | Instruction::MakeFunction
            | Instruction::GetYieldFromIter
            | Instruction::PopIter
            | Instruction::EndSend
            | Instruction::DeleteSubscr
            | Instruction::LoadLocals
            | Instruction::LoadBuildClass
            | Instruction::BuildTemplate
            | Instruction::Copy { .. }
            | Instruction::BinarySlice
            | Instruction::StoreSlice
            | Instruction::ContainsOp { .. }
            | Instruction::IsOp { .. }
            | Instruction::Swap { .. }
            | Instruction::BuildList { .. }
            | Instruction::BuildSet { .. }
            | Instruction::BuildString { .. }
            | Instruction::BuildMap { .. }
            | Instruction::BuildSlice { .. }
            | Instruction::LoadFastAndClear { .. }
            | Instruction::ListAppend { .. }
            | Instruction::ListExtend { .. }
            | Instruction::SetAdd { .. }
            | Instruction::SetUpdate { .. }
            | Instruction::MapAdd { .. }
            | Instruction::DictUpdate { .. }
            | Instruction::DictMerge { .. }
            | Instruction::SetFunctionAttribute { .. }
            | Instruction::UnpackEx { .. }
            // Instruction::LoadName / StoreName excluded: the trait
            // handlers (`load_name_value` / `store_name_value`) carry the
            // celldict ModuleDictStrategy fast path (LOAD_GLOBAL_cached
            // celldict.py:285-322: QUASIIMMUT_FIELD version dep + elidable
            // cell fold; STORE_GLOBAL_cached celldict.py:328-333:
            // layout-agnostic setitem_str). The walker's jitcode arm for
            // these aborts with ResidualCallArgUnbound on the &str name
            // argument — it was never exercised before module-level frames
            // became portals.
            | Instruction::StoreGlobal { .. }
            | Instruction::DeleteAttr { .. }
            | Instruction::ImportName { .. }
            | Instruction::ImportFrom { .. }
            | Instruction::LoadSuperAttr { .. }
            | Instruction::BuildInterpolation { .. }
            | Instruction::CallIntrinsic1 { .. }
            | Instruction::CallIntrinsic2 { .. }
            | Instruction::GetLen
            | Instruction::LoadSpecial { .. }
            | Instruction::LoadFromDictOrGlobals { .. }
            | Instruction::LoadFromDictOrDeref { .. }
            | Instruction::LoadDeref { .. }
            | Instruction::LoadFastCheck { .. }
            | Instruction::LoadCommonConstant { .. }
            | Instruction::GetAiter
            | Instruction::GetAwaitable { .. }
            | Instruction::StoreDeref { .. }
            | Instruction::YieldValue { .. }
            | Instruction::ReturnGenerator
            | Instruction::Send { .. }
            | Instruction::GetAnext
            | Instruction::EndAsyncFor
            | Instruction::CleanupThrow
            | Instruction::WithExceptStart
            | Instruction::DeleteFast { .. }
            | Instruction::DeleteDeref { .. }
            | Instruction::DeleteGlobal { .. }
            | Instruction::DeleteName { .. }
            | Instruction::CopyFreeVars { .. }
            | Instruction::MakeCell { .. }
            | Instruction::ConvertValue { .. }
            | Instruction::Reraise { .. } // walker hook in dispatch_via_walker_for_opcode delegates to MIFrame::reraise which returns Err(PyError) with reraise_lasti seeded — `step_result.err().map(|e| e.reraise_lasti)` extraction works the same on either leg
            | Instruction::PopJumpIfNone { .. }
            | Instruction::PopJumpIfNotNone { .. }
            | Instruction::ForIter { .. }
            | Instruction::CallKw { .. }
            | Instruction::CallFunctionEx
            // Instruction::LoadAttr is walker-routed EXCEPT the foldable
            // builtin list-method form (append/pop/reverse on a list
            // receiver), which the dispatch gate carves out to the trait
            // leg so MIFrame::load_method can resolve it to a
            // class-guarded Const unbound function (the list
            // specialization); see is_foldable_list_method_load_attr.
            | Instruction::LoadAttr { .. }
            | Instruction::StoreAttr { .. }
            // Instruction::StoreFastStoreFast excluded: a probe flip
            // (2026-06-13) confirmed the GcVec-fnaddr SIGBUS the prior
            // comment described is gone (the #406 result_exc lowering
            // drained the Result-shell residual chains in pop_value /
            // store_local_value).  The remaining blocker is shared with
            // BuildTuple / UnpackSequence: `ResidualCallArgUnbound`
            // (var_nums payload register unbound) because arm entry seeds
            // only r0.  Re-enable once arm-entry operand seeding (#405)
            // lands on dispatch_via_miframe_at_opcode_entry.
            | Instruction::StoreSubscr // handled by dispatch_via_walker_for_opcode entry hook
            // Instruction::PopExcept excluded: same bridge-tracing
            // rationale as PushExcInfo below — its arm manages the
            // exception-info stack via impure helper jitcodes the walker
            // cannot execute concretely, so on the bridge leg the handler's
            // stack is left inflated and the loop-back `Jump` carries the
            // wrong arg count.  Trait dispatch executes it concretely.
            // Instruction::PushExcInfo excluded: its codewriter arm
            // `inline_call`s the exception-info-stack helper jitcodes,
            // which branch on the concrete result of an impure /
            // may-raise `residual_call`.  The production walker only folds
            // the concrete result of *pure* calls
            // (`try_fold_pure_call_via_executor`), so the helper's
            // post-call `goto_if_not` reads a `Null` concrete and mis-routes
            // into the helper's `raise/r` tail — surfacing a spurious
            // `SubRaise { exc_concrete: Null }` that aborts every bridge
            // trace of an exception handler.  The walker path was never
            // exercised before (the main loop only traces the no-exception
            // path; the bridge that reaches the handler never compiled
            // until the exc-value threading fix landed).  Keep PUSH_EXC_INFO
            // on trait dispatch — which executes the opcode concretely — the
            // same rationale that excludes `Reraise` above, until the walker
            // can execute impure residual calls during tracing.
            // Instruction::PopTop excluded: in the exception-handler
            // region it pops the matched exception value off the stack the
            // PUSH_EXC_INFO / POP_EXCEPT helpers manage.  Leaving it on the
            // walker while those are trait-dispatched desyncs the bridge's
            // handler-region stack, producing a loop-back `Jump` with the
            // wrong arg count (bridge fails to compile) and a backend
            // regalloc panic.  Keep the whole handler region on one concrete
            // (trait) leg so the bridge's framestate matches the loop entry.
            // PushNull is handled by the dispatch_via_walker_for_opcode
            // entry hook (direct const-NULL symbolic push): its auto-gen
            // arm is an inert residual wrapper (`opcode_push_null` is not
            // in the runtime fnaddr registry), so the arm-walk +
            // vsd-resync route loses the push and underflows the
            // following CALL.
            | Instruction::PushNull
    )
}

/// Phase-5 flip-completion probe: is `PYRE_FLIP_PROBE` set?  Read once.
/// When set, the trait-dispatch fallback callers record each unique
/// (opcode, reason) that still bypasses the walker, so the residual flip
/// surface is measurable against the production workload.  Inert (a
/// single cached bool load) when unset.
fn flip_probe_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FLIP_PROBE").is_some())
}

/// Record one trait-dispatch fallback hit, deduplicated per
/// (opcode-variant, reason) so each distinct residual case prints once
/// instead of once-per-opcode-execution.  `reason` distinguishes
/// not-in-allowlist vs the context gates (inline-frame / foldable
/// list-method LOAD_ATTR).
fn flip_probe_record(instruction: &Instruction, reason: &str) {
    use std::cell::RefCell;
    use std::collections::HashSet;
    thread_local! {
        static SEEN: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
    }
    let dbg = format!("{instruction:?}");
    let name = dbg
        .split(|c: char| c == ' ' || c == '{' || c == '(')
        .next()
        .unwrap_or(&dbg);
    let key = format!("{name}|{reason}");
    SEEN.with(|seen| {
        if seen.borrow_mut().insert(key) {
            eprintln!("[flip-probe] trait-fallback opcode={name} reason={reason}");
        }
    });
}

/// Apply the symbolic-tracker side effects of a walker-handled opcode.
///
/// The walker arm records IR ops for stack pushes/pops by walking the
/// per-opcode arm body (which contains `setarrayitem_vable_r` /
/// `setfield_vable_i` against the user-side virtualizable).  Those IR
/// ops correctly mutate the live `PyFrame` at trace-execution time.
/// What the walker arm does NOT do is update the tracer's symbolic
/// shadow that subsequent trait-dispatched opcodes consult:
///
/// * `sym.valuestackdepth` — the symbolic mirror of
///   `PyFrame.valuestackdepth`.  Trait dispatch advances it via
///   `MIFrame::push_typed_value` / `MIFrame::pop_value` (trace_opcode.rs
///   :1797..1925); the walker arm bypasses both.
/// * `sym.vable_valuestackdepth` — the IR `OpRef` cache pointing at
///   the current `const_int(vsd)` value.  Snapshot encoders read this
///   OpRef into the guard's resume-data scalar header; if it lags the
///   IR the blackhole resume restores a stale vsd into PyFrame.
/// * `virtualizable_boxes[NUM_VABLE_SCALARS + semantic_idx]` — the
///   heap-mirror shadow for the pushed/popped stack slot.  `pop_value`
///   clears the popped slot to NULL (line 1901-1906); `push_typed_value`
///   writes the new value via `write_stack_slot` (line 572-611).  The
///   walker arm's `setarrayitem_vable_r` IR op updates the runtime
///   heap, but the tracer's shadow needs the same update applied here.
///
/// Resync `sym.valuestackdepth` from the live `PyFrame` after a walker
/// arm with non-zero net stack effect has run.
///
/// The walker arm emits `setfield_vable_i(valuestackdepth, …)` for any
/// push/pop; that op routes through `vable_setfield` →
/// `synchronize_virtualizable()` (pyjitpl.py:3470-3474), which writes
/// the new vsd into the heap `PyFrame` at trace time.  Pyre's tracer
/// maintains a parallel `sym.valuestackdepth` shadow that the walker
/// path does not touch (pyre-only adaptation; PyPy's MIFrame has no
/// such counter — rpython/jit/metainterp/pyjitpl.py:65-93).  Read the
/// live depth back so the tracer's symbolic counter stays consistent
/// with subsequent trait-dispatched opcodes that consume
/// `sym.valuestackdepth`.
fn resync_sym_vsd_from_concrete_frame(state: &mut MIFrame) {
    let new_vsd = match state.concrete_valuestackdepth() {
        Some(v) => v,
        // Inline-callee tracing has no live `PyFrame`; the walker path
        // is not reachable for inlined opcodes today (production walker
        // dispatch only fires on top-level opcode entries), so a None
        // here indicates the seed contract was violated upstream.
        None => return,
    };
    // Mirror `sym.concrete_stack` from the live `PyFrame` so trait-
    // dispatched callers reading `concrete_stack[stack_idx]` after a
    // walker push/pop see the post-opcode value instead of the
    // pre-opcode placeholder.  Same pattern as `PyreSym::setup_call`
    // (state.rs:3336-3347).
    let frame_addr = state.concrete_frame_addr;
    let nlocals = state.sym().nlocals;
    let new_stack_len = new_vsd.saturating_sub(nlocals);
    let s = state.sym_mut();
    let old_stack_len = s.concrete_stack.len();
    if new_stack_len < old_stack_len {
        s.concrete_stack.truncate(new_stack_len);
    } else if new_stack_len > old_stack_len {
        s.concrete_stack
            .resize(new_stack_len, crate::state::ConcreteValue::Null);
        for stack_idx in old_stack_len..new_stack_len {
            let abs_idx = nlocals + stack_idx;
            let obj = crate::state::concrete_stack_value(frame_addr, abs_idx)
                .unwrap_or(pyre_object::PY_NULL);
            s.concrete_stack[stack_idx] = crate::state::concrete_value_from_slot(obj);
        }
    }
    s.valuestackdepth = new_vsd;
}

/// Mirrors `liveness.rs:528..588 _opcode_stack_effect` (Python
/// `dis.stack_effect` parity).  Walker-handled opcodes must remain a
/// closed set listed here AND in `production_walker_handles`.
fn apply_walker_stack_effect(state: &mut MIFrame, instruction: &Instruction) {
    let _ = state;
    match instruction {
        Instruction::Nop
        | Instruction::ExtendedArg
        | Instruction::Resume { .. }
        | Instruction::Cache
        | Instruction::NotTaken
        | Instruction::ExitInitCheck
        | Instruction::EndFor => {
            // delta = 0, no shadow mutation.
        }
        Instruction::UnaryNot
        | Instruction::UnaryInvert
        | Instruction::GetIter
        | Instruction::MatchMapping
        | Instruction::MatchSequence
        | Instruction::SetupAnnotations
        | Instruction::FormatSimple
        | Instruction::FormatWithSpec
        | Instruction::MakeFunction
        | Instruction::GetYieldFromIter => {
            // 1-in-1-out at the TOS slot. The walker arm emits
            // `inline_call_r_r/dR>r` whose dst writeback to
            // `concrete_registers_r[dst]` replaces the TOS register
            // operand in-place; `sym.valuestackdepth` is invariant.
        }
        Instruction::PopIter
        | Instruction::EndSend
        | Instruction::DeleteSubscr
        | Instruction::LoadLocals
        | Instruction::LoadBuildClass
        | Instruction::BuildTemplate
        | Instruction::Copy { .. }
        | Instruction::BinarySlice
        | Instruction::StoreSlice
        | Instruction::ContainsOp { .. }
        | Instruction::IsOp { .. }
        | Instruction::Swap { .. }
        | Instruction::BuildTuple { .. }
        | Instruction::BuildList { .. }
        | Instruction::BuildSet { .. }
        | Instruction::BuildString { .. }
        | Instruction::BuildMap { .. }
        | Instruction::BuildSlice { .. }
        | Instruction::LoadFastAndClear { .. }
        | Instruction::ListAppend { .. }
        | Instruction::ListExtend { .. }
        | Instruction::SetAdd { .. }
        | Instruction::SetUpdate { .. }
        | Instruction::MapAdd { .. }
        | Instruction::DictUpdate { .. }
        | Instruction::DictMerge { .. }
        | Instruction::SetFunctionAttribute { .. }
        | Instruction::UnpackSequence { .. }
        | Instruction::UnpackEx { .. }
        | Instruction::LoadName { .. }
        | Instruction::StoreName { .. }
        | Instruction::StoreGlobal { .. }
        | Instruction::DeleteAttr { .. }
        | Instruction::ImportName { .. }
        | Instruction::ImportFrom { .. }
        | Instruction::LoadSuperAttr { .. }
        | Instruction::BuildInterpolation { .. }
        | Instruction::CallIntrinsic1 { .. }
        | Instruction::CallIntrinsic2 { .. }
        | Instruction::GetLen
        | Instruction::LoadSpecial { .. }
        | Instruction::LoadFromDictOrGlobals { .. }
        | Instruction::LoadFromDictOrDeref { .. }
        | Instruction::LoadDeref { .. }
        | Instruction::LoadFastCheck { .. }
        | Instruction::LoadCommonConstant { .. }
        | Instruction::GetAiter
        | Instruction::GetAwaitable { .. }
        | Instruction::StoreDeref { .. }
        | Instruction::YieldValue { .. }
        | Instruction::ReturnGenerator
        | Instruction::Send { .. }
        | Instruction::GetAnext
        | Instruction::EndAsyncFor
        | Instruction::CleanupThrow
        | Instruction::WithExceptStart
        | Instruction::DeleteFast { .. }
        | Instruction::DeleteDeref { .. }
        | Instruction::DeleteGlobal { .. }
        | Instruction::DeleteName { .. }
        | Instruction::CopyFreeVars { .. }
        | Instruction::MakeCell { .. }
        | Instruction::ConvertValue { .. }
        | Instruction::PopJumpIfNone { .. }
        | Instruction::PopJumpIfNotNone { .. }
        | Instruction::ForIter { .. }
        | Instruction::CallKw { .. }
        | Instruction::CallFunctionEx
        | Instruction::LoadAttr { .. }
        | Instruction::StoreAttr { .. }
        // | Instruction::StoreSubscr { .. } // see production_walker_handles for the walker hook rationale
        // | Instruction::PushNull // handled by the dispatch_via_walker_for_opcode entry hook; push_value advances vsd
        | Instruction::StoreFastStoreFast { .. } => {
            // Non-zero stack delta. The walker arm's
            // `setfield_vable_i(valuestackdepth)` emit routes through
            // `vable_setfield` (trace_ctx.rs:2608-2655) which calls
            // `synchronize_virtualizable()` (`pyjitpl.py:3470-3474
            // synchronize_virtualizable` parity) — that writes the new
            // vsd back into the live `PyFrame` at trace time.  Resync
            // `sym.valuestackdepth` from the concrete `PyFrame` so the
            // tracer's parallel counter follows the live frame, mirroring
            // PyPy's tracer which has no `MIFrame.valuestackdepth` shadow
            // (rpython/jit/metainterp/pyjitpl.py:65-93) and reads stack
            // depth directly from `metainterp.virtualizable_boxes`.
            resync_sym_vsd_from_concrete_frame(state);
        }
        other => panic!(
            "apply_walker_stack_effect: missing handler for {:?}; every entry \
             in production_walker_handles must list its symbolic effect here",
            other,
        ),
    }
}

fn classify_concrete(cv: ConcreteValue) -> (bool, bool) {
    match cv {
        ConcreteValue::Int(_) => (true, false),
        ConcreteValue::Float(_) => (false, true),
        ConcreteValue::Ref(obj) if !obj.is_null() => unsafe { (is_int(obj), is_float(obj)) },
        _ => (false, false),
    }
}

pub(crate) fn trace_step_result_to_action(
    state: &mut MIFrame,
    result: Result<pyre_interpreter::StepResult<FrontendOp>, PyError>,
) -> TraceAction {
    match result {
        Ok(pyre_interpreter::StepResult::Continue) => {
            // opimpl_jit_merge_point portal_call_depth>0 orthodox path
            // (pyjitpl.py:1579-1602): the inline-frame back-edge targets a
            // loop with compiled code; surface the signal so the metainterp
            // pops the inline frame and records a CALL_ASSEMBLER from the
            // parent (finishframe + do_recursive_call assembler_call=True).
            if let Some((green_key, target_pc)) = state.ctx().take_recursive_call_assembler() {
                return TraceAction::RecursiveCallAssembler {
                    green_key,
                    target_pc,
                };
            }
            // opimpl_jit_merge_point portal_call_depth>0 safe subset: a
            // back-edge reached inside an inline callee frame was flagged in
            // close_loop_args (it cannot be unrolled — its JUMP arg-set is
            // built from the callee frame shape, diverging from the root
            // LABEL). Abort this trace and stop tracing the root loop so the
            // callee compiles standalone instead of being re-inlined. Reuses
            // the trace-too-long recovery primitive (disable_noninlinable_-
            // function, below) but targets the root key, not the callee.
            if state.ctx().take_inline_loop_abort() {
                let root_green_key = state.with_ctx(|_, ctx| ctx.root_green_key());
                let callee_key = biggest_inline_trace_key(state);
                let (driver, _) = crate::driver::driver_pair();
                let warm_state = driver.meta_interp_mut().warm_state_mut();
                // Stop tracing the root loop that inlined this callee, so the
                // callee compiles as its own LOOP and runs in JIT code while
                // the (trivial) root loop stays interpreted.
                //
                // The orthodox endpoint is a residual CALL_ASSEMBLER from the
                // root into the callee's compiled loop (do_recursive_call,
                // assembler_call=True). This safe subset stops short of
                // emitting that call. Marking the *callee* non-inlinable
                // instead would route it through a function-entry PROCEDURE
                // compile, which is broken for a callee first reached by an
                // aborted inline trace (guard-failure storm → crash); marking
                // the *root* lets the callee's hot loop compile exactly as it
                // does when the driver lives at module scope and never inlines
                // it (the working baseline).
                warm_state.disable_noninlinable_function(root_green_key);
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][inline-loop-abort] disable root={} (callee={:?})",
                        root_green_key, callee_key
                    );
                }
                return majit_metainterp::TraceAction::Abort;
            }
            let compile_trace_succeeded = {
                let (driver, _) = crate::driver::driver_pair();
                driver.compile_trace_success_pending()
            };
            if compile_trace_succeeded {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!("[jit][compile-trace] pending success seen in trace_step_result");
                }
                return TraceAction::CompileTrace;
            }
            if state.ctx().is_too_long() {
                let green_key = state.ctx().green_key();
                let root_green_key = state.with_ctx(|_, ctx| ctx.root_green_key());
                if let Some(biggest_key) = biggest_inline_trace_key(state) {
                    let (driver, _) = crate::driver::driver_pair();
                    let warm_state = driver.meta_interp_mut().warm_state_mut();
                    warm_state.disable_noninlinable_function(biggest_key);
                    warm_state.trace_next_iteration(root_green_key);
                    if majit_metainterp::majit_log_enabled() {
                        eprintln!(
                            "[jit][trace-too-long] biggest_inline_key={} trace_next_iteration root_key={}",
                            biggest_key, root_green_key
                        );
                    }
                    return majit_metainterp::TraceAction::Abort;
                }
                let force_finish_trace = {
                    let (driver, _) = crate::driver::driver_pair();
                    driver.meta_interp().force_finish_trace_enabled()
                };
                if force_finish_trace {
                    let jump_args = state.with_ctx(|this, ctx| this.close_loop_args(ctx));
                    if majit_metainterp::majit_log_enabled() {
                        eprintln!(
                            "[jit] force_finish_trace: closing loop early at key={}",
                            root_green_key
                        );
                    }
                    return TraceAction::CloseLoopWithArgs {
                        jump_args,
                        loop_header_pc: None,
                    };
                }
                note_root_trace_too_long(root_green_key);
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][abort-reason] trace_too_long key={} root_key={}",
                        green_key, root_green_key
                    );
                }
                TraceAction::Abort
            } else {
                TraceAction::Continue
            }
        }
        Ok(pyre_interpreter::StepResult::CloseLoop {
            jump_args,
            loop_header_pc,
        }) => TraceAction::CloseLoopWithArgs {
            jump_args: jump_args.iter().map(|fop| fop.opref).collect(),
            loop_header_pc: Some(loop_header_pc),
        },
        Ok(pyre_interpreter::StepResult::Return(fop)) => {
            // pyjitpl.py:2489-2502 finishframe + compile_done_with_this_frame:
            //   result_type = self.jitdriver_sd.result_type  (STATIC property)
            //   elif result_type == history.REF:
            //       raise jitexc.DoneWithThisFrameRef(resultbox.getref_base())
            //
            // pyre eval_loop_jit sets result_type = Type::Ref. Every FINISH
            // at the portal exit must carry Type::Ref. If the optimizer
            // unboxed the return value to Int/Float, ensure_boxed_for_ca
            // re-boxes it (NewWithVtable + SetfieldGc).
            let value = fop.opref;
            let finish_value =
                state.with_ctx(|this, ctx| crate::state::ensure_boxed_for_ca(ctx, this, value));
            // pyjitpl.py:3222 store_token_in_vable
            state.with_ctx(|this, ctx| {
                this.store_token_in_vable(ctx);
            });
            TraceAction::Finish {
                finish_args: vec![finish_value],
                finish_arg_types: vec![Type::Ref],
                exit_with_exception: false,
            }
        }
        Ok(pyre_interpreter::StepResult::Yield(fop)) => {
            // pyjitpl.py:3198 compile_done_with_this_frame parity:
            // Yield uses the same Ref result_type.
            let value = fop.opref;
            let finish_value =
                state.with_ctx(|this, ctx| crate::state::ensure_boxed_for_ca(ctx, this, value));
            // pyjitpl.py:3222 store_token_in_vable (same as return path)
            state.with_ctx(|this, ctx| {
                this.store_token_in_vable(ctx);
            });
            TraceAction::Finish {
                finish_args: vec![finish_value],
                finish_arg_types: vec![Type::Ref],
                exit_with_exception: false,
            }
        }
        Err(err) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] step_error key={} err={}",
                    state.ctx().green_key(),
                    err
                );
            }
            TraceAction::Abort
        }
    }
}

impl TraceHelperAccess for MIFrame {
    fn with_trace_ctx<R>(&mut self, f: impl FnOnce(&mut TraceCtx) -> R) -> R {
        self.with_ctx(|_, ctx| f(ctx))
    }

    fn trace_frame(&self) -> OpRef {
        self.frame()
    }

    fn trace_globals_ptr(&mut self) -> OpRef {
        self.with_ctx(|this, ctx| crate::state::frame_get_globals_obj(ctx, this.frame()))
    }

    fn trace_record_not_forced_guard(&mut self) {
        self.with_ctx(|this, ctx| {
            this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
            // heapcache.py: invalidate_caches after non-pure calls.
            // The call may have mutated heap state, so cached field
            // values for escaped objects are no longer reliable.
            ctx.heap_cache_mut().invalidate_caches_for_escaped();
        });
    }

    fn trace_record_no_exception_guard(&mut self) {
        self.with_ctx(|this, ctx| {
            this.generate_guard(ctx, OpCode::GuardNoException, &[]);
        });
    }

    fn trace_call_callable(&mut self, callable: OpRef, args: &[OpRef]) -> Result<OpRef, PyError> {
        let frame = self.trace_frame();
        let result = self.with_ctx(|this, ctx| {
            let boxed_args = box_args_for_python_helper(this, ctx, args);
            crate::helpers::emit_trace_call_callable(ctx, frame, callable, &boxed_args)
        })?;
        // may-force CanRaise: GuardNotForced + GuardNoException, matching
        // PyPy `execute_varargs` ordering.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(result)
    }

    fn trace_binary_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: pyre_interpreter::bytecode::BinaryOperator,
    ) -> Result<OpRef, PyError> {
        let result = self.with_ctx(|this, ctx| {
            let lhs = box_value_for_python_helper(this, ctx, a);
            let rhs = box_value_for_python_helper(this, ctx, b);
            crate::helpers::emit_trace_binary_value(ctx, lhs, rhs, op)
        })?;
        // pyjitpl.py:2079: the generic binary op may invoke `__add__` etc.,
        // forcing the virtualizable.
        self.trace_record_not_forced_guard();
        self.trace_record_no_exception_guard();
        Ok(result)
    }
}

// `impl SharedOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl LocalOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl NamespaceOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl StackOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl IterOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl TruthOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl ControlFlowOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl BranchOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl ArithmeticOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl ConstantOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

impl MIFrame {
    /// `callmethod.py:25-85 LOAD_METHOD` JIT fast path.  Returns `true` after
    /// recording the folded method-dispatch sequence and pushing the two
    /// stack values; returns `false` (recording nothing) when the predicate
    /// does not hold, so the caller records the residual getattr fallback.
    ///
    /// Scope: a user-class instance whose method resolves to a plain
    /// `FUNCTION_TYPE` function with no shadowing instance attribute — the
    /// common `obj.m` dispatch.  Every other receiver / descriptor shape
    /// (type objects, builtin-type methods, staticmethod / classmethod /
    /// property / member descriptors, instance-dict shadows, custom
    /// `__getattribute__`) falls back; the fallback is the unchanged getattr
    /// path, so behaviour is identical, only foldability differs.
    fn try_load_method_fast_path(
        &mut self,
        obj: FrontendOp,
        concrete_obj: PyObjectRef,
        name: &str,
    ) -> Result<bool, PyError> {
        // callmethod.py:60-78 fast-path decision, shared with the concrete
        // interpreter (`eval::load_method`) so the symbolic trace and the
        // concrete frame agree on whether the `[w_descr, w_obj]` shape is
        // pushed — the two desync at the following `CALL` otherwise.  `None`
        // (every other receiver / descriptor shape) falls back to the
        // residual getattr path below; behaviour is identical, only
        // foldability differs.
        let (w_type, version_tag, w_descr) =
            match unsafe { pyre_interpreter::load_method_fast_path(concrete_obj, name) } {
                Some(triple) => triple,
                None => return Ok(false),
            };

        // ── Record the fast path (callmethod.py:55-68). ──
        let lookup_fn = crate::helpers::jit_lookup_where_with_method_cache as *const ();
        let getdict_fn = crate::helpers::jit_instance_getdictvalue as *const ();
        // The interned, immortal name (`box_str_constant`: content-keyed,
        // never freed) is the green token the lookup folds on, and the
        // pointer `jit_instance_getdictvalue` reads back via `w_str_get_value`.
        let w_name_ptr = pyre_object::strobject::box_str_constant(rustpython_wtf8::Wtf8::new(name));

        let w_descr_op = self.with_ctx(|this, ctx| {
            // callmethod.py:32 `w_type = space.type(w_obj)` → pin the receiver
            // type so it (and `version_tag` below) is a green constant.
            this.guard_class(ctx, obj.opref, w_type as *const PyType);
            let w_type_const = ctx.const_ref(w_type as i64);
            // typeobject.py:506 `promote(self.version_tag())`: read the live
            // `_version_tag` field and `guard_value` it to a constant so the
            // lookup folds; the guard deopts when `mutated()` bumps it.
            let vt_op =
                opimpl_getfield_gc_i(ctx, w_type_const, crate::descr::type_version_tag_descr());
            this.implement_guard_value(ctx, vt_op, version_tag as i64);
            let w_name_const = ctx.const_ref(w_name_ptr as i64);
            // typeobject.py:510 `_pure_lookup_where_with_method_cache` →
            // `CALL_PURE_R`: type const + interned name + promoted
            // `version_tag` are all green, so the optimizer folds it to
            // `ConstPtr(w_descr)` and the lookup leaves the loop.
            let lookup_args = [w_type_const, w_name_const, vt_op];
            let lookup_arg_types = [Type::Ref, Type::Ref, Type::Int];
            ctx.record_known_result_typed(
                w_descr as i64,
                lookup_fn,
                &lookup_args,
                &lookup_arg_types,
                Type::Ref,
                majit_metainterp::EffectInfoSlot::ElidableCannotRaise,
            );
            let lookup_concrete_args = [
                Value::Int(lookup_fn as usize as i64),
                Value::Ref(GcRef(w_type as usize)),
                Value::Ref(GcRef(w_name_ptr as usize)),
                Value::Int(version_tag as i64),
            ];
            let w_descr_op = crate::helpers::emit_trace_call_ref_typed_elidable_cannot_raise(
                ctx,
                lookup_fn,
                &lookup_args,
                &lookup_arg_types,
                &lookup_concrete_args,
                Value::Ref(GcRef(w_descr as usize)),
            );
            // callmethod.py:66 `w_value = w_obj.getdictvalue(space, name)`:
            // residual instance-dict read (not pure — the dict mutates),
            // guarded `null` so a later shadowing attribute deopts.
            let gv_op = crate::helpers::emit_trace_call_ref_typed(
                ctx,
                getdict_fn,
                &[obj.opref, w_name_const],
                &[Type::Ref, Type::Ref],
            );
            this.implement_guard_value(ctx, gv_op, pyre_object::PY_NULL as i64);
            w_descr_op
        });

        // callmethod.py:68 `f.pushvalue(w_descr); f.pushvalue(w_obj)`.
        // `call()` prepends the non-null receiver, dispatching
        // `w_descr(w_obj, *args)` — identical to the bound method the
        // fallback getattr would build.
        <Self as SharedOpcodeHandler>::push_value(
            self,
            FrontendOp::new(w_descr_op, ConcreteValue::Ref(w_descr)),
        )?;
        <Self as SharedOpcodeHandler>::push_value(self, obj)?;
        Ok(true)
    }
}

impl OpcodeStepExecutor for MIFrame {
    fn pop_jump_if_none(&mut self, target: usize) -> Result<(), PyError> {
        let value = SharedOpcodeHandler::pop_value(self)?;
        if self.value_type(value.opref) != Type::Ref {
            return Err(PyError::type_error("pop_jump_if_none expects a ref value"));
        }
        let concrete_obj = value.concrete.to_pyobj();
        if concrete_obj.is_null() {
            return Err(PyError::type_error(
                "missing concrete value for pop_jump_if_none during trace",
            ));
        }
        let truth = self.with_ctx(|_this, ctx| {
            let expected = ctx.const_ref(pyre_object::w_none() as i64);
            ctx.record_op(OpCode::PtrEq, &[value.opref, expected])
        });
        let should_jump = unsafe { pyre_object::is_none(concrete_obj) };
        let fallthrough = self.fallthrough_pc();
        if !should_jump {
            self.with_ctx(|this, ctx| {
                MIFrame::set_next_instr(this, ctx, target);
                Ok::<(), PyError>(())
            })?;
        }
        let other_target = if should_jump { fallthrough } else { target };
        self.with_ctx(|this, ctx| {
            MIFrame::record_branch_guard(this, ctx, value.opref, truth, should_jump, other_target);
            Ok::<(), PyError>(())
        })?;
        let next_target = if should_jump { target } else { fallthrough };
        self.with_ctx(|this, ctx| {
            MIFrame::set_next_instr(this, ctx, next_target);
            Ok::<(), PyError>(())
        })
    }

    fn pop_jump_if_not_none(&mut self, target: usize) -> Result<(), PyError> {
        let value = SharedOpcodeHandler::pop_value(self)?;
        if self.value_type(value.opref) != Type::Ref {
            return Err(PyError::type_error(
                "pop_jump_if_not_none expects a ref value",
            ));
        }
        let concrete_obj = value.concrete.to_pyobj();
        if concrete_obj.is_null() {
            return Err(PyError::type_error(
                "missing concrete value for pop_jump_if_not_none during trace",
            ));
        }
        let truth = self.with_ctx(|_this, ctx| {
            let expected = ctx.const_ref(pyre_object::w_none() as i64);
            ctx.record_op(OpCode::PtrNe, &[value.opref, expected])
        });
        let should_jump = !unsafe { pyre_object::is_none(concrete_obj) };
        let fallthrough = self.fallthrough_pc();
        if !should_jump {
            self.with_ctx(|this, ctx| {
                MIFrame::set_next_instr(this, ctx, target);
                Ok::<(), PyError>(())
            })?;
        }
        let other_target = if should_jump { fallthrough } else { target };
        self.with_ctx(|this, ctx| {
            MIFrame::record_branch_guard(this, ctx, value.opref, truth, should_jump, other_target);
            Ok::<(), PyError>(())
        })?;
        let next_target = if should_jump { target } else { fallthrough };
        self.with_ctx(|this, ctx| {
            MIFrame::set_next_instr(this, ctx, next_target);
            Ok::<(), PyError>(())
        })
    }

    /// Fix fusion opcode: load two locals with correct concrete tracking.
    /// FrontendOp carries concrete directly — no pending_concrete_push needed.
    fn load_fast_pair_checked(
        &mut self,
        idx1: usize,
        _name1: &str,
        idx2: usize,
        _name2: &str,
    ) -> Result<(), PyError> {
        let c1 = self
            .sym()
            .concrete_locals
            .get(idx1)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        let c2 = self
            .sym()
            .concrete_locals
            .get(idx2)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        let v1 = self.with_ctx(|this, ctx| MIFrame::load_local_value(this, ctx, idx1))?;
        SharedOpcodeHandler::push_value(self, FrontendOp::new(v1, c1))?;
        let v2 = self.with_ctx(|this, ctx| MIFrame::load_local_value(this, ctx, idx2))?;
        SharedOpcodeHandler::push_value(self, FrontendOp::new(v2, c2))?;
        Ok(())
    }

    fn to_bool(&mut self) -> Result<(), PyError> {
        Ok(())
    }

    /// CPython/PyPy LOOKUP_METHOD parity.
    ///
    /// Mirror eval.rs:1543-1608: push `(attr, null_or_self)` where
    /// `null_or_self` is the bound receiver for instance/class methods and
    /// `PY_NULL` for already-bound methods / static methods / plain attrs.
    fn load_method(&mut self, name: &str) -> Result<(), PyError> {
        let obj = SharedOpcodeHandler::pop_value(self)?;
        let concrete_obj = obj.concrete.to_pyobj();

        // callmethod.py:25-85 LOAD_METHOD JIT fast path: for a user-instance
        // method-dispatch the type lookup folds to a constant `w_descr` via
        // the `@elidable` method cache, leaving only the receiver-type guard,
        // the `version_tag` guard, and the per-iteration instance-dict
        // shadowing check in the loop.  Falls back to the residual getattr
        // below for every other receiver / descriptor shape.
        if self.try_load_method_fast_path(obj, concrete_obj, name)? {
            return Ok(());
        }

        // Resolve the foldable builtin list methods (append/pop/reverse) to
        // a Const unbound function guarded by class, with self in the
        // receiver slot, instead of a residual `jit_getattr` that
        // materialises a fresh bound `W_MethodObject` every iteration. A
        // Const callable is trivially reconstructed at guard-failure resume
        // (a residual bound method is not — it resolves to a null callable
        // on the blackhole CALL re-execution) and routes the following CALL
        // through the resolved-builtin folding shape (`call_callable_value`).
        if !concrete_obj.is_null()
            && matches!(name, "append" | "pop" | "reverse")
            && unsafe { is_list(concrete_obj) }
        {
            let list_type = pyre_interpreter::typedef::gettypeobject(&LIST_TYPE);
            if let Some(unbound) = unsafe { pyre_interpreter::lookup_in_type(list_type, name) } {
                if unsafe { is_function(unbound) }
                    && unsafe {
                        is_builtin_code(
                            pyre_interpreter::getcode(unbound) as pyre_object::PyObjectRef
                        )
                    }
                {
                    let method_op = self.with_ctx(|this, ctx| {
                        this.guard_class(ctx, obj.opref, &LIST_TYPE as *const PyType);
                        ctx.const_ref(unbound as i64)
                    });
                    <Self as SharedOpcodeHandler>::push_value(
                        self,
                        FrontendOp::new(method_op, ConcreteValue::Ref(unbound)),
                    )?;
                    return <Self as SharedOpcodeHandler>::push_value(self, obj);
                }
            }
        }

        let attr = <Self as SharedOpcodeHandler>::load_attr(self, obj, name)?;
        <Self as SharedOpcodeHandler>::push_value(self, attr)?;

        let null_value = self.with_ctx(|_this, ctx| {
            FrontendOp::new(
                ctx.const_ref(pyre_object::PY_NULL as i64),
                ConcreteValue::Ref(pyre_object::PY_NULL),
            )
        });
        if concrete_obj.is_null() {
            return <Self as SharedOpcodeHandler>::push_value(self, null_value);
        }

        let concrete_attr = attr.concrete.to_pyobj();
        if !concrete_attr.is_null() && unsafe { pyre_object::is_method(concrete_attr) } {
            return <Self as SharedOpcodeHandler>::push_value(self, null_value);
        }

        let bound = unsafe {
            if pyre_object::is_instance(concrete_obj) {
                let w_type = pyre_object::w_instance_get_type(concrete_obj);
                let raw = pyre_interpreter::lookup_in_type(w_type, name);
                match raw {
                    Some(d) if pyre_object::is_staticmethod(d) => pyre_object::PY_NULL,
                    Some(d) if pyre_object::is_classmethod(d) => w_type,
                    Some(d) if pyre_object::is_type(d) => pyre_object::PY_NULL,
                    Some(d) if pyre_object::is_property(d) => pyre_object::PY_NULL,
                    Some(d) if pyre_object::is_member(d) => pyre_object::PY_NULL,
                    Some(d) if pyre_interpreter::is_function(d) => {
                        let ob_type = (*d).ob_type;
                        if std::ptr::eq(
                            ob_type,
                            &pyre_interpreter::BUILTIN_FUNCTION_TYPE as *const PyType,
                        ) {
                            pyre_object::PY_NULL
                        } else if std::ptr::eq(
                            ob_type,
                            &pyre_interpreter::FUNCTION_TYPE as *const PyType,
                        ) && pyre_interpreter::is_builtin_code(
                            pyre_interpreter::function_get_code(d) as pyre_object::PyObjectRef,
                        ) {
                            concrete_obj
                        } else {
                            concrete_obj
                        }
                    }
                    Some(_) => concrete_obj,
                    None => pyre_object::PY_NULL,
                }
            } else if pyre_object::is_type(concrete_obj) {
                let raw = pyre_interpreter::lookup_in_type(concrete_obj, name);
                match raw {
                    Some(d) if pyre_object::is_classmethod(d) => concrete_obj,
                    Some(_) => pyre_object::PY_NULL,
                    None => concrete_obj,
                }
            } else if pyre_interpreter::typedef::r#type(concrete_obj).is_some()
                && !pyre_object::is_module(concrete_obj)
            {
                concrete_obj
            } else {
                pyre_object::PY_NULL
            }
        };

        let receiver = if bound.is_null() {
            null_value
        } else if bound == concrete_obj {
            obj
        } else {
            let bound_opref = self.with_ctx(|_this, ctx| ctx.const_ref(bound as i64));
            FrontendOp::new(bound_opref, ConcreteValue::Ref(bound))
        };
        <Self as SharedOpcodeHandler>::push_value(self, receiver)
    }

    /// CPython/PyPy CALL_FUNCTION parity for the `load_method()` stack shape.
    ///
    /// shared_opcode.rs intentionally discards `null_or_self`; the concrete
    /// interpreter override in eval.rs:1640-1652 prepends it for instance
    /// method calls. Trace-time execution must do the same so `CALL` after
    /// `LOAD_METHOD` sees the same argument list as the interpreter.
    fn call(&mut self, nargs: usize) -> Result<(), PyError> {
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(<Self as SharedOpcodeHandler>::pop_value(self)?);
        }
        args.reverse();
        let null_or_self = <Self as SharedOpcodeHandler>::pop_value(self)?;
        let callable = <Self as SharedOpcodeHandler>::pop_value(self)?;

        let result = if null_or_self.concrete.to_pyobj().is_null() {
            <Self as SharedOpcodeHandler>::call_callable(self, callable, &args)?
        } else {
            let mut full_args = Vec::with_capacity(args.len() + 1);
            full_args.push(null_or_self);
            full_args.extend_from_slice(&args);
            <Self as SharedOpcodeHandler>::call_callable(self, callable, &full_args)?
        };
        // If the call inlined the callee, the parent concrete frame's cleanup
        // (metainterp `step_inline_frame`) pops `pending.nargs` bytecode args
        // plus callable + null_or_self.  `call_callable` set `nargs` from the
        // callee arg list, which for a method dispatch (non-null null_or_self)
        // has the receiver prepended above; the parent cf only holds the
        // bytecode operands `[callable, null_or_self, args]`, so reset it to
        // the bytecode arg count (a no-op when nothing was prepended).
        if let Some(pending) = self.pending_inline_frame.as_mut() {
            pending.nargs = nargs;
        }
        <Self as SharedOpcodeHandler>::push_value(self, result)
    }

    fn call_kw(&mut self, nargs: usize) -> Result<(), PyError> {
        use pyre_interpreter::bytecode::CodeFlags;

        let kwarg_names_val = <Self as SharedOpcodeHandler>::pop_value(self)?;
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(<Self as SharedOpcodeHandler>::pop_value(self)?);
        }
        args.reverse();
        let null_or_self = <Self as SharedOpcodeHandler>::pop_value(self)?;
        let callable = <Self as SharedOpcodeHandler>::pop_value(self)?;

        let concrete_kwnames = kwarg_names_val.concrete.to_pyobj();
        let concrete_callable = callable.concrete.to_pyobj();

        if null_or_self.concrete.to_pyobj() != PY_NULL {
            args.insert(0, null_or_self);
        }

        // Determine nkw from the concrete kwarg_names tuple. PyPy's
        // `CALL_FUNCTION_KW` immediately `interp_w`s the stack value as a
        // tuple (pyopcode.py:1391), so treating an unavailable/non-tuple
        // concrete value as "no kwargs" would record a plain positional call
        // that PyPy would never execute.
        if concrete_kwnames.is_null() || unsafe { !pyre_object::is_tuple(concrete_kwnames) } {
            return Err(trace_abort_error(
                "abort tracing CALL_KW without concrete keyword tuple",
            ));
        }
        self.with_ctx(|this, ctx| {
            this.implement_guard_value(ctx, kwarg_names_val.opref, concrete_kwnames as i64);
        });
        let nkw = unsafe { w_tuple_len(concrete_kwnames) };
        if nkw > args.len() {
            return Err(trace_abort_error(
                "abort tracing CALL_KW with malformed keyword tuple",
            ));
        }

        // Only trace the direct user-function keyword path here.  PyPy's
        // `CALL_FUNCTION_KW` builds `Arguments(keyword_names_w, keywords_w)`
        // and dispatches the same object through `space.call_args`
        // (pyopcode.py:1386-1410), so methods, builtins, type calls, and
        // arbitrary `__call__` objects all receive structured kwargs.
        // Pyre's trace-side residual helpers still expose only flat
        // positional slices for those callable shapes; forcing them through
        // `call_callable` here would either discard the keyword-name tuple
        // or re-bind receivers differently from PyPy.  Keep this as a
        // structural adaptation until the trace ABI can carry `Arguments`.
        let target_func = if nkw == 0 {
            concrete_callable
        } else if !concrete_callable.is_null()
            && unsafe { is_function(concrete_callable) }
            && unsafe {
                !is_builtin_code(
                    pyre_interpreter::getcode(concrete_callable) as pyre_object::PyObjectRef
                )
            }
        {
            concrete_callable
        } else {
            return Err(trace_abort_error(
                "abort tracing CALL_KW for non-user-function callable",
            ));
        };

        if nkw == 0 {
            // No kwargs or not a user function — fall through to plain call.
            let result = <Self as SharedOpcodeHandler>::call_callable(self, callable, &args)?;
            return <Self as SharedOpcodeHandler>::push_value(self, result);
        }

        // Resolve kwargs to positional order at trace time.
        let code_ptr = unsafe { pyre_interpreter::get_pycode(target_func) };
        let code = unsafe { &*(code_ptr as *const CodeObject) };
        let total_params = (code.arg_count + code.kwonlyarg_count) as usize;
        let n_pos_params = code.arg_count as usize;
        let n_posonly_params = code.posonlyarg_count as usize;
        let has_varargs = code.flags.contains(CodeFlags::VARARGS);
        let has_varkw = code.flags.contains(CodeFlags::VARKEYWORDS);
        let n_pos = args.len() - nkw;

        if has_varargs || has_varkw {
            // PyPy's `Arguments._match_signature` writes a fully resolved
            // frame scope for `*args` / `**kwargs` (argument.py:222-242).
            // The trace-side residual helper below still calls the normal
            // positional user-function path, whose `call_user_function`
            // re-packs varargs/kwargs. Passing a pre-packed scope there
            // double-packs `*args` and drops non-empty `**kwargs`. Until the
            // JIT grows a resolved-scope helper matching
            // `call_user_function_resolved`, keep this as a structural
            // adaptation and let the interpreter handle these calls.
            return Err(trace_abort_error(
                "abort tracing CALL_KW for *args/**kwargs user function",
            ));
        }

        // PyPy would raise the exact `ArgErr*` TypeError from
        // `Arguments._match_signature` / `_match_keywords`
        // (argument.py:259-321, 464-501).  The trace recorder cannot yet
        // emit those exception paths with the right keyword metadata, and
        // recording a partially resolved call would be worse than falling
        // back to the interpreter.  Treat argument-mismatch keyword calls as
        // structural aborts rather than residual calls.
        if n_pos > n_pos_params {
            return Err(trace_abort_error(
                "abort tracing CALL_KW with too many positional args",
            ));
        }

        let mut resolved: Vec<Option<FrontendOp>> = vec![None; total_params];

        // Fill positional args.
        for i in 0..n_pos.min(n_pos_params) {
            resolved[i] = Some(args[i].clone());
        }

        // Match keyword args to parameter positions.
        for ki in 0..nkw {
            let kw_name_obj = unsafe { pyre_object::w_tuple_getitem(concrete_kwnames, ki as i64) };
            let Some(kw_name_obj) = kw_name_obj else {
                continue;
            };
            if !unsafe { pyre_object::is_str(kw_name_obj) } {
                return Err(trace_abort_error(
                    "abort tracing CALL_KW with non-string keyword name",
                ));
            }
            let kw_str = unsafe { pyre_object::w_str_get_value(kw_name_obj) };
            let kw_val = args[n_pos + ki].clone();

            let mut matched = false;
            for pi in 0..total_params {
                if &*code.varnames[pi] == kw_str {
                    if pi < n_posonly_params {
                        return Err(trace_abort_error(
                            "abort tracing CALL_KW with positional-only keyword",
                        ));
                    }
                    if pi < n_pos.min(n_pos_params) {
                        return Err(trace_abort_error(
                            "abort tracing CALL_KW with duplicate keyword value",
                        ));
                    }
                    // PyPy argument.py:_match_keywords overwrites
                    // kwds_mapping[j - input_argcount] for duplicate
                    // keyword names in malformed bytecode; the last value
                    // then wins when _match_signature fills scope_w. Only
                    // conflicts with already-filled positional parameters
                    // raise ArgErrMultipleValues.
                    resolved[pi] = Some(kw_val.clone());
                    matched = true;
                    break;
                }
            }
            if !matched {
                return Err(trace_abort_error(
                    "abort tracing CALL_KW with unknown keyword",
                ));
            }
        }

        // Fill positional defaults.
        let defaults_obj = unsafe { function_get_defaults(target_func) };
        if !defaults_obj.is_null() {
            self.with_ctx(|this, ctx| {
                let defaults = ctx.call_ref_typed_with_effect(
                    trace_function_get_defaults as *const (),
                    &[callable.opref],
                    &[Type::Ref],
                    CANNOT_RAISE_NO_HEAP_EFFECT_INFO.clone(),
                );
                this.implement_guard_value(ctx, defaults, defaults_obj as i64);
            });
            let defaults_obj = pyre_interpreter::baseobjspace::unwrap_cell(defaults_obj);
            if unsafe { pyre_object::is_tuple(defaults_obj) } {
                let ndefaults = unsafe { w_tuple_len(defaults_obj) };
                let first_default = n_pos_params.saturating_sub(ndefaults);
                for pi in first_default..n_pos_params {
                    if resolved[pi].is_none() {
                        if let Some(v) = unsafe {
                            pyre_object::w_tuple_getitem(defaults_obj, (pi - first_default) as i64)
                        } {
                            let opref = self.with_ctx(|_this, ctx| ctx.const_ref(v as i64));
                            resolved[pi] =
                                Some(FrontendOp::new(opref, ConcreteValue::from_pyobj(v)));
                        }
                    }
                }
            }
        }

        // Fill keyword-only defaults.
        let kwdefaults = pyre_interpreter::baseobjspace::unwrap_cell(unsafe {
            pyre_interpreter::function_get_kwdefaults(target_func)
        });
        if !kwdefaults.is_null() && unsafe { pyre_object::is_dict(kwdefaults) } {
            let kwdefaults_opref = self.with_ctx(|this, ctx| {
                let runtime_kwdefaults = ctx.call_ref_typed_with_effect(
                    trace_function_get_kwdefaults as *const (),
                    &[callable.opref],
                    &[Type::Ref],
                    CANNOT_RAISE_NO_HEAP_EFFECT_INFO.clone(),
                );
                this.implement_guard_value(ctx, runtime_kwdefaults, kwdefaults as i64);
                runtime_kwdefaults
            });
            let nkwonly = code.kwonlyarg_count as usize;
            for ki in 0..nkwonly {
                let pi = n_pos_params + ki;
                if resolved[pi].is_none() {
                    let param_name = &code.varnames[pi];
                    let key = pyre_object::w_str_new(param_name);
                    if let Some(val) = unsafe { pyre_object::w_dict_lookup(kwdefaults, key) } {
                        let opref = self.with_ctx(|this, ctx| {
                            let key_opref = ctx.const_ref(key as i64);
                            let val_opref = ctx.call_ref_typed_with_effect(
                                trace_dict_lookup_jit as *const (),
                                &[kwdefaults_opref, key_opref],
                                &[Type::Ref, Type::Ref],
                                CANNOT_RAISE_NO_HEAP_EFFECT_INFO.clone(),
                            );
                            this.implement_guard_value(ctx, val_opref, val as i64);
                            val_opref
                        });
                        resolved[pi] = Some(FrontendOp::new(opref, ConcreteValue::from_pyobj(val)));
                    }
                }
            }
        }

        if resolved.iter().any(Option::is_none) {
            return Err(trace_abort_error(
                "abort tracing CALL_KW with missing required arguments",
            ));
        }

        // Build the resolved call scope. Missing required arguments were
        // rejected above so no PY_NULL placeholder reaches the compiled call.
        let mut final_args: Vec<FrontendOp> = Vec::with_capacity(total_params + 2);
        for slot in resolved {
            match slot {
                Some(val) => final_args.push(val),
                None => {
                    let opref = self.with_ctx(|_this, ctx| ctx.const_ref(PY_NULL as i64));
                    final_args.push(FrontendOp::new(opref, ConcreteValue::Null));
                }
            }
        }

        let result = <Self as SharedOpcodeHandler>::call_callable(self, callable, &final_args)?;
        <Self as SharedOpcodeHandler>::push_value(self, result)
    }

    // RPython exception handler tracing (pyjitpl.py:2506 finishframe_exception):
    // handle_possible_exception emits GUARD_EXCEPTION and continues at the
    // handler PC. These three overrides trace the handler-entry bytecodes.
    //
    //   PUSH_EXC_INFO   → pop exc, save prev CURRENT_EXCEPTION, set
    //                     CURRENT_EXCEPTION = exc, push prev, push exc
    //                     (pyopcode.py:786 / eval.rs:1220-1229)
    //   CHECK_EXC_MATCH → opimpl_goto_if_exception_mismatch (pyjitpl.py:1677)
    //   POP_EXCEPT      → pop prev, restore CURRENT_EXCEPTION = prev,
    //                     clear tracer last_exc_value (pyopcode.py:778 /
    //                     eval.rs:1243-1249)

    fn push_exc_info(&mut self) -> Result<(), PyError> {
        let exc = <Self as SharedOpcodeHandler>::pop_value(self)?;
        let exc_obj = exc.concrete.to_pyobj();
        // Emit the save/restore pair as GETFIELD_GC_R / SETFIELD_GC on the
        // per-thread EC's `sys_exc_value` slot (the single source of truth;
        // see eval::get_current_exception). This performs the same
        // `prev = ec.sys_exc_value; ec.sys_exc_value = exc` sequence at
        // runtime (pyopcode.py:786 / eval.rs:1220-1229) as the former
        // residual TLS helpers, but as heap ops the optimizer can reason
        // about: a balanced PUSH_EXC_INFO save + POP_EXCEPT restore with no
        // intervening read is dead-store-eliminated, so a non-escaping
        // exception stays virtual. `ec_sys_exc_value_descr` is a non-pointer
        // flagged Ref field, so SETFIELD_GC emits NO write barrier (the EC
        // is non-GC and its slot is a forwarded GC root — see the descr).
        let ec = self.with_ctx(|this, ctx| this.ensure_execution_context(ctx));
        let prev_exc_opref = self.with_ctx(|_this, ctx| {
            ctx.record_op_with_descr(
                majit_ir::OpCode::GetfieldGcR,
                &[ec],
                crate::descr::ec_sys_exc_value_descr(),
            )
        });
        self.with_ctx(|_this, ctx| {
            ctx.record_op_with_descr(
                majit_ir::OpCode::SetfieldGc,
                &[ec, exc.opref],
                crate::descr::ec_sys_exc_value_descr(),
            );
        });
        let prev_exc = get_current_exception();
        set_current_exception(exc_obj);
        <Self as SharedOpcodeHandler>::push_value(
            self,
            FrontendOp::new(prev_exc_opref, ConcreteValue::Ref(prev_exc)),
        )?;
        <Self as SharedOpcodeHandler>::push_value(self, exc)?;
        {
            let s = self.sym_mut();
            s.current_exc_value = exc_obj;
            s.current_exc_box = exc.opref;
        }
        let frame =
            unsafe { &mut *(self.concrete_frame_addr as *mut pyre_interpreter::pyframe::PyFrame) };
        let _ = frame.pop();
        frame.push(prev_exc);
        frame.push(exc_obj);
        Ok(())
    }

    fn pop_except(&mut self) -> Result<(), PyError> {
        let prev_exc = <Self as SharedOpcodeHandler>::pop_value(self)?;
        // Restore the saved sys_exc_info as SETFIELD_GC on the EC's
        // `sys_exc_value` slot (pyopcode.py:778 / eval.rs:1243-1249).
        // Paired with the PUSH_EXC_INFO save above on the same EC field,
        // this lets the heap optimizer dead-store-eliminate a balanced,
        // never-read save/restore (no write barrier — see the descr).
        let ec = self.with_ctx(|this, ctx| this.ensure_execution_context(ctx));
        self.with_ctx(|_this, ctx| {
            ctx.record_op_with_descr(
                majit_ir::OpCode::SetfieldGc,
                &[ec, prev_exc.opref],
                crate::descr::ec_sys_exc_value_descr(),
            );
        });
        set_current_exception(prev_exc.concrete.to_pyobj());
        {
            let s = self.sym_mut();
            s.current_exc_value = prev_exc.concrete.to_pyobj();
            s.current_exc_box = prev_exc.opref;
        }
        // POP_EXCEPT pops the saved exc_info (prev_exc). The matching
        // POP_TOP that discards the caught exception runs through the
        // generic opcode executor, which updates the symbolic stack
        // (`pop_value`) but not this concrete frame — so the exception
        // object pushed at handler entry (`finishframe_exception` /
        // `push_exc_info`) is still on the concrete frame above prev_exc.
        // Unwind to the symbolic baseline (authoritative once the handler
        // block closes) rather than popping a single slot, so the
        // closing-jump's `concrete_valuestackdepth()` matches the loop
        // entry instead of carrying the stale exception slot.
        let target_vsd = self.sym().valuestackdepth;
        let frame =
            unsafe { &mut *(self.concrete_frame_addr as *mut pyre_interpreter::pyframe::PyFrame) };
        while frame.valuestackdepth > target_vsd {
            let _ = frame.pop();
        }
        // RPython pyjitpl.py:2751 clear_exception: exception fully handled.
        let s = self.sym_mut();
        s.last_exc_value = std::ptr::null_mut();
        s.class_of_last_exc_is_const = false;
        Ok(())
    }

    /// RPython pyjitpl.py:1677 opimpl_goto_if_exception_mismatch +
    /// CPython `CHECK_EXC_MATCH` semantics.  Pops the expected exception
    /// type, checks against last_exc_value, and pushes the concrete
    /// match result.  GUARD_EXCEPTION already verified the class so this
    /// usually produces True, but multi-except blocks may produce False
    /// for non-matching clauses.
    ///
    /// Logic mirrors `pyre_interpreter::eval::check_exc_match_against`
    /// (`pyre/pyre-interpreter/src/eval.rs:81-130`) — including tuple-of-
    /// types and `is_type` + MRO walk — so the recorded boolean matches
    /// what the interpreter would compute.  RPython
    /// `pyjitpl.py:1680-1681` asserts last_exc_value + class_of_last_exc
    /// are both set; pyre aborts tracing instead of silently emitting a
    /// constant True if either operand is null at trace time.
    fn check_exc_match(&mut self) -> Result<(), PyError> {
        let exc_type_val = <Self as SharedOpcodeHandler>::pop_value(self).ok();
        let exc_type_obj = exc_type_val
            .as_ref()
            .map(|v| v.concrete.to_pyobj())
            .unwrap_or(std::ptr::null_mut());

        let last_exc = self.sym().last_exc_value;
        if last_exc.is_null() || exc_type_obj.is_null() {
            // RPython `pyjitpl.py:1680-1681` `assert last_exc_value;
            // assert class_of_last_exc_is_const`.  A null operand here
            // means a tracer invariant was broken upstream (typically
            // CHECK_EXC_MATCH ran without prior PUSH_EXC_INFO /
            // GUARD_EXCEPTION).  Bail out loudly rather than fold a
            // wrong constant into the trace.
            return Err(PyError::runtime_error(
                "CHECK_EXC_MATCH with null last_exc or exc_type during tracing",
            ));
        }

        // pyopcode.py:1034-1039 validity gate. The PyError raised here
        // is `PyError = PyError` at trace_opcode.rs:7091 and aborts
        // the trace (the interpreter re-runs CHECK_EXC_MATCH freshly to
        // surface the TypeError without baking it into the recorded
        // trace). Matches `pyre_interpreter::eval::check_exc_match` BC
        // handler split.
        pyre_interpreter::eval::validate_check_exc_match_class(exc_type_obj)?;
        let matched = unsafe { trace_check_exc_match_against(last_exc, exc_type_obj) };

        let result_obj = pyre_object::w_bool_from(matched);
        let result_opref = self.with_ctx(|_this, ctx| ctx.const_ref(result_obj as i64));
        <Self as SharedOpcodeHandler>::push_value(
            self,
            FrontendOp::new(result_opref, ConcreteValue::Ref(result_obj)),
        )?;
        Ok(())
    }

    /// `pyopcode.py:704-720 RAISE_VARARGS` + `eval.rs:1049-1128` parity.
    ///
    /// pyre's tracer plays the role RPython splits between
    /// `pypy/interpreter/pyopcode.py` (operand handling + normalization)
    /// and `rpython/jit/metainterp/pyjitpl.py:1688 opimpl_raise`
    /// (box-level bookkeeping). Both responsibilities live here:
    ///
    /// 1. Normalize the operand — `raise Type` must call the exception
    ///    class to obtain an instance, matching
    ///    `eval.rs:1049-1088` / `eval.rs:1091-1127`.
    /// 2. For `argc == 2`, pop cause (TOS) before exception (TOS1) and
    ///    attach it via `attach_raise_cause` so `raise X from Y` produces
    ///    the same observable `__cause__` state as the interpreter.
    /// 3. Emit a residual helper that performs the same normalization +
    ///    `jit_exc_raise` sequence at runtime, then seed metainterp
    ///    state (pyjitpl.py:1690-1696) from the trace-time concrete
    ///    value so `handle_raise_varargs` → `finishframe_exception`
    ///    keeps seeing a pending exception on both paths.
    fn raise_varargs(&mut self, argc: usize) -> Result<(), PyError> {
        if argc == 0 {
            // `eval.rs:1032-1048 RAISE_VARARGS 0` — reraise from the
            // active exception. Prefer the tracer-seeded
            // `last_exc_value` (RPython pyjitpl.py:1701 opimpl_reraise),
            // fall back to the `CURRENT_EXCEPTION` TLS slot the
            // interpreter reads.
            let exc = self.sym().last_exc_value;
            if !exc.is_null() {
                return Err(unsafe { PyError::from_exc_object(exc) });
            }
            let exc = get_current_exception();
            if unsafe { pyre_object::is_exception(exc) } {
                return Err(unsafe { PyError::from_exc_object(exc) });
            }
            return Err(PyError::runtime_error("No active exception to reraise"));
        }
        if argc > 2 {
            return Err(PyError::type_error("too many arguments for raise"));
        }
        // `eval.rs:1091-1094 RAISE_VARARGS 2` pops cause (TOS) first,
        // then exception (TOS1). `attach_raise_cause` runs only for
        // argc == 2.
        let (exc_val, cause, cause_opref) = if argc == 2 {
            let raw_cause = <Self as SharedOpcodeHandler>::pop_value(self)?;
            let exc_val = <Self as SharedOpcodeHandler>::pop_value(self)?;
            let raw_cause_obj = raw_cause.concrete.to_pyobj();
            let cause = if raw_cause_obj.is_null() {
                None
            } else {
                // pyopcode.py:706 `w_cause = space.call_function(w_cause)`
                // runs on the interpreter path; in pyre's tracer context
                // the call must be forced onto the plain eval loop so it
                // does not re-enter the tracer. Mirrors the exc-path
                // guard at `call_function(exc, &[])` below (commit
                // bef2ee2035 symmetrized JIT + blackhole helpers but
                // missed this tracer-time site).
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                Some(normalize_raise_cause(raw_cause_obj)?)
            };
            (exc_val, cause, raw_cause.opref)
        } else {
            let exc_val = <Self as SharedOpcodeHandler>::pop_value(self)?;
            let null_cause = self.with_ctx(|_this, ctx| ctx.const_ref(pyre_object::PY_NULL as i64));
            (exc_val, None, null_cause)
        };
        let exc = exc_val.concrete.to_pyobj();
        // E1: a trace-built exception (`try_trace_exception_new`) leaves the
        // type-call result with concrete=Null; recover the fresh instance
        // recorded at construction so the raise takes the instance fast path.
        // Consumed (not just read): after this raise the instance is no
        // longer fresh — its `w_context` is stamped below — so a second
        // raise of the same object must take the residual path, whose
        // runtime `attach_raise_cause` keeps an existing `__context__`
        // and avoids the self-cycle.
        let trace_built = self.sym_mut().trace_built_exc.remove(&exc_val.opref);
        let exc = if exc.is_null() {
            trace_built.unwrap_or(exc)
        } else {
            exc
        };
        let emit_runtime_raise = |slf: &mut Self| {
            slf.with_ctx(|this, ctx| {
                ctx.call_ref_typed_with_effect(
                    normalize_raise_varargs_jit as *const (),
                    &[this.frame(), exc_val.opref, cause_opref],
                    &[Type::Ref, Type::Ref, Type::Ref],
                    default_effect_info(),
                )
            })
        };
        if exc.is_null() {
            let _ = emit_runtime_raise(self);
            return Err(PyError::value_error("raised during tracing"));
        }
        unsafe {
            if pyre_object::is_exception(exc) {
                // `eval.rs:1053-1055 / :1096-1098`: already an instance.
                //
                // E1 fast path: a freshly trace-built exception
                // (`NewWithVtable`) raised with no explicit cause needs
                // neither the residual `normalize_raise_varargs_jit` publish
                // nor `GUARD_EXCEPTION`.  `seed_raised_exception` sets
                // `last_exc_box`, so the dispatch routes through
                // `handle_raise_varargs` → `finishframe_exception` (a local
                // handler continues with `last_exc_box`; a root-frame escape
                // FINISHes with it via `compile_exit_frame_with_exception`).
                // Skipping the publish leaves the New op with no escape, so the
                // optimizer can virtualize it.
                //
                // `__context__` chaining is emitted as a SetField on the
                // (virtualizable) exception instead of inside the residual: for
                // a fresh exception `w_context` is null and the self-cycle is
                // impossible, so `attach_raise_cause`'s conditional
                // `w_context = active` reduces to an unconditional
                // `exc.w_context = ec.sys_exc_value` (storing null when no
                // exception is active is a no-op that DCEs).
                if trace_built.is_some() && cause.is_none() {
                    let kind = pyre_object::excobject::w_exception_get_kind(exc);
                    let ec = self.with_ctx(|this, ctx| this.ensure_execution_context(ctx));
                    let active = self.with_ctx(|_this, ctx| {
                        ctx.record_op_with_descr(
                            majit_ir::OpCode::GetfieldGcR,
                            &[ec],
                            crate::descr::ec_sys_exc_value_descr(),
                        )
                    });
                    self.with_ctx(|_this, ctx| {
                        ctx.record_op_with_descr(
                            majit_ir::OpCode::SetfieldGc,
                            &[exc_val.opref, active],
                            crate::descr::w_exception_context_descr(kind),
                        );
                    });
                } else {
                    let _ = emit_runtime_raise(self);
                }
                attach_raise_cause(exc, cause)?;
                self.seed_raised_exception(exc_val.opref, exc);
                return Err(PyError::from_exc_object(exc));
            }
            // `eval.rs:1056-1083 / :1099-1121`: `raise SomeType`
            // normalizes by invoking the exception class with no args.
            // Run the call at trace time to get a concrete instance for
            // `GUARD_CLASS` + state seeding, and emit a residual
            // `call_ref` so the compiled trace re-invokes the constructor
            // each iteration instead of folding the trace-time heap
            // address into a `const_ref`.
            if pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(exc) {
                // Mirror `normalize_raise_value` (pyopcode.py:707/:713): the
                // trace-time constructor call must stay on the plain
                // interpreter path so it does not re-enter the tracer.
                let result = {
                    let _plain_guard = pyre_interpreter::call::force_plain_eval();
                    call_function(exc, &[])
                };
                if !pyre_object::is_exception(result) {
                    return Err(PyError::type_error(
                        "exceptions must derive from BaseException",
                    ));
                }
                let exc_box = emit_runtime_raise(self);
                attach_raise_cause(result, cause)?;
                self.seed_raised_exception(exc_box, result);
                return Err(PyError::from_exc_object(result));
            }
        }
        // Non-null, not an instance / exception class — matches
        // `eval.rs:1084-1087` / `eval.rs:1122-1126` fall-through.
        Err(PyError::type_error(
            "exceptions must derive from BaseException",
        ))
    }

    /// `pypy/interpreter/pyopcode.py:1348-1376 RERAISE`.
    fn reraise(&mut self, oparg: u32) -> Result<(), PyError> {
        // pyopcode.py:1357-1363
        let reraise_lasti: i32 = if oparg != 0 {
            // pyopcode.py:1361 — self.space.int_w(self.peekvalue(oparg))
            //
            // Trace-time peek: the lasti slot was pushed by a prior
            // exception-table dispatch (`finishframe_exception` lasti push)
            // as a const-int box `w_int_new(pc as i64)`, so the symbolic
            // `concrete_stack` carries the value verbatim.  When the slot
            // has been displaced and the trace can no longer fold a
            // const-int, signal abort with -1; the dispatcher's RERAISE
            // branch detects `oparg != 0 && reraise_lasti < 0` and routes
            // to the interpreter via `TraceAction::Abort`.
            let s = self.sym();
            match s
                .valuestackdepth
                .checked_sub(s.nlocals + oparg as usize + 1)
            {
                Some(stack_idx) => match s.concrete_stack.get(stack_idx).copied() {
                    Some(crate::state::ConcreteValue::Int(v)) => v as i32,
                    Some(crate::state::ConcreteValue::Ref(obj))
                        if !obj.is_null() && unsafe { pyre_object::is_int(obj) } =>
                    unsafe { pyre_object::w_int_get_value(obj) as i32 },
                    _ => -1,
                },
                None => -1,
            }
        } else {
            -1
        };
        // pyopcode.py:1364 — w_exc = self.popvalue()
        //
        // PyPy's `popvalue()` returns the concrete W_Root that was on TOS;
        // type validation and OperationError construction run against THAT
        // object (`:1367-1369`).  Our `pop_value` returns the symbolic
        // OpRef only — the concrete is in `concrete_stack[TOS]`.  Snapshot
        // the TOS concrete first, then pop, then validate.  Matching the
        // popped value (not `sym.last_exc_value`) keeps trace semantics
        // identical to PyPy even when the stack and the tracker drift
        // (malformed bytecode or a buggy upstream opcode).
        let w_exc: PyObjectRef = {
            let s = self.sym();
            s.valuestackdepth
                .checked_sub(s.nlocals + 1)
                .and_then(|idx| s.concrete_stack.get(idx).copied())
                .map(|cv| cv.to_pyobj())
                .unwrap_or(pyre_object::PY_NULL)
        };
        let _ = self.with_ctx(|this, ctx| this.pop_value(ctx))?;
        // pyopcode.py:1367 — w_value = space.interp_w(W_BaseException, w_exc)
        if w_exc.is_null() || !unsafe { pyre_object::is_exception(w_exc) } {
            return Err(PyError::type_error(
                "exception must derive from BaseException",
            ));
        }
        // pyopcode.py:1368-1369 — w_type = space.type(w_exc); operr = OperationError(...)
        let mut err = unsafe { PyError::from_exc_object(w_exc) };
        // pyopcode.py:1376 — raise RaiseWithExplicitTraceback(operr, reraise_lasti)
        err.attach_tb = false;
        err.reraise_lasti = reraise_lasti;
        Err(err)
    }

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<pyre_interpreter::StepResult<FrontendOp>, PyError> {
        Err(PyError::type_error(format!(
            "unsupported instruction during trace: {instruction:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_miframe<'a>(ctx: &'a mut TraceCtx, sym: &'a mut PyreSym) -> MIFrame {
        MIFrame {
            ctx,
            sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        }
    }

    #[cfg(feature = "cranelift")]
    fn clear_pending_jit_exception() {
        majit_backend_cranelift::jit_exc_raise(0);
    }

    #[cfg(all(not(feature = "cranelift"), feature = "dynasm"))]
    fn clear_pending_jit_exception() {
        majit_backend_dynasm::jit_exc_raise(0);
    }

    #[cfg(not(any(feature = "cranelift", feature = "dynasm")))]
    fn clear_pending_jit_exception() {}

    #[cfg(feature = "cranelift")]
    fn pending_jit_exception_raw() -> i64 {
        majit_backend_cranelift::jit_exc_value_raw()
    }

    #[cfg(all(not(feature = "cranelift"), feature = "dynasm"))]
    fn pending_jit_exception_raw() -> i64 {
        majit_backend_dynasm::jit_exc_value_raw()
    }

    #[cfg(not(any(feature = "cranelift", feature = "dynasm")))]
    fn pending_jit_exception_raw() -> i64 {
        0
    }

    #[test]
    fn exact_w_class_guard_skips_primitive_values() {
        let mut ctx = TraceCtx::for_test_types(&[Type::Int, Type::Float]);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        let mut frame = test_miframe(&mut ctx, &mut sym);

        let fake_type = 0x1234usize as PyObjectRef;
        let before = unsafe { &*frame.ctx }.num_ops();
        let frame_ptr: *mut MIFrame = &mut frame;
        let ctx_ptr = frame.ctx;
        unsafe {
            trace_guard_exact_w_class(
                &mut *frame_ptr,
                &mut *ctx_ptr,
                OpRef::input_arg_int(0),
                fake_type,
            );
            trace_guard_exact_w_class(
                &mut *frame_ptr,
                &mut *ctx_ptr,
                OpRef::input_arg_float(1),
                fake_type,
            );
        }

        assert_eq!(unsafe { &*frame.ctx }.num_ops(), before);
    }

    #[test]
    fn normalize_raise_varargs_jit_null_frame_still_publishes_pending_exception() {
        clear_pending_jit_exception();
        let code = pyre_interpreter::compile_exec("x = ValueError\n").expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new(code);
        frame
            .execute_frame(None, None)
            .expect("module body should execute");
        let exc_class = unsafe {
            (*frame.fget_w_globals())
                .get("x")
                .copied()
                .expect("namespace should contain ValueError")
        };

        let result = normalize_raise_varargs_jit(0, exc_class as i64, pyre_object::PY_NULL as i64);

        assert_eq!(result, pending_jit_exception_raw());
        let err = unsafe { pyre_interpreter::PyError::from_exc_object(result as PyObjectRef) };
        assert_eq!(err.kind, pyre_interpreter::PyErrorKind::RuntimeError);
        assert_eq!(err.message_text(), "raise helper missing current frame");
        clear_pending_jit_exception();
    }

    #[test]
    fn get_list_of_active_boxes_reads_kind_specific_register_banks() {
        use majit_ir::VecAssoc;
        use majit_translate::liveness::encode_liveness;
        use std::sync::Arc;

        let mut all_liveness = vec![1, 1, 1];
        all_liveness.extend(encode_liveness(&[2]));
        all_liveness.extend(encode_liveness(&[1]));
        all_liveness.extend(encode_liveness(&[3]));
        let mut insns: VecAssoc<String, u8> = VecAssoc::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &all_liveness, all_liveness.len(), 1);

        let runtime_jc = {
            let inner = majit_metainterp::jitcode::JitCode::new("get_list_of_active_boxes_test");
            inner.set_body(majit_translate::jitcode::JitCodeBody {
                code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
                c_num_regs_i: 4,
                c_num_regs_r: 4,
                c_num_regs_f: 4,
                // RPython `jitcode.py:85-90` `assert pc in self._startpoints`:
                // hand-crafted bodies must declare each opcode's offset.  The
                // single BC_LIVE here sits at byte 0.
                startpoints: Some([0_usize].into_iter().collect()),
                ..Default::default()
            });
            inner
        };
        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null(), std::ptr::null(), None);
        pyjit.jitcode = Arc::new(runtime_jc);
        pyjit.metadata.pc_map.push(0);
        let inner_jc = crate::state::JitCode {
            code: std::ptr::null(),
            index: 0,
            payload: Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let int_box = OpRef::int_op(10);
        let ref_box = OpRef::ref_op(20);
        let float_box = OpRef::float_op(30);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;
        // SSA-authoritative live_r: the encoder reads the
        // color-indexed Ref bank. Set nlocals=2 so the Ref liveness
        // color 1 reads the temporary bank at index 1 (identity for locals).
        // Int and Float banks stay kind-specific (no unification),
        // so their bank-indexed setup is unchanged.
        sym.nlocals = 2;
        sym.valuestackdepth = 2;
        sym.registers_i = vec![OpRef::NONE, OpRef::NONE, int_box];
        sym.registers_r = vec![OpRef::NONE, ref_box];
        sym.registers_f = vec![OpRef::NONE, OpRef::NONE, OpRef::NONE, float_box];

        let mut ctx = TraceCtx::for_test(1);
        let mut frame = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let active = frame.get_list_of_active_boxes(&mut ctx, false, false, None);
        assert_eq!(active, vec![int_box, ref_box, float_box]);

        unsafe {
            let _ = Box::from_raw(inner_jc_ptr as *mut crate::state::JitCode);
        }
    }

    #[test]
    fn pre_opcode_snapshot_reads_coalesced_stack_color_by_semantic_slot() {
        use majit_ir::VecAssoc;
        use majit_translate::liveness::encode_liveness;
        use std::sync::Arc;

        let mut all_liveness = vec![0, 1, 0];
        all_liveness.extend(encode_liveness(&[0]));
        let mut insns: VecAssoc<String, u8> = VecAssoc::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &all_liveness, all_liveness.len(), 1);

        let runtime_jc = {
            let inner = majit_metainterp::jitcode::JitCode::new(
                "pre_opcode_snapshot_coalesced_stack_color_test",
            );
            inner.set_body(majit_translate::jitcode::JitCodeBody {
                code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
                c_num_regs_r: 3,
                startpoints: Some([0_usize].into_iter().collect()),
                ..Default::default()
            });
            inner
        };
        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null(), std::ptr::null(), None);
        pyjit.jitcode = Arc::new(runtime_jc);
        pyjit.metadata.pc_map.push(0);
        pyjit.metadata.depth_at_py_pc.push(1);
        pyjit.metadata.pyre_color_for_semantic_local = vec![0, 1];
        pyjit.metadata.stack_slot_color_map = vec![0];
        let inner_jc = crate::state::JitCode {
            code: std::ptr::null(),
            index: 0,
            payload: Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let local0 = OpRef::ref_op(10);
        let local1 = OpRef::ref_op(11);
        let stack0 = OpRef::ref_op(20);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;
        sym.nlocals = 2;
        sym.valuestackdepth = 3;
        // Semantic mirror: local0 is at slot 0, while stack depth 0 is at
        // semantic slot 2. Liveness color 0 belongs to the live stack slot,
        // reusing dead local0's color.
        sym.registers_r = vec![local0, local1, stack0];

        let mut ctx = TraceCtx::for_test(1);
        let mut frame = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: Some(vec![local0, local1, stack0]),
            pre_opcode_semantic_depth: Some(3),
        };

        let active = frame.get_list_of_active_boxes(&mut ctx, false, false, None);
        assert_eq!(active, vec![stack0]);

        unsafe {
            let _ = Box::from_raw(inner_jc_ptr as *mut crate::state::JitCode);
        }
    }

    /// Issue #143 M-core2: the resize guard synthesizes a 2-frame snapshot
    /// whose innermost frame is the Rust runtime-helper callee (top) and
    /// whose outer frame is the demoted Python caller resuming at its
    /// post-CALL fallthrough.  Frames come back outermost-first
    /// (`recorder.rs:56`), so `frames[0]` is the caller and `frames[1]` the
    /// synthetic helper.  Exercises `build_synthetic_callee_frames` (the
    /// frame-chain novelty) — NOT the full `_with_synthetic_callee` wrapper,
    /// whose vable/vref tail would deref a skeleton helper's null code.
    #[test]
    fn build_synthetic_callee_frames_synthesizes_self_caller_and_helper_top() {
        use majit_ir::VecAssoc;
        use majit_metainterp::recorder::SnapshotTagged;
        use std::sync::Arc;

        // Empty liveness banks at offset 0: the demoted caller resumes at
        // fallthrough_pc=3 with no live boxes.
        let all_liveness = vec![0u8, 0, 0];
        let mut insns: VecAssoc<String, u8> = VecAssoc::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &all_liveness, all_liveness.len(), 1);

        // Outer (caller) jitcode: BC_LIVE at byte 0 and byte 3.  Identity
        // pc_map so resume_jitcode_pc_for(fallthrough_pc=3) == Some(3)
        // (the M-core1 sibling-test pattern); the operand bytes [4],[5]=0,0
        // select liveness offset 0 -> empty banks.
        let runtime_jc = {
            let inner = majit_metainterp::jitcode::JitCode::new("synth_callee_outer");
            inner.set_body(majit_translate::jitcode::JitCodeBody {
                code: vec![
                    majit_metainterp::jitcode::insns::BC_LIVE,
                    0,
                    0,
                    majit_metainterp::jitcode::insns::BC_LIVE,
                    0,
                    0,
                ],
                c_num_regs_r: 1,
                startpoints: Some([0_usize, 3_usize].into_iter().collect()),
                ..Default::default()
            });
            inner
        };
        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null(), std::ptr::null(), None);
        pyjit.jitcode = Arc::new(runtime_jc);
        pyjit.metadata.pc_map = (0..6).collect();
        const OUTER_INDEX: i32 = 4;
        let inner_jc = crate::state::JitCode {
            code: std::ptr::null(),
            index: OUTER_INDEX,
            payload: Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;
        sym.nlocals = 0;
        sym.valuestackdepth = 0;
        // registers_r left empty: no live boxes for the caller frame.

        let mut ctx = TraceCtx::for_test(1);
        let mut frame = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 3,
            parent_frames: Vec::new(), // empty -> exactly 2 frames out
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            orgpc: 3,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        // Synthetic helper boxes: list = Ref op, value = unboxed Int op.
        let list = OpRef::ref_op(20);
        let value = OpRef::int_op(21);
        const HELPER_INDEX: u32 = 7;
        const GUARD_PC: usize = 99; // guard byte offset (identity pc_map)

        let frames = frame.build_synthetic_callee_frames(
            &mut ctx,
            HELPER_INDEX,
            GUARD_PC,
            &[list, value],
            None,
            None,
        );

        // Outermost-first: [caller (demoted self), helper (innermost/top)].
        assert_eq!(frames.len(), 2);

        // frames[0] = self demoted to caller, resuming at fallthrough_pc.
        assert_eq!(frames[0].jitcode_index, OUTER_INDEX as u32);
        assert_eq!(frames[0].pc, 3); // self.fallthrough_pc
        assert!(frames[0].boxes.is_empty()); // empty liveness, header sliced

        // frames[1] = synthetic Rust-helper callee (new innermost/top frame).
        assert_eq!(frames[1].jitcode_index, HELPER_INDEX);
        assert_eq!(frames[1].pc, GUARD_PC as u32); // guard byte offset, raw
        assert_eq!(
            frames[1].boxes,
            vec![
                SnapshotTagged::Box(list, majit_ir::Type::Ref), // list -> Ref
                SnapshotTagged::Box(value, majit_ir::Type::Int), // value -> Int (no header)
            ]
        );

        unsafe {
            let _ = Box::from_raw(inner_jc_ptr as *mut crate::state::JitCode);
        }
    }
}

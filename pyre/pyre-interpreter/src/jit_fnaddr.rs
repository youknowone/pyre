//! Build-time fnaddr registry for pyre's traced helper surface.
//!
//! `pyre-jit-trace/build.rs` runs the source-only codewriter. Unlike the
//! proc-macro path, it cannot call `#[jit_module]::__majit_helper_trace_fnaddrs()`
//! on the analyzed sources, so pyre publishes the same shape explicitly here.

fn push_fnaddr(entries: &mut Vec<(&'static str, i64)>, full_path: &'static str, fnptr: *const ()) {
    let fnaddr = fnptr as usize as i64;
    if fnaddr != 0 {
        entries.push((full_path, fnaddr));
    }
}

fn push_alias_pair(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    fnptr: *const (),
) {
    push_fnaddr(entries, module_path, fnptr);
    push_fnaddr(entries, root_path, fnptr);
}

const CALLABLE_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_0",
        "pyre_interpreter::jit_call_callable_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_1",
        "pyre_interpreter::jit_call_callable_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_2",
        "pyre_interpreter::jit_call_callable_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_3",
        "pyre_interpreter::jit_call_callable_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_4",
        "pyre_interpreter::jit_call_callable_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_5",
        "pyre_interpreter::jit_call_callable_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_6",
        "pyre_interpreter::jit_call_callable_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_7",
        "pyre_interpreter::jit_call_callable_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_8",
        "pyre_interpreter::jit_call_callable_8",
    ),
];

const KNOWN_BUILTIN_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_0",
        "pyre_interpreter::jit_call_known_builtin_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_1",
        "pyre_interpreter::jit_call_known_builtin_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_2",
        "pyre_interpreter::jit_call_known_builtin_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_3",
        "pyre_interpreter::jit_call_known_builtin_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_4",
        "pyre_interpreter::jit_call_known_builtin_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_5",
        "pyre_interpreter::jit_call_known_builtin_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_6",
        "pyre_interpreter::jit_call_known_builtin_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_7",
        "pyre_interpreter::jit_call_known_builtin_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_8",
        "pyre_interpreter::jit_call_known_builtin_8",
    ),
];

const KNOWN_FUNCTION_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_0",
        "pyre_interpreter::jit_call_known_function_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_1",
        "pyre_interpreter::jit_call_known_function_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_2",
        "pyre_interpreter::jit_call_known_function_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_3",
        "pyre_interpreter::jit_call_known_function_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_4",
        "pyre_interpreter::jit_call_known_function_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_5",
        "pyre_interpreter::jit_call_known_function_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_6",
        "pyre_interpreter::jit_call_known_function_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_7",
        "pyre_interpreter::jit_call_known_function_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_8",
        "pyre_interpreter::jit_call_known_function_8",
    ),
];

const LIST_BUILD_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_build_list_0",
        "pyre_interpreter::jit_build_list_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_1",
        "pyre_interpreter::jit_build_list_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_2",
        "pyre_interpreter::jit_build_list_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_3",
        "pyre_interpreter::jit_build_list_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_4",
        "pyre_interpreter::jit_build_list_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_5",
        "pyre_interpreter::jit_build_list_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_6",
        "pyre_interpreter::jit_build_list_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_7",
        "pyre_interpreter::jit_build_list_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_8",
        "pyre_interpreter::jit_build_list_8",
    ),
];

const TUPLE_BUILD_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_0",
        "pyre_interpreter::jit_build_tuple_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_1",
        "pyre_interpreter::jit_build_tuple_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_2",
        "pyre_interpreter::jit_build_tuple_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_3",
        "pyre_interpreter::jit_build_tuple_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_4",
        "pyre_interpreter::jit_build_tuple_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_5",
        "pyre_interpreter::jit_build_tuple_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_6",
        "pyre_interpreter::jit_build_tuple_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_7",
        "pyre_interpreter::jit_build_tuple_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_8",
        "pyre_interpreter::jit_build_tuple_8",
    ),
];

const MAP_BUILD_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_build_map_0",
        "pyre_interpreter::jit_build_map_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_1",
        "pyre_interpreter::jit_build_map_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_2",
        "pyre_interpreter::jit_build_map_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_3",
        "pyre_interpreter::jit_build_map_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_4",
        "pyre_interpreter::jit_build_map_4",
    ),
];

/// Build-time equivalent of `#[jit_module]::__majit_helper_trace_fnaddrs()`.
///
/// The registry includes both the module-qualified path produced by the
/// source analyzer (`runtime_ops::foo`) and the crate-root re-export path
/// (`foo`) that pyre's runtime helper code often calls directly.
pub fn jit_trace_fnaddrs() -> Vec<(&'static str, i64)> {
    let mut entries = Vec::new();

    push_alias_pair(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_make_function_from_globals",
        "pyre_interpreter::jit_make_function_from_globals",
        crate::runtime_ops::jit_make_function_from_globals as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_load_name_from_namespace",
        "pyre_interpreter::jit_load_name_from_namespace",
        crate::runtime_ops::jit_load_name_from_namespace as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_store_name_to_namespace",
        "pyre_interpreter::jit_store_name_to_namespace",
        crate::runtime_ops::jit_store_name_to_namespace as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_sequence_getitem",
        "pyre_interpreter::jit_sequence_getitem",
        crate::runtime_ops::jit_sequence_getitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_range_iter_next_or_null",
        "pyre_interpreter::jit_range_iter_next_or_null",
        crate::runtime_ops::jit_range_iter_next_or_null as *const (),
    );

    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_truth_value",
        "pyre_interpreter::jit_truth_value",
        crate::opcode_ops::jit_truth_value as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_bool_value_from_truth",
        "pyre_interpreter::jit_bool_value_from_truth",
        crate::opcode_ops::jit_bool_value_from_truth as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_binary_value_from_tag",
        "pyre_interpreter::jit_binary_value_from_tag",
        crate::opcode_ops::jit_binary_value_from_tag as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_compare_value_from_tag",
        "pyre_interpreter::jit_compare_value_from_tag",
        crate::opcode_ops::jit_compare_value_from_tag as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_unary_negative_value",
        "pyre_interpreter::jit_unary_negative_value",
        crate::opcode_ops::jit_unary_negative_value as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_unary_invert_value",
        "pyre_interpreter::jit_unary_invert_value",
        crate::opcode_ops::jit_unary_invert_value as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_getitem",
        "pyre_interpreter::jit_getitem",
        crate::opcode_ops::jit_getitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_setitem",
        "pyre_interpreter::jit_setitem",
        crate::opcode_ops::jit_setitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_getattr",
        "pyre_interpreter::jit_getattr",
        crate::opcode_ops::jit_getattr as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_setattr",
        "pyre_interpreter::jit_setattr",
        crate::opcode_ops::jit_setattr as *const (),
    );

    // Production walker's `Instruction::StoreSubscr` arm emits a
    // `residual_call_r_r` whose funcptr resolves at codewriter time
    // through the bare path `["execute_store_subscr"]` (the dispatch-
    // table entry at `pyopcode.rs:2909`).  Without a runtime fnaddr
    // entry the codewriter mints a `symbolic_fnaddr_for_path` hash
    // that the `runtime_fnaddr_patch` cannot rewrite and sub-slice 4's
    // 47-bit sanity gate rejects, causing the walker to skip the heap
    // mutation and SIGBUS on the next read.  `bh_execute_store_subscr`
    // is the C-ABI bridge over the generic
    // `execute_store_subscr::<PyFrame>` whose `Result<StepResult<_>,
    // PyError>` cannot ride the residual_call's single-register Ref
    // result slot.  Registering the bare path here lets the codewriter
    // bake the wrapper address directly into `JitCode.constants_i`,
    // mirroring PyPy's `cpu.bh_call_*` -> linker-resolved C symbol
    // contract (`pyjitpl.py:1346 _opimpl_residual_call*`).
    push_fnaddr(
        &mut entries,
        "execute_store_subscr",
        crate::opcode_ops::bh_execute_store_subscr as *const (),
    );

    // `cpu.store_subscr_fn` binding (`pyre-jit/src/jit/cpu.rs:151`)
    // bound via `pyre_interpreter::opcode_ops::bh_store_subscr_fn`
    // (relocated from `pyre-jit/src/call_jit.rs` in 5.5c).
    // Registered here so `pyre-jit-trace`'s walker specialization gate
    // (`try_walker_store_subscr_specialization`) can recover the
    // runtime address via `jit_trace_fnaddrs()` lookup without a
    // cross-crate `pyre-jit-trace → pyre-jit` dependency edge.
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::bh_store_subscr_fn",
        "pyre_interpreter::bh_store_subscr_fn",
        crate::opcode_ops::bh_store_subscr_fn as *const (),
    );

    for (nargs, (module_path, root_path)) in CALLABLE_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::callable_call_helper(nargs) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (nargs, (module_path, root_path)) in KNOWN_BUILTIN_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::known_builtin_call_helper(nargs) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (nargs, (module_path, root_path)) in KNOWN_FUNCTION_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::known_function_call_helper(nargs) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (count, (module_path, root_path)) in LIST_BUILD_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::list_build_helper(count) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (count, (module_path, root_path)) in TUPLE_BUILD_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::tuple_build_helper(count) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (count, (module_path, root_path)) in MAP_BUILD_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::map_build_helper(count) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }

    push_alias_pair(
        &mut entries,
        "pyre_object::intobject::jit_w_int_new",
        "pyre_object::jit_w_int_new",
        pyre_object::jit_w_int_new as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::floatobject::jit_w_float_new",
        "pyre_object::jit_w_float_new",
        pyre_object::jit_w_float_new as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::listobject::jit_list_append",
        "pyre_object::jit_list_append",
        pyre_object::jit_list_append as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::listobject::jit_list_getitem",
        "pyre_object::jit_list_getitem",
        pyre_object::jit_list_getitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::listobject::jit_list_setitem",
        "pyre_object::jit_list_setitem",
        pyre_object::jit_list_setitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::listobject::jit_list_reverse",
        "pyre_object::jit_list_reverse",
        pyre_object::jit_list_reverse as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::tupleobject::jit_tuple_getitem",
        "pyre_object::jit_tuple_getitem",
        pyre_object::jit_tuple_getitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::strobject::jit_str_concat",
        "pyre_object::jit_str_concat",
        pyre_object::jit_str_concat as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::strobject::jit_str_repeat",
        "pyre_object::jit_str_repeat",
        pyre_object::jit_str_repeat as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::strobject::jit_str_compare",
        "pyre_object::jit_str_compare",
        pyre_object::jit_str_compare as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::strobject::jit_str_is_true",
        "pyre_object::jit_str_is_true",
        pyre_object::jit_str_is_true as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::rangeobject::jit_range_iter_new",
        "pyre_object::jit_range_iter_new",
        pyre_object::jit_range_iter_new as *const (),
    );

    // `@jit.elidable`-decorated inherent methods that show up as
    // `residual_call_*` in the codewriter (`call.py:181-187
    // getfunctionptr(graph)` parity).  Without an entry here
    // `direct_funcptr_value` (`jtransform.rs:614-623`) falls back to
    // `symbolic_fnaddr_for_path`, which is a deterministic hash but NOT
    // a valid function address — invoking it at the walker's
    // `execute_residual_call` (`jitcode_dispatch.rs:3192-3239`) is an
    // immediate SEGV.  Path shape matches
    // `target_to_path` for inherent method calls
    // (`call.rs:3024-3028 CallPath::for_impl_method(impl_type_joined,
    // method)`): the `register_macro_helper_trace_fnaddr` string-strip
    // drops the leading crate segment, leaving `[module, Type, method]`
    // which is the exact 3-segment shape `for_impl_method` produces.
    //
    // PyFrame::nlocals — invoked by `eval.rs:840 pop_value` and is the
    // funcptr the walker reaches when dispatching `PopTop`'s nested
    // `pop_value` sub-jitcode.
    let pyframe_nlocals: fn(&crate::pyframe::PyFrame) -> usize = crate::pyframe::PyFrame::nlocals;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::pyframe::PyFrame::nlocals",
        pyframe_nlocals as *const (),
    );

    // `PyFrame::pop` — invoked by `<PyFrame as SharedOpcodeHandler>::pop_value`
    // at `eval.rs:844 Ok(self.pop())`.  Two CallPath shapes need binding:
    //
    // 1. The qualified `PyFrame::pop(self)` spelling resolves to the
    //    2-segment CallPath `["PyFrame", "pop"]` via `for_impl_method`.
    // 2. The bare `self.pop()` spelling goes through `target_to_path`'s
    //    suffix-match fallback (call.rs:3069-3112), which returns the
    //    3-segment module-qualified key `["pyframe", "PyFrame", "pop"]`
    //    that `function_graphs` actually stores inherent impl methods
    //    under (per `parse::extract_inherent_impl_methods`).
    //
    // `register_macro_helper_trace_fnaddr` strips the leading segment,
    // so we register both spellings via `push_alias_pair`: the 3-segment
    // input `pyre_interpreter::PyFrame::pop` produces the 2-segment
    // canonical, and the 4-segment input `pyre_interpreter::pyframe::PyFrame::pop`
    // produces the 3-segment module-qualified form.  Without the second
    // binding, `fnaddr_for_target` for `self.pop()` falls back to the
    // symbolic hash from [`symbolic_fnaddr_for_path`], which SEGVs at
    // trace-time call.
    let pyframe_pop: fn(&mut crate::pyframe::PyFrame) -> pyre_object::PyObjectRef =
        crate::pyframe::PyFrame::pop;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyframe::PyFrame::pop",
        "pyre_interpreter::PyFrame::pop",
        pyframe_pop as *const (),
    );

    // `pop_value`'s underflow guard returns `stack_underflow_error(...)`
    // when the value stack is empty (`eval.rs:840-845` +
    // `eval.rs:847-852 peek_at`). The codewriter follows that call edge
    // into the `stack_underflow_error` helper jitcode, whose `constants_i[0]`
    // is the function pointer the walker invokes at residual_call time.
    // Without this binding the codewriter falls back to
    // `symbolic_fnaddr_for_path`, which is a deterministic hash and
    // SEGVs when called.  `lib.rs:72 pub use shared_opcode::*` re-exports
    // the helper at the crate root, so the codewriter resolves
    // `stack_underflow_error` to `pyre_interpreter::stack_underflow_error`
    // when it appears as a bare identifier in `eval.rs`; register both
    // the module-qualified path and the root re-export path via
    // [`push_alias_pair`].
    let stack_underflow: fn(&str) -> crate::PyError = crate::shared_opcode::stack_underflow_error;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::shared_opcode::stack_underflow_error",
        "pyre_interpreter::stack_underflow_error",
        stack_underflow as *const (),
    );

    // `pyframe_get_pycode` / `ncells` / `npure_cellvars` / `PyFrame::ncells`
    // carry `#[elidable_cannot_raise]`.  `call.rs:has_cannot_raise_assertion`
    // only honours the assertion when `function_fnaddrs.contains_key(p)`,
    // so without a registration the descr falls back to
    // `EF_ELIDABLE_CAN_RAISE`.
    let pyframe_get_pycode_fn: unsafe fn(&crate::pyframe::PyFrame) -> *const crate::CodeObject =
        crate::pyframe::pyframe_get_pycode;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::pyframe::pyframe_get_pycode",
        pyframe_get_pycode_fn as *const (),
    );

    let pyframe_ncells_free: fn(&crate::CodeObject) -> usize = crate::pyframe::ncells;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::pyframe::ncells",
        pyframe_ncells_free as *const (),
    );

    let pyframe_npure_cellvars: fn(&crate::CodeObject) -> usize = crate::pyframe::npure_cellvars;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::pyframe::npure_cellvars",
        pyframe_npure_cellvars as *const (),
    );

    let pyframe_ncells_method: fn(&crate::pyframe::PyFrame) -> usize =
        crate::pyframe::PyFrame::ncells;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::pyframe::PyFrame::ncells",
        pyframe_ncells_method as *const (),
    );

    // B1 LoadFast/LoadFastBorrow/LoadFastCheck arm folding helpers.  Both
    // carry `#[elidable_cannot_raise]` so `has_cannot_raise_assertion`
    // requires the fnaddr registration to fire (`call.rs:3626-3631`
    // gates the assertion on `function_fnaddrs.contains_key(p)`).
    // Without these the chained `Arg::get` / `VarNum::as_usize` /
    // `Vec::len` third-party helpers reach the walker as unfolded
    // `residual_call` ops and the walker's `goto_if_not` bounds-check
    // aborts with `GotoIfNotValueNotConcrete`.
    //
    // `push_alias_pair` (vs plain `push_fnaddr`) is required because the
    // in-module call site `load_fast_var_num_to_index(var_num, op_arg)`
    // inside `pyopcode.rs` resolves to a bare-segment `CallPath`
    // (`["load_fast_var_num_to_index"]`) that the assertion-aware hint
    // walker DOES populate but the module-qualified-only fnaddr
    // registration would miss.  Register the bare alias alongside the
    // canonical `pyopcode::name` form so the assertion gate fires.
    let load_fast_var_num_to_index: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::VarNum>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::load_fast_var_num_to_index;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::load_fast_var_num_to_index",
        "pyre_interpreter::load_fast_var_num_to_index",
        load_fast_var_num_to_index as *const (),
    );

    let code_varnames_len: fn(&crate::CodeObject) -> usize = crate::pyopcode::code_varnames_len;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::code_varnames_len",
        "pyre_interpreter::code_varnames_len",
        code_varnames_len as *const (),
    );

    // `PyError::type_error` — invoked by `stack_underflow_error`'s body
    // (`shared_opcode.rs:181-183`). The codewriter resolves it to the
    // 2-segment CallPath `["PyError", "type_error"]` (impl-method shape:
    // type segment + method segment).  `register_macro_helper_trace_fnaddr`
    // strips the leading crate segment, so the input string must have
    // exactly 3 segments to produce the desired 2-segment canonical form.
    let pyerror_type_error: fn(String) -> crate::PyError = |msg| crate::PyError::type_error(msg);
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::PyError::type_error",
        pyerror_type_error as *const (),
    );

    // RPython convention (cross-reference `support.py:255-271` for
    // the C-trunc helpers, `rint.py:398/495` for the Python-floor
    // ones) is to keep the two semantic flavours under DISTINCT
    // canonical names:
    //
    //   - bare `int_mod` / `int_floordiv` — the lltype-level
    //     truncating primitive (canonical names of the
    //     `_ll_2_int_mod` / `_ll_2_int_floordiv` no-branch reverse).
    //     C-truncating output.
    //   - `int.py_mod` / `int.py_div` — the Python-semantic
    //     `@jit.oopspec("int.py_mod")` / `@jit.oopspec("int.py_div")`
    //     names that decorate `ll_int_py_mod` / `ll_int_py_div`.
    //     Python-floor output.
    //
    // Pyre's `jtransform.rs` BinOp{mod,floordiv,Int} arm emits a
    // `CallTarget::function_path(["_ll_2_int_mod"])` /
    // `CallTarget::function_path(["_ll_2_int_floordiv"])` per
    // `jtransform.py:576-577 rewrite_op_int_floordiv =
    // _do_builtin_call` (which resolves the helper through
    // `support.py:266` `_ll_2_int_mod` / `:255` `_ll_2_int_floordiv`).
    // The C-trunc residual call below is what the trace path sees;
    // the Python-floor `ll_int_py_*` helpers stay available for the
    // future route-(b) emitter (Python-bytecode `int.py_mod` /
    // `int.py_div` direct calls) under the dotted-name keys.
    //
    // `register_macro_helper_trace_fnaddr` strips the leading segment
    // from `full_path`; for a single-segment path (no `::`) the entire
    // string survives as the canonical CallPath, matching the segment
    // shape jtransform produces.
    //
    // The Rust-source graphs for the integer helpers are NOT
    // registered in `CallControl::function_graphs` (pyre has no
    // `MixLevelHelperAnnotator` to materialise a graph from a `pub
    // extern "C"` function pointer), so `call.rs:1620-1670`
    // `find_all_graphs_bfs` finds the function pointer via
    // `function_fnaddrs` lookup but cannot seed the BFS through the
    // helper's body — the helpers stay opaque to the inliner,
    // matching upstream behaviour for any `@dont_look_inside`
    // oopspec helper.  Two `support.py:inline_calls_to` entries
    // are intentionally NOT bound:
    //   * `_ll_1_int_abs` — RPython `inline_calls_to` seeds the
    //     `int_abs` helper *graph* into the BFS for actual inlining
    //     at `call.py:60-64 todo.append(c_func.value._obj.graph)`.
    //     Pyre can register the fnaddr but cannot fabricate the
    //     helper body graph from an `extern "C"` function pointer
    //     (no `MixLevelHelperAnnotator.constfunc` analogue), so a
    //     fnaddr-only binding would make `int_abs` an opaque extern
    //     helper — the opposite of the upstream inlining intent.
    //     No production pyre rewrite emits `direct_call(_ll_1_int_abs)`
    //     so the binding is omitted until the rtyper-equivalent
    //     can synthesise the body graph.
    //   * `_ll_1_ll_math_ll_math_sqrt` — `rpython/rtyper/lltypesystem/
    //     module/ll_math.py:317-322 ll_math_sqrt` raises
    //     `ValueError("math domain error")` on negative input, and
    //     Rust's `f64::sqrt()` returns NaN; making the fnaddr
    //     reachable would be a silent semantic regression.
    // See the TODO block at
    // `call.rs::find_all_graphs_bfs` for the convergence path.
    push_fnaddr(
        &mut entries,
        "_ll_2_int_floordiv",
        majit_metainterp::blackhole::_ll_2_int_floordiv as *const (),
    );
    push_fnaddr(
        &mut entries,
        "_ll_2_int_mod",
        majit_metainterp::blackhole::_ll_2_int_mod as *const (),
    );

    // `support.py:274 _ll_1_cast_uint_to_float` / `_ll_1_cast_float_to_uint`
    // residual-call targets emitted by
    // `jit_codewriter/jtransform.rs:cast_*_to_*` (mirroring
    // `jtransform.py:587-588 _do_builtin_call`).  Without these the
    // codewriter falls back to `symbolic_fnaddr_for_path`, which
    // produces a deterministic but unbound hash — fine for source
    // analysis but unreachable at runtime.  The 1-segment root_path
    // alias is what `CallTarget::function_path(["cast_uint_to_float"])`
    // resolves against after `register_macro_helper_trace_fnaddr`
    // strips the crate segment.
    push_alias_pair(
        &mut entries,
        "majit_metainterp::blackhole::cast_uint_to_float",
        "majit_metainterp::cast_uint_to_float",
        majit_metainterp::blackhole::cast_uint_to_float as *const (),
    );
    push_alias_pair(
        &mut entries,
        "majit_metainterp::blackhole::cast_float_to_uint",
        "majit_metainterp::cast_float_to_uint",
        majit_metainterp::blackhole::cast_float_to_uint as *const (),
    );

    // `_ll_2_str_eq_nonnull` (`rpython/jit/codewriter/support.py:526-
    // 538`) is the helper canonically registered by `jtransform.py:
    // 620-624 _register_extra_helper(OS_STREQ_NONNULL, "str.eq_nonnull",
    // ...)` and `:637-641 _register_extra_helper(OS_UNIEQ_NONNULL,
    // "str.eq_nonnull", ...)`.  Pyre intentionally does NOT register
    // a host fnaddr for it: there is no `rstr.STR`-equivalent GC
    // layout in pyre-object today, so a registration would have to
    // point at a panic-stub that fails at runtime — a parity
    // violation against `support.py:526-538`'s real `s.chars[i]`
    // comparison body.
    //
    // Pyre's type state has no `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`
    // channel yet: the elidable-promote dual hint (`PromoteOrString`)
    // falls through to the plain `<kind>_guard_value` arm, and direct
    // `hint_promote_string` / `hint_promote_unicode` calls fail loud
    // in `jit_codewriter/jtransform.rs`.  Re-introduce the
    // registration here together with a line-by-line port of
    // `_ll_2_str_eq_nonnull`'s body in `majit-metainterp::blackhole`
    // once pyre grows the backing GC struct.

    entries
}

/// Build-time addresses of the prebuilt static `PyType` singletons that
/// pyre source carries through the flowgraph as opaque `LOAD_GLOBAL`
/// constants (`flowcontext.py:856` pushes the per-module-globals entry
/// as `Constant(value)`).  The codewriter bakes each into
/// `JitCode.constants_i` as a build-time `ConstValue::Int(addr)`.
///
/// The translator (`majit-translate`) sits in `rpython/` layer terms
/// below the object space and must not import `pyre-object`; the driver
/// supplies these prebuilt-instance addresses across the translation
/// boundary the same way `rpython/jit` receives `Constant(GCREF)` from
/// the host rather than importing `pypy/objspace`.  Resolved here in the
/// same build-script process that runs the translator, so the captured
/// addresses are identical to a direct `&pyre_object::X` read at the
/// codewriter call site.
///
/// Keys are the in-source `module::NAME` spelling the front-end
/// `Expr::Path` arm looks up in `KnownStaticsCatalogue`.
pub fn jit_static_pytype_addrs() -> Vec<(&'static str, i64)> {
    macro_rules! pytype_addr {
        ($key:literal, $($path:tt)::+) => {
            ($key, &pyre_object::$($path)::+ as *const _ as i64)
        };
    }
    vec![
        pytype_addr!(
            "bytearrayobject::BYTEARRAY_TYPE",
            bytearrayobject::BYTEARRAY_TYPE
        ),
        pytype_addr!("bytesobject::BYTES_TYPE", bytesobject::BYTES_TYPE),
        pytype_addr!(
            "celldict::OBJECT_MUTABLE_CELL_TYPE",
            celldict::OBJECT_MUTABLE_CELL_TYPE
        ),
        pytype_addr!(
            "celldict::INT_MUTABLE_CELL_TYPE",
            celldict::INT_MUTABLE_CELL_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::MODULE_DICT_TYPE",
            dictmultiobject::MODULE_DICT_TYPE
        ),
        pytype_addr!(
            "dictviewobject::DICT_KEYS_TYPE",
            dictviewobject::DICT_KEYS_TYPE
        ),
        pytype_addr!(
            "dictviewobject::DICT_VALUES_TYPE",
            dictviewobject::DICT_VALUES_TYPE
        ),
        pytype_addr!(
            "dictviewobject::DICT_ITEMS_TYPE",
            dictviewobject::DICT_ITEMS_TYPE
        ),
        pytype_addr!(
            "dictviewobject::DICT_KEYITERATOR_TYPE",
            dictviewobject::DICT_KEYITERATOR_TYPE
        ),
        pytype_addr!(
            "dictviewobject::DICT_VALUEITERATOR_TYPE",
            dictviewobject::DICT_VALUEITERATOR_TYPE
        ),
        pytype_addr!(
            "dictviewobject::DICT_ITEMITERATOR_TYPE",
            dictviewobject::DICT_ITEMITERATOR_TYPE
        ),
        pytype_addr!("excobject::EXCEPTION_TYPE", excobject::EXCEPTION_TYPE),
        pytype_addr!(
            "excobject::EXC_EXCEPTION_TYPE",
            excobject::EXC_EXCEPTION_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_ARITHMETIC_ERROR_TYPE",
            excobject::EXC_ARITHMETIC_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_OVERFLOW_ERROR_TYPE",
            excobject::EXC_OVERFLOW_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_ZERO_DIVISION_ERROR_TYPE",
            excobject::EXC_ZERO_DIVISION_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_TYPE_ERROR_TYPE",
            excobject::EXC_TYPE_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_VALUE_ERROR_TYPE",
            excobject::EXC_VALUE_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_NAME_ERROR_TYPE",
            excobject::EXC_NAME_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_INDEX_ERROR_TYPE",
            excobject::EXC_INDEX_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_KEY_ERROR_TYPE",
            excobject::EXC_KEY_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_ATTRIBUTE_ERROR_TYPE",
            excobject::EXC_ATTRIBUTE_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_RUNTIME_ERROR_TYPE",
            excobject::EXC_RUNTIME_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_STOP_ITERATION_TYPE",
            excobject::EXC_STOP_ITERATION_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_IMPORT_ERROR_TYPE",
            excobject::EXC_IMPORT_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_NOT_IMPLEMENTED_ERROR_TYPE",
            excobject::EXC_NOT_IMPLEMENTED_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_ASSERTION_ERROR_TYPE",
            excobject::EXC_ASSERTION_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_REFERENCE_ERROR_TYPE",
            excobject::EXC_REFERENCE_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_GENERATOR_EXIT_TYPE",
            excobject::EXC_GENERATOR_EXIT_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_RECURSION_ERROR_TYPE",
            excobject::EXC_RECURSION_ERROR_TYPE
        ),
        pytype_addr!("excobject::EXC_OS_ERROR_TYPE", excobject::EXC_OS_ERROR_TYPE),
        pytype_addr!(
            "excobject::EXC_FILE_NOT_FOUND_ERROR_TYPE",
            excobject::EXC_FILE_NOT_FOUND_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_UNICODE_DECODE_ERROR_TYPE",
            excobject::EXC_UNICODE_DECODE_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_UNICODE_ENCODE_ERROR_TYPE",
            excobject::EXC_UNICODE_ENCODE_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_UNICODE_TRANSLATE_ERROR_TYPE",
            excobject::EXC_UNICODE_TRANSLATE_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_SYSTEM_EXIT_TYPE",
            excobject::EXC_SYSTEM_EXIT_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_MEMORY_ERROR_TYPE",
            excobject::EXC_MEMORY_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_SYSTEM_ERROR_TYPE",
            excobject::EXC_SYSTEM_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_LOOKUP_ERROR_TYPE",
            excobject::EXC_LOOKUP_ERROR_TYPE
        ),
        pytype_addr!(
            "excobject::EXC_UNICODE_ERROR_TYPE",
            excobject::EXC_UNICODE_ERROR_TYPE
        ),
        pytype_addr!(
            "generatorobject::GENERATOR_TYPE",
            generatorobject::GENERATOR_TYPE
        ),
        pytype_addr!("pyobject::INT_TYPE", pyobject::INT_TYPE),
        pytype_addr!("pyobject::BOOL_TYPE", pyobject::BOOL_TYPE),
        pytype_addr!("pyobject::FLOAT_TYPE", pyobject::FLOAT_TYPE),
        pytype_addr!("pyobject::STR_TYPE", pyobject::STR_TYPE),
        pytype_addr!("pyobject::LIST_TYPE", pyobject::LIST_TYPE),
        pytype_addr!("pyobject::TUPLE_TYPE", pyobject::TUPLE_TYPE),
        pytype_addr!("pyobject::DICT_TYPE", pyobject::DICT_TYPE),
        pytype_addr!("pyobject::LONG_TYPE", pyobject::LONG_TYPE),
        pytype_addr!("pyobject::NONE_TYPE", pyobject::NONE_TYPE),
        pytype_addr!(
            "pyobject::NOTIMPLEMENTED_TYPE",
            pyobject::NOTIMPLEMENTED_TYPE
        ),
        pytype_addr!("pyobject::ELLIPSIS_TYPE", pyobject::ELLIPSIS_TYPE),
        pytype_addr!("pyobject::MODULE_TYPE", pyobject::MODULE_TYPE),
        pytype_addr!("pyobject::MAPPING_PROXY_TYPE", pyobject::MAPPING_PROXY_TYPE),
        pytype_addr!("pyobject::TYPE_TYPE", pyobject::TYPE_TYPE),
        pytype_addr!("pyobject::INSTANCE_TYPE", pyobject::INSTANCE_TYPE),
        pytype_addr!("setobject::SET_TYPE", setobject::SET_TYPE),
        pytype_addr!("setobject::FROZENSET_TYPE", setobject::FROZENSET_TYPE),
        pytype_addr!(
            "specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE",
            specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE
        ),
        pytype_addr!(
            "specialisedtupleobject::SPECIALISED_TUPLE_FF_TYPE",
            specialisedtupleobject::SPECIALISED_TUPLE_FF_TYPE
        ),
        pytype_addr!(
            "specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE",
            specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE
        ),
        pytype_addr!("weakref::GC_WEAKREF_TYPE", weakref::GC_WEAKREF_TYPE),
    ]
}

/// Build-time addresses of the prebuilt dict-strategy singletons pyre
/// source references as opaque ref constants.  Same translation-boundary
/// contract as [`jit_static_pytype_addrs`]; the front-end records these
/// under `ValueType::Ref(None)`.
pub fn jit_static_ref_addrs() -> Vec<(&'static str, i64)> {
    macro_rules! ref_addr {
        ($key:literal, $($path:tt)::+) => {
            ($key, &pyre_object::$($path)::+ as *const _ as i64)
        };
    }
    vec![
        ref_addr!(
            "dictstrategy::OBJECT_DICT_STRATEGY",
            dictstrategy::OBJECT_DICT_STRATEGY
        ),
        ref_addr!(
            "dictstrategy::EMPTY_DICT_STRATEGY",
            dictstrategy::EMPTY_DICT_STRATEGY
        ),
        ref_addr!(
            "dictstrategy::EMPTY_KWARGS_DICT_STRATEGY",
            dictstrategy::EMPTY_KWARGS_DICT_STRATEGY
        ),
        ref_addr!(
            "dictstrategy::BYTES_DICT_STRATEGY",
            dictstrategy::BYTES_DICT_STRATEGY
        ),
        ref_addr!(
            "dictstrategy::UNICODE_DICT_STRATEGY",
            dictstrategy::UNICODE_DICT_STRATEGY
        ),
        ref_addr!(
            "dictstrategy::INT_DICT_STRATEGY",
            dictstrategy::INT_DICT_STRATEGY
        ),
        ref_addr!(
            "identitydict::IDENTITY_DICT_STRATEGY",
            identitydict::IDENTITY_DICT_STRATEGY
        ),
        ref_addr!(
            "kwargsdict::KWARGS_DICT_STRATEGY",
            kwargsdict::KWARGS_DICT_STRATEGY
        ),
        // Prebuilt object singletons (`None` / `NotImplemented` /
        // `Ellipsis` / `True` / `False`).  The accessors `w_none`,
        // `w_ellipsis`, `w_not_implemented`, `w_bool_from` read these
        // statics as a bare same-file `LOAD_GLOBAL` and return their
        // address; supplying the captured address lets the front-end
        // `Expr::Path` same-file fold emit `ConstRefAddr` with the real
        // runtime identity instead of a cross-block body-`Input`.  The
        // statics are private (callers route through the accessors), so
        // the address is captured through the accessor rather than the
        // `ref_addr!` `&pyre_object::X` path form.
        (
            "noneobject::NONE_SINGLETON",
            pyre_object::w_none() as usize as i64,
        ),
        (
            "noneobject::NOT_IMPLEMENTED_SINGLETON",
            pyre_object::w_not_implemented() as usize as i64,
        ),
        (
            "noneobject::ELLIPSIS_SINGLETON",
            pyre_object::w_ellipsis() as usize as i64,
        ),
        (
            "boolobject::TRUE_SINGLETON",
            pyre_object::w_bool_from(true) as usize as i64,
        ),
        (
            "boolobject::FALSE_SINGLETON",
            pyre_object::w_bool_from(false) as usize as i64,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::jit_trace_fnaddrs;
    use std::collections::HashMap;

    #[test]
    fn jit_trace_fnaddrs_contains_root_and_module_aliases() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let make_fn =
            crate::runtime_ops::jit_make_function_from_globals as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::runtime_ops::jit_make_function_from_globals"],
            make_fn
        );
        assert_eq!(
            bindings["pyre_interpreter::jit_make_function_from_globals"],
            make_fn
        );

        let list_append = pyre_object::jit_list_append as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::listobject::jit_list_append"],
            list_append
        );
        assert_eq!(bindings["pyre_object::jit_list_append"], list_append);
    }

    #[test]
    fn jit_trace_fnaddrs_covers_generated_runtime_helper_families() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let callable3 =
            crate::runtime_ops::callable_call_helper(3).expect("callable helper") as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::runtime_ops::jit_call_callable_3"],
            callable3
        );
        assert_eq!(bindings["pyre_interpreter::jit_call_callable_3"], callable3);

        let tuple2 =
            crate::runtime_ops::tuple_build_helper(2).expect("tuple build helper") as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::runtime_ops::jit_build_tuple_2"],
            tuple2
        );
        assert_eq!(bindings["pyre_interpreter::jit_build_tuple_2"], tuple2);
    }

    #[test]
    fn jit_trace_fnaddrs_covers_store_subscr_helpers() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let execute_store_subscr =
            crate::opcode_ops::bh_execute_store_subscr as *const () as usize as i64;
        assert_eq!(bindings["execute_store_subscr"], execute_store_subscr);

        let store_subscr_fn = crate::opcode_ops::bh_store_subscr_fn as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::opcode_ops::bh_store_subscr_fn"],
            store_subscr_fn
        );
        assert_eq!(
            bindings["pyre_interpreter::bh_store_subscr_fn"],
            store_subscr_fn
        );
    }

    /// Negative parity guard: pyre intentionally does NOT publish a
    /// host fnaddr for `_ll_2_str_eq_nonnull` (see the comment block
    /// at `jit_trace_fnaddrs` next to the `cast_float_to_uint`
    /// registration).  A stub registration would fail at runtime
    /// inside any guard-failure recovery; better to surface the
    /// missing helper at codewriter time via the fail-loud
    /// `PromoteString` / `PromoteUnicode` rewrite arms.
    #[test]
    fn jit_trace_fnaddrs_omits_str_eq_nonnull_helper_until_rstr_str_layout_lands() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        assert_eq!(
            bindings.get("_ll_2_str_eq_nonnull").copied(),
            None,
            "no `_ll_2_str_eq_nonnull` fnaddr should be published while pyre \
             lacks an `rstr.STR`-equivalent GC layout — registering one would \
             point at a panic-stub that fails at runtime, contradicting \
             `rpython/jit/codewriter/support.py:526-538`'s real comparison body"
        );
    }
}

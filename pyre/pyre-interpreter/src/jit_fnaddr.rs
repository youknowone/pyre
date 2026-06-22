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

/// Returns `true` when `addr` is the runtime address of a `PyFrame`
/// operand-stack accessor (`pop` / `push` / `peek` / `peek_at`).
///
/// The full-body-walk tracer concretely executes plain residual calls during
/// tracing to fold their results, but a residual targeting one of these
/// accessors reads or mutates the live frame's operand stack — which during a
/// walk is empty, because the walk tracks operand values symbolically in its
/// register banks rather than on the real frame.  Executing one underflows
/// (`pop` asserts `valuestackdepth > stack_base()`).  The walker uses this
/// predicate to leave such a residual symbolic so it runs at runtime against a
/// frame whose operand stack is populated.
///
/// The accessors are `#[inline]`, so a fresh `PyFrame::pop as *const ()` is
/// not address-stable across call sites — it can resolve to a distinct
/// out-of-line copy than the one the codewriter baked into the JitCode
/// constant pool.  Match instead against the exact funcptrs the codewriter
/// bakes: the values [`jit_trace_fnaddrs`] records for the accessor paths,
/// computed through the very same coercion site (cached once, addresses are
/// process-stable).
///
/// Today only `PyFrame::pop` is registered in [`jit_trace_fnaddrs`] (the only
/// accessor a residual call currently reaches — `pop_value`'s sub-jitcode).
/// The `push` / `peek` / `peek_at` arms below are dormant defensive guards:
/// their paths never appear in the registry, so they never match.  They
/// activate (still as a SAFE leave-symbolic decline) only if those accessors
/// are later registered; an unregistered helper is already declined upstream
/// by the funcptr-hash gate, so registering them is unnecessary for soundness.
pub fn is_pyframe_operand_stack_accessor(addr: usize) -> bool {
    use std::sync::OnceLock;
    static ACCESSOR_ADDRS: OnceLock<Vec<i64>> = OnceLock::new();
    let addrs = ACCESSOR_ADDRS.get_or_init(|| {
        jit_trace_fnaddrs()
            .into_iter()
            .filter(|(path, _)| {
                path.ends_with("::PyFrame::pop")
                    || path.ends_with("::PyFrame::push")
                    || path.ends_with("::PyFrame::peek")
                    || path.ends_with("::PyFrame::peek_at")
            })
            .map(|(_, fnaddr)| fnaddr)
            .collect()
    });
    addrs.contains(&(addr as i64))
}

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
    // that the `runtime_fnaddr_patch` cannot rewrite; the walker rejects
    // the unresolved address and skips the heap mutation, leaving the next
    // read to observe stale container state.  `bh_execute_store_subscr`
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
    // bound via `pyre_interpreter::opcode_ops::bh_store_subscr_fn`.
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

    // `dont_look_inside` runtime-state accessors residualised at trace
    // time (TLS / per-type atomic the tracer cannot model).  Their
    // residual call resolves its address here by qualified path; a
    // missing entry would fall back to a symbolic hash that SEGVs at
    // trace time.  `shadow_stack_len` carries a JIT-representable
    // `-> int` signature and binds its Rust `fn` directly (the
    // `PyFrame::nlocals` / `get_current_exception` precedent);
    // `w_type_set_uses_object_setattr` rides a C-ABI bridge that
    // normalises its `bool` argument.
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_roots::shadow_stack_len",
        "pyre_object::shadow_stack_len",
        pyre_object::gc_roots::shadow_stack_len as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::typeobject::w_type_set_uses_object_setattr",
        "pyre_object::w_type_set_uses_object_setattr",
        crate::opcode_ops::bh_w_type_set_uses_object_setattr as *const (),
    );
    // `lookup_exc_class_for_kind` reads the TLS `EXC_CLASS_BY_KIND`
    // registry the tracer cannot model; its residual call rides a C-ABI
    // bridge that reconstructs the `ExcKind` from the integer arg slot.
    push_alias_pair(
        &mut entries,
        "pyre_object::excobject::lookup_exc_class_for_kind",
        "pyre_object::lookup_exc_class_for_kind",
        crate::opcode_ops::bh_lookup_exc_class_for_kind as *const (),
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
        "pyre_object::strobject::jit_int_str",
        "pyre_object::jit_int_str",
        pyre_object::jit_int_str as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::rangeobject::jit_range_iter_new",
        "pyre_object::jit_range_iter_new",
        pyre_object::jit_range_iter_new as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::pyobject::ensure_object_subclass_ranges_initialized",
        "pyre_object::ensure_object_subclass_ranges_initialized",
        pyre_object::pyobject::ensure_object_subclass_ranges_initialized as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_hook::try_gc_write_barrier",
        "pyre_object::try_gc_write_barrier",
        pyre_object::gc_hook::try_gc_write_barrier as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dict_eq_hook::has_hash_w_hook",
        "pyre_object::has_hash_w_hook",
        pyre_object::dict_eq_hook::has_hash_w_hook as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dict_eq_hook::hash_w_hooked",
        "pyre_object::hash_w_hooked",
        pyre_object::dict_eq_hook::hash_w_hooked as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dict_eq_hook::has_compares_by_identity_hook",
        "pyre_object::has_compares_by_identity_hook",
        pyre_object::dict_eq_hook::has_compares_by_identity_hook as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dict_eq_hook::compares_by_identity_hooked",
        "pyre_object::compares_by_identity_hooked",
        pyre_object::dict_eq_hook::compares_by_identity_hooked as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dict_eq_hook::signal_hash_error",
        "pyre_object::signal_hash_error",
        pyre_object::dict_eq_hook::signal_hash_error as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dict_eq_hook::take_hash_error",
        "pyre_object::take_hash_error",
        pyre_object::dict_eq_hook::take_hash_error as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dict_eq_hook::signal_eq_error",
        "pyre_object::signal_eq_error",
        pyre_object::dict_eq_hook::signal_eq_error as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dict_eq_hook::take_eq_error",
        "pyre_object::take_eq_error",
        pyre_object::dict_eq_hook::take_eq_error as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::stack_check::stack_almost_full",
        "pyre_interpreter::stack_almost_full",
        crate::stack_check::stack_almost_full as *const (),
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
    // `pop_value` sub-jitcode.  Same dual-shape binding as
    // `PyFrame::pop` below: the bare `self.nlocals()` spelling inside
    // the MIR-lowered `pop_value` graph resolves through
    // `impl_method_owner` to the 2-segment `["PyFrame", "nlocals"]`,
    // while the module-qualified form is the 3-segment
    // `["pyframe", "PyFrame", "nlocals"]` — register both.
    let pyframe_nlocals: fn(&crate::pyframe::PyFrame) -> usize = crate::pyframe::PyFrame::nlocals;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyframe::PyFrame::nlocals",
        "pyre_interpreter::PyFrame::nlocals",
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

    // `get_current_exception` / `set_current_exception` — the named TLS
    // accessors `PyFrame::push_exc_info` / `pop_except` (`eval.rs`) call
    // for the per-thread `CURRENT_EXCEPTION` slot.  Both carry
    // `#[dont_look_inside]` (the `LocalKey::with` closure inside has no
    // extractable graph), so the codewriter classifies the calls
    // `Residual` and needs these bindings to bake real funcptrs instead
    // of `symbolic_fnaddr_for_path` hashes.  These are the
    // interpreter-side twins of the trace-side
    // `get_current_exception_fn` / `set_current_exception_fn` cpu
    // helpers — same TLS slot, same flat read/write semantics.
    let get_current_exc: fn() -> pyre_object::PyObjectRef = crate::eval::get_current_exception;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::eval::get_current_exception",
        "pyre_interpreter::get_current_exception",
        get_current_exc as *const (),
    );
    let set_current_exc: fn(pyre_object::PyObjectRef) = crate::eval::set_current_exception;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::eval::set_current_exception",
        "pyre_interpreter::set_current_exception",
        set_current_exc as *const (),
    );

    // `pyframe_get_pycode` / `ncells` / `npure_cellvars` / `PyFrame::ncells`
    // carry `#[elidable_cannot_raise]`.  `call.rs:has_cannot_raise_assertion`
    // only honours the assertion when `function_fnaddrs.contains_key(p)`,
    // so without a registration the descr falls back to
    // `EF_ELIDABLE_CAN_RAISE`.
    //
    // These free functions are also called unqualified inside `pyframe.rs`
    // itself (`pyframe_get_pycode(self)` / `ncells(code)` / `npure_cellvars(code)`).
    // `target_to_path` for a `FunctionPath` returns the segments verbatim,
    // so an in-module bare call resolves to a 1-segment CallPath
    // `["<name>"]` while a cross-module qualified call resolves to
    // `["pyframe", "<name>"]`.  Use `push_alias_pair` to register both
    // shapes via the strip-one-segment rule in
    // `register_macro_helper_trace_fnaddr`.
    let pyframe_get_pycode_fn: unsafe fn(&crate::pyframe::PyFrame) -> *const crate::CodeObject =
        crate::pyframe::pyframe_get_pycode;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyframe::pyframe_get_pycode",
        "pyre_interpreter::pyframe_get_pycode",
        pyframe_get_pycode_fn as *const (),
    );

    let pyframe_ncells_free: fn(&crate::CodeObject) -> usize = crate::pyframe::ncells;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyframe::ncells",
        "pyre_interpreter::ncells",
        pyframe_ncells_free as *const (),
    );

    let pyframe_npure_cellvars: fn(&crate::CodeObject) -> usize = crate::pyframe::npure_cellvars;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyframe::npure_cellvars",
        "pyre_interpreter::npure_cellvars",
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

    // Paired-local index decode helpers for the LoadFastLoadFast /
    // StoreFastLoadFast / StoreFastStoreFast /
    // LoadFastBorrowLoadFastBorrow arms — same `push_alias_pair`
    // rationale as `load_fast_var_num_to_index` above.
    let var_nums_to_first_index: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::VarNums>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::var_nums_to_first_index;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::var_nums_to_first_index",
        "pyre_interpreter::var_nums_to_first_index",
        var_nums_to_first_index as *const (),
    );

    let var_nums_to_second_index: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::VarNums>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::var_nums_to_second_index;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::var_nums_to_second_index",
        "pyre_interpreter::var_nums_to_second_index",
        var_nums_to_second_index as *const (),
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
        pytype_addr!("cellobject::CELL_TYPE", cellobject::CELL_TYPE),
        pytype_addr!("sliceobject::SLICE_TYPE", sliceobject::SLICE_TYPE),
        pytype_addr!("rangeobject::RANGE_TYPE", rangeobject::RANGE_TYPE),
        pytype_addr!("rangeobject::RANGE_ITER_TYPE", rangeobject::RANGE_ITER_TYPE),
        pytype_addr!("rangeobject::SEQ_ITER_TYPE", rangeobject::SEQ_ITER_TYPE),
        pytype_addr!("methodobject::METHOD_TYPE", methodobject::METHOD_TYPE),
        pytype_addr!("memberobject::MEMBER_TYPE", memberobject::MEMBER_TYPE),
        pytype_addr!(
            "propertyobject::PROPERTY_TYPE",
            propertyobject::PROPERTY_TYPE
        ),
        pytype_addr!(
            "propertyobject::STATICMETHOD_TYPE",
            propertyobject::STATICMETHOD_TYPE
        ),
        pytype_addr!(
            "propertyobject::CLASSMETHOD_TYPE",
            propertyobject::CLASSMETHOD_TYPE
        ),
        pytype_addr!(
            "getsetproperty::GETSET_DESCRIPTOR_TYPE",
            getsetproperty::GETSET_DESCRIPTOR_TYPE
        ),
        pytype_addr!(
            "enumerateobject::ENUMERATE_TYPE",
            enumerateobject::ENUMERATE_TYPE
        ),
        pytype_addr!(
            "reversedobject::REVERSED_TYPE",
            reversedobject::REVERSED_TYPE
        ),
        pytype_addr!("filterobject::FILTER_TYPE", filterobject::FILTER_TYPE),
        pytype_addr!("mapobject::MAP_TYPE", mapobject::MAP_TYPE),
        pytype_addr!("zipobject::ZIP_TYPE", zipobject::ZIP_TYPE),
        pytype_addr!(
            "callableiteratorobject::CALLABLE_ITERATOR_TYPE",
            callableiteratorobject::CALLABLE_ITERATOR_TYPE
        ),
        pytype_addr!("itertoolsmodule::COUNT_TYPE", itertoolsmodule::COUNT_TYPE),
        pytype_addr!("itertoolsmodule::REPEAT_TYPE", itertoolsmodule::REPEAT_TYPE),
        pytype_addr!(
            "itertoolsmodule::TAKEWHILE_TYPE",
            itertoolsmodule::TAKEWHILE_TYPE
        ),
        pytype_addr!(
            "itertoolsmodule::DROPWHILE_TYPE",
            itertoolsmodule::DROPWHILE_TYPE
        ),
        pytype_addr!(
            "itertoolsmodule::FILTERFALSE_TYPE",
            itertoolsmodule::FILTERFALSE_TYPE
        ),
        pytype_addr!(
            "itertoolsmodule::PAIRWISE_TYPE",
            itertoolsmodule::PAIRWISE_TYPE
        ),
        pytype_addr!("sreobject::SRE_SCANNER_TYPE", sreobject::SRE_SCANNER_TYPE),
        pytype_addr!(
            "rangeobject::LONG_RANGE_ITER_TYPE",
            rangeobject::LONG_RANGE_ITER_TYPE
        ),
        pytype_addr!("sreobject::SRE_MATCH_TYPE", sreobject::SRE_MATCH_TYPE),
        pytype_addr!("sreobject::SRE_PATTERN_TYPE", sreobject::SRE_PATTERN_TYPE),
        pytype_addr!(
            "genericaliasobject::GENERIC_ALIAS_TYPE",
            genericaliasobject::GENERIC_ALIAS_TYPE
        ),
        pytype_addr!("superobject::SUPER_TYPE", superobject::SUPER_TYPE),
        pytype_addr!("unionobject::UNION_TYPE", unionobject::UNION_TYPE),
        // `pyre_interpreter`-local `PyType` singletons.  The `pytype_addr!`
        // macro emits `&pyre_object::$path` and cannot reach these
        // crate-local statics, so capture their addresses directly.  The
        // keys match the front-end `["pyre_interpreter", module, NAME]`
        // global-read segments via the `static_key_matches` `::`-suffix
        // rule, so the `module::NAME` form suffices.  All five are
        // compile-time `static … : PyType = new_pytype(…)` so the captured
        // address is the stable runtime identity.
        (
            "function::FUNCTION_TYPE",
            &crate::function::FUNCTION_TYPE as *const _ as i64,
        ),
        (
            "function::BUILTIN_FUNCTION_TYPE",
            &crate::function::BUILTIN_FUNCTION_TYPE as *const _ as i64,
        ),
        (
            "gateway::BUILTIN_CODE_TYPE",
            &crate::gateway::BUILTIN_CODE_TYPE as *const _ as i64,
        ),
        (
            "pycode::CODE_TYPE",
            &crate::pycode::CODE_TYPE as *const _ as i64,
        ),
        (
            "pytraceback::PYTRACEBACK_TYPE",
            &crate::pytraceback::PYTRACEBACK_TYPE as *const _ as i64,
        ),
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

/// Build-time *values* of the immutable size constants pyre source reads
/// through the flowgraph as opaque `LOAD_GLOBAL` constants.  Unlike the
/// `refs`/`pytypes` siblings (which carry a static's *address*), these are
/// compile-time `const`s whose initializer is a `size_of::<T>()` the
/// front-end cannot evaluate (Charon leaves the target-dependent layout
/// symbolic).  The value is identical at the codewriter call site, so the
/// front-end bakes it directly as a `ConstInt` instead of minting an
/// accessor call no registry can resolve.
///
/// Resolved in the same build-script process the translator runs in, so
/// the captured size matches a direct `size_of::<T>()` at the call site
/// (the JIT is native — host target == runtime target).  Keys are the
/// crate-stripped `module::NAME` spelling `front::mir::static_int_value_op`
/// matches against the `FunctionPath` segments.
pub fn jit_static_int_values() -> Vec<(&'static str, i64)> {
    vec![
        (
            "function::FUNCTION_OBJECT_SIZE",
            crate::function::FUNCTION_OBJECT_SIZE as i64,
        ),
        (
            "dictmultiobject::W_DICT_OBJECT_SIZE",
            pyre_object::dictmultiobject::W_DICT_OBJECT_SIZE as i64,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::{is_pyframe_operand_stack_accessor, jit_trace_fnaddrs};
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

    /// These path spellings are what keep the `pop_value` / paired-local
    /// / exception-TLS residual calls off the `symbolic_fnaddr_for_path`
    /// fallback (which SEGVs at trace time); a typo in either the
    /// module-qualified or root alias would silently regress to a
    /// symbolic hash, so pin both spellings against the live fnaddr.
    #[test]
    fn jit_trace_fnaddrs_covers_pop_value_and_exception_tls_helpers() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let nlocals: fn(&crate::pyframe::PyFrame) -> usize = crate::pyframe::PyFrame::nlocals;
        let nlocals = nlocals as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::pyframe::PyFrame::nlocals"],
            nlocals
        );
        assert_eq!(bindings["pyre_interpreter::PyFrame::nlocals"], nlocals);

        let get_exc: fn() -> pyre_object::PyObjectRef = crate::eval::get_current_exception;
        let get_exc = get_exc as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::eval::get_current_exception"],
            get_exc
        );
        assert_eq!(bindings["pyre_interpreter::get_current_exception"], get_exc);

        let set_exc: fn(pyre_object::PyObjectRef) = crate::eval::set_current_exception;
        let set_exc = set_exc as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::eval::set_current_exception"],
            set_exc
        );
        assert_eq!(bindings["pyre_interpreter::set_current_exception"], set_exc);

        let first: fn(
            crate::bytecode::Arg<crate::bytecode::oparg::VarNums>,
            crate::bytecode::OpArg,
        ) -> usize = crate::pyopcode::var_nums_to_first_index;
        let first = first as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::pyopcode::var_nums_to_first_index"],
            first
        );
        assert_eq!(bindings["pyre_interpreter::var_nums_to_first_index"], first);

        let second: fn(
            crate::bytecode::Arg<crate::bytecode::oparg::VarNums>,
            crate::bytecode::OpArg,
        ) -> usize = crate::pyopcode::var_nums_to_second_index;
        let second = second as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::pyopcode::var_nums_to_second_index"],
            second
        );
        assert_eq!(
            bindings["pyre_interpreter::var_nums_to_second_index"],
            second
        );
    }

    /// `is_pyframe_operand_stack_accessor` must recognise the funcptr the
    /// codewriter bakes for `PyFrame::pop` — the `pop_value` sub-jitcode
    /// residual the full-body walk must not concretely execute against the
    /// paused outer frame — and must NOT flag `PyFrame::nlocals`, a registered
    /// `PyFrame` method that is a constant read, safe to fold during a walk.
    #[test]
    fn is_pyframe_operand_stack_accessor_matches_registered_pop() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        let pop = bindings["pyre_interpreter::pyframe::PyFrame::pop"];
        assert!(is_pyframe_operand_stack_accessor(pop as usize));
        let nlocals = bindings["pyre_interpreter::pyframe::PyFrame::nlocals"];
        assert!(!is_pyframe_operand_stack_accessor(nlocals as usize));
        assert!(!is_pyframe_operand_stack_accessor(0));
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

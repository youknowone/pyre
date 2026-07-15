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
        "pyre_interpreter::runtime_ops::jit_next",
        "pyre_interpreter::jit_next",
        crate::runtime_ops::jit_next as *const (),
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
        "pyre_object::gc_roots::shadow_stack_get",
        "pyre_object::shadow_stack_get",
        pyre_object::gc_roots::shadow_stack_get as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::typeobject::w_type_set_uses_object_setattr",
        "pyre_object::w_type_set_uses_object_setattr",
        crate::opcode_ops::bh_w_type_set_uses_object_setattr as *const (),
    );
    // `w_type_issubtype` is the MRO membership scan (`_issubtype`,
    // typeobject.py:1640), run under the JIT inside `_pure_issubtype`
    // (`@elidable_promote`, typeobject.py:1657).  Its `#[dont_look_inside]`
    // residualises the call; bind the `-> bool` Rust `fn` directly by
    // qualified path (2-pointer args, JIT-representable, no C-ABI bridge).
    let w_type_issubtype: unsafe fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) -> bool =
        pyre_object::w_type_issubtype;
    push_alias_pair(
        &mut entries,
        "pyre_object::typeobject::w_type_issubtype",
        "pyre_object::w_type_issubtype",
        w_type_issubtype as *const (),
    );
    // `lookup_exc_class_for_kind` reads the TLS `EXC_CLASS_BY_KIND`
    // registry the tracer cannot model; its residual call rides a C-ABI
    // bridge that reconstructs the `ExcKind` from the integer arg slot.
    push_alias_pair(
        &mut entries,
        "pyre_object::interp_exceptions::lookup_exc_class_for_kind",
        "pyre_object::lookup_exc_class_for_kind",
        crate::opcode_ops::bh_lookup_exc_class_for_kind as *const (),
    );
    // `pin_root` pushes onto the TLS `SHADOW_STACK` (the `shadow_stack_len`
    // twin), `dereference` reads the weakref `w_obj_weak` slot
    // (`@jit.dont_look_inside` upstream, the `proxy_type` twin), and
    // `_obj_setdict` writes the per-instance `INSTANCE_DICT` side table —
    // all through closures the tracer cannot model.  Their `#[dont_look_inside]`
    // calls bind the Rust `fn` directly by qualified path (pointer / `-> ()`
    // / `-> Result<(), PyError>` signatures are JIT-representable).
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_roots::pin_root",
        "pyre_object::pin_root",
        pyre_object::gc_roots::pin_root as *const (),
    );
    // `mark_prebuilt_roots_dirty` sets the static `PREBUILT_ROOTS_DIRTY`
    // bit, `box_str_constant` reads the TLS `STRING_CONSTANT_CACHE`, and
    // `try_gc_add_root` dispatches the TLS `GC_ADD_ROOT_HOOK` — all through
    // state the tracer cannot model (the `pin_root` / `try_gc_write_barrier`
    // twins).  Their `#[dont_look_inside]` calls bind the Rust `fn` directly
    // by qualified path (`-> ()` / pointer / `-> bool` signatures are
    // JIT-representable).
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_roots::mark_prebuilt_roots_dirty",
        "pyre_object::mark_prebuilt_roots_dirty",
        pyre_object::gc_roots::mark_prebuilt_roots_dirty as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::unicodeobject::box_str_constant",
        "pyre_object::box_str_constant",
        pyre_object::unicodeobject::box_str_constant as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::unicodeobject::w_str_new",
        "pyre_object::w_str_new",
        pyre_object::unicodeobject::w_str_new as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::longobject::w_long_new",
        "pyre_object::w_long_new",
        pyre_object::longobject::w_long_new as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_hook::try_gc_add_root",
        "pyre_object::try_gc_add_root",
        pyre_object::gc_hook::try_gc_add_root as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_hook::try_gc_remove_root",
        "pyre_object::try_gc_remove_root",
        pyre_object::gc_hook::try_gc_remove_root as *const (),
    );
    // #346: four direct `malloc_typed` (`NewWithVtable`) roots residualised
    // via `#[dont_look_inside]`; each binds both the qualified module path and
    // the glob-re-exported root alias. `function_new_impl` lives in this crate
    // so it binds through `crate::`.
    push_alias_pair(
        &mut entries,
        "pyre_object::bytesobject::w_bytes_from_bytes",
        "pyre_object::w_bytes_from_bytes",
        pyre_object::bytesobject::w_bytes_from_bytes as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::alloc_dict_object",
        "pyre_object::alloc_dict_object",
        pyre_object::dictmultiobject::alloc_dict_object as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_len",
        "pyre_object::w_dict_len",
        pyre_object::dictmultiobject::w_dict_len as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_setitem_str",
        "pyre_object::w_dict_setitem_str",
        pyre_object::dictmultiobject::w_dict_setitem_str as *const (),
    );
    // The typed int/bytes dict-storage leaves residualise their
    // `IndexMap::{insert,get}` (an external-crate heap store/lookup the tracer
    // cannot model): the stores return `()`, the lookups `Option<PyObjectRef>`.
    push_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_store_int_strategy",
        "pyre_object::w_dict_store_int_strategy",
        pyre_object::dictmultiobject::w_dict_store_int_strategy as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_lookup_int_strategy",
        "pyre_object::w_dict_lookup_int_strategy",
        pyre_object::dictmultiobject::w_dict_lookup_int_strategy as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_store_bytes_strategy",
        "pyre_object::w_dict_store_bytes_strategy",
        pyre_object::dictmultiobject::w_dict_store_bytes_strategy as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_lookup_bytes_strategy",
        "pyre_object::w_dict_lookup_bytes_strategy",
        pyre_object::dictmultiobject::w_dict_lookup_bytes_strategy as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::w_module_dict_new",
        "pyre_object::w_module_dict_new",
        pyre_object::dictmultiobject::w_module_dict_new as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::module::w_module_new_aliasing_dict",
        "pyre_object::w_module_new_aliasing_dict",
        pyre_object::module::w_module_new_aliasing_dict as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::function::function_new_impl",
        "pyre_interpreter::function_new_impl",
        crate::function::function_new_impl as *const (),
    );
    // #346: null-collapsing stable-alloc primitive residualised via
    // `#[dont_look_inside]`, keeping the thread-local GC hook dispatch out of
    // the trace.
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_hook::try_gc_alloc_stable_raw",
        "pyre_object::try_gc_alloc_stable_raw",
        pyre_object::gc_hook::try_gc_alloc_stable_raw as *const (),
    );
    // The interp-alloc boxing tail's two GC-hook toucher residuals: `note_alloc`
    // bumps the runtime-mutable `ALLOC_SINCE_GC` atomic, and
    // `try_gc_charge_oldgen_external` dispatches through a thread-local `Cell`.
    // Neither is a build-time constant, so both carry `#[dont_look_inside]` and
    // bind their `()`-returning `fn` directly by qualified path (siblings of
    // `try_gc_alloc_stable_raw` / `gc_interp::enabled`).
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_interp::note_alloc",
        "pyre_object::note_alloc",
        pyre_object::gc_interp::note_alloc as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_hook::try_gc_charge_oldgen_external",
        "pyre_object::try_gc_charge_oldgen_external",
        pyre_object::gc_hook::try_gc_charge_oldgen_external as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::module::_weakref::interp__weakref::dereference",
        crate::module::_weakref::interp__weakref::dereference as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::std::mapdict::_obj_setdict",
        crate::objspace::std::mapdict::_obj_setdict as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::std::mapdict::_obj_getdict",
        crate::objspace::std::mapdict::_obj_getdict as *const (),
    );
    // #346: the C3 linearization core `compute_mro` carries `#[dont_look_inside]`
    // — MRO computation is opaque, MRO iteration stays traced. The self-recursive
    // C3 walk bottoms out in the `vec![w_type]` foreign alloc intrinsic, so it
    // residualizes; its public wrapper `compute_default_mro` residualizes with
    // it. Both are Vec-returning residuals with no build-time constant, so bind
    // their `fn` directly by qualified path.
    let compute_mro: unsafe fn(pyre_object::PyObjectRef) -> Vec<pyre_object::PyObjectRef> =
        crate::baseobjspace::compute_mro;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::baseobjspace::compute_mro",
        "pyre_interpreter::compute_mro",
        compute_mro as *const (),
    );
    let compute_default_mro: unsafe fn(pyre_object::PyObjectRef) -> Vec<pyre_object::PyObjectRef> =
        crate::baseobjspace::compute_default_mro;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::baseobjspace::compute_default_mro",
        "pyre_interpreter::compute_default_mro",
        compute_default_mro as *const (),
    );
    // #346: `memoryview_gather_bytes` is the sole `.gather()` call surface —
    // the buffer-protocol copy leaf whose geometry walk + `Vec<u8>` growth
    // + `Range`-indexed sub-slices are opaque host plumbing. Residualized
    // (`#[dont_look_inside]`), it is a `Vec`-returning residual like
    // `compute_mro`; bind its `fn` directly by qualified path.
    let memoryview_gather_bytes: unsafe fn(pyre_object::PyObjectRef) -> Vec<u8> =
        crate::builtins::memoryview_gather_bytes;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::builtins::memoryview_gather_bytes",
        "pyre_interpreter::memoryview_gather_bytes",
        memoryview_gather_bytes as *const (),
    );
    // #346: `lookup_in_type_where_uncached` is the scalar-`Option` residual
    // boundary over the cold uncached MRO walk. With `compute_mro` opaque,
    // `lookup_where`'s `mro` phi-merges the cached slice against the opaque
    // `Vec` borrow (`<other> ∪ _ptr`), a union the annotator cannot model, so
    // this projection carries `#[dont_look_inside]`. No build-time constant, so
    // bind its `fn` directly by qualified path.
    let lookup_in_type_where_uncached: unsafe fn(
        pyre_object::PyObjectRef,
        &str,
    ) -> Option<pyre_object::PyObjectRef> = crate::baseobjspace::lookup_in_type_where_uncached;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::baseobjspace::lookup_in_type_where_uncached",
        "pyre_interpreter::lookup_in_type_where_uncached",
        lookup_in_type_where_uncached as *const (),
    );
    // `gc_interp::enabled` reads (and lazily inits) the `STATE` atomic and
    // `longobject::bigint_gc_type_id` reads the init-assigned `BIGINT_GC_TYPE_ID`
    // atomic — neither is a build-time constant, so both carry
    // `#[dont_look_inside]` and bind their `-> bool` / `-> u32` Rust `fn`
    // directly by qualified path.
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_interp::enabled",
        "pyre_object::enabled",
        pyre_object::gc_interp::enabled as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::longobject::bigint_gc_type_id",
        "pyre_object::bigint_gc_type_id",
        pyre_object::longobject::bigint_gc_type_id as *const (),
    );
    // The dispatch-loop safepoint's four toucher residuals plus the frame-entry
    // odometer bump and the items-block strategy gate: each reads a
    // runtime-mutable global (`COLLECT_STATE` / `EVAL_NESTING` atomics, the two
    // GC hook fn-pointer cells, `FRAME_ENTRY_COUNT` TLS, the `PYRE_GC_ITEMSBLOCK`
    // `OnceLock`) — none a build-time constant — so all carry
    // `#[dont_look_inside]` and bind their `-> bool` / `()` Rust `fn` directly by
    // qualified path (siblings of `gc_interp::enabled` / `note_alloc`).
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_interp::collect_enabled",
        "pyre_object::collect_enabled",
        pyre_object::gc_interp::collect_enabled as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_interp::at_outermost_activation",
        "pyre_object::at_outermost_activation",
        pyre_object::gc_interp::at_outermost_activation as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_hook::try_gc_collect_oldgen",
        "pyre_object::try_gc_collect_oldgen",
        pyre_object::gc_hook::try_gc_collect_oldgen as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_hook::try_gc_jitframe_empty",
        "pyre_object::try_gc_jitframe_empty",
        pyre_object::gc_hook::try_gc_jitframe_empty as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::object_array::itemsblock_gc_enabled",
        "pyre_object::itemsblock_gc_enabled",
        pyre_object::object_array::itemsblock_gc_enabled as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::call::bump_frame_entry_count",
        crate::call::bump_frame_entry_count as *const (),
    );
    // The dispatch-loop safepoint entry itself reads `ALLOC_SINCE_GC` inline and
    // dispatches to the collection hook.
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_interp::safepoint",
        "pyre_object::safepoint",
        pyre_object::gc_interp::safepoint as *const (),
    );
    // `jit_bigint_div` / `jit_bigint_rem` residualize the `div_rem()` tuple
    // synth (`front::bigint_div_rem`): the foreign malachite `div_rem` returns
    // a `(BigInt, BigInt)` the tracer models as a `__pos_0`/`__pos_1` tuple
    // sourced from these two `#[dont_look_inside]` calls, bound by path.
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_div",
        crate::objspace::descroperation::jit_bigint_div as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_rem",
        crate::objspace::descroperation::jit_bigint_rem as *const (),
    );
    // `jit_bigint_div_floor` / `jit_bigint_mod_floor` residualize the
    // `div_mod_floor()` tuple synth (`front::bigint_div_mod_floor`): the foreign
    // malachite `div_mod_floor` returns a floored `(BigInt, BigInt)` the tracer
    // models as a `__pos_0`/`__pos_1` tuple sourced from these two
    // `#[dont_look_inside]` calls, bound by path.
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_div_floor",
        crate::objspace::descroperation::jit_bigint_div_floor as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_mod_floor",
        crate::objspace::descroperation::jit_bigint_mod_floor as *const (),
    );
    // `jit_bigint_{and,or,xor,sub,mul}` residualize the foreign BigInt binary
    // operators (`<BigInt as BitAnd>::bitand`, …) the `front::mir` retarget
    // (`front::bigint_binop`) redirects when both operands are the opaque
    // `BigInt` ADT.  Each returns a fresh `*mut BigInt` (as i64), bound by path.
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_and",
        crate::objspace::descroperation::jit_bigint_and as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_or",
        crate::objspace::descroperation::jit_bigint_or as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_xor",
        crate::objspace::descroperation::jit_bigint_xor as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_sub",
        crate::objspace::descroperation::jit_bigint_sub as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_mul",
        crate::objspace::descroperation::jit_bigint_mul as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_add",
        crate::objspace::descroperation::jit_bigint_add as *const (),
    );
    // `jit_bigint_neg` residualizes the unary `<BigInt as Neg>::neg` operator;
    // a single operand pointer.
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_neg",
        crate::objspace::descroperation::jit_bigint_neg as *const (),
    );
    // `jit_bigint_{shl,shr}` residualize the BigInt shift-by-`usize` operators
    // (`<BigInt as Shl<usize>>::shl`, …); `b` is the machine shift count.
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_shl",
        crate::objspace::descroperation::jit_bigint_shl as *const (),
    );
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_shr",
        crate::objspace::descroperation::jit_bigint_shr as *const (),
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
        "pyre_object::longobject::jit_bigint_to_i64_fits",
        "pyre_object::jit_bigint_to_i64_fits",
        pyre_object::jit_bigint_to_i64_fits as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_i64_value",
        "pyre_object::jit_bigint_to_i64_value",
        pyre_object::jit_bigint_to_i64_value as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_u64_fits",
        "pyre_object::jit_bigint_to_u64_fits",
        pyre_object::jit_bigint_to_u64_fits as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_u64_value",
        "pyre_object::jit_bigint_to_u64_value",
        pyre_object::jit_bigint_to_u64_value as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::longobject::jit_bigint_sign_i64",
        "pyre_object::jit_bigint_sign_i64",
        pyre_object::jit_bigint_sign_i64 as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_f64_or_inf",
        "pyre_object::jit_bigint_to_f64_or_inf",
        pyre_object::jit_bigint_to_f64_or_inf as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_f64_or_nan",
        "pyre_object::jit_bigint_to_f64_or_nan",
        pyre_object::jit_bigint_to_f64_or_nan as *const (),
    );
    // The #171 object-append fold descends `w_list_append` and folds the
    // store leaves to native ops, leaving `list_write_barrier(l)` as a
    // residual call (the off-GC ItemsBlock is reached by the collector only
    // through the remembered W_ListObject). Register it so the codewriter
    // resolves the residual to a runtime-patchable address instead of a
    // `symbolic_fnaddr_for_path` hash the inline sub-walk must decline.
    push_alias_pair(
        &mut entries,
        "pyre_object::listobject::list_write_barrier",
        "pyre_object::list_write_barrier",
        pyre_object::list_write_barrier as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::tupleobject::jit_tuple_getitem",
        "pyre_object::jit_tuple_getitem",
        pyre_object::jit_tuple_getitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::unicodeobject::jit_str_concat",
        "pyre_object::jit_str_concat",
        pyre_object::jit_str_concat as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::unicodeobject::jit_str_repeat",
        "pyre_object::jit_str_repeat",
        pyre_object::jit_str_repeat as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::unicodeobject::jit_str_compare",
        "pyre_object::jit_str_compare",
        pyre_object::jit_str_compare as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::unicodeobject::jit_str_is_true",
        "pyre_object::jit_str_is_true",
        pyre_object::jit_str_is_true as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::unicodeobject::jit_int_str",
        "pyre_object::jit_int_str",
        pyre_object::jit_int_str as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::functional::jit_range_iter_new",
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
        "pyre_object::gc_hook::try_gc_owns_object",
        "pyre_object::try_gc_owns_object",
        pyre_object::gc_hook::try_gc_owns_object as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::gc_hook::maybe_register_finalizer",
        "pyre_object::maybe_register_finalizer",
        pyre_object::gc_hook::maybe_register_finalizer as *const (),
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
        "pyre_object::dict_eq_hook::has_hash_str_hook",
        "pyre_object::has_hash_str_hook",
        pyre_object::dict_eq_hook::has_hash_str_hook as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::dict_eq_hook::hash_str_hooked",
        "pyre_object::hash_str_hooked",
        pyre_object::dict_eq_hook::hash_str_hooked as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_float_abs",
        "pyre_interpreter::jit_float_abs",
        crate::objspace::descroperation::jit_float_abs as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::call::pyre_debug_call_enabled",
        "pyre_interpreter::pyre_debug_call_enabled",
        crate::call::pyre_debug_call_enabled as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::executioncontext::execution_context_builtin_cache_get",
        "pyre_interpreter::execution_context_builtin_cache_get",
        crate::executioncontext::execution_context_builtin_cache_get as *const (),
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
        "pyre_object::dict_eq_hook::eq_error_pending",
        "pyre_object::eq_error_pending",
        pyre_object::dict_eq_hook::eq_error_pending as *const (),
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

    // `w_type` / `w_object` — the `type` / `object` typeobject accessors
    // read the `W_TYPE_TYPEOBJECT` / `W_OBJECT_TYPEOBJECT` `OnceLock<usize>`
    // slots set once at startup.  Both carry `#[dont_look_inside]` (the
    // `OnceLock::get` read has no registry-resolvable accessor graph), so
    // the codewriter classifies the calls `Residual` and needs these
    // bindings to bake real funcptrs instead of `symbolic_fnaddr_for_path`
    // hashes.  Callers spell them `crate::typedef::w_type()`, the sole
    // path form, with no crate-root re-export.
    let w_type: fn() -> pyre_object::PyObjectRef = crate::typedef::w_type;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::typedef::w_type",
        w_type as *const (),
    );
    let w_object: fn() -> pyre_object::PyObjectRef = crate::typedef::w_object;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::typedef::w_object",
        w_object as *const (),
    );

    // Thread-local / `OnceLock` accessors that carry `#[dont_look_inside]`
    // (the `.with` closure read has no extractable graph): the codewriter
    // classifies the calls `Residual` and needs real funcptrs instead of
    // `symbolic_fnaddr_for_path` hashes.  Error-slot twins of
    // `get_current_exception` / `set_current_exception`, plus the weakref
    // proxy type singletons (twins of `w_type` / `w_object`).
    let set_call_error: fn(crate::PyError) = crate::call::set_call_error;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::call::set_call_error",
        set_call_error as *const (),
    );
    let take_call_error: fn() -> Option<crate::PyError> = crate::call::take_call_error;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::call::take_call_error",
        take_call_error as *const (),
    );
    let clear_call_error: fn() = crate::call::clear_call_error;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::call::clear_call_error",
        clear_call_error as *const (),
    );
    // `#[dont_look_inside]` execution-context thread-local reads
    // (`BUILD_CLASS_EXEC_CTX` / `LAST_EXEC_CTX`), twins of the call-error
    // slot accessors above; front::mir const-folds the `ThreadLocal` global
    // to None, so their bodies have no extractable graph and the call stays
    // a residual read via the registered fnaddr.
    let build_class_exec_ctx: fn() -> *const crate::PyExecutionContext =
        crate::call::build_class_exec_ctx;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::call::build_class_exec_ctx",
        build_class_exec_ctx as *const (),
    );
    let take_last_exec_ctx: fn() -> *const crate::PyExecutionContext =
        crate::call::take_last_exec_ctx;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::call::take_last_exec_ctx",
        take_last_exec_ctx as *const (),
    );
    let take_pending_hash_error: fn() -> crate::PyError =
        crate::baseobjspace::take_pending_hash_error;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::baseobjspace::take_pending_hash_error",
        take_pending_hash_error as *const (),
    );
    let proxy_type: fn() -> pyre_object::PyObjectRef =
        crate::module::_weakref::interp__weakref::proxy_type;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::module::_weakref::interp__weakref::proxy_type",
        proxy_type as *const (),
    );
    let callable_proxy_type: fn() -> pyre_object::PyObjectRef =
        crate::module::_weakref::interp__weakref::callable_proxy_type;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::module::_weakref::interp__weakref::callable_proxy_type",
        callable_proxy_type as *const (),
    );

    // Stack-overflow / JIT-pending-exception bookkeeping accessors, all
    // `#[dont_look_inside]` (PYRE_STACKTOOBIG static / TL_JIT_PENDING_EXCEPTION
    // thread-local reads with no extractable graph).  The slowpath is
    // already a C-ABI residual the backend calls directly; the wrappers
    // become residual Calls.
    let stack_slowpath: extern "C" fn(usize) -> u8 =
        crate::stack_check::pyre_stack_too_big_slowpath;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::stack_check::pyre_stack_too_big_slowpath",
        stack_slowpath as *const (),
    );
    let stack_check: fn() -> Result<(), crate::PyError> = crate::stack_check::stack_check;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::stack_check::stack_check",
        stack_check as *const (),
    );
    let drain_jit_pending: fn() -> Result<(), crate::PyError> =
        crate::stack_check::drain_jit_pending_exception;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::stack_check::drain_jit_pending_exception",
        drain_jit_pending as *const (),
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

    // Opcode oparg decode helpers for two-phase lifting. These wrap
    // RustPython's generic `Arg::get` and `CodeUnits::deref` surfaces
    // behind first-party residual calls whose return values are the
    // scalar/enum values consumed by the opcode handlers.
    let label_arg_to_usize: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::Label>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::label_arg_to_usize;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::label_arg_to_usize",
        "pyre_interpreter::label_arg_to_usize",
        label_arg_to_usize as *const (),
    );

    let jump_target_forward_decoded: fn(
        &crate::CodeObject,
        usize,
        crate::bytecode::Arg<crate::bytecode::oparg::Label>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::jump_target_forward_decoded;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::jump_target_forward_decoded",
        "pyre_interpreter::jump_target_forward_decoded",
        jump_target_forward_decoded as *const (),
    );

    let jump_target_forward_from_oparg: fn(
        &crate::CodeObject,
        usize,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::jump_target_forward_from_oparg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::jump_target_forward_from_oparg",
        "pyre_interpreter::jump_target_forward_from_oparg",
        jump_target_forward_from_oparg as *const (),
    );

    let jump_target_backward_decoded: fn(
        &crate::CodeObject,
        usize,
        crate::bytecode::Arg<crate::bytecode::oparg::Label>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::jump_target_backward_decoded;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::jump_target_backward_decoded",
        "pyre_interpreter::jump_target_backward_decoded",
        jump_target_backward_decoded as *const (),
    );

    let binary_op_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::BinaryOperator>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::BinaryOperator = crate::pyopcode::binary_op_arg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::binary_op_arg",
        "pyre_interpreter::binary_op_arg",
        binary_op_arg as *const (),
    );

    let comparison_op_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::ComparisonOperator>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::ComparisonOperator = crate::pyopcode::comparison_op_arg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::comparison_op_arg",
        "pyre_interpreter::comparison_op_arg",
        comparison_op_arg as *const (),
    );

    let invert_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::Invert>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::Invert = crate::pyopcode::invert_arg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::invert_arg",
        "pyre_interpreter::invert_arg",
        invert_arg as *const (),
    );

    let build_slice_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::BuildSliceArgCount>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::BuildSliceArgCount = crate::pyopcode::build_slice_arg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::build_slice_arg",
        "pyre_interpreter::build_slice_arg",
        build_slice_arg as *const (),
    );

    let common_constant_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::CommonConstant>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::CommonConstant = crate::pyopcode::common_constant_arg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::common_constant_arg",
        "pyre_interpreter::common_constant_arg",
        common_constant_arg as *const (),
    );

    let convert_value_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::ConvertValueOparg>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::ConvertValueOparg = crate::pyopcode::convert_value_arg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::convert_value_arg",
        "pyre_interpreter::convert_value_arg",
        convert_value_arg as *const (),
    );

    let special_method_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::SpecialMethod>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::SpecialMethod = crate::pyopcode::special_method_arg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::special_method_arg",
        "pyre_interpreter::special_method_arg",
        special_method_arg as *const (),
    );

    let make_function_flag_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::MakeFunctionFlag>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::MakeFunctionFlag = crate::pyopcode::make_function_flag_arg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::make_function_flag_arg",
        "pyre_interpreter::make_function_flag_arg",
        make_function_flag_arg as *const (),
    );

    let intrinsic_function_1_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::IntrinsicFunction1>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::IntrinsicFunction1 = crate::pyopcode::intrinsic_function_1_arg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::intrinsic_function_1_arg",
        "pyre_interpreter::intrinsic_function_1_arg",
        intrinsic_function_1_arg as *const (),
    );

    let intrinsic_function_2_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::IntrinsicFunction2>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::IntrinsicFunction2 = crate::pyopcode::intrinsic_function_2_arg;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::intrinsic_function_2_arg",
        "pyre_interpreter::intrinsic_function_2_arg",
        intrinsic_function_2_arg as *const (),
    );

    let raise_kind_arg_as_usize: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::RaiseKind>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::raise_kind_arg_as_usize;
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::raise_kind_arg_as_usize",
        "pyre_interpreter::raise_kind_arg_as_usize",
        raise_kind_arg_as_usize as *const (),
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

    // `PyError::to_exc_object` — residual exception materialization emitted by
    // the two-phase rtyper for `PyError.to_exc_object()` call sites.  This uses
    // the same impl-method CallPath shape as `type_error`, resolving to
    // `["PyError", "to_exc_object"]` after the crate segment is stripped.
    let pyerror_to_exc_object: fn(&crate::PyError) -> pyre_object::PyObjectRef =
        crate::PyError::to_exc_object;
    push_fnaddr(
        &mut entries,
        "pyre_interpreter::PyError::to_exc_object",
        pyerror_to_exc_object as *const (),
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
    // `codewriter/jtransform.rs:cast_*_to_*` (mirroring
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
    // in `codewriter/jtransform.rs`.  Re-introduce the
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
        pytype_addr!("interp_array::ARRAY_TYPE", interp_array::ARRAY_TYPE),
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
            "dictmultiobject::DICT_KEYS_TYPE",
            dictmultiobject::DICT_KEYS_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_VALUES_TYPE",
            dictmultiobject::DICT_VALUES_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_ITEMS_TYPE",
            dictmultiobject::DICT_ITEMS_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_KEYITERATOR_TYPE",
            dictmultiobject::DICT_KEYITERATOR_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_VALUEITERATOR_TYPE",
            dictmultiobject::DICT_VALUEITERATOR_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_ITEMITERATOR_TYPE",
            dictmultiobject::DICT_ITEMITERATOR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXCEPTION_TYPE",
            interp_exceptions::EXCEPTION_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_EXCEPTION_TYPE",
            interp_exceptions::EXC_EXCEPTION_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE",
            interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_OVERFLOW_ERROR_TYPE",
            interp_exceptions::EXC_OVERFLOW_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_ZERO_DIVISION_ERROR_TYPE",
            interp_exceptions::EXC_ZERO_DIVISION_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_TYPE_ERROR_TYPE",
            interp_exceptions::EXC_TYPE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_VALUE_ERROR_TYPE",
            interp_exceptions::EXC_VALUE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_NAME_ERROR_TYPE",
            interp_exceptions::EXC_NAME_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_INDEX_ERROR_TYPE",
            interp_exceptions::EXC_INDEX_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_KEY_ERROR_TYPE",
            interp_exceptions::EXC_KEY_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_ATTRIBUTE_ERROR_TYPE",
            interp_exceptions::EXC_ATTRIBUTE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_RUNTIME_ERROR_TYPE",
            interp_exceptions::EXC_RUNTIME_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_STOP_ITERATION_TYPE",
            interp_exceptions::EXC_STOP_ITERATION_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_IMPORT_ERROR_TYPE",
            interp_exceptions::EXC_IMPORT_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_NOT_IMPLEMENTED_ERROR_TYPE",
            interp_exceptions::EXC_NOT_IMPLEMENTED_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_ASSERTION_ERROR_TYPE",
            interp_exceptions::EXC_ASSERTION_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_REFERENCE_ERROR_TYPE",
            interp_exceptions::EXC_REFERENCE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_GENERATOR_EXIT_TYPE",
            interp_exceptions::EXC_GENERATOR_EXIT_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_RECURSION_ERROR_TYPE",
            interp_exceptions::EXC_RECURSION_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_OS_ERROR_TYPE",
            interp_exceptions::EXC_OS_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_FILE_NOT_FOUND_ERROR_TYPE",
            interp_exceptions::EXC_FILE_NOT_FOUND_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_UNICODE_DECODE_ERROR_TYPE",
            interp_exceptions::EXC_UNICODE_DECODE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_UNICODE_ENCODE_ERROR_TYPE",
            interp_exceptions::EXC_UNICODE_ENCODE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_UNICODE_TRANSLATE_ERROR_TYPE",
            interp_exceptions::EXC_UNICODE_TRANSLATE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_SYSTEM_EXIT_TYPE",
            interp_exceptions::EXC_SYSTEM_EXIT_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_MEMORY_ERROR_TYPE",
            interp_exceptions::EXC_MEMORY_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_SYSTEM_ERROR_TYPE",
            interp_exceptions::EXC_SYSTEM_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_LOOKUP_ERROR_TYPE",
            interp_exceptions::EXC_LOOKUP_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_UNICODE_ERROR_TYPE",
            interp_exceptions::EXC_UNICODE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_MODULE_NOT_FOUND_ERROR_TYPE",
            interp_exceptions::EXC_MODULE_NOT_FOUND_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_SYNTAX_ERROR_TYPE",
            interp_exceptions::EXC_SYNTAX_ERROR_TYPE
        ),
        pytype_addr!("generator::GENERATOR_TYPE", generator::GENERATOR_TYPE),
        pytype_addr!("pyobject::INT_TYPE", pyobject::INT_TYPE),
        pytype_addr!("pyobject::BOOL_TYPE", pyobject::BOOL_TYPE),
        pytype_addr!("pyobject::FLOAT_TYPE", pyobject::FLOAT_TYPE),
        pytype_addr!("pyobject::COMPLEX_TYPE", pyobject::COMPLEX_TYPE),
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
        pytype_addr!("weakref::GC_WEAKREF_BOX_TYPE", weakref::GC_WEAKREF_BOX_TYPE),
        pytype_addr!("nestedscope::CELL_TYPE", nestedscope::CELL_TYPE),
        pytype_addr!("sliceobject::SLICE_TYPE", sliceobject::SLICE_TYPE),
        pytype_addr!("functional::RANGE_TYPE", functional::RANGE_TYPE),
        pytype_addr!("functional::RANGE_ITER_TYPE", functional::RANGE_ITER_TYPE),
        pytype_addr!("memoryview::MEMORYVIEW_TYPE", memoryview::MEMORYVIEW_TYPE),
        pytype_addr!("iterobject::SEQ_ITER_TYPE", iterobject::SEQ_ITER_TYPE),
        pytype_addr!("function::METHOD_TYPE", function::METHOD_TYPE),
        pytype_addr!("typedef::MEMBER_TYPE", MEMBER_TYPE),
        pytype_addr!("descriptor::PROPERTY_TYPE", descriptor::PROPERTY_TYPE),
        pytype_addr!("function::STATICMETHOD_TYPE", function::STATICMETHOD_TYPE),
        pytype_addr!("function::CLASSMETHOD_TYPE", function::CLASSMETHOD_TYPE),
        pytype_addr!("typedef::GETSET_DESCRIPTOR_TYPE", GETSET_DESCRIPTOR_TYPE),
        pytype_addr!("functional::ENUMERATE_TYPE", functional::ENUMERATE_TYPE),
        pytype_addr!("functional::REVERSED_TYPE", functional::REVERSED_TYPE),
        pytype_addr!("functional::FILTER_TYPE", functional::FILTER_TYPE),
        pytype_addr!("functional::MAP_TYPE", functional::MAP_TYPE),
        pytype_addr!("functional::ZIP_TYPE", functional::ZIP_TYPE),
        pytype_addr!(
            "operation::CALLABLE_ITERATOR_TYPE",
            operation::CALLABLE_ITERATOR_TYPE
        ),
        pytype_addr!("interp_itertools::COUNT_TYPE", interp_itertools::COUNT_TYPE),
        pytype_addr!(
            "interp_itertools::REPEAT_TYPE",
            interp_itertools::REPEAT_TYPE
        ),
        pytype_addr!(
            "interp_itertools::TAKEWHILE_TYPE",
            interp_itertools::TAKEWHILE_TYPE
        ),
        pytype_addr!(
            "interp_itertools::DROPWHILE_TYPE",
            interp_itertools::DROPWHILE_TYPE
        ),
        pytype_addr!(
            "interp_itertools::FILTERFALSE_TYPE",
            interp_itertools::FILTERFALSE_TYPE
        ),
        pytype_addr!(
            "interp_itertools::PAIRWISE_TYPE",
            interp_itertools::PAIRWISE_TYPE
        ),
        pytype_addr!("interp_itertools::CYCLE_TYPE", interp_itertools::CYCLE_TYPE),
        pytype_addr!("interp_itertools::CHAIN_TYPE", interp_itertools::CHAIN_TYPE),
        pytype_addr!("interp_sre::SRE_SCANNER_TYPE", interp_sre::SRE_SCANNER_TYPE),
        pytype_addr!(
            "functional::LONG_RANGE_ITER_TYPE",
            functional::LONG_RANGE_ITER_TYPE
        ),
        pytype_addr!("interp_sre::SRE_MATCH_TYPE", interp_sre::SRE_MATCH_TYPE),
        pytype_addr!("interp_sre::SRE_PATTERN_TYPE", interp_sre::SRE_PATTERN_TYPE),
        pytype_addr!(
            "_pypy_generic_alias::GENERIC_ALIAS_TYPE",
            _pypy_generic_alias::GENERIC_ALIAS_TYPE
        ),
        pytype_addr!("descriptor::SUPER_TYPE", descriptor::SUPER_TYPE),
        pytype_addr!(
            "_pypy_generic_alias::UNION_TYPE",
            _pypy_generic_alias::UNION_TYPE
        ),
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
        (
            "interp_buffer::PICKLEBUFFER_TYPE",
            &crate::module::__pypy__::interp_buffer::PICKLEBUFFER_TYPE as *const _ as i64,
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
            "dictmultiobject::OBJECT_DICT_STRATEGY",
            dictmultiobject::OBJECT_DICT_STRATEGY
        ),
        ref_addr!(
            "dictmultiobject::EMPTY_DICT_STRATEGY",
            dictmultiobject::EMPTY_DICT_STRATEGY
        ),
        ref_addr!(
            "dictmultiobject::EMPTY_KWARGS_DICT_STRATEGY",
            dictmultiobject::EMPTY_KWARGS_DICT_STRATEGY
        ),
        ref_addr!(
            "dictmultiobject::BYTES_DICT_STRATEGY",
            dictmultiobject::BYTES_DICT_STRATEGY
        ),
        ref_addr!(
            "dictmultiobject::UNICODE_DICT_STRATEGY",
            dictmultiobject::UNICODE_DICT_STRATEGY
        ),
        ref_addr!(
            "dictmultiobject::INT_DICT_STRATEGY",
            dictmultiobject::INT_DICT_STRATEGY
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
            "special::NOT_IMPLEMENTED_SINGLETON",
            pyre_object::w_not_implemented() as usize as i64,
        ),
        (
            "special::ELLIPSIS_SINGLETON",
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
        (
            "specialisedtupleobject::SPECIALISED_TUPLE_II_OBJECT_SIZE",
            pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_II_OBJECT_SIZE as i64,
        ),
        (
            "specialisedtupleobject::SPECIALISED_TUPLE_FF_OBJECT_SIZE",
            pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_FF_OBJECT_SIZE as i64,
        ),
        (
            "specialisedtupleobject::SPECIALISED_TUPLE_OO_OBJECT_SIZE",
            pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_OBJECT_SIZE as i64,
        ),
        (
            "objectobject::W_OBJECT_OBJECT_SIZE",
            pyre_object::objectobject::W_OBJECT_OBJECT_SIZE as i64,
        ),
        // `pub const CAN_BE_TAGGED: bool` (tagged-int scaffolding, currently
        // `false`); Charon emits the read as an opaque global rather than
        // folding it, so bake the build-time value (`false as i64` == 0).
        (
            "tagged_int::CAN_BE_TAGGED",
            pyre_object::tagged_int::CAN_BE_TAGGED as i64,
        ),
        // `i64::MAX` reached as `core::num::<Impl>::MAX` in `getindex_w`'s
        // overflow clamp. Charon leaves the associated const as a global
        // accessor path, so bake the native signed max value.
        ("core::num::<Impl>::MAX", i64::MAX),
        // `compares_by_identity_status` tri-state markers, read as opaque
        // global accessor paths in `mutated` / the `__eq__`/`__hash__`
        // fast paths. Bake the build-time `u8` values.
        (
            "typeobject::COMPARES_BY_IDENTITY_UNKNOWN",
            pyre_object::typeobject::COMPARES_BY_IDENTITY_UNKNOWN as i64,
        ),
        (
            "typeobject::COMPARES_BY_IDENTITY_YES",
            pyre_object::typeobject::COMPARES_BY_IDENTITY_YES as i64,
        ),
        (
            "typeobject::COMPARES_BY_IDENTITY_NO",
            pyre_object::typeobject::COMPARES_BY_IDENTITY_NO as i64,
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

    /// The dispatch-loop safepoint's global-state readers residualize by
    /// qualified path; a typo in either the module-qualified or root alias
    /// silently regresses the `#[dont_look_inside]` call to a symbolic hash,
    /// so pin both spellings against the live fnaddr (siblings of the
    /// `gc_interp::enabled` / `note_alloc` registrations).
    #[test]
    fn jit_trace_fnaddrs_covers_interp_gc_safepoint_readers() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let collect_enabled = pyre_object::gc_interp::collect_enabled as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::gc_interp::collect_enabled"],
            collect_enabled
        );
        assert_eq!(bindings["pyre_object::collect_enabled"], collect_enabled);

        let at_outermost =
            pyre_object::gc_interp::at_outermost_activation as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::gc_interp::at_outermost_activation"],
            at_outermost
        );
        assert_eq!(
            bindings["pyre_object::at_outermost_activation"],
            at_outermost
        );

        let collect_oldgen =
            pyre_object::gc_hook::try_gc_collect_oldgen as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::gc_hook::try_gc_collect_oldgen"],
            collect_oldgen
        );
        assert_eq!(
            bindings["pyre_object::try_gc_collect_oldgen"],
            collect_oldgen
        );

        let jitframe_empty =
            pyre_object::gc_hook::try_gc_jitframe_empty as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::gc_hook::try_gc_jitframe_empty"],
            jitframe_empty
        );
        assert_eq!(
            bindings["pyre_object::try_gc_jitframe_empty"],
            jitframe_empty
        );

        let itemsblock =
            pyre_object::object_array::itemsblock_gc_enabled as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::object_array::itemsblock_gc_enabled"],
            itemsblock
        );
        assert_eq!(bindings["pyre_object::itemsblock_gc_enabled"], itemsblock);

        let bump = crate::call::bump_frame_entry_count as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::call::bump_frame_entry_count"],
            bump
        );

        let safepoint = pyre_object::gc_interp::safepoint as *const () as usize as i64;
        assert_eq!(bindings["pyre_object::gc_interp::safepoint"], safepoint);
        assert_eq!(bindings["pyre_object::safepoint"], safepoint);
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

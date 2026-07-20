//! RPython: `rpython/jit/backend/model.py` `AbstractCPU` /
//! `rpython/jit/backend/llgraph/runner.py` `LLGraphCPU`.
//!
//! pyre-side bundle of the blackhole helper function pointers used by
//! `transform_graph_to_jitcode` to resolve `bhimpl_residual_call` and
//! the per-arity / per-opcode helpers.
//!
//! In RPython the same `cpu` object is referenced from both
//! `CodeWriter.cpu` (codewriter.py:21) and `CallControl.cpu`
//! (call.py:27); pyre owns it on `CallControl` and exposes a
//! convenience accessor `CodeWriter::cpu(&self)` so the upstream
//! attribute access pattern still works.
//!
//! Note: pyre's "cpu" is much smaller than RPython's
//! `LLGraphCPU` — there is no `calldescrof`, no `setup_descrs`, no
//! vector-extension support, and no GC integration at this layer.
//! All those concerns either live in `pyre_jit_trace::state` (descrs)
//! or are handled by the metainterp directly. This struct holds only
//! the helpers that the codewriter needs at compile time to emit the
//! correct fn-pointer indices into the `JitCode` table.

/// `rpython/jit/backend/model.py:11` `class AbstractCPU(object)`.
///
/// pyre-side: blackhole `bhimpl_*` helper trampolines, all `extern "C"`
/// so the compiled `JitCode` can call them via raw fn-pointer slots.
#[derive(Debug)]
pub struct Cpu {
    /// `bhimpl_residual_call` general entry point.
    /// `(callable, null_or_self, arg0) → result`.
    pub call_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// Per-arity `bhimpl_residual_call_<n>` helpers
    /// (`call_fn_0(callable, null_or_self)` ...
    /// `call_fn_14(callable, null_or_self, a0..a13)`).  RPython
    /// `bhimpl_residual_call_r_r` carries no frame; the parent frame is
    /// resolved from the execution context inside `bh_call_fn_impl`.  A
    /// non-null `null_or_self` is the method receiver — the helper
    /// prepends it as arg0 (eval.rs:3216-3226).  The arity ceiling is
    /// nargs=14: each helper takes `callable + null_or_self + nargs`
    /// i64s, and the backend dispatch table (`call_stub.rs::
    /// dispatch_arity_body!`, `MAX_HOST_CALL_ARITY` = 16) tops out at
    /// 16 i64 arguments.
    pub call_fn_0: extern "C" fn(i64, i64) -> i64,
    pub call_fn_2: extern "C" fn(i64, i64, i64, i64) -> i64,
    pub call_fn_3: extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub call_fn_4: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_5: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_6: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_7: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_8: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_9: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_10:
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_11:
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_12:
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_13: extern "C" fn(
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
    ) -> i64,
    pub call_fn_14: extern "C" fn(
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
    ) -> i64,
    /// `bhimpl_load_global` — namespace/code from getfield_vable_r plus live frame.
    pub load_global_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// `bh_load_from_dict_or_globals_fn(dict, code, frame, namei)` —
    /// LOAD_FROM_DICT_OR_GLOBALS: try the popped mapping then frame globals.
    pub load_from_dict_or_globals_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// `bh_call_function_ex_fn(callable, self_or_null, starargs, kwargs_or_null)`
    /// — CALL_FUNCTION_EX: unpack `*`/`**` and dispatch.
    pub call_function_ex_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// Per-arity `bh_call_kw_<n>` helpers for CALL_KW, ABI
    /// `(callable, null_or_self, kwnames, arg0..arg{n-1})` = 3 + n i64.
    /// The kwnames slot leaves room for nargs 0..=13 within the backend's
    /// `MAX_HOST_CALL_ARITY` = 16 ceiling; CALL_KW with nargs > 13 aborts.
    pub call_kw_fn_0: extern "C" fn(i64, i64, i64) -> i64,
    pub call_kw_fn_1: extern "C" fn(i64, i64, i64, i64) -> i64,
    pub call_kw_fn_2: extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub call_kw_fn_3: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64,
    pub call_kw_fn_4: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_kw_fn_5: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_kw_fn_6: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_kw_fn_7: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_kw_fn_8: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_kw_fn_9:
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_kw_fn_10:
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_kw_fn_11:
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_kw_fn_12: extern "C" fn(
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
    ) -> i64,
    pub call_kw_fn_13: extern "C" fn(
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
    ) -> i64,
    /// LOOKUP_METHOD attribute half — `(obj, code, name_idx) → attr`.
    /// Reproduces `PyFrame::load_method`'s `getattr` for blackhole resume.
    pub load_attr_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// LOOKUP_METHOD `null_or_self` half — `(obj, attr, code, name_idx) →
    /// bound`. Pure binding decision shared with the interpreter.
    pub load_method_self_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// LOAD_SPECIAL attribute half — `(obj, method_kind) → attr`.
    pub load_special_fn: extern "C" fn(i64, i64) -> i64,
    /// LOAD_SPECIAL `null_or_self` half — `(obj, attr, method_kind) → bound`.
    pub load_special_self_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// STORE_ATTR residual — `(obj, value, code, name_idx) → void`.
    /// Resolves the name from the code object and runs generic `setattr`.
    pub store_attr_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// BINARY_SLICE residual — `(obj, start, stop) → obj[start:stop]`.
    pub binary_slice_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// STORE_SLICE residual — `(obj, start, stop, value) → void`
    /// (`obj[start:stop] = value`).
    pub store_slice_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// DELETE_SUBSCR residual — `(obj, index) → void` (`del obj[index]`).
    pub delete_subscr_fn: extern "C" fn(i64, i64) -> i64,
    /// LIST_EXTEND residual — `(list, iterable) → void` (`list.extend(iterable)`,
    /// list peeked + mutated in place).
    pub list_extend_fn: extern "C" fn(i64, i64) -> i64,
    /// SET_ADD residual — `(set, value) → void` (`set.add(value)`, peeked).
    pub set_add_fn: extern "C" fn(i64, i64) -> i64,
    /// SET_UPDATE residual — `(set, iterable) → void` (`set.update`, peeked).
    pub set_update_fn: extern "C" fn(i64, i64) -> i64,
    /// DICT_UPDATE residual — `(dict, source) → void` (`dict.update`, peeked).
    pub dict_update_fn: extern "C" fn(i64, i64) -> i64,
    /// MAP_ADD residual — `(dict, key, value) → void` (`dict[key]=value`, peeked).
    pub map_add_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// DICT_MERGE residual — `(dict, source, callable) → void` (`**` merge,
    /// peeked; callable only for error-message prefixes).
    pub dict_merge_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// LIST_APPEND residual — `(list, value) → void` (`list.append(value)`,
    /// list peeked + mutated in place).  The full-body walker's #171 fold
    /// intercepts it (`PyreHelperKind::ListAppendValue`); this is the decline
    /// fallback, identical to the residual the trait tracer records.
    pub list_append_fn: extern "C" fn(i64, i64) -> i64,
    /// DELETE_ATTR residual — `(obj, code, name_idx) → void` (`del obj.name`).
    /// Resolves the name from the code object and runs generic `delattr`.
    pub delete_attr_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// FORMAT_SIMPLE residual — `value → str` (`f"{x}"`, empty spec).
    /// User `__format__` may run Python (fallible).
    pub format_simple_fn: extern "C" fn(i64) -> i64,
    /// FORMAT_WITH_SPEC residual — `(value, spec) → str` (`f"{x:.2f}"`).
    /// User `__format__` may run Python (fallible).
    pub format_with_spec_fn: extern "C" fn(i64, i64) -> i64,
    /// CONVERT_VALUE residual — `(value, conv) → str` (`f"{x!r}"`).
    /// `conv` is a `runtime_ops::convert_value_code`; user `__str__` /
    /// `__repr__` may run Python (fallible).
    pub convert_value_fn: extern "C" fn(i64, i64) -> i64,
    /// `bh_import_name_fn(fromlist, level, code, frame, name_idx)` —
    /// IMPORT_NAME `__import__` residual; resolves the module name from the
    /// code object, reads `__name__`/`__package__` for relative imports from
    /// the threaded `frame`, and imports through the TLS-pinned execution
    /// context (may run module top-level Python → fallible).
    pub import_name_fn: extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    /// `bh_import_from_fn(module, code, name_idx)` — IMPORT_FROM residual;
    /// resolves the attribute name from the code object and runs
    /// `importing::import_from` on the peeked module (namespace lookup, then a
    /// submodule-import fallback that may run module top-level Python →
    /// fallible) through the TLS-pinned execution context.
    pub import_from_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `bh_load_super_attr_fn(global_super, self, cls, code, name_idx,
    /// is_two_arg)` — LOAD_SUPER_ATTR residual. Calls the actual global
    /// callable with zero or two args, then resolves the attribute.
    pub load_super_attr_fn: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64,
    /// `bh_super_attr_unwrap_fn(raw, which)` — LOAD_SUPER_ATTR method-form
    /// unwrap (`which` 0 = func slot, 1 = self slot); pure / infallible.
    pub super_attr_unwrap_fn: extern "C" fn(i64, i64) -> i64,
    /// `bh_load_deref_value_fn(cell, code, deref_idx)` — LOAD_DEREF
    /// dereference residual (cell contents, raising the named unbound-variable
    /// `NameError` resolved via `code` + `deref_idx`).
    pub load_deref_value_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `bh_store_deref_value_fn(cell, value)` — STORE_DEREF residual: mutate
    /// the cell's contents (returning the unchanged cell) or return the raw
    /// `value` for a non-cell slot; infallible.
    pub store_deref_value_fn: extern "C" fn(i64, i64) -> i64,
    /// `bh_make_cell_fn(current)` — MAKE_CELL residual: wrap a raw slot value
    /// in a fresh cell (or return an existing cell unchanged); infallible.
    pub make_cell_fn: extern "C" fn(i64) -> i64,
    /// `jit_make_function_from_globals(globals, code)` — MAKE_FUNCTION residual:
    /// wrap a code object into a function using the given globals object;
    /// allocates but runs no user code and never raises.
    pub make_function_fn: extern "C" fn(i64, i64) -> i64,
    /// `jit_set_function_attribute(func, attr, flag)` — SET_FUNCTION_ATTRIBUTE
    /// residual: stamp one attribute (`flag` discriminant) on `func`, returning
    /// `func`; sets a typed field but runs no user code and never raises.
    pub set_function_attribute_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `bh_get_iter_fn(obj)` — GET_ITER `iter(obj)` residual
    /// (a user `__iter__` may run Python → fallible).
    pub get_iter_fn: extern "C" fn(i64) -> i64,
    /// `jit_next(iter)` — FOR_ITER `next(iter)` residual; returns the next
    /// item, PY_NULL on StopIteration exhaustion (the trailing for-iter
    /// GuardNonnull catches it), or publishes a real exception into the
    /// backend exception cells on error.
    pub for_iter_next_fn: extern "C" fn(i64) -> i64,
    /// `bh_unary_negative_fn(value)` — UNARY_NEGATIVE `-value` residual
    /// (a user `__neg__` may run Python → fallible).
    pub unary_negative_fn: extern "C" fn(i64) -> i64,
    /// `bh_unary_invert_fn(value)` — UNARY_INVERT `~value` residual
    /// (a user `__invert__` may run Python → fallible).
    pub unary_invert_fn: extern "C" fn(i64) -> i64,
    /// `bh_unary_positive_fn(value)` — UNARY_POSITIVE `+value` residual
    /// (a user `__pos__` may run Python → fallible).
    pub unary_positive_fn: extern "C" fn(i64) -> i64,
    /// `bh_load_common_constant_fn(disc)` — LOAD_COMMON_CONSTANT residual
    /// resolving a `CommonConstant` discriminant to its pushed object
    /// (allocates for the `all`/`any` builtin variants → `MayForce`).
    pub load_common_constant_fn: extern "C" fn(i64) -> i64,
    /// `bh_list_to_tuple_fn(value)` — CALL_INTRINSIC_1 ListToTuple residual
    /// (`list_to_tuple`, allocates a fresh tuple; non-list → TypeError).
    pub list_to_tuple_fn: extern "C" fn(i64) -> i64,
    /// `bh_unary_not_fn(value)` — UNARY_NOT `not value` residual returning a
    /// bool (a user `__bool__` / `__len__` may run Python; infallible).
    pub unary_not_fn: extern "C" fn(i64) -> i64,
    /// `bh_load_fast_check_fn(value, code, name_idx)` — LOAD_FAST_CHECK
    /// unbound-local guard (returns `value`, or raises `NameError`).
    pub load_fast_check_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `bh_unbound_local_error_fn(code, name_idx)` — construct and return the
    /// UnboundLocalError value for DELETE_FAST without publishing it.
    pub unbound_local_error_fn: extern "C" fn(i64, i64) -> i64,
    /// `bhimpl_compare_op` — RPython compare_op opcodes.
    pub compare_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `bhimpl_binary_op` — RPython binary_op opcodes.
    pub binary_op_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `bhimpl_w_int_new` — box a raw integer into a PyObject.
    pub box_int_fn: extern "C" fn(i64) -> i64,
    /// `bhimpl_truth` — PyObjectRef → raw 0 or 1.
    pub truth_fn: extern "C" fn(i64) -> i64,
    /// `bhimpl_load_const` — load constant from frame's code object.
    pub load_const_fn: extern "C" fn(i64, i64) -> i64,
    /// `bhimpl_store_subscr` — obj[key] = value.
    pub store_subscr_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `bhimpl_getattr` — `getattr(obj, w_name)`.
    /// `(obj: Ref, w_name: Ref) → Ref` with `w_name` an interned str
    /// constant.  Blackhole/deopt lowering of `LOAD_ATTR`
    /// (`rclass.py:838 rtype_getattr`).
    pub getattr_fn: extern "C" fn(i64, i64) -> i64,
    /// `bhimpl_load_name` — `(frame: Ref, w_name: Ref, namei: Int) → Ref`
    /// with `w_name` an interned str constant.  Blackhole/deopt lowering
    /// of `LOAD_NAME` (`pyopcode.py:945`).
    pub load_name_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `bhimpl_store_name` — `(frame: Ref, w_name: Ref, value: Ref) → Void`
    /// with `w_name` an interned str constant.  Blackhole/deopt lowering
    /// of `STORE_NAME` (`pyopcode.py:855`).
    pub store_name_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `bhimpl_store_global` — `(frame: Ref, w_name: Ref, value: Ref) →
    /// Void` with `w_name` an interned str constant.  Blackhole/deopt
    /// lowering of `STORE_GLOBAL` (`pyopcode.py:567`); writes directly
    /// into `w_globals`, bypassing `w_locals`.
    pub store_global_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `newtuple(list_w)` (`objspace.py:332`) — (ref array) → new tuple.
    /// The array is the forced `popvalues` list; length travels inside
    /// the array, so any arity fits.
    pub newtuple_from_array_fn: extern "C" fn(i64) -> i64,
    /// BUILD_MAP — the forced `[k0, v0, ...]` pair array → dict.  Length
    /// travels inside the array, so any arity fits.
    pub build_map_from_array_fn: extern "C" fn(i64) -> i64,
    /// BUILD_SET — the forced element array → set (fallible: element
    /// hashing may run user `__hash__` / raise on a non-hashable element).
    pub build_set_from_array_fn: extern "C" fn(i64) -> i64,
    /// BUILD_STRING — the forced fragment array → concatenated str.
    /// Fragments are already strings (formatted first), so this runs no
    /// user code and is infallible (`Plain`).
    pub build_string_from_array_fn: extern "C" fn(i64) -> i64,
    /// `newlist(list_w)` (`objspace.py`) — (ref array) → new list.  The
    /// array is the forced `popvalues_mutable` list; length travels
    /// inside the array, so any arity fits.
    pub newlist_from_array_fn: extern "C" fn(i64) -> i64,
    /// `bhimpl_unpack_sequence` — (count, seq) → validated tuple of items.
    pub unpack_sequence_fn: extern "C" fn(i64, i64) -> i64,
    /// Read item `index` out of the validated unpack tuple — (index, seq) → item.
    pub unpack_item_fn: extern "C" fn(i64, i64) -> i64,
    /// UNPACK_EX residual — `(before, after, seq) → tuple` of the
    /// `before + 1 + after` slots (head items, starred list, tail items)
    /// in TOS order; read back with `unpack_item_fn`.
    pub unpack_ex_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `bhimpl_build_slice` — (argc, start, stop, step) → new slice.
    pub build_slice_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// `RAISE_VARARGS` normalization helper used before `raise/r`.
    /// `(frame: Ref, exc: Ref, cause: Ref) → Ref` — the explicit frame
    /// pointer feeds `frame.execution_context` directly.
    pub normalize_raise_varargs_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// Read per-thread `CURRENT_EXCEPTION` — used by `PUSH_EXC_INFO`.
    pub get_current_exception_fn: extern "C" fn() -> i64,
    /// `raise_varargs(0)` value — the active exception, or a fresh
    /// `RuntimeError("No active exception to reraise")` when none is live.
    /// Used by a bare `RAISE_VARARGS(0)` with no static `last_exception` pair.
    pub reraise_varargs_zero_fn: extern "C" fn() -> i64,
    /// Write per-thread `CURRENT_EXCEPTION` — used by `PUSH_EXC_INFO`
    /// (set to new exc) and `POP_EXCEPT` (restore saved prev).
    pub set_current_exception_fn: extern "C" fn(i64),
    /// `rpython/jit/backend/llgraph/runner.py:LLGraphCPU.rtyper` —
    /// upstream `flatten_graph` reaches `cpu.rtyper.exceptiondata.
    /// get_standard_ll_exc_instance_by_class(OverflowError)` at
    /// `flatten.py:166-170` (the `handling_ovf=True` arm of
    /// `make_exception_link`).  Pyre's rtyper shim
    /// ([`super::exceptiondata::Rtyper`]) exposes only that attribute
    /// chain; other rtyper machinery is intentionally absent because
    /// pyre operates on the flowspace graph directly, without a
    /// typed-low-level rewrite.
    pub rtyper: super::exceptiondata::Rtyper,
    /// Retired-family fn-idx pool — pyre-specific extension carried
    /// on `Cpu` so the canonical `flatten_graph(graph, regallocs,
    /// _include_all_exc_links, cpu)` entry can derive the
    /// dispatcher's `LoweringContext` from `cpu` alone (matching
    /// upstream's "cpu carries everything flatten needs" contract).
    /// Each `Option<u16>` is populated by `CodeWriter::transform_
    /// graph_to_jitcode` from `descrs.intern_int_method_index`.
    /// **Note**: upstream's `cpu` does not carry
    /// these because upstream's rtyper rewrites the graph to post-
    /// rtype shape before `flatten_graph` runs, so the dispatcher has
    /// no upstream analog.  These fields retire once pyre's
    /// walker stops emitting pre-rtype HLOps in favour of post-rtype
    /// residual_call SpaceOperations on the graph.
    pub lowering_ctx: std::sync::RwLock<Option<super::flatten::LoweringContext>>,
}

impl Cpu {
    /// Default pyre `Cpu` — wires the production `bh_*` thunks from
    /// `crate::call_jit`. Matches the implicit `cpu = LLGraphCPU(...)`
    /// constructor in `warmspot.py:243` for the standard JIT.
    pub fn new() -> Self {
        let mut rtyper = super::exceptiondata::Rtyper::new();
        // Install a **lazy** resolver for standard exception instance
        // pointers — RPython's `RPythonTyper.specialize` ->
        // `ExceptionData.finish` invokes `get_standard_ll_exc_instance(
        // rtyper, clsdef)` at rtyper construction time
        // (`rpython/rtyper/exceptiondata.py:34-38`), which calls
        // `r_inst.get_reusable_prebuilt_instance()` and returns the
        // **prebuilt INSTANCE** pointer (not the class pointer).
        // Downstream `get_standard_ll_exc_instance_by_class` wraps
        // that pointer in a `Constant` and `flatten.py:165-170` emits
        // it directly as the operand of `raise/r`; feeding the
        // **class** pointer here would be semantically wrong
        // (`raise CLASS` is not the same as `raise INSTANCE`).
        //
        // Pyre's analog is `pyre_interpreter::lookup_exc_instance`
        // which materialises a process-global singleton
        // `W_BaseException` per `ExcKind` (see
        // `pyre_object::interp_exceptions::standard_exc_instance`).  We install
        // the resolver here but defer the per-`ExcKind` singleton
        // allocation to the first
        // `get_standard_ll_exc_instance_by_class` lookup; under the
        // current walker-driven pipeline the canonical
        // `flatten_graph` `_ovf` direct-raise rewrite (the sole
        // consumer of this table in production) never fires, so the
        // singletons stay unallocated.  The deferral matters because
        // the cranelift backend's trace compilation is sensitive to
        // heap layout — eagerly allocating 16 `W_BaseException`
        // singletons at `Cpu::new` time shifts subsequent heap
        // addresses enough to consistently push raise_catch_loop
        // tracing into a slow recompile path.
        rtyper.exceptiondata.set_lazy_resolver(|name| {
            let ptr = pyre_interpreter::lookup_exc_instance(name)?;
            Some(ptr as i64)
        });
        Self {
            call_fn: crate::call_jit::bh_call_fn,
            call_fn_0: crate::call_jit::bh_call_fn_0,
            call_fn_2: crate::call_jit::bh_call_fn_2,
            call_fn_3: crate::call_jit::bh_call_fn_3,
            call_fn_4: crate::call_jit::bh_call_fn_4,
            call_fn_5: crate::call_jit::bh_call_fn_5,
            call_fn_6: crate::call_jit::bh_call_fn_6,
            call_fn_7: crate::call_jit::bh_call_fn_7,
            call_fn_8: crate::call_jit::bh_call_fn_8,
            call_fn_9: crate::call_jit::bh_call_fn_9,
            call_fn_10: crate::call_jit::bh_call_fn_10,
            call_fn_11: crate::call_jit::bh_call_fn_11,
            call_fn_12: crate::call_jit::bh_call_fn_12,
            call_fn_13: crate::call_jit::bh_call_fn_13,
            call_fn_14: crate::call_jit::bh_call_fn_14,
            load_global_fn: crate::call_jit::bh_load_global_fn,
            load_from_dict_or_globals_fn: crate::call_jit::bh_load_from_dict_or_globals_fn,
            call_function_ex_fn: crate::call_jit::bh_call_function_ex_fn,
            call_kw_fn_0: crate::call_jit::bh_call_kw_0,
            call_kw_fn_1: crate::call_jit::bh_call_kw_1,
            call_kw_fn_2: crate::call_jit::bh_call_kw_2,
            call_kw_fn_3: crate::call_jit::bh_call_kw_3,
            call_kw_fn_4: crate::call_jit::bh_call_kw_4,
            call_kw_fn_5: crate::call_jit::bh_call_kw_5,
            call_kw_fn_6: crate::call_jit::bh_call_kw_6,
            call_kw_fn_7: crate::call_jit::bh_call_kw_7,
            call_kw_fn_8: crate::call_jit::bh_call_kw_8,
            call_kw_fn_9: crate::call_jit::bh_call_kw_9,
            call_kw_fn_10: crate::call_jit::bh_call_kw_10,
            call_kw_fn_11: crate::call_jit::bh_call_kw_11,
            call_kw_fn_12: crate::call_jit::bh_call_kw_12,
            call_kw_fn_13: crate::call_jit::bh_call_kw_13,
            load_attr_fn: crate::call_jit::bh_load_attr_fn,
            load_method_self_fn: crate::call_jit::bh_load_method_self_fn,
            load_special_fn: crate::call_jit::bh_load_special_fn,
            load_special_self_fn: crate::call_jit::bh_load_special_self_fn,
            store_attr_fn: crate::call_jit::bh_store_attr_fn,
            binary_slice_fn: crate::call_jit::bh_binary_slice_fn,
            store_slice_fn: crate::call_jit::bh_store_slice_fn,
            delete_subscr_fn: crate::call_jit::bh_delete_subscr_fn,
            delete_attr_fn: crate::call_jit::bh_delete_attr_fn,
            list_extend_fn: crate::call_jit::bh_list_extend_fn,
            set_add_fn: crate::call_jit::bh_set_add_fn,
            set_update_fn: crate::call_jit::bh_set_update_fn,
            dict_update_fn: crate::call_jit::bh_dict_update_fn,
            map_add_fn: crate::call_jit::bh_map_add_fn,
            dict_merge_fn: crate::call_jit::bh_dict_merge_fn,
            list_append_fn: pyre_object::listobject::jit_list_append,
            format_simple_fn: crate::call_jit::bh_format_simple_fn,
            format_with_spec_fn: crate::call_jit::bh_format_with_spec_fn,
            convert_value_fn: crate::call_jit::bh_convert_value_fn,
            import_name_fn: crate::call_jit::bh_import_name_fn,
            import_from_fn: crate::call_jit::bh_import_from_fn,
            load_super_attr_fn: crate::call_jit::bh_load_super_attr_fn,
            super_attr_unwrap_fn: crate::call_jit::bh_super_attr_unwrap_fn,
            load_deref_value_fn: crate::call_jit::bh_load_deref_value_fn,
            store_deref_value_fn: crate::call_jit::bh_store_deref_value_fn,
            make_cell_fn: crate::call_jit::bh_make_cell_fn,
            make_function_fn: pyre_interpreter::runtime_ops::jit_make_function_from_globals,
            set_function_attribute_fn: pyre_interpreter::runtime_ops::jit_set_function_attribute,
            get_iter_fn: crate::call_jit::bh_get_iter_fn,
            for_iter_next_fn: pyre_interpreter::runtime_ops::jit_next,
            unary_negative_fn: crate::call_jit::bh_unary_negative_fn,
            unary_invert_fn: crate::call_jit::bh_unary_invert_fn,
            unary_positive_fn: crate::call_jit::bh_unary_positive_fn,
            load_common_constant_fn: crate::call_jit::bh_load_common_constant_fn,
            list_to_tuple_fn: crate::call_jit::bh_list_to_tuple_fn,
            unary_not_fn: crate::call_jit::bh_unary_not_fn,
            load_fast_check_fn: crate::call_jit::bh_load_fast_check_fn,
            unbound_local_error_fn: crate::call_jit::bh_unbound_local_error_fn,
            compare_fn: crate::call_jit::bh_compare_fn,
            binary_op_fn: crate::call_jit::bh_binary_op_fn,
            box_int_fn: crate::call_jit::bh_box_int_fn,
            truth_fn: crate::call_jit::bh_truth_fn,
            load_const_fn: crate::call_jit::bh_load_const_fn,
            store_subscr_fn: pyre_interpreter::opcode_ops::bh_store_subscr_fn,
            getattr_fn: crate::call_jit::bh_getattr_fn,
            load_name_fn: crate::call_jit::bh_load_name_fn,
            store_name_fn: crate::call_jit::bh_store_name_fn,
            store_global_fn: crate::call_jit::bh_store_global_fn,
            newtuple_from_array_fn: crate::call_jit::bh_newtuple_from_array,
            build_map_from_array_fn: crate::call_jit::bh_build_map_from_array,
            build_set_from_array_fn: crate::call_jit::bh_build_set_from_array,
            build_string_from_array_fn: crate::call_jit::bh_build_string_from_array,
            newlist_from_array_fn: crate::call_jit::bh_newlist_from_array,
            unpack_sequence_fn: crate::call_jit::bh_unpack_sequence_fn,
            unpack_item_fn: crate::call_jit::bh_unpack_item_fn,
            unpack_ex_fn: crate::call_jit::bh_unpack_ex_fn,
            build_slice_fn: crate::call_jit::bh_build_slice_fn,
            normalize_raise_varargs_fn: crate::call_jit::bh_normalize_raise_varargs_with_frame,
            get_current_exception_fn: crate::call_jit::bh_get_current_exception,
            reraise_varargs_zero_fn: crate::call_jit::bh_reraise_varargs_zero,
            set_current_exception_fn: crate::call_jit::bh_set_current_exception,
            rtyper,
            lowering_ctx: std::sync::RwLock::new(None),
        }
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

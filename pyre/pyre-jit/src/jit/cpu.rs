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
    /// `call_fn_8(callable, null_or_self, a0..a7)`).  RPython
    /// `bhimpl_residual_call_r_r` carries no frame; the parent frame is
    /// resolved from the execution context inside `bh_call_fn_impl`.  A
    /// non-null `null_or_self` is the method receiver — the helper
    /// prepends it as arg0 (eval.rs:3216-3226).
    pub call_fn_0: extern "C" fn(i64, i64) -> i64,
    pub call_fn_2: extern "C" fn(i64, i64, i64, i64) -> i64,
    pub call_fn_3: extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub call_fn_4: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_5: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_6: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_7: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_8: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    /// `bhimpl_load_global` — namespace/code from getfield_vable_r plus live frame.
    pub load_global_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// LOOKUP_METHOD attribute half — `(obj, code, name_idx) → attr`.
    /// Reproduces `PyFrame::load_method`'s `getattr` for blackhole resume.
    pub load_attr_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// LOOKUP_METHOD `null_or_self` half — `(obj, attr, code, name_idx) →
    /// bound`. Pure binding decision shared with the interpreter.
    pub load_method_self_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
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
    /// `bhimpl_build_list` — (argc, item0, item1, item2) → new list.
    pub build_list_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// `bhimpl_build_tuple` — (argc, item0, item1, item2) → new tuple.
    pub build_tuple_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// `bhimpl_unpack_sequence` — (count, seq) → validated tuple of items.
    pub unpack_sequence_fn: extern "C" fn(i64, i64) -> i64,
    /// Read item `index` out of the validated unpack tuple — (index, seq) → item.
    pub unpack_item_fn: extern "C" fn(i64, i64) -> i64,
    /// `bhimpl_build_slice` — (argc, start, stop, step) → new slice.
    pub build_slice_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// `RAISE_VARARGS` normalization helper used before `raise/r`.
    /// `(frame: Ref, exc: Ref, cause: Ref) → Ref` — the explicit frame
    /// pointer feeds `frame.execution_context` directly.
    pub normalize_raise_varargs_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// Read per-thread `CURRENT_EXCEPTION` — used by `PUSH_EXC_INFO`.
    pub get_current_exception_fn: extern "C" fn() -> i64,
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
        // `W_ExceptionObject` per `ExcKind` (see
        // `pyre_object::excobject::standard_exc_instance`).  We install
        // the resolver here but defer the per-`ExcKind` singleton
        // allocation to the first
        // `get_standard_ll_exc_instance_by_class` lookup; under the
        // current walker-driven pipeline the canonical
        // `flatten_graph` `_ovf` direct-raise rewrite (the sole
        // consumer of this table in production) never fires, so the
        // singletons stay unallocated.  The deferral matters because
        // the cranelift backend's trace compilation is sensitive to
        // heap layout — eagerly allocating 16 `W_ExceptionObject`
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
            load_global_fn: crate::call_jit::bh_load_global_fn,
            load_attr_fn: crate::call_jit::bh_load_attr_fn,
            load_method_self_fn: crate::call_jit::bh_load_method_self_fn,
            compare_fn: crate::call_jit::bh_compare_fn,
            binary_op_fn: crate::call_jit::bh_binary_op_fn,
            box_int_fn: crate::call_jit::bh_box_int_fn,
            truth_fn: crate::call_jit::bh_truth_fn,
            load_const_fn: crate::call_jit::bh_load_const_fn,
            store_subscr_fn: pyre_interpreter::opcode_ops::bh_store_subscr_fn,
            getattr_fn: crate::call_jit::bh_getattr_fn,
            build_list_fn: crate::call_jit::bh_build_list_fn,
            build_tuple_fn: crate::call_jit::bh_build_tuple_fn,
            unpack_sequence_fn: crate::call_jit::bh_unpack_sequence_fn,
            unpack_item_fn: crate::call_jit::bh_unpack_item_fn,
            build_slice_fn: crate::call_jit::bh_build_slice_fn,
            normalize_raise_varargs_fn: crate::call_jit::bh_normalize_raise_varargs_with_frame,
            get_current_exception_fn: crate::call_jit::bh_get_current_exception,
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

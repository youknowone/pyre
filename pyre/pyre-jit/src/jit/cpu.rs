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
    pub call_fn: extern "C" fn(i64, i64) -> i64,
    /// Per-arity `bhimpl_residual_call_<n>` helpers
    /// (call_fn_0(callable) ... call_fn_8(callable, a0..a7)).
    pub call_fn_0: extern "C" fn(i64) -> i64,
    pub call_fn_2: extern "C" fn(i64, i64, i64) -> i64,
    pub call_fn_3: extern "C" fn(i64, i64, i64, i64) -> i64,
    pub call_fn_4: extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub call_fn_5: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_6: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_7: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_8: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    /// `bhimpl_load_global` — namespace/code from getfield_vable_r plus live frame.
    pub load_global_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
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
    /// `bhimpl_build_list` — (argc, item0, item1, item2) → new list.
    pub build_list_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// `bhimpl_build_slice` — (argc, start, stop, step) → new slice.
    pub build_slice_fn: extern "C" fn(i64, i64, i64, i64) -> i64,
    /// `RAISE_VARARGS` normalization helper used before `raise/r`.
    pub normalize_raise_varargs_fn: extern "C" fn(i64, i64) -> i64,
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
    /// no upstream analog.  Phase 5 retires these fields once pyre's
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
        // Resolve standard exception class pointers from the live
        // interpreter registry.  Matches RPython's
        // `RPythonTyper.specialize` -> `ExceptionData.make_helpers`
        // (`exceptiondata.py:34-42`) where `get_standard_ll_exc_instance(
        // rtyper, clsdef)` is invoked at rtyper construction so every
        // downstream `get_standard_ll_exc_instance_by_class` returns
        // the resolved LL instance pointer.  Pyre's analog reads the
        // installed builtin exception class via
        // `pyre_interpreter::lookup_exc_class(name)`; the lookup is
        // thread-local and may return `None` when `Cpu::new` runs
        // before `install_default_builtins` (test setup paths that
        // never reach the `_ovf` rewrite are unaffected — the panic
        // surfaces only if `get_standard_ll_exc_instance_by_class` is
        // ever called while unresolved).
        rtyper
            .exceptiondata
            .resolve_standard_exception_pointers(|name| {
                let ptr = pyre_interpreter::lookup_exc_class(name)?;
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
            compare_fn: crate::call_jit::bh_compare_fn,
            binary_op_fn: crate::call_jit::bh_binary_op_fn,
            box_int_fn: crate::call_jit::bh_box_int_fn,
            truth_fn: crate::call_jit::bh_truth_fn,
            load_const_fn: crate::call_jit::bh_load_const_fn,
            store_subscr_fn: crate::call_jit::bh_store_subscr_fn,
            build_list_fn: crate::call_jit::bh_build_list_fn,
            build_slice_fn: crate::call_jit::bh_build_slice_fn,
            normalize_raise_varargs_fn: crate::call_jit::bh_normalize_raise_varargs_fn,
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

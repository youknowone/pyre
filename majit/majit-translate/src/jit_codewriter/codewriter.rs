//! Code generation pipeline — majit's equivalent of `rpython/jit/codewriter/`.
//!
//! ```text
//! rpython/jit/codewriter/          majit-translate/src/codewriter/
//! ├── codewriter.py          →     ├── mod.rs (CodeWriter struct)
//! ├── jtransform.py          →     ├── jtransform.rs
//! ├── flatten.py + assembler.py →  ├── codegen.rs
//! └── call.py                →     └── (call.rs)
//! ```

pub mod codegen;

pub use codegen::{
    CodegenValueKind, IoShim, JitDriverConfig, VirtualizableCodegenConfig, generate_jitcode,
};

use std::collections::HashMap;
use std::sync::Arc;

use crate::assembler::Assembler;
use crate::call::CallControl;
use crate::jitcode::JitCode;
use crate::jtransform::GraphTransformConfig;
use crate::model::FunctionGraph;
use crate::parse::CallPath;

/// Output of `CodeWriter::make_jitcodes` — the populated JitCode registry
/// paired with its alloc-order index view.
///
/// Parity shape: RPython `rpython/jit/codewriter/call.py:87-88` holds two
/// fields on `CallControl` itself — `self.jitcodes = {}` (graph-keyed dict)
/// and `self.all_jitcodes = []` (alloc-order list). Pyre preserves both
/// fields on `CallControl` and additionally returns them as a single value
/// from `make_jitcodes`. The wrapper is a PRE-EXISTING-ADAPTATION: upstream
/// callers (`warmspot.py:281-282`) read `self.all_jitcodes` off the
/// CallControl directly; pyre pairs the two views so that downstream
/// consumers (e.g. `crate::generated::all_jitcodes`) receive one value and
/// do not need to pass `CallControl` around to perform later lookups.
pub struct AllJitCodes {
    /// RPython `call.py:87 self.jitcodes = {}` (graph-keyed dict). Pyre's
    /// graph identity at this boundary is `CallPath`.
    pub by_path: HashMap<CallPath, Arc<JitCode>>,
    /// RPython `call.py:88 self.all_jitcodes = []` (alloc-order list). The
    /// `jitcode.index == i` invariant from `codewriter.py:80` is enforced
    /// by `CallControl::collect_jitcodes_in_alloc_order`.
    pub in_order: Vec<Arc<JitCode>>,
}

impl AllJitCodes {
    pub fn len(&self) -> usize {
        self.in_order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.in_order.is_empty()
    }

    pub fn get(&self, path: &CallPath) -> Option<&Arc<JitCode>> {
        self.by_path.get(path)
    }
}

/// RPython: `codewriter.py::CodeWriter`.
///
/// Orchestrates the full JitCode generation pipeline:
///   annotate → rtype → jtransform → regalloc → flatten → liveness → assemble
///
/// RPython's CodeWriter owns both the Assembler and CallControl.
/// In majit, CallControl is passed by `&mut` reference to avoid
/// lifetime entanglement with the Transformer's borrows.
pub struct CodeWriter {
    /// RPython: `self.assembler = Assembler()` (codewriter.py:22).
    pub assembler: Assembler,
    /// RPython: `self.debug = True` (codewriter.py:18).
    pub debug: bool,
    /// Slice 6 — lazy program-wide `PyreCallRegistry` for the
    /// PYRE_RTYPER dual-gate (cf. `transform_graph_to_jitcode`).
    /// Built on first use from `callcontrol.function_graphs()` and
    /// reused across every dual-gated graph in the same CodeWriter
    /// run, mirroring upstream `Bookkeeper.descs` lifecycle (one
    /// descs map per `Translator` driving an `RPythonAnnotator`).
    real_rtyper_registry: std::cell::RefCell<
        Option<std::rc::Rc<crate::translator::rtyper::pyre_call_registry::PyreCallRegistry>>,
    >,
}

impl CodeWriter {
    /// RPython: `CodeWriter.__init__(cpu, jitdrivers_sd)` (codewriter.py:20-23).
    ///
    /// `debug` mirrors the class-level default `debug = True`
    /// (`codewriter.py:18`). Upstream always produces per-jitcode
    /// diagnostic output (`log.dot()` in `print_ssa_repr`), so pyre
    /// matches by defaulting `debug: true`. Tests that want silent
    /// operation may flip the field after construction.
    pub fn new() -> Self {
        Self {
            assembler: Assembler::new(),
            debug: true,
            real_rtyper_registry: std::cell::RefCell::new(None),
        }
    }

    /// `codewriter.py:91-94 CodeWriter.setup_vrefinfo(self, vrefinfo)`.
    ///
    /// ```python
    /// def setup_vrefinfo(self, vrefinfo):
    ///     # must be called at most once
    ///     assert self.callcontrol.virtualref_info is None
    ///     self.callcontrol.virtualref_info = vrefinfo
    /// ```
    ///
    /// RPython's `CodeWriter` owns `self.callcontrol`; pyre keeps
    /// `CallControl` outside the `CodeWriter` (passed by `&mut` to
    /// `transform_graph_to_jitcode`), so the wrapper takes
    /// `callcontrol` as an explicit parameter and delegates to
    /// [`CallControl::setup_vrefinfo`].  The at-most-once assertion
    /// lives in the delegate.
    pub fn setup_vrefinfo(
        &self,
        callcontrol: &mut CallControl,
        vrefinfo: Arc<dyn crate::call::VirtualRefInfoHandle>,
    ) {
        callcontrol.setup_vrefinfo(vrefinfo);
    }

    /// Get-or-build the program-wide `PyreCallRegistry` for the
    /// PYRE_RTYPER dual-gate.  Builds once per CodeWriter run and
    /// caches; subsequent calls reuse the same `Rc`.  The registry
    /// is populated from `callcontrol.function_graphs()` — pyre's
    /// program-wide graph map, the production analog of
    /// `SemanticProgram.functions`.
    fn dual_gate_registry(
        &self,
        callcontrol: &CallControl,
    ) -> std::rc::Rc<crate::translator::rtyper::pyre_call_registry::PyreCallRegistry> {
        if let Some(existing) = self.real_rtyper_registry.borrow().as_ref() {
            return existing.clone();
        }
        let registry = std::rc::Rc::new(
            crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(std::rc::Rc::new(
                crate::annotator::bookkeeper::Bookkeeper::new(),
            )),
        );
        // Thread the program-wide `use_imports` aggregate (one entry
        // per `(source_module, alias)` pair across all parsed files,
        // built in `lib.rs::analyze_files`) into the registry so
        // `flowspace_adapter::translate_op` can recover the caller's
        // per-file lexical scope when resolving an
        // `OpKind::Call::FunctionPath` (mirrors RPython
        // `flowcontext.py:845-866 LOAD_GLOBAL` consulting
        // `func.__globals__` before the builtins fallback).
        registry.set_use_imports(callcontrol.use_imports.clone());
        // PyPy's `Bookkeeper.compute_at_fixpoint` raises through to
        // the caller (`bookkeeper.py:108-127`); pyre's dual-gate
        // mirrors that propagation by routing the populate `TyperError`
        // through `is_known_unported`.  Known-unported categories
        // leave a partial registry (the per-graph dual-gate will
        // Skip-classify the affected callsites via the cascade
        // `"not registered in PyreCallRegistry"`).  Unknown errors
        // panic immediately so parity bugs surface here rather than
        // silently shifting downstream behind a Skip mask.
        let populate_result =
            crate::translator::rtyper::cutover::populate_call_registry_from_call_graphs(
                callcontrol.function_graphs(),
                &registry,
            );
        if let Err(err) = populate_result {
            let msg = format!("{err}");
            if !crate::translator::rtyper::cutover::is_known_unported(&msg) {
                panic!("populate_call_registry_from_call_graphs failed: {msg}");
            }
        }
        // RPython parity: `Translator.buildannotator()` /
        // `Translator.buildrtyper()` (`translator.py:69-83`) construct
        // exactly one annotator and one rtyper per Translator and assert
        // on re-entry.  Pyre mirrors that contract through
        // [`PyreCallRegistry::ensure_session`]
        // (`pyre_call_registry.rs:210-234`), which lazily builds a single
        // `(RPythonAnnotator, RPythonTyper)` pair on first use and
        // returns the cached pair on every subsequent
        // `specialize_legacy_graph_with_registry_returning_value_to_var`
        // call.  The
        // registry itself is cached on the `CodeWriter`
        // (`real_rtyper_registry`), so all per-graph subjects of one
        // CodeWriter share the same annotator + rtyper, matching
        // upstream's "one annotator + one rtyper per Translator"
        // semantics.
        *self.real_rtyper_registry.borrow_mut() = Some(registry.clone());
        registry
    }

    /// RPython: `CodeWriter.transform_graph_to_jitcode()` (codewriter.py:33-72).
    ///
    /// Transforms a FunctionGraph into a JitCode through the 4-step pipeline.
    /// Upstream signature `(self, graph, jitcode, verbose, index)`. Pyre adds
    /// `path` / `callcontrol` / `config` as PRE-EXISTING-ADAPTATION:
    ///   - `path`: graph identity surrogate (upstream uses `graph` object
    ///     identity; pyre uses `CallPath`).
    ///   - `callcontrol`: upstream has `self.callcontrol`; pyre passes
    ///     `&mut` due to Rust borrow-checker constraints (the Transformer
    ///     borrows callcontrol during `transform()`).
    ///   - `config`: pyre's `GraphTransformConfig` carries options that
    ///     upstream keeps in globals / command-line flags.
    ///
    /// Steps:
    ///   0. annotate + rtype (majit-specific; RPython does this before codewriter)
    ///   1. jtransform — `transform_graph()` (codewriter.py:42)
    ///   2. regalloc — `perform_register_allocation()` per kind (codewriter.py:45-47)
    ///   3. flatten — `flatten_graph()` (codewriter.py:53)
    ///   3b. liveness — `compute_liveness()` (codewriter.py:56, called inside assemble)
    ///   4. assemble — `assembler.assemble()` (codewriter.py:67)
    ///   5. `jitcode.index = index` (codewriter.py:68)
    ///   6. `if self.debug: self.print_ssa_repr(ssarepr, portal_jd, verbose)`
    ///      (codewriter.py:71-72)
    ///
    /// **Type-source contract (post graph-side concretetype migration)**
    ///
    /// `regalloc`/`flatten`/`assemble`/`liveness`/`format` all read
    /// kinds via `FunctionGraph::concretetype_of(&v)`, which routes
    /// straight to the backing `Variable.concretetype` cell stored in
    /// [`crate::model::FunctionGraph::value_variables`] — RPython's
    /// `Variable.concretetype` (`flowspace/model.py:280`) is the
    /// single source of truth for every slot.  No type side-table
    /// parameter survives across stages: the post-rtyper merge
    /// below (`merge_synth_kinds_into_graph`) stamps each synth
    /// Variable's `.concretetype` cell via
    /// `set_concretetype_of_inline`, then `apply_from_flowspace_variables`
    /// rebinds per-Variable from the `value_to_var` map so the
    /// rtyper's authoritative `Variable.concretetype` overrides
    /// any synthetic stamp.  Slots without a rtyper-bound Variable
    /// keep the synthetic canonical type the merge wrote.
    ///
    /// **Remaining structural divergence** — pyre's codewriter still
    /// consumes [`crate::model::FunctionGraph`] (a slot-indexed legacy
    /// IR) instead of [`crate::flowspace::model::FunctionGraph`] (the
    /// upstream `Variable`-based shape).  Operand identity in
    /// `SpaceOperation` payloads is already `Variable` (Slice 2.9
    /// onward); `FlatOp` operands are `Register` (regalloc color +
    /// kind).  What still ties the codewriter to the legacy IR is the
    /// dense slot index that `FunctionGraph` uses to key
    /// `value_variables` and the side tables in `value_map` /
    /// `alias_map` paths.  Migrating to the `Variable`-based IR
    /// throughout would let pyre drop the `value_to_var` bridge and
    /// consume the rtyper's Variable graph directly — multi-week
    /// scope tracked separately.
    /// Slice 12.2 / 12.4 — shared dual-gate type-resolve entry.
    ///
    /// Runs [`dual_gate_check_with_registry`] against the
    /// program-wide `PyreCallRegistry`; on Match the real path's
    /// `SlotToVariable` map (with each `Variable.concretetype`
    /// cell populated by `RPythonTyper::specialize`) is returned
    /// directly, on Skip the legacy walker (`legacy_annotator::annotate` +
    /// `legacy_resolve::resolve_types`) commits kinds to
    /// `graph.concretetype` cells so non-portal jitcodes that didn't
    /// pass the real path still get sound kinds.
    ///
    /// `diag_label` is appended to the optional Skip log line and the
    /// real-path panic message; production callers pass
    /// `path.canonical_key()`-style identification, the lib.rs
    /// debug-snapshot path passes `graph.name`.
    /// Run the dual-gate type resolver and commit every resolved kind
    /// to each backing `Variable.concretetype` cell on `graph` (RPython
    /// `rtyper.py:258 v.concretetype = ...`).  Returns the
    /// `SlotToVariable` map produced by the Match arm so the
    /// post-jtransform path can rebind operand Variables to the
    /// upstream-typed ones; Skip arm returns `None`.
    pub fn dual_gate_publish_concretetypes(
        &mut self,
        graph: &FunctionGraph,
        callcontrol: &mut CallControl,
        diag_label: &str,
    ) -> Option<crate::translator::rtyper::flowspace_adapter::SlotToVariable> {
        let dual_gate_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let registry = self.dual_gate_registry(callcontrol);
            crate::translator::rtyper::cutover::dual_gate_check_with_registry(graph, &registry)
        }));
        let outcome = match dual_gate_outcome {
            Ok(result) => result,
            Err(payload) => {
                let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "<unrecognised panic payload>".to_string()
                };
                if crate::translator::rtyper::cutover::is_known_unported(&msg) {
                    Ok(crate::translator::rtyper::cutover::DualGateOutcome::Skip(
                        format!("registry build panicked: {msg}"),
                    ))
                } else {
                    Err(format!("registry build panicked: {msg}"))
                }
            }
        };
        match outcome {
            Ok(crate::translator::rtyper::cutover::DualGateOutcome::Match {
                real_value_to_var,
            }) => {
                // Commit each real-rtyper Variable's `concretetype`
                // (LowLevelType) onto its placeholder on the graph's
                // value table.  Mirrors RPython `rtyper.py:258 v.concretetype = ...`
                // attribute aliasing; reads via `FunctionGraph::concretetype_of(&v)`
                // then match upstream `getkind(v.concretetype)`.
                for var in real_value_to_var.values() {
                    if let Some(lltype) = var.concretetype().as_ref() {
                        crate::model::FunctionGraph::set_concretetype_of_inline(
                            var,
                            crate::model::getkind(lltype),
                        );
                    }
                }
                Some(real_value_to_var)
            }
            Ok(crate::translator::rtyper::cutover::DualGateOutcome::Skip(reason)) => {
                if std::env::var_os("PYRE_RTYPER_VERBOSE").is_some_and(|v| v == "1") {
                    eprintln!(
                        "[PYRE_RTYPER skip] graph {diag_label:?} ({:?}): {reason}",
                        graph.name,
                    );
                }
                crate::translator::rtyper::legacy_annotator::annotate(graph);
                // `resolve_types` commits per-Variable concretetype
                // cells at the resolver boundary.  Skip arm has no
                // flowspace Variable surface — the legacy walker stays
                // the only source of types for these graphs.  Reads
                // annotations off `Variable.annotation` populated by
                // the preceding `annotate` post-loop publish.
                crate::translator::rtyper::legacy_resolve::resolve_types(graph);
                None
            }
            Err(diff) => panic!(
                "PYRE_RTYPER real-path failure on graph {diag_label:?} ({:?}): {diff}",
                graph.name,
            ),
        }
    }

    pub fn transform_graph_to_jitcode(
        &mut self,
        graph: &FunctionGraph,
        path: &crate::parse::CallPath,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
        jitcode: &std::sync::Arc<JitCode>,
        verbose: bool,
        index: usize,
    ) {
        // RPython: graph = copygraph(graph, shallowvars=True) (codewriter.py:38)
        // In Rust, Transformer.transform() already clones the graph.

        // Step 0: annotate + rtype (majit-specific)
        // RPython: types are already on Variable.concretetype from the rtyper.
        //
        // Slice 12.2 narrowed the legacy walker to the Skip arm; Slice
        // 12.4 lifted the dual-gate logic into the shared
        // [`Self::dual_gate_type_state`] helper so the parser-level
        // debug snapshot in `build_canonical_opcode_dispatch`
        // (lib.rs:898) can route through the same path.
        let canonical_diag = path.canonical_key().to_string();
        // `dual_gate_publish_concretetypes` commits every resolved
        // kind to each backing `Variable.concretetype` cell in both
        // arms, so downstream consumers read kinds via
        // `FunctionGraph::concretetype_of(&v)` directly.
        let real_value_to_var =
            self.dual_gate_publish_concretetypes(graph, callcontrol, &canonical_diag);

        // Step 0b: rtyper-equivalent indirect_call lowering
        // (`translator/rtyper/rpbc.rs::lower_indirect_calls`).
        // RPython rpbc.py:199-217 emits `indirect_call(funcptr, *args,
        // c_graphs)` during rtype; pyre runs the same pass here so
        // jtransform sees `OpKind::IndirectCall` (with funcptr already
        // a regular Variable), never `CallTarget::Indirect`.
        let mut graph_owned = graph.clone();
        crate::translator::rtyper::rpbc::lower_indirect_calls(&mut graph_owned, callcontrol);
        #[cfg(debug_assertions)]
        crate::translator::rtyper::rpbc::assert_no_indirect_call_targets(&graph_owned);
        // `resolve_types` (called upstream by the rtyper) already
        // commits each backing Variable's `concretetype` cell as it
        // resolves, so jtransform reads kinds via
        // `FunctionGraph::concretetype_of(&v)` (the upstream
        // `getkind(v.concretetype)` path) without a separate publish
        // step here.
        // Annotator monomorphization producer — for every Call(Method)
        // op in the graph, look up the receiver's annotated flowspace
        // Variable via `real_value_to_var` and stamp its concrete
        // `ClassDefKey` onto `CallTarget::Method.classdef_hint`. The
        // resolver fast path in `CallControl::resolve_method` /
        // `resolve_method_impl_type` then bypasses the receiver-root
        // heuristic (mirrors `bookkeeper.py:431-442 getmethoddesc`
        // classdef-keyed dispatch). Ops whose receiver carries no
        // SomeInstance annotation are left untouched.
        if let Some(value_to_var) = real_value_to_var.as_ref() {
            stamp_classdef_hints_on_graph(&mut graph_owned, value_to_var, callcontrol);
        }
        let graph = &graph_owned;

        // RPython codewriter.py:37 `portal_jd =
        // self.callcontrol.jitdriver_sd_from_portal_graph(graph)` — look
        // up the JitDriverStaticData keyed by portal graph identity. In
        // pyre `CallPath` is the surrogate, so resolve via `path`. Store
        // only the index to avoid a second borrow against
        // `callcontrol` during Transformer construction; the
        // Transformer can re-dereference `jitdrivers_sd[portal_jd_index]`
        // on demand if it needs the full record.
        let portal_jd_index = callcontrol
            .jitdriver_sd_from_portal_graph(path)
            .map(|jd| jd.index);

        // Step 1: jtransform (codewriter.py:42)
        // RPython: transform_graph(graph, cpu, callcontrol, portal_jd)
        //
        // No `with_type_state(&type_state)` — `dual_gate_type_state`
        // already committed every kind to each backing Variable's
        // `concretetype` cell, and `Variable::clone` Rc-shares that
        // cell so jtransform's internal `rewritten = graph.clone()`
        // carries it through.
        let mut rewritten = {
            let mut transformer = crate::jtransform::Transformer::new(config)
                .with_callcontrol(callcontrol)
                .with_portal_jd(portal_jd_index);
            transformer.transform(graph)
        };
        // Transformer is dropped here, releasing the &mut CallControl borrow.

        // RPython stores `.concretetype` on each Variable. Pyre merges
        // the pre-jtransform rtyper table with explicit result kinds
        // introduced by jtransform (`CallResidual.result_kind`, etc.)
        // without letting stale pre-rewrite entries override them.
        //
        // The merge precedence is `stamped > post_result > original`:
        //   - `original` = pre-jtransform `type_state` from the real
        //     RPythonTyper (already typed every pre-rewrite Variable).
        //   - `post_result` = `authoritative_result_types(rewritten)`
        //     reads the rewritten ops' declared `result_ty`/`result_
        //     kind` fields (per-op ground-truth).
        //   - `stamped` = `synth_kinds` jtransform produced for
        //     freshly-synthesised values.
        //
        // The legacy `resolve_types(rewritten_graph, annotations)`
        // walker that previously fed `merge_synth_kinds`'s `post_
        // resolve` lane is no longer called: every post-rewrite
        // Variable either reuses a pre-rewrite identity (covered by
        // `original`), is an op.result with declared kind (covered by
        // `post_result`), or is a synth jtransform value (covered by
        // `stamped`).  Block inputargs introduced by jtransform pick
        // up types via `original` (jtransform reuses the pre-rewrite
        // inputarg Variables when splicing new control flow).
        // Merge the four post-rtyper kind sources directly into
        // each backing `Variable.concretetype` (via
        // `FunctionGraph::set_concretetype_of_inline`) — pyre's
        // analogue of RPython's "rtyper finishes, every Variable
        // has `.concretetype` inline" handoff (`rtyper.py`).
        // Precedence stack
        // `stamped > post_result > post_resolve > original` is
        // preserved; the graph IS the merge target, no intermediate
        // type side-table survives the call.
        let post_result_types =
            crate::jit_codewriter::type_state::authoritative_result_types(&rewritten.graph);
        // `post_resolve` is intentionally empty here: jtransform now
        // writes resolved kinds straight to each backing
        // `Variable.concretetype` (see `Transformer::transform` →
        // `apply_to_graph`), so the legacy `resolve_rewritten_types`
        // walk is structurally dead in the production path.  The
        // parameter is kept on the API because legacy_pipeline.rs still
        // funnels its `resolve_rewritten_types` output through this
        // function for the dual-gate baseline comparison.
        // Commit jtransform-induced op-result kinds (`result_ty` /
        // `result_kind` declarations) to each backing
        // `Variable.concretetype` cell.  Pre-jtransform kinds are
        // already on the graph cells (rtyper boundary via
        // `apply_to_graph` / `resolve_types`); this overlay handles
        // the post-jtransform op-result deltas.  Precedence stack
        // `post_result > pre-jtransform` falls out of
        // `set_concretetype_of_inline`'s "preserve richer existing
        // iff getkind matches" semantic.
        for (var, kind) in &post_result_types {
            if !matches!(kind, crate::model::ConcreteType::Unknown) {
                crate::model::FunctionGraph::set_concretetype_of_inline(var, kind.clone());
            }
        }
        // Long-term parity hydration: when the dual-gate Match arm
        // surfaced a `SlotToVariable` map, rebind each slot to the
        // upstream-typed `Variable` so `FunctionGraph::concretetype_of(&v)`
        // reads its `concretetype` cell directly.  Upstream parity:
        // `history.py:46-71 getkind` reads `v.concretetype` from the
        // Variable, so this rebinding makes pyre's read path
        // line-for-line equivalent.
        if let Some(value_to_var) = real_value_to_var.as_ref() {
            crate::jit_codewriter::type_state::apply_from_flowspace_variables(
                &mut rewritten.graph,
                value_to_var,
            );
        }

        // Step 2: regalloc (codewriter.py:45-47)
        // RPython: for kind in KINDS: regallocs[kind] = perform_register_allocation(graph, kind)
        // Pyre reads each per-value kind via
        // `FunctionGraph::concretetype_of(&v)` (set up by `apply_to_graph`
        // / `apply_from_flowspace_variables` above), matching upstream's
        // `getkind(v.concretetype)` access.
        // Stamp canonical exceptblock kinds first so the rtyper-skip
        // path still gets `(etype=Int, evalue=Ref)`.
        crate::regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten.graph);
        let mut regallocs = crate::regalloc::perform_all_register_allocations(&rewritten.graph);

        // Step 3: flatten (codewriter.py:53)
        // RPython: ssarepr = flatten_graph(graph, regallocs, cpu=cpu)
        // Each Variable's `.concretetype` cell is the kind source
        // after the merge/hydration steps above; flatten reads it via
        // `FunctionGraph::concretetype_of(&var)`.  `flatten_graph`
        // itself runs `enforce_input_args` (flatten.py:88-100) so the
        // startblock inputarg colors land in the dense `0..N` prefix
        // of each kind, and the rotation persists into the assembler
        // call below — matching upstream `flatten.py:63-66`
        // invocation order verbatim.
        let mut ssarepr = crate::flatten::flatten_graph(&rewritten.graph, &mut regallocs);

        // Step 3b + 4: liveness + assemble (codewriter.py:56,67)
        // RPython: compute_liveness(ssarepr) then assembler.assemble(ssarepr, jitcode, num_regs)
        // In majit, assemble() calls compute_liveness() internally and now
        // returns the body so the codewriter can fill calldescr before
        // committing the shell via `set_body`.
        let mut body = self.assembler.assemble_with_callcontrol_and_graph(
            &mut ssarepr,
            &regallocs,
            Some(callcontrol),
            &rewritten.graph,
        );

        // call.py:174-187 get_jitcode_calldescr:
        //   FUNC = lltype.typeOf(fnptr).TO
        //   NON_VOID_ARGS = [ARG for ARG in FUNC.ARGS if ARG is not lltype.Void]
        //   calldescr = self.cpu.calldescrof(FUNC, tuple(NON_VOID_ARGS),
        //                                    FUNC.RESULT, EffectInfo.MOST_GENERAL)
        // Source of truth for `result_type` is the declared return type
        // registered on `CallControl` (mirrors RPython's `FUNC.RESULT`,
        // which comes from `getfunctionptr(graph)._obj`'s lltype). The
        // CFG terminator scan stays as a `debug_assert!` cross-check so
        // graphs that disagree with their declared signature surface
        // immediately.
        {
            let start_block = rewritten.graph.block(rewritten.graph.startblock);
            let mut arg_classes = String::new();
            // RPython `call.py:181-187 get_jitcode_calldescr` derives
            // `FUNC.ARGS` from `lltype.typeOf(fnptr).TO.ARGS`
            // directly.  Pyre's source-of-truth analogue is each
            // start-block inputarg's backing `Variable.concretetype`:
            // `FunctionGraph::concretetype_of(&v)` projects
            // `getkind(v.concretetype)` verbatim.  Reading from the
            // Variable matches the upstream's "type-source" provenance
            // instead of going through regalloc as a side-channel.
            for arg in &start_block.inputargs {
                use crate::model::ConcreteType;
                let class = match crate::model::FunctionGraph::concretetype_of(arg) {
                    ConcreteType::Signed => 'i',
                    ConcreteType::GcRef => 'r',
                    ConcreteType::Float => 'f',
                    ConcreteType::Void => 'v',
                    ConcreteType::Unknown => 'v',
                };
                arg_classes.push(class);
            }
            let cfg_kind = graph_result_kind(&rewritten.graph);
            let declared_kind = callcontrol.declared_return_kind(path);
            let result_type = declared_kind.unwrap_or(cfg_kind);
            // Cross-check: when both sources are present they must agree,
            // with one pre-existing exception. RPython `call.py:182-187
            // get_jitcode_calldescr` derives FUNC.RESULT from the declared
            // Rust-side return type (via `function_return_types`).
            // `graph_result_kind` independently walks the CFG and reports
            // the coloring produced by the rtyper. In pyre these two
            // sources can disagree specifically on `i ↔ r`: PyObjectRef
            // is a pointer (declared as `r`) but some helper graphs
            // (e.g. `unwrap_cell`, several `CellObject` accessors) return
            // it through an integer-tagged path that the rtyper colors
            // as `i`. Neither side is wrong — RPython's `lltype`
            // unification chooses `r` for the call descriptor while
            // pyre's coloring chooses `i` for the SSA value. `v` (void)
            // mismatches are also allowed for synthesized graphs. Any
            // OTHER mismatch (e.g. i ↔ f, r ↔ f) is still a bug.
            debug_assert!(
                declared_kind.is_none_or(|d| {
                    d == cfg_kind
                        || cfg_kind == 'v'
                        || (d == 'r' && cfg_kind == 'i')
                        || (d == 'i' && cfg_kind == 'r')
                }),
                "graph {} declared FUNC.RESULT={} but CFG return kind is {}",
                rewritten.graph.name,
                declared_kind.unwrap(),
                cfg_kind,
            );
            body.calldescr = crate::jitcode::BhCallDescr::from_arg_classes(
                arg_classes,
                result_type,
                majit_ir::descr::EffectInfo::MOST_GENERAL,
            );
        }

        // Commit the body to the pre-allocated `Arc<JitCode>` shell.
        // RPython mutates the JitCode in place; pyre uses `OnceLock`
        // so that shells handed out earlier (e.g. into
        // `JitDriverStaticData.mainjitcode` by `grab_initial_jitcodes`)
        // see the same body without locking.
        jitcode.set_body(body);

        // RPython `codewriter.py:68 jitcode.index = index` — assign the
        // dense position in `all_jitcodes[]` immediately after
        // `assembler.assemble(...)` (upstream line 67). The
        // `in_order[i].index == i` invariant asserted by
        // `CallControl::collect_jitcodes_in_alloc_order` (and mirrored in
        // `test_phase_f_all_jitcodes::all_jitcodes_indices_match_alloc_order`)
        // depends on the caller having passed `index = finished_jitcodes.len()`
        // BEFORE appending.
        jitcode.set_index(index);

        if self.debug {
            // RPython `codewriter.py:71-72` → `print_ssa_repr(ssarepr,
            // portal_jd, verbose)` → `log.dot()` (default, verbose=False)
            // or `print(format_assembler(ssarepr))` (verbose=True). Pyre
            // currently mirrors only the low-noise branch: one line per
            // jitcode with the name, analogous to upstream's udir
            // filename (`codewriter.py:122-125
            // dir.join(name+extra).write(format_assembler(ssarepr))`).
            // The `verbose` parameter is plumbed to match the upstream
            // signature; the high-verbosity branch lands when
            // `format_assembler` is ported.
            let _ = verbose;
            let _ = &ssarepr;
            // RPython `codewriter.py:72 log.dot()` is unconditional —
            // the only gate is the `CodeWriter.debug` instance flag
            // (`codewriter.py:18 debug = True`).  Keep that semantics:
            // a single `self.debug` gate, no additional `MAJIT_LOG`
            // dependency.
            eprintln!("[CodeWriter] {}", jitcode.name);
        }
    }

    /// RPython: `CodeWriter.make_jitcodes(verbose)` (codewriter.py:74-89).
    ///
    /// Full pipeline: grab_initial_jitcodes → enum_pending_graphs loop → finished.
    pub fn make_jitcodes(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) -> AllJitCodes {
        // RPython: self.callcontrol.grab_initial_jitcodes() (codewriter.py:76)
        callcontrol.grab_initial_jitcodes();
        self.make_jitcodes_pending(callcontrol, config)
    }

    /// Drain pending graphs and fill each `Arc<JitCode>` shell's body.
    ///
    /// RPython codewriter.py:79-84: the enum_pending_graphs loop. Pyre
    /// stores the allocated `Arc<JitCode>` shells inside
    /// `CallControl::jitcodes`; this loop pops one path at a time and
    /// commits its body via `JitCode::set_body`. The
    /// `all_jitcodes[i].index == i` invariant (RPython codewriter.py:80)
    /// is guaranteed by `CallControl::collect_jitcodes_in_alloc_order`.
    ///
    /// `verbose` threads through to `transform_graph_to_jitcode`'s debug
    /// branch; upstream `codewriter.py:74 make_jitcodes(verbose=False)`
    /// treats `False` as the default, matching pyre's call site in
    /// `make_jitcodes` which does not expose the knob to callers yet.
    pub fn drain_pending_graphs(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) {
        // RPython: for graph, jitcode in self.callcontrol.enum_pending_graphs():
        //            self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
        //
        // RPython's enum_pending_graphs() pops from unfinished_graphs (LIFO).
        // During transform, new graphs may be discovered and added via
        // get_jitcode(). We pop one at a time to match RPython's yield semantics.
        loop {
            let Some((path, jitcode)) = callcontrol.enum_pending_graphs() else {
                break;
            };
            let Some(graph) = callcontrol.function_graphs().get(&path).cloned() else {
                // RPython `enum_pending_graphs` (codewriter.py:79-84)
                // never yields a jitcode whose graph is missing —
                // `get_jitcode()` only allocates shells for paths that
                // already live in `function_graphs`. Phase I3 restored
                // this invariant for pyre by routing
                // `handle_regular_call` through the qualified
                // `CallControl::target_to_path` (jtransform.rs:970).
                // If this branch fires, a new producer has been added
                // that bypasses `target_to_path` or inserts under an
                // alias key. Producer-side bug, not an expected
                // runtime condition.
                panic!(
                    "drain_pending_graphs: jitcode shell has no matching graph — \
                     path={:?} idx={:?} name={:?}. Producer allocated a jitcode \
                     under a path that `function_graphs` does not contain. \
                     Fix the producer (prefer `CallControl::target_to_path`) \
                     instead of silently skipping.",
                    path.segments,
                    jitcode.try_index(),
                    jitcode.name,
                );
            };
            // RPython `codewriter.py:80` passes `index = len(all_jitcodes)`,
            // i.e. the slot this jitcode will occupy AFTER the append on
            // line 81. `finished_jitcodes_len()` is pyre's equivalent read.
            let index = callcontrol.finished_jitcodes_len();
            self.transform_graph_to_jitcode(
                &graph,
                &path,
                callcontrol,
                config,
                &jitcode,
                /* verbose = */ false,
                index,
            );
            // RPython `codewriter.py:81 all_jitcodes.append(jitcode)`.
            // `transform_graph_to_jitcode` already set `jitcode.index`.
            callcontrol.finish_jitcode(jitcode.clone());

            // RPython call.py:148: jd.mainjitcode.jitdriver_sd = jd
            for jd in callcontrol.jitdrivers_sd() {
                if jd.portal_graph == path {
                    jitcode.set_jitdriver_sd(jd.index);
                }
            }
        }
    }

    /// Process all pending graphs and finalize.
    ///
    /// RPython codewriter.py:79-85: enum_pending_graphs loop + finished.
    pub fn make_jitcodes_pending(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) -> AllJitCodes {
        self.drain_pending_graphs(callcontrol, config);
        self.assembler.finished(&callcontrol.callinfocollection);
        let in_order = callcontrol.collect_jitcodes_in_alloc_order();
        let by_path = callcontrol.jitcodes().clone();
        AllJitCodes { by_path, in_order }
    }
}

impl Default for CodeWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Codewriter-time producer for `CallTarget::Method.classdef_hint`.
///
/// Walks every `OpKind::Call { target: CallTarget::Method { .. }, .. }`
/// in `graph`, looks up the receiver's annotated flowspace `Variable`
/// via the dual-gate `value_to_var` map, extracts
/// `SomeInstance.classdef` (`model.py:1054`) when present, and stamps
/// the resulting `ClassDefKey` onto the op's `classdef_hint` field.
/// The corresponding `(ClassDefKey → shortname)` pair is registered on
/// `callcontrol` so [`crate::call::CallControl::resolve_method`] /
/// [`crate::call::CallControl::resolve_method_impl_type`] can take the
/// classdef-keyed fast path — mirroring upstream's
/// `bookkeeper.py:431-442 getmethoddesc` keying on the concrete
/// `ClassDef` reference rather than a source-syntax string.
///
/// Receivers whose annotation is `None` (annotator did not bind them)
/// or whose `SomeInstance` carries `classdef = None` (`object`-only
/// instances per `model.py:1056`) leave the hint untouched; the
/// existing receiver-root heuristic still runs for those sites until
/// slices C4-C6 retire it.
fn stamp_classdef_hints_on_graph(
    graph: &mut FunctionGraph,
    value_to_var: &crate::translator::rtyper::flowspace_adapter::SlotToVariable,
    callcontrol: &mut CallControl,
) {
    use crate::annotator::description::ClassDefKey;
    use crate::annotator::model::SomeValue;
    use crate::model::{CallTarget, OpKind};
    let mut stamps: Vec<(usize, usize, ClassDefKey, String)> = Vec::new();
    for (b_idx, block) in graph.blocks.iter().enumerate() {
        for (o_idx, op) in block.operations.iter().enumerate() {
            let OpKind::Call {
                target:
                    CallTarget::Method {
                        classdef_hint: None,
                        name: method_name,
                        ..
                    },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            let Some(receiver) = args.first() else {
                continue;
            };
            let Some(slot) = graph.slot_of(receiver) else {
                continue;
            };
            let Some(annotated_var) = value_to_var.get(&slot) else {
                continue;
            };
            let annotation = annotated_var.annotation.borrow();
            let Some(rc_someval) = annotation.as_ref() else {
                continue;
            };
            let SomeValue::Instance(inst) = &**rc_someval else {
                continue;
            };
            let Some(classdef_rc) = &inst.classdef else {
                continue;
            };
            let key = ClassDefKey::from_classdef(classdef_rc);
            // Prime the upstream-orthodox `bookkeeper.methoddescs`
            // cache by routing the receiver classdef + method name
            // through `Bookkeeper.getmethoddesc_for_attribute`, which
            // mirrors the regular-method branch of `bookkeeper.py:383-397
            // getdesc`. The returned MethodDesc rc is discarded — the
            // call is invoked for its cache-priming side-effect only,
            // so the upstream-orthodox MethodDesc identity is reachable
            // for any future PyPy-orthodox consumer that navigates the
            // receiver classdef's bookkeeper backlink.
            if let Some(bk) = classdef_rc.borrow().bookkeeper.upgrade() {
                let _ = bk.getmethoddesc_for_attribute(classdef_rc, method_name);
            }
            // Store the fully qualified `ClassDef.name` (RPython
            // `classdef.py:36 self.name = self.classdesc.name`,
            // `module.Class` form) rather than `shortname`. The leaf-only
            // spelling collapses distinct classdefs that share the same
            // unqualified name (`pkg1.C` vs `pkg2.C`) onto a single
            // side-table value, which can hand back the wrong registered
            // graph when `trait_method_graphs` carries multiple impls
            // under bare `C`. PRE-EXISTING-ADAPTATION on the
            // `classdef_impl_types` map itself (no upstream basis in
            // `bookkeeper.py`); retired when `trait_method_graphs` is
            // re-keyed on the upstream-orthodox `MethodDesc` identity
            // directly so dispatch consumes `bookkeeper.methoddescs`
            // primed above.
            let impl_type = classdef_rc.borrow().name.clone();
            stamps.push((b_idx, o_idx, key, impl_type));
        }
    }
    for (b_idx, o_idx, key, impl_type) in stamps {
        callcontrol.register_classdef_impl_type(key, impl_type);
        let op = &mut graph.blocks[b_idx].operations[o_idx];
        if let OpKind::Call {
            target: CallTarget::Method { classdef_hint, .. },
            ..
        } = &mut op.kind
        {
            *classdef_hint = Some(key);
        }
    }
}

/// Mirror of `FUNC.RESULT` in `rpython/jit/codewriter/call.py:181-187`.
///
/// Upstream reads `lltype.typeOf(fnptr).TO.RESULT` from the function
/// pointer type; the graph-level surface is
/// `flowspace/model.py:17-18` `graph.returnblock = Block([return_var])`,
/// where `return_var.concretetype` carries the same information.
/// Pyre reads `graph.concretetype(returnblock.inputargs[0])`,
/// which routes straight to the backing
/// [`crate::flowspace::model::Variable::concretetype`] cell stored
/// on the graph's `value_variables`.  The Variable IS the type source.
fn graph_result_kind(graph: &FunctionGraph) -> char {
    let returnblock = graph.block(graph.returnblock);
    let Some(arg) = returnblock.inputargs.first() else {
        return 'v';
    };
    use crate::model::ConcreteType;
    match FunctionGraph::concretetype_of(arg) {
        ConcreteType::Signed => 'i',
        ConcreteType::GcRef => 'r',
        ConcreteType::Float => 'f',
        ConcreteType::Void => 'v',
        ConcreteType::Unknown => 'v',
    }
}

#[cfg(test)]
mod stamp_classdef_hints_tests {
    use super::*;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::classdesc::{ClassDef, ClassDesc};
    use crate::annotator::description::ClassDefKey;
    use crate::annotator::model::{SomeInstance, SomeValue};
    use crate::flowspace::model::{HostObject, Variable};
    use crate::model::{CallTarget, OpKind, ValueType};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn make_classdef(name: &str) -> Rc<RefCell<ClassDef>> {
        let bk = Rc::new(Bookkeeper::new());
        let pyobj = HostObject::new_class(name, vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(&bk, pyobj, name.into())));
        ClassDef::new(&bk, &desc)
    }

    /// End-to-end producer test: a Call(Method) op whose receiver
    /// carries an annotated `SomeInstance(classdef=PyFrame)` gets a
    /// `classdef_hint` stamped, and the `(key → "PyFrame")` binding is
    /// registered into `CallControl.classdef_impl_types` so the
    /// resolver can take the classdef-keyed fast path. Mirrors
    /// `bookkeeper.py:431-442 getmethoddesc` keying on concrete
    /// classdef rather than receiver source-syntax string.
    #[test]
    fn stamp_classdef_hints_on_graph_stamps_hint_and_registers_impl_type() {
        let classdef = make_classdef("PyFrame");
        let expected_key = ClassDefKey::from_classdef(&classdef);

        // Synthetic flowspace Variable annotated SomeInstance(PyFrame).
        let recv_var = Variable::new();
        *recv_var.annotation.borrow_mut() = Some(Rc::new(SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            Default::default(),
        ))));

        let mut graph = FunctionGraph::new("producer_test");
        graph.ensure_variable_registered_void(&recv_var);
        let recv_slot = graph.slot_of(&recv_var).expect("just registered above");
        let _result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("push_value", Some("H".to_string())),
                    args: vec![recv_var.clone()],
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .expect("push_op_var should succeed");
        let mut value_to_var = std::collections::HashMap::new();
        value_to_var.insert(recv_slot, recv_var.clone());

        let mut callcontrol = CallControl::new();
        stamp_classdef_hints_on_graph(&mut graph, &value_to_var, &mut callcontrol);

        // Op's classdef_hint must be Some(expected_key).
        let op = &graph.blocks[graph.startblock.0].operations[0];
        let OpKind::Call { target, .. } = &op.kind else {
            panic!("expected Call op");
        };
        assert_eq!(target.classdef_hint(), Some(expected_key));

        // Side-table must have the (key → "PyFrame") binding so the
        // resolver fast path resolves to the registered impl.
        assert_eq!(
            callcontrol.classdef_impl_type_for_test(expected_key),
            Some("PyFrame"),
        );
    }

    /// Receivers whose annotation is `None` (annotator did not bind
    /// them) leave the hint untouched and do not poison the
    /// side-table. The string-keyed heuristic still runs for those
    /// sites until slices C4-C6 retire it.
    #[test]
    fn stamp_classdef_hints_on_graph_skips_unannotated_receiver() {
        let recv_var = Variable::new();
        // No annotation bound.
        let mut graph = FunctionGraph::new("producer_test");
        graph.ensure_variable_registered_void(&recv_var);
        let recv_slot = graph.slot_of(&recv_var).expect("just registered above");
        graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("push_value", Some("H".to_string())),
                    args: vec![recv_var.clone()],
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .expect("push_op_var should succeed");
        let mut value_to_var = std::collections::HashMap::new();
        value_to_var.insert(recv_slot, recv_var.clone());

        let mut callcontrol = CallControl::new();
        stamp_classdef_hints_on_graph(&mut graph, &value_to_var, &mut callcontrol);

        let op = &graph.blocks[graph.startblock.0].operations[0];
        let OpKind::Call { target, .. } = &op.kind else {
            panic!("expected Call op");
        };
        assert_eq!(target.classdef_hint(), None);
        // No registration must have happened either.
        assert_eq!(
            callcontrol.classdef_impl_type_for_test(ClassDefKey::from_raw(0)),
            None,
        );
    }

    /// When the receiver classdef's `attrs[method_name].s_value` carries
    /// a `SomePBC` with a `MethodDesc`, the producer routes the receiver
    /// classdef + method name through `Bookkeeper.getmethoddesc_for_attribute`
    /// so the upstream-orthodox `bookkeeper.methoddescs` cache (per
    /// `bookkeeper.py:431-442 getmethoddesc`) is populated. The cache
    /// is the real PyPy structure; pyre stores no separate
    /// MethodDescKey side-table on CallControl.
    #[test]
    fn stamp_classdef_hints_on_graph_primes_bookkeeper_methoddescs() {
        use crate::annotator::bookkeeper::MethodDescKey;
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::{DescEntry, DescKey, FunctionDesc};
        use crate::annotator::model::SomePBC;
        use crate::flowspace::argument::Signature;

        let bk = Rc::new(Bookkeeper::new());
        let pyobj = HostObject::new_class("PyFrame", vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            pyobj,
            "PyFrame".into(),
        )));
        let classdef = ClassDef::new(&bk, &desc);
        let classdef_key = ClassDefKey::from_classdef(&classdef);

        // Mint an upstream-shaped MethodDesc via bookkeeper.getmethoddesc
        // and bind it as a Method PBC under `attrs[push_value]`.
        let funcdesc = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            "push_value",
            Signature::new(vec!["self".into(), "v".into()], None, None),
            None,
            None,
        )));
        let md = bk.getmethoddesc(
            &funcdesc,
            classdef_key,
            Some(classdef_key),
            "push_value",
            std::collections::BTreeMap::new(),
        );
        let pbc = SomePBC::new([DescEntry::Method(md.clone())], false);
        let mut attr = Attribute::new("push_value");
        attr.s_value = SomeValue::PBC(pbc);
        classdef
            .borrow_mut()
            .attrs
            .insert("push_value".into(), attr);

        // Receiver Variable annotated SomeInstance(PyFrame).
        let recv_var = Variable::new();
        *recv_var.annotation.borrow_mut() = Some(Rc::new(SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            Default::default(),
        ))));

        let mut graph = FunctionGraph::new("producer_test");
        graph.ensure_variable_registered_void(&recv_var);
        let recv_slot = graph.slot_of(&recv_var).expect("just registered above");
        graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("push_value", Some("H".to_string())),
                    args: vec![recv_var.clone()],
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .expect("push_op_var should succeed");
        let mut value_to_var = std::collections::HashMap::new();
        value_to_var.insert(recv_slot, recv_var.clone());

        let mut callcontrol = CallControl::new();
        stamp_classdef_hints_on_graph(&mut graph, &value_to_var, &mut callcontrol);

        // Producer primed bookkeeper.methoddescs (the upstream-orthodox
        // cache) so the MethodDescKey shape resolves to the same
        // MethodDesc Rc through the bookkeeper backlink. PyPy's actual
        // structure — no separate pyre side-table.
        let md_borrow = md.borrow();
        let key = MethodDescKey {
            funcdesc_id: DescKey::from_rc(&md_borrow.funcdesc),
            originclassdef: md_borrow.originclassdef,
            selfclassdef: md_borrow.selfclassdef,
            name: md_borrow.name.clone(),
            flags: md_borrow
                .flags
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        };
        let cached_md = bk
            .methoddescs
            .borrow()
            .get(&key)
            .cloned()
            .expect("producer should prime bookkeeper.methoddescs");
        assert!(
            Rc::ptr_eq(&cached_md, &md),
            "primed entry should be the same MethodDesc rc as the originally-minted one",
        );

        // The legacy `classdef_impl_types` side-table still carries the
        // cached qualified `classdef.name` for the existing
        // string-keyed dispatch. PRE-EXISTING-ADAPTATION; retired when
        // `trait_method_graphs` is re-keyed on MethodDesc identity.
        assert_eq!(
            callcontrol.classdef_impl_type_for_test(classdef_key),
            Some("PyFrame"),
        );
    }
}

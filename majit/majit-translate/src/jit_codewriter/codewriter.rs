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

use crate::assembler::Assembler;
use crate::call::CallControl;
use crate::flatten::{RegKind, SSARepr, flatten_with_types};
use crate::jitcode::JitCode;
use crate::jtransform::GraphTransformConfig;
use crate::model::{FunctionGraph, Terminator, ValueId};

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
}

impl CodeWriter {
    /// RPython: `CodeWriter.__init__(cpu, jitdrivers_sd)` (codewriter.py:20-23).
    pub fn new() -> Self {
        Self {
            assembler: Assembler::new(),
            debug: false,
        }
    }

    /// RPython: `CodeWriter.transform_graph_to_jitcode()` (codewriter.py:33-72).
    ///
    /// Transforms a FunctionGraph into a JitCode through the 4-step pipeline.
    /// Returns both the flattened SSARepr (with liveness) and the assembled JitCode.
    ///
    /// Steps:
    ///   0. annotate + rtype (majit-specific; RPython does this before codewriter)
    ///   1. jtransform — `transform_graph()` (codewriter.py:42)
    ///   2. regalloc — `perform_register_allocation()` per kind (codewriter.py:45-47)
    ///   3. flatten — `flatten_graph()` (codewriter.py:53)
    ///   3b. liveness — `compute_liveness()` (codewriter.py:56, called inside assemble)
    ///   4. assemble — `assembler.assemble()` (codewriter.py:67)
    pub fn transform_graph_to_jitcode(
        &mut self,
        graph: &FunctionGraph,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
        jitcode: &std::sync::Arc<JitCode>,
    ) -> SSARepr {
        // RPython: graph = copygraph(graph, shallowvars=True) (codewriter.py:38)
        // In Rust, Transformer.transform() already clones the graph.

        // Step 0: annotate + rtype (majit-specific)
        // RPython: types are already on Variable.concretetype from the rtyper.
        let annotations = crate::translate_legacy::annotator::annrpython::annotate(graph);
        let mut type_state =
            crate::translate_legacy::rtyper::rtyper::resolve_types(graph, &annotations);

        // Step 0b: rtyper-equivalent indirect_call lowering
        // (`translator/rtyper/rpbc.rs::lower_indirect_calls`).
        // RPython rpbc.py:199-217 emits `indirect_call(funcptr, *args,
        // c_graphs)` during rtype; pyre runs the same pass here so
        // jtransform sees `OpKind::IndirectCall` (with funcptr already
        // a regular ValueId), never `CallTarget::Indirect`.
        let mut graph_owned = graph.clone();
        crate::translator::rtyper::rpbc::lower_indirect_calls(
            &mut graph_owned,
            &mut type_state,
            callcontrol,
        );
        #[cfg(debug_assertions)]
        crate::translator::rtyper::rpbc::assert_no_indirect_call_targets(&graph_owned);
        let graph = &graph_owned;

        // Step 1: jtransform (codewriter.py:42)
        // RPython: transform_graph(graph, cpu, callcontrol, portal_jd)
        let rewritten = {
            let mut transformer = crate::jtransform::Transformer::new(config)
                .with_callcontrol(callcontrol)
                .with_type_state(&type_state);
            transformer.transform(graph)
        };
        // Transformer is dropped here, releasing the &mut CallControl borrow.

        // Re-resolve types on the post-jtransform graph: jtransform synthesizes
        // fresh ValueIds (e.g. ConstInt funcptrs from `direct_funcptr_value`)
        // that the original `type_state` never saw. The rerun supplies kinds
        // for those new IDs so `build_value_kinds` covers every operand
        // reaching regalloc/flatten. RPython has no analogue because its
        // jtransform preserves Variable.concretetype on every newly created
        // Variable; pyre's side-table needs the rebuild.
        let rewritten_type_state =
            crate::translate_legacy::rtyper::rtyper::resolve_types(&rewritten.graph, &annotations);

        // Step 2: regalloc (codewriter.py:45-47)
        // RPython: for kind in KINDS: regallocs[kind] = perform_register_allocation(graph, kind)
        let value_kinds =
            crate::translate_legacy::rtyper::rtyper::build_value_kinds(&rewritten_type_state);
        let regallocs =
            crate::regalloc::perform_all_register_allocations(&rewritten.graph, &value_kinds);

        // Step 3: flatten (codewriter.py:53)
        // RPython: ssarepr = flatten_graph(graph, regallocs, cpu=cpu)
        let mut ssarepr = flatten_with_types(&rewritten.graph, &rewritten_type_state);

        // Step 3b + 4: liveness + assemble (codewriter.py:56,67)
        // RPython: compute_liveness(ssarepr) then assembler.assemble(ssarepr, jitcode, num_regs)
        // In majit, assemble() calls compute_liveness() internally and now
        // returns the body so the codewriter can fill calldescr before
        // committing the shell via `set_body`.
        let mut body = self.assembler.assemble(&mut ssarepr, &regallocs);

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
            for arg_id in &start_block.inputargs {
                match ssarepr.value_kinds.get(arg_id) {
                    Some(RegKind::Int) => arg_classes.push('i'),
                    Some(RegKind::Ref) => arg_classes.push('r'),
                    Some(RegKind::Float) => arg_classes.push('f'),
                    None => arg_classes.push('i'),
                }
            }
            let cfg_kind = graph_result_kind(&rewritten.graph, &ssarepr.value_kinds);
            let declared_kind = callcontrol
                .path_for_jitcode_index(jitcode.index)
                .and_then(|p| callcontrol.declared_return_kind(p));
            let result_type = declared_kind.unwrap_or(cfg_kind);
            // Cross-check: when both sources are present they must agree.
            // Synthesized graphs from tests / inline arms may legitimately
            // omit the registration; ignore the assert in that case.
            debug_assert!(
                declared_kind.is_none_or(|d| d == cfg_kind || cfg_kind == 'v'),
                "graph {} declared FUNC.RESULT={} but CFG return kind is {}",
                rewritten.graph.name,
                declared_kind.unwrap(),
                cfg_kind,
            );
            body.calldescr = crate::jitcode::BhCallDescr {
                arg_classes,
                result_type,
            };
        }

        // Commit the body to the pre-allocated `Arc<JitCode>` shell.
        // RPython mutates the JitCode in place; pyre uses `OnceLock`
        // so that shells handed out earlier (e.g. into
        // `JitDriverStaticData.mainjitcode` by `grab_initial_jitcodes`)
        // see the same body without locking.
        jitcode.set_body(body);

        if self.debug {
            eprintln!(
                "[CodeWriter] {} → {} ops, {} bytes, regs i={} r={} f={}",
                jitcode.name,
                ssarepr.insns.len(),
                jitcode.code.len(),
                jitcode.num_regs_i(),
                jitcode.num_regs_r(),
                jitcode.num_regs_f(),
            );
        }

        ssarepr
    }

    /// RPython: `CodeWriter.make_jitcodes(verbose)` (codewriter.py:74-89).
    ///
    /// Full pipeline: grab_initial_jitcodes → enum_pending_graphs loop → finished.
    pub fn make_jitcodes(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) -> Vec<std::sync::Arc<JitCode>> {
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
                continue;
            };
            let _ssarepr = self.transform_graph_to_jitcode(&graph, callcontrol, config, &jitcode);

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
    ) -> Vec<std::sync::Arc<JitCode>> {
        self.drain_pending_graphs(callcontrol, config);
        self.assembler.finished(&callcontrol.callinfocollection);
        callcontrol.collect_jitcodes_in_alloc_order()
    }
}

impl Default for CodeWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Mirror of `FUNC.RESULT` in `rpython/jit/codewriter/call.py:182-187`.
///
/// Walks every `Terminator::Return(Some(_))` in the graph and reports the
/// kind char of the returned value. Returns `'v'` when every terminator is
/// `Return(None)` (or there are no `Return` terminators at all). Panics
/// when two `Return` terminators disagree — RPython's `FUNC.RESULT` is a
/// single type per graph, so a conflict means an upstream pass produced a
/// graph that no `BhCallDescr` can describe.
fn graph_result_kind(
    graph: &FunctionGraph,
    value_kinds: &std::collections::HashMap<ValueId, RegKind>,
) -> char {
    let mut found: Option<char> = None;
    for block in &graph.blocks {
        if let Terminator::Return(Some(vid)) = &block.terminator {
            let kind = match value_kinds.get(vid) {
                Some(RegKind::Int) => 'i',
                Some(RegKind::Ref) => 'r',
                Some(RegKind::Float) => 'f',
                None => 'i',
            };
            match found {
                None => found = Some(kind),
                Some(prev) if prev == kind => {}
                Some(prev) => panic!(
                    "graph {} has inconsistent return kinds: {prev} vs {kind}",
                    graph.name
                ),
            }
        }
    }
    found.unwrap_or('v')
}

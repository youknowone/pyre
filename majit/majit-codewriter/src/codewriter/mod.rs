//! Code generation pipeline — majit's equivalent of `rpython/jit/codewriter/`.
//!
//! ```text
//! rpython/jit/codewriter/          majit-codewriter/src/codewriter/
//! ├── codewriter.py          →     ├── mod.rs (CodeWriter struct)
//! ├── jtransform.py          →     ├── (passes/jtransform.rs)
//! ├── flatten.py + assembler.py →  ├── codegen.rs
//! └── call.py                →     └── (call.rs)
//! ```

pub mod codegen;

pub use codegen::{
    BinopMapping, CodegenValueKind, IoShim, JitDriverConfig, StorageConfig,
    VirtualizableCodegenConfig, generate_jitcode,
};

use crate::assembler::{Assembler, JitCode};
use crate::call::CallControl;
use crate::model::FunctionGraph;
use crate::passes::GraphTransformConfig;
use crate::passes::flatten::SSARepr;

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
        index: usize,
    ) -> (SSARepr, JitCode) {
        // RPython: graph = copygraph(graph, shallowvars=True) (codewriter.py:38)
        // In Rust, Transformer.transform() already clones the graph.

        // Step 0: annotate + rtype (majit-specific)
        // RPython: types are already on Variable.concretetype from the rtyper.
        let annotations = crate::passes::annotate::annotate(graph);
        let type_state = crate::passes::rtype::resolve_types(graph, &annotations);

        // Step 1: jtransform (codewriter.py:42)
        // RPython: transform_graph(graph, cpu, callcontrol, portal_jd)
        let rewritten = {
            let mut transformer = crate::passes::Transformer::new(config)
                .with_callcontrol(callcontrol)
                .with_type_state(&type_state);
            transformer.transform(graph)
        };
        // Transformer is dropped here, releasing the &mut CallControl borrow.

        // Step 2: regalloc (codewriter.py:45-47)
        // RPython: for kind in KINDS: regallocs[kind] = perform_register_allocation(graph, kind)
        let value_kinds = crate::passes::rtype::build_value_kinds(&type_state);
        let regallocs =
            crate::regalloc::perform_all_register_allocations(&rewritten.graph, &value_kinds);

        // Step 3: flatten (codewriter.py:53)
        // RPython: ssarepr = flatten_graph(graph, regallocs, cpu=cpu)
        let mut ssarepr = crate::passes::flatten::flatten_with_types(&rewritten.graph, &type_state);

        // Step 3b + 4: liveness + assemble (codewriter.py:56,67)
        // RPython: compute_liveness(ssarepr) then assembler.assemble(ssarepr, jitcode, num_regs)
        // In majit, assemble() calls compute_liveness() internally.
        let mut jitcode = self.assembler.assemble(&mut ssarepr, &regallocs);
        // RPython: jitcode.index = index (codewriter.py:68)
        jitcode.index = index;

        if self.debug {
            eprintln!(
                "[CodeWriter] {} → {} ops, {} bytes, regs i={} r={} f={}",
                jitcode.name,
                ssarepr.insns.len(),
                jitcode.code.len(),
                jitcode.num_regs_i,
                jitcode.num_regs_r,
                jitcode.num_regs_f,
            );
        }

        (ssarepr, jitcode)
    }

    /// RPython: `CodeWriter.make_jitcodes(verbose)` (codewriter.py:74-89).
    ///
    /// Full pipeline: grab_initial_jitcodes → enum_pending_graphs loop → finished.
    pub fn make_jitcodes(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) -> Vec<JitCode> {
        // RPython: self.callcontrol.grab_initial_jitcodes() (codewriter.py:76)
        callcontrol.grab_initial_jitcodes();
        self.make_jitcodes_pending(callcontrol, config)
    }

    /// Drain pending graphs and transform each into a JitCode.
    ///
    /// RPython codewriter.py:79-84: the enum_pending_graphs loop.
    ///
    /// Places each JitCode at `all_jitcodes[jitcode.index]` directly,
    /// guaranteeing the RPython invariant `all_jitcodes[i].index == i`
    /// without post-hoc sorting.
    ///
    /// RPython achieves this naturally because `jitcode.index = len(all_jitcodes)`
    /// at append time (codewriter.py:80). majit uses `get_jitcode()` allocation
    /// indices instead, so we place by index rather than append.
    pub fn drain_pending_graphs(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
        all_jitcodes: &mut Vec<Option<JitCode>>,
    ) {
        // RPython: for graph, jitcode in self.callcontrol.enum_pending_graphs():
        //            self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
        //
        // RPython's enum_pending_graphs() pops from unfinished_graphs (LIFO).
        // During transform, new graphs may be discovered and added via
        // get_jitcode(). We pop one at a time to match RPython's yield semantics.
        loop {
            let Some((path, alloc_index)) = callcontrol.pop_one_graph() else {
                break;
            };
            let Some(graph) = callcontrol.function_graphs().get(&path).cloned() else {
                continue;
            };
            let (_ssarepr, mut jitcode) =
                self.transform_graph_to_jitcode(&graph, callcontrol, config, alloc_index);

            // RPython call.py:148: jd.mainjitcode.jitdriver_sd = jd
            for jd in callcontrol.jitdrivers_sd() {
                if jd.portal_graph == path {
                    jitcode.jitdriver_sd = Some(jd.index);
                }
            }

            // Place at the correct index position.
            // RPython: all_jitcodes[jitcode.index] == jitcode
            if alloc_index >= all_jitcodes.len() {
                all_jitcodes.resize_with(alloc_index + 1, || None);
            }
            all_jitcodes[alloc_index] = Some(jitcode);
        }
    }

    /// Process all pending graphs and finalize.
    ///
    /// RPython codewriter.py:79-85: enum_pending_graphs loop + finished.
    pub fn make_jitcodes_pending(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) -> Vec<JitCode> {
        let mut all_jitcodes: Vec<Option<JitCode>> = Vec::new();
        self.drain_pending_graphs(callcontrol, config, &mut all_jitcodes);
        self.assembler.finished(&callcontrol.callinfocollection);
        // Unwrap: every slot must be filled.
        all_jitcodes.into_iter().flatten().collect()
    }
}

impl Default for CodeWriter {
    fn default() -> Self {
        Self::new()
    }
}

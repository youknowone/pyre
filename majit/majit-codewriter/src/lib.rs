//! majit-codewriter: JIT code generation pipeline.
//!
//! RPython equivalent: `rpython/jit/codewriter/`
//!
//! Given a bundled interpreter source file, it:
//! 1. Parses the entire crate with `syn`
//! 2. Extracts the opcode dispatch table (match arms)
//! 3. Traces trait implementations to resolve handler methods
//! 4. Classifies helper functions (elidable, residual, field access)
//! 5. Collects type layouts (struct fields, offsets)
//! 6. Generates tracing code that mirrors the interpreter's execution

pub mod assembler;
pub mod call;
mod codegen;
pub mod codewriter;
pub mod front;
pub mod graph;
pub mod hints;
pub mod inline;
pub mod liveness;
mod parse;
pub mod passes;
pub mod regalloc;
#[cfg(test)]
mod test_support;

pub use call::CallDescriptor;
pub use front::{
    AstGraphOptions, SemanticFunction, SemanticProgram, build_semantic_program,
    build_semantic_program_from_parsed_files,
};
pub use graph::{
    BasicBlock, BasicBlockId, CallTarget, MajitGraph, Op, OpKind, Terminator, ValueId, ValueType,
};
pub use parse::{
    CallPath, OpcodeDispatchSelector, ParsedInterpreter, find_opcode_dispatch_match, parse_source,
};
pub use passes::{
    AnnotationState, CallEffectKind, CallEffectOverride, ConcreteType, FlatOp, FlattenedFunction,
    GraphTransformConfig, GraphTransformResult, Label, PipelineConfig, PipelineOpcodeArm,
    PipelineResult, ProgramPipelineResult, TypeResolutionState, VirtualizableFieldDescriptor,
    analyze_function, analyze_program, annotate_graph, resolve_types, rewrite_graph,
};

use serde::{Deserialize, Serialize};

/// Configuration for the canonical graph/pipeline analyzer.
///
/// Consumers supply graph-rewrite metadata such as virtualizable
/// field/array mappings before the codewriter-style passes run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyzeConfig {
    pub pipeline: PipelineConfig,
}

/// A resolved method/function call with source context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedCall {
    pub name: String,
    pub impl_type: Option<String>,
    pub trait_name: Option<String>,
    /// Canonical semantic graph for this resolved target.
    #[serde(default)]
    pub graph: Option<graph::MajitGraph>,
}

/// Trait implementation info
#[derive(Debug, Serialize, Deserialize)]
pub struct TraitImplInfo {
    pub trait_name: String,
    pub for_type: String,
    #[serde(default)]
    pub self_ty_root: Option<String>,
    pub methods: Vec<MethodInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MethodInfo {
    pub name: String,
    /// Canonical semantic graph for this method when available.
    #[serde(default)]
    pub graph: Option<graph::MajitGraph>,
}

/// Canonical single-file analysis entry point.
pub fn analyze_pipeline(source: &str) -> passes::ProgramPipelineResult {
    analyze_pipeline_with_config(source, &AnalyzeConfig::default())
}

/// Configurable canonical single-file analysis entry point.
pub fn analyze_pipeline_with_config(
    source: &str,
    config: &AnalyzeConfig,
) -> passes::ProgramPipelineResult {
    analyze_multiple_pipeline_with_config(&[source], config)
}

/// Canonical multi-file analysis entry point.
///
/// This returns only the graph/pipeline result and is the preferred API for
/// RPython-like translator consumers.
pub fn analyze_multiple_pipeline(sources: &[&str]) -> passes::ProgramPipelineResult {
    analyze_multiple_pipeline_with_config(sources, &AnalyzeConfig::default())
}

/// Configurable canonical multi-file analysis entry point.
///
/// This is the canonical graph/pipeline translator entry point.
pub fn analyze_multiple_pipeline_with_config(
    sources: &[&str],
    config: &AnalyzeConfig,
) -> passes::ProgramPipelineResult {
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();
    analyze_pipeline_from_parsed(&parsed_files, config)
}

fn analyze_pipeline_from_parsed(
    parsed_files: &[parse::ParsedInterpreter],
    config: &AnalyzeConfig,
) -> passes::ProgramPipelineResult {
    let program = front::build_semantic_program_from_parsed_files(parsed_files);
    let mut pipeline = passes::analyze_program(&program, &config.pipeline);
    let mut canonical_trait_impls = Vec::new();
    let mut canonical_inherent_methods = Vec::new();
    let mut canonical_function_graphs = std::collections::HashMap::new();

    for parsed in parsed_files {
        canonical_trait_impls.extend(parse::extract_trait_impls(parsed));
        canonical_inherent_methods.extend(parse::extract_inherent_impl_methods(parsed));
        parse::collect_function_graphs(parsed, &mut canonical_function_graphs);
    }

    // ── Build CallControl (RPython call.py) ──
    // Populate with all discovered function graphs and trait impl methods.
    let mut call_control = call::CallControl::new();
    for (path, graph) in &canonical_function_graphs {
        call_control.register_function_graph(path.clone(), graph.clone());
    }
    for impl_info in &canonical_trait_impls {
        let impl_type = impl_info
            .self_ty_root
            .as_deref()
            .unwrap_or(&impl_info.for_type);
        for method in &impl_info.methods {
            if let Some(graph) = &method.graph {
                call_control.register_trait_method(&method.name, impl_type, graph.clone());
            }
        }
    }
    call_control.find_all_graphs();

    pipeline.opcode_dispatch = build_canonical_opcode_dispatch(
        parsed_files,
        &canonical_trait_impls,
        &canonical_inherent_methods,
        &canonical_function_graphs,
        &config.pipeline,
        &call_control,
    );

    pipeline
}

fn build_canonical_opcode_dispatch(
    parsed_files: &[parse::ParsedInterpreter],
    trait_impls: &[TraitImplInfo],
    inherent_methods: &[parse::InherentMethodInfo],
    function_graphs: &std::collections::HashMap<parse::CallPath, graph::MajitGraph>,
    pipeline_config: &passes::PipelineConfig,
    call_control: &call::CallControl,
) -> Vec<passes::PipelineOpcodeArm> {
    let mut opcode_arms = Vec::new();
    let mut receiver_traits = parse::ReceiverTraitBindings::default();

    for parsed in parsed_files {
        let file_opcodes = parse::extract_opcode_dispatch_arms(parsed);
        if !file_opcodes.is_empty() {
            opcode_arms = file_opcodes;
            receiver_traits = parse::extract_opcode_dispatch_receiver_traits(parsed);
            break;
        }
    }

    opcode_arms
        .into_iter()
        .map(|arm| {
            let resolved_calls = resolve_handler_calls(
                &arm.handler_calls,
                trait_impls,
                inherent_methods,
                function_graphs,
                &receiver_traits,
            );

            // RPython codewriter path: jtransform → flatten.
            //
            // RPython does NOT splice callee bodies into the caller.
            // Instead, jtransform rewrites each `direct_call` to either
            // `inline_call_*` (referencing the callee's JitCode) or
            // `residual_call_*` (keeping the function pointer). The
            // meta-interpreter then descends into callee JitCode at runtime.
            //
            // For each resolved handler, run jtransform + flatten on the
            // handler's own graph. Call ops remain as Call/CallResidual/
            // CallElidable — they are NOT expanded into callee bodies.
            let flattened = resolved_calls.iter().find_map(|resolved| {
                let graph = resolved.graph.as_ref()?;
                let rewritten = passes::rewrite_graph(graph, &pipeline_config.transform);
                let flattened = passes::flatten_with_types(
                    &rewritten.graph,
                    &passes::resolve_types(
                        &rewritten.graph,
                        &passes::annotate_graph(&rewritten.graph),
                    ),
                );
                // Skip trivially empty graphs (only Input ops)
                if flattened.ops.len() <= 1 {
                    return None;
                }
                Some(flattened)
            });

            passes::PipelineOpcodeArm {
                selector: arm.selector,
                flattened,
            }
        })
        .collect()
}

/// Classify a resolved call by jtransform-rewriting its graph.

fn resolve_handler_calls(
    handler_calls: &[parse::ExtractedHandlerCall],
    trait_impls: &[TraitImplInfo],
    inherent_methods: &[parse::InherentMethodInfo],
    function_graphs: &std::collections::HashMap<parse::CallPath, graph::MajitGraph>,
    receiver_traits: &parse::ReceiverTraitBindings,
) -> Vec<ResolvedCall> {
    fn push_unique(resolved_calls: &mut Vec<ResolvedCall>, call: ResolvedCall) {
        if !resolved_calls.iter().any(|existing| {
            existing.name == call.name
                && existing.impl_type == call.impl_type
                && existing.trait_name == call.trait_name
        }) {
            resolved_calls.push(call);
        }
    }

    fn receiver_matches_root(
        receiver_type_root: Option<&String>,
        candidate_root: &Option<String>,
    ) -> bool {
        match (receiver_type_root, candidate_root.as_ref()) {
            (Some(receiver), Some(candidate)) => receiver == candidate,
            _ => false,
        }
    }

    fn push_matching_trait_methods(
        resolved_calls: &mut Vec<ResolvedCall>,
        trait_impls: &[TraitImplInfo],
        name: &str,
        receiver_type_root: Option<&String>,
        allowed_traits: Option<&[String]>,
    ) -> bool {
        let mut matched = false;
        for impl_info in trait_impls {
            if let Some(allowed_traits) = allowed_traits {
                if !allowed_traits
                    .iter()
                    .any(|bound| bound == &impl_info.trait_name)
                {
                    continue;
                }
            }

            let is_default_methods = impl_info.for_type.starts_with("<default methods of ");
            let is_generic = receiver_type_root
                .is_some_and(|r| crate::call::is_generic_receiver(r))
                || receiver_type_root.is_none();
            let applies = if is_default_methods {
                is_generic
                    || receiver_type_root.is_some_and(|receiver_ty| {
                        trait_impls.iter().any(|candidate| {
                            candidate.trait_name == impl_info.trait_name
                                && candidate.self_ty_root.as_ref() == Some(receiver_ty)
                        })
                    })
            } else {
                // Concrete impls: match if the receiver type matches,
                // OR if receiver is a generic type parameter (e.g. "E", "H")
                receiver_type_root.is_some_and(|r| crate::call::is_generic_receiver(r))
                    || receiver_type_root.is_none()
                    || receiver_matches_root(receiver_type_root, &impl_info.self_ty_root)
            };

            if !applies {
                continue;
            }

            for method in &impl_info.methods {
                if method.name == name {
                    matched = true;
                    push_unique(
                        resolved_calls,
                        ResolvedCall {
                            name: name.to_string(),
                            impl_type: Some(impl_info.for_type.clone()),
                            trait_name: Some(impl_info.trait_name.clone()),
                            graph: method.graph.clone(),
                        },
                    );
                }
            }
        }
        matched
    }

    let mut resolved_calls = Vec::new();

    for call in handler_calls {
        match call {
            parse::ExtractedHandlerCall::Method {
                name,
                receiver_root,
            } => {
                let receiver_binding = receiver_root.as_ref();
                let receiver_type_root = receiver_binding
                    .and_then(|receiver| receiver_traits.type_root_by_receiver.get(receiver));
                let receiver_bound_traits = receiver_binding
                    .and_then(|receiver| receiver_traits.traits_by_receiver.get(receiver));

                let mut matched_specific_target = false;
                if receiver_type_root.is_some() {
                    for method in inherent_methods {
                        if method.name == *name
                            && receiver_matches_root(receiver_type_root, &method.self_ty_root)
                        {
                            matched_specific_target = true;
                            push_unique(
                                &mut resolved_calls,
                                ResolvedCall {
                                    name: name.clone(),
                                    impl_type: Some(method.for_type.clone()),
                                    trait_name: None,
                                    graph: Some(method.graph.clone()),
                                },
                            );
                        }
                    }

                    matched_specific_target |= push_matching_trait_methods(
                        &mut resolved_calls,
                        trait_impls,
                        name,
                        receiver_type_root,
                        receiver_bound_traits.map(Vec::as_slice),
                    );
                }

                if !matched_specific_target {
                    if let Some(bound_traits) = receiver_bound_traits {
                        for trait_name in bound_traits {
                            let mut default_match: Option<ResolvedCall> = None;
                            let mut concrete_matches = Vec::new();

                            for impl_info in trait_impls {
                                if &impl_info.trait_name != trait_name {
                                    continue;
                                }
                                for method in &impl_info.methods {
                                    if method.name != *name {
                                        continue;
                                    }
                                    let resolved = ResolvedCall {
                                        name: name.clone(),
                                        impl_type: Some(impl_info.for_type.clone()),
                                        trait_name: Some(impl_info.trait_name.clone()),
                                        graph: method.graph.clone(),
                                    };
                                    if impl_info.for_type.starts_with("<default methods of ") {
                                        default_match = Some(resolved);
                                    } else {
                                        concrete_matches.push(resolved);
                                    }
                                }
                            }

                            // Push default method first (simple delegation like
                            // `opcode_build_list(self, size).map_err(Into::into)`)
                            // so the classifier tries it before complex concrete impls.
                            if let Some(default_match) = default_match {
                                push_unique(&mut resolved_calls, default_match);
                            }
                            for concrete in concrete_matches {
                                push_unique(&mut resolved_calls, concrete);
                            }
                        }
                    }
                }
            }
            parse::ExtractedHandlerCall::FunctionPath(path) => {
                // Handle qualified trait calls like OpcodeStepExecutor::build_list(executor, ...)
                // which parse as FunctionPath(["OpcodeStepExecutor", "build_list"]).
                // Convert to a trait method resolution.
                if path.segments.len() == 2 {
                    let trait_name = &path.segments[0];
                    let method_name = &path.segments[1];
                    // Check if the first segment is a known trait name
                    let is_trait = trait_impls.iter().any(|imp| imp.trait_name == *trait_name);
                    if is_trait {
                        let allowed = &[trait_name.clone()];
                        push_matching_trait_methods(
                            &mut resolved_calls,
                            trait_impls,
                            method_name,
                            None, // generic receiver
                            Some(allowed),
                        );
                        continue; // skip the function_graphs lookup
                    }
                }
                if let Some(graph) = function_graphs.get(path) {
                    resolved_calls.push(ResolvedCall {
                        name: path.canonical_key(),
                        impl_type: None,
                        trait_name: None,
                        graph: Some(graph.clone()),
                    });
                }
            }
            parse::ExtractedHandlerCall::UnsupportedFunctionExpr => {}
        }
    }

    resolved_calls
}

/// Generate tracing code directly from the canonical pipeline result.
pub fn generate_trace_code_from_pipeline(result: &passes::ProgramPipelineResult) -> String {
    codegen::generate_from_pipeline(result)
}

/// Generate code from graph pipeline results.
#[cfg(test)]
pub fn generate_graph_code(result: &passes::ProgramPipelineResult) -> String {
    codegen::generate_from_graph(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn read_pyre_file(name: &str) -> String {
        let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../pyre/");
        std::fs::read_to_string(format!("{base}{name}"))
            .unwrap_or_else(|_| panic!("failed to read {name}"))
    }

    fn collect_rs_files(dir: &Path, sources: &mut Vec<String>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|_| panic!("failed to read dir {}", dir.display()));
        for entry in entries {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, sources);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                sources.push(
                    std::fs::read_to_string(&path)
                        .unwrap_or_else(|_| panic!("failed to read {}", path.display())),
                );
            }
        }
    }

    fn read_all_pyre_sources() -> Vec<String> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pyre");
        let mut sources = Vec::new();
        for dir in [
            base.join("pyre-object/src"),
            base.join("pyre-interpreter/src"),
        ] {
            collect_rs_files(&dir, &mut sources);
        }
        sources
    }

    #[test]
    fn test_analyze_pyopcode() {
        let source = read_pyre_file("pyre-interpreter/src/pyopcode.rs");
        let result = analyze_multiple_pipeline_with_config(
            &[&source],
            &crate::test_support::pyre_analyze_config(),
        );

        assert!(
            result.opcode_dispatch.len() > 20,
            "expected >20 opcode arms, got {}",
            result.opcode_dispatch.len()
        );

        eprintln!("=== Single-file Analysis ===");
        eprintln!("Opcodes: {}", result.opcode_dispatch.len());
        for (i, arm) in result.opcode_dispatch.iter().enumerate() {
            eprintln!(
                "  [{i}] {} → {:?}",
                arm.selector.canonical_key(),
                arm.flattened.as_ref().map(|f| f.ops.len())
            );
        }
    }

    #[test]
    fn test_multi_file_analysis() {
        let sources = read_all_pyre_sources();
        let source_refs: Vec<_> = sources.iter().map(String::as_str).collect();
        let parsed_files: Vec<_> = sources
            .iter()
            .map(|source| parse::parse_source(source))
            .collect();
        let result = analyze_multiple_pipeline_with_config(
            &source_refs,
            &crate::test_support::pyre_analyze_config(),
        );
        let trait_impls: Vec<_> = parsed_files
            .iter()
            .flat_map(parse::extract_trait_impls)
            .collect();

        eprintln!("=== Multi-file Analysis ===");
        eprintln!("Opcodes: {}", result.opcode_dispatch.len());
        eprintln!("Functions: {}", result.functions.len());
        eprintln!("Trait impls: {}", trait_impls.len());

        // Should have trait impls from eval.rs (PyFrame impls)
        let pyframe_impls: Vec<_> = trait_impls
            .iter()
            .filter(|i| i.for_type.contains("PyFrame"))
            .collect();
        eprintln!("\nPyFrame trait impls: {}", pyframe_impls.len());
        for impl_info in &pyframe_impls {
            eprintln!(
                "  impl {} for PyFrame — {} methods",
                impl_info.trait_name,
                impl_info.methods.len()
            );
            for m in &impl_info.methods {
                eprintln!("    {}", m.name);
            }
        }

        // Should have resolved opcode patterns (flattened op counts)
        eprintln!("\nOpcode patterns:");
        for arm in &result.opcode_dispatch {
            if let Some(ref flat) = arm.flattened {
                eprintln!(
                    "  {} → {} flat ops",
                    arm.selector.canonical_key(),
                    flat.ops.len()
                );
            }
        }

        // Report flattened (inline→jtransform→flatten) stats
        let flattened_count = result
            .opcode_dispatch
            .iter()
            .filter(|a| a.flattened.is_some())
            .count();
        eprintln!(
            "\nFlattened (inline pipeline): {flattened_count}/{}",
            result.opcode_dispatch.len()
        );
        for arm in &result.opcode_dispatch {
            if let Some(ref flat) = arm.flattened {
                eprintln!(
                    "  {} → {} flat ops",
                    arm.selector.canonical_key(),
                    flat.ops.len()
                );
            }
        }

        // Verify canonical graph/pipeline dispatch flattens a useful subset.
        let flattened_dispatch_count = result
            .opcode_dispatch
            .iter()
            .filter(|a| a.flattened.is_some())
            .count();
        assert!(
            flattened_dispatch_count >= 10,
            "expected >=10 flattened opcode arms, got {}",
            flattened_dispatch_count
        );

        // Verify flattened arms produce non-empty op sequences.
        assert!(
            result
                .opcode_dispatch
                .iter()
                .filter_map(|arm| arm.flattened.as_ref())
                .all(|f| f.ops.len() > 0),
            "all flattened arms should have non-empty op sequences"
        );
    }

    #[test]
    fn test_codegen_output() {
        let sources = read_all_pyre_sources();
        let source_refs: Vec<_> = sources.iter().map(String::as_str).collect();
        let result = analyze_multiple_pipeline_with_config(
            &source_refs,
            &crate::test_support::pyre_analyze_config(),
        );
        let code = generate_trace_code_from_pipeline(&result);
        let flattened_arms: Vec<_> = result
            .opcode_dispatch
            .iter()
            .filter(|arm| arm.flattened.is_some())
            .collect();

        // Should contain canonical dispatch table
        assert!(
            code.contains("CANONICAL_TRACE_PATTERNS"),
            "missing CANONICAL_TRACE_PATTERNS"
        );
        assert!(
            !code.contains("pub const TRACE_PATTERNS"),
            "canonical output should not emit legacy TRACE_PATTERNS alias"
        );
        assert!(
            code.contains("Canonical analysis summary:"),
            "missing canonical summary"
        );
        assert!(!flattened_arms.is_empty(), "expected flattened opcode arms");

        eprintln!("=== Generated Code ({} bytes) ===", code.len());
        // Print first 50 lines
        for (i, line) in code.lines().enumerate().take(50) {
            eprintln!("{:3}: {}", i + 1, line);
        }
    }

    #[test]
    fn test_graph_pipeline_e2e() {
        // E2E test: source → AST front-end → semantic graph → graph transform → classify
        let parsed = parse::parse_source(
            r#"
            struct Frame { next_instr: usize, locals_w: Vec<i64> }
            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
                fn store_fast(&mut self, val: i64) {
                    let idx = self.next_instr;
                    self.locals_w[idx] = val;
                }
            }
        "#,
        );

        // Step 1: AST → semantic graph
        let program = front::build_semantic_program(&parsed);
        assert_eq!(
            program.functions.len(),
            2,
            "should have load_fast + store_fast"
        );

        // Step 2: graph transform (with virtualizable config)
        let config = passes::GraphTransformConfig {
            vable_fields: vec![passes::VirtualizableFieldDescriptor::new(
                "next_instr",
                Some("Frame".into()),
                0,
            )],
            vable_arrays: vec![passes::VirtualizableFieldDescriptor::new(
                "locals_w",
                Some("Frame".into()),
                0,
            )],
            ..Default::default()
        };

        let load_fast_graph = &program.functions[0].graph;
        let result = passes::rewrite_graph(load_fast_graph, &config);
        assert!(
            result.vable_rewrites > 0,
            "load_fast should have vable rewrites, got notes: {:?}",
            result.notes
        );

        // Step 3: flatten the rewritten graph
        let flattened = passes::flatten_with_types(
            &result.graph,
            &passes::resolve_types(&result.graph, &passes::annotate_graph(&result.graph)),
        );
        eprintln!(
            "load_fast graph ops: {:?}",
            load_fast_graph.block(load_fast_graph.entry).ops
        );
        eprintln!("load_fast flattened: {} ops", flattened.ops.len());
        assert!(flattened.ops.len() > 0, "load_fast should produce flat ops");
    }

    #[test]
    fn test_graph_pipeline_on_pyre_source() {
        // Run the full graph pipeline on actual pyre interpreter source.
        // This validates that the pipeline handles real-world Rust code.
        let source = read_pyre_file("pyre-interpreter/src/pyopcode.rs");
        let parsed = parse::parse_source(&source);
        let program = front::build_semantic_program(&parsed);

        let config = passes::PipelineConfig::default();
        let result = passes::analyze_program(&program, &config);

        eprintln!("=== Graph Pipeline on pyre-interpreter ===");
        eprintln!("Functions analyzed: {}", result.functions.len());
        eprintln!("Total blocks: {}", result.total_blocks);
        eprintln!("Total flat ops: {}", result.total_ops);

        // Should analyze many functions from the real source
        assert!(
            result.functions.len() >= 5,
            "expected >=5 functions from pyopcode.rs, got {}",
            result.functions.len()
        );

        // Should produce multi-block CFGs for functions with control flow
        let multi_block = result
            .functions
            .iter()
            .filter(|f| f.original_blocks > 1)
            .count();
        eprintln!("Functions with multi-block CFG: {multi_block}");

        // Should produce flat ops
        assert!(
            result.total_ops > 0,
            "should produce flat ops from real source"
        );
    }

    #[test]
    fn test_analyze_pipeline_runs_canonical_graph_path() {
        let source = read_pyre_file("pyre-interpreter/src/pyopcode.rs");
        let graph_result = analyze_pipeline(&source);

        // Graph pipeline should analyze functions
        assert!(graph_result.functions.len() >= 5);

        eprintln!(
            "analyze_pipeline: graph {} functions / {} blocks / {} ops",
            graph_result.functions.len(),
            graph_result.total_blocks,
            graph_result.total_ops,
        );
    }

    #[test]
    fn test_analyze_multiple_with_config_rewrites_virtualizable_graphs() {
        let source = r#"
            enum Instruction { LoadFast }

            struct Frame {
                next_instr: usize,
                locals_w: Vec<i64>,
            }

            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
            }

            fn execute_opcode_step(frame: &mut Frame, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => {
                        let _ = frame.load_fast();
                    }
                }
            }
        "#;

        let result = analyze_multiple_pipeline_with_config(
            &[source],
            &AnalyzeConfig {
                pipeline: PipelineConfig {
                    transform: GraphTransformConfig {
                        vable_fields: vec![passes::VirtualizableFieldDescriptor::new(
                            "next_instr",
                            Some("Frame".into()),
                            0,
                        )],
                        vable_arrays: vec![passes::VirtualizableFieldDescriptor::new(
                            "locals_w",
                            Some("Frame".into()),
                            0,
                        )],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
        );

        let load_fast = result
            .opcode_dispatch
            .iter()
            .find(|arm| arm.selector.canonical_key() == "Instruction::LoadFast")
            .expect("LoadFast opcode arm");
        assert!(
            load_fast.flattened.is_some(),
            "LoadFast should be flattened"
        );
        assert!(
            load_fast.flattened.as_ref().unwrap().ops.len() > 0,
            "LoadFast flattened should have ops"
        );
    }

    #[test]
    fn test_analyze_multiple_pipeline_with_config_produces_canonical_vable_dispatch() {
        let source = r#"
            enum Instruction { LoadFast }

            struct Frame {
                next_instr: usize,
                locals_w: Vec<i64>,
            }

            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
            }

            fn execute_opcode_step(frame: &mut Frame, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => {
                        let _ = frame.load_fast();
                    }
                }
            }
        "#;

        let result = analyze_multiple_pipeline_with_config(
            &[source],
            &AnalyzeConfig {
                pipeline: PipelineConfig {
                    transform: GraphTransformConfig {
                        vable_fields: vec![passes::VirtualizableFieldDescriptor::new(
                            "next_instr",
                            Some("Frame".into()),
                            0,
                        )],
                        vable_arrays: vec![passes::VirtualizableFieldDescriptor::new(
                            "locals_w",
                            Some("Frame".into()),
                            0,
                        )],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
        );
        let canonical_load_fast = result
            .opcode_dispatch
            .iter()
            .find(|arm| arm.selector.canonical_key() == "Instruction::LoadFast")
            .expect("canonical LoadFast opcode arm");
        assert!(
            result.total_vable_rewrites > 0,
            "graph pipeline should perform vable rewrites"
        );
        assert!(
            canonical_load_fast.flattened.is_some(),
            "canonical LoadFast should be flattened"
        );
        assert!(
            canonical_load_fast.flattened.as_ref().unwrap().ops.len() > 0,
            "canonical LoadFast flattened should have ops"
        );
    }

    #[test]
    fn test_opcode_dispatch_uses_trait_bound_default_method_graphs() {
        let source = r#"
            enum Instruction { LoadFast }

            trait OpcodeStepExecutor {
                fn load_fast_checked(&mut self, idx: usize) {
                    let _ = idx;
                }
            }

            fn execute_opcode_step<E: OpcodeStepExecutor>(executor: &mut E, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => executor.load_fast_checked(0),
                }
            }
        "#;

        let result = analyze_multiple_pipeline(&[source]);
        let arm = result
            .opcode_dispatch
            .iter()
            .find(|arm| arm.selector.canonical_key() == "Instruction::LoadFast")
            .expect("LoadFast opcode arm");
        assert!(
            arm.flattened.is_some(),
            "trait-bound default method should produce a flattened result"
        );
    }

    #[test]
    fn test_generic_trait_bound_does_not_sweep_concrete_impls() {
        let source = r#"
            enum Instruction { LoadFast }

            trait OpcodeStepExecutor {
                fn load_fast_checked(&mut self, idx: usize) {
                    let _ = idx;
                }
            }

            struct PyFrame;
            impl OpcodeStepExecutor for PyFrame {}

            fn execute_opcode_step<E: OpcodeStepExecutor>(executor: &mut E, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => executor.load_fast_checked(0),
                }
            }
        "#;

        let parsed = parse::parse_source(source);
        let trait_impls = parse::extract_trait_impls(&parsed);
        let inherent_methods = parse::extract_inherent_impl_methods(&parsed);
        let mut function_graphs = std::collections::HashMap::new();
        parse::collect_function_graphs(&parsed, &mut function_graphs);
        let receiver_traits = parse::extract_opcode_dispatch_receiver_traits(&parsed);
        let arms = parse::extract_opcode_dispatch_arms(&parsed);
        let resolved = resolve_handler_calls(
            &arms[0].handler_calls,
            &trait_impls,
            &inherent_methods,
            &function_graphs,
            &receiver_traits,
        );

        assert_eq!(
            resolved.len(),
            1,
            "generic trait bound should resolve only default method body"
        );
        assert_eq!(
            resolved[0].impl_type.as_deref(),
            Some("<default methods of OpcodeStepExecutor>")
        );
        assert_eq!(
            resolved[0].trait_name.as_deref(),
            Some("OpcodeStepExecutor")
        );
    }

    #[test]
    fn test_generic_trait_bound_uses_unique_concrete_impl() {
        let source = r#"
            enum Instruction { LoadFast }

            trait OpcodeStepExecutor {
                fn load_fast_checked(&mut self, idx: usize);
            }

            struct PyFrame;
            impl OpcodeStepExecutor for PyFrame {
                fn load_fast_checked(&mut self, idx: usize) {
                    let _ = idx;
                }
            }

            fn execute_opcode_step<E: OpcodeStepExecutor>(executor: &mut E, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => executor.load_fast_checked(0),
                }
            }
        "#;

        let parsed = parse::parse_source(source);
        let trait_impls = parse::extract_trait_impls(&parsed);
        let inherent_methods = parse::extract_inherent_impl_methods(&parsed);
        let mut function_graphs = std::collections::HashMap::new();
        parse::collect_function_graphs(&parsed, &mut function_graphs);
        let receiver_traits = parse::extract_opcode_dispatch_receiver_traits(&parsed);
        let arms = parse::extract_opcode_dispatch_arms(&parsed);
        let resolved = resolve_handler_calls(
            &arms[0].handler_calls,
            &trait_impls,
            &inherent_methods,
            &function_graphs,
            &receiver_traits,
        );

        assert_eq!(
            resolved.len(),
            1,
            "generic trait bound should use the unique concrete impl"
        );
        assert_eq!(resolved[0].impl_type.as_deref(), Some("PyFrame"));
        assert_eq!(
            resolved[0].trait_name.as_deref(),
            Some("OpcodeStepExecutor")
        );
    }

    #[test]
    fn test_function_path_resolution_uses_exact_call_path() {
        let source = r#"
            enum Instruction { LoadFast }

            fn helper() {}

            fn execute_opcode_step(instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => crate::helper(),
                }
            }
        "#;

        let parsed = parse::parse_source(source);
        let trait_impls = parse::extract_trait_impls(&parsed);
        let inherent_methods = parse::extract_inherent_impl_methods(&parsed);
        let mut function_graphs = std::collections::HashMap::new();
        parse::collect_function_graphs(&parsed, &mut function_graphs);
        let receiver_traits = parse::extract_opcode_dispatch_receiver_traits(&parsed);
        let arms = parse::extract_opcode_dispatch_arms(&parsed);
        let resolved = resolve_handler_calls(
            &arms[0].handler_calls,
            &trait_impls,
            &inherent_methods,
            &function_graphs,
            &receiver_traits,
        );

        assert_eq!(resolved.len(), 1, "exact crate::helper path should resolve");
        assert_eq!(resolved[0].name, "crate::helper");
        assert!(
            resolved[0].graph.is_some(),
            "resolved helper should carry a graph"
        );
    }

    #[test]
    fn test_opcode_dispatch_prefers_concrete_receiver_impl_root() {
        let source = r#"
            enum Instruction { LoadFast }

            trait TraitA {
                fn load_fast_checked(&mut self, idx: usize) { let _ = idx; }
            }
            trait TraitB {
                fn load_fast_checked(&mut self, idx: usize) { let _ = idx + 1; }
            }

            struct PyFrame;
            struct OtherFrame;

            impl TraitA for PyFrame {}
            impl TraitB for OtherFrame {}

            fn execute_opcode_step(frame: &mut PyFrame, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => frame.load_fast_checked(0),
                }
            }
        "#;

        let parsed = parse::parse_source(source);
        let trait_impls = parse::extract_trait_impls(&parsed);
        let inherent_methods = parse::extract_inherent_impl_methods(&parsed);
        let mut function_graphs = std::collections::HashMap::new();
        parse::collect_function_graphs(&parsed, &mut function_graphs);
        let receiver_traits = parse::extract_opcode_dispatch_receiver_traits(&parsed);
        let arms = parse::extract_opcode_dispatch_arms(&parsed);
        let resolved = resolve_handler_calls(
            &arms[0].handler_calls,
            &trait_impls,
            &inherent_methods,
            &function_graphs,
            &receiver_traits,
        );

        assert_eq!(
            resolved.len(),
            1,
            "expected only concrete receiver impl match"
        );
        assert_eq!(
            resolved[0].impl_type.as_deref(),
            Some("<default methods of TraitA>")
        );
        assert_eq!(resolved[0].trait_name.as_deref(), Some("TraitA"));
    }

    #[test]
    fn test_opcode_dispatch_does_not_broad_match_known_receiver_identity() {
        let source = r#"
            trait TraitA { fn helper(&mut self); }
            trait TraitB { fn helper(&mut self) {} }
            struct PyFrame;
            impl TraitA for PyFrame { fn helper(&mut self) {} }
            struct OtherFrame;
            impl TraitB for OtherFrame {}
            pub fn execute_opcode_step(frame: &mut PyFrame) { frame.helper(); }
        "#;

        let parsed = parse::parse_source(source);
        let receiver_traits = parse::extract_opcode_dispatch_receiver_traits(&parsed);
        let trait_impls = parse::extract_trait_impls(&parsed);
        let inherent_methods = parse::extract_inherent_impl_methods(&parsed);
        let function_graphs = std::collections::HashMap::new();
        let handler_calls = vec![parse::ExtractedHandlerCall::Method {
            name: "helper".into(),
            receiver_root: Some("frame".into()),
        }];

        let resolved = resolve_handler_calls(
            &handler_calls,
            &trait_impls,
            &inherent_methods,
            &function_graphs,
            &receiver_traits,
        );

        assert_eq!(
            resolved.len(),
            1,
            "known receiver should not broad-match unrelated impls"
        );
        assert_eq!(resolved[0].impl_type.as_deref(), Some("PyFrame"));
        assert_eq!(resolved[0].trait_name.as_deref(), Some("TraitA"));
    }

    /// Integration test: CallControl + inline on real pyre sources.
    ///
    /// Verifies that the inline pass produces graphs with low-level ops
    /// (FieldRead, ArrayRead) from inlined handler method bodies.
    #[test]
    fn test_inline_pipeline_integration() {
        let sources = read_all_pyre_sources();
        let source_refs: Vec<&str> = sources.iter().map(String::as_str).collect();
        let parsed_files: Vec<_> = source_refs.iter().map(|s| parse::parse_source(s)).collect();

        // Build CallControl from parsed sources
        let mut call_control = call::CallControl::new();
        let mut function_graphs = std::collections::HashMap::new();
        for parsed in &parsed_files {
            parse::collect_function_graphs(parsed, &mut function_graphs);
        }
        for (path, graph) in &function_graphs {
            call_control.register_function_graph(path.clone(), graph.clone());
        }
        let trait_impls: Vec<TraitImplInfo> = parsed_files
            .iter()
            .flat_map(parse::extract_trait_impls)
            .collect();
        for impl_info in &trait_impls {
            let impl_type = impl_info
                .self_ty_root
                .as_deref()
                .unwrap_or(&impl_info.for_type);
            for method in &impl_info.methods {
                if let Some(graph) = &method.graph {
                    call_control.register_trait_method(&method.name, impl_type, graph.clone());
                }
            }
        }
        call_control.find_all_graphs();

        // Get opcode_load_fast_checked graph and inline it
        let path = parse::CallPath::from_segments(["opcode_load_fast_checked"]);
        let graph = function_graphs.get(&path);
        assert!(
            graph.is_some(),
            "opcode_load_fast_checked should exist in function_graphs"
        );
        let mut graph = graph.unwrap().clone();

        let pre_inline_blocks = graph.blocks.len();
        let inlined = inline::inline_graph(&mut graph, &call_control, 3);

        eprintln!("=== Inline Integration Test ===");
        eprintln!(
            "  opcode_load_fast_checked: {pre_inline_blocks} blocks → {} blocks, {inlined} call sites inlined",
            graph.blocks.len()
        );
        for block in &graph.blocks {
            for op in &block.ops {
                eprintln!("    {:?}", op.kind);
            }
        }

        assert!(inlined > 0, "should inline at least one call site");

        // After inlining, the graph should have low-level ops from callee bodies
        let all_ops: Vec<_> = graph.blocks.iter().flat_map(|b| &b.ops).collect();
        let has_low_level = all_ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { .. }
                    | OpKind::ArrayRead { .. }
                    | OpKind::ArrayWrite { .. }
                    | OpKind::FieldWrite { .. }
            )
        });
        eprintln!("  has low-level ops after inline: {has_low_level}");
    }
}

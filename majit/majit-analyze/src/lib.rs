//! majit-analyze: Static program analyzer for meta-tracing JIT generation.
//!
//! This is the Rust equivalent of RPython's translation toolchain.
//! Given a bundled interpreter source file, it:
//! 1. Parses the entire crate with `syn`
//! 2. Extracts the opcode dispatch table (match arms)
//! 3. Traces trait implementations to resolve handler methods
//! 4. Classifies helper functions (elidable, residual, field access)
//! 5. Collects type layouts (struct fields, offsets)
//! 6. Generates tracing code that mirrors the interpreter's execution

mod classify;
mod codegen;
pub mod codewriter;
pub mod front;
pub mod graph;
pub mod interp_extract;
mod parse;
pub mod passes;
mod patterns;

pub use classify::{FunctionInfo, HelperClassification};
pub use front::{AstGraphOptions, SemanticFunction, SemanticProgram, build_semantic_program};
pub use graph::{BasicBlock, BasicBlockId, MajitGraph, Op, OpKind, Terminator, ValueId, ValueType};
pub use parse::ParsedInterpreter;
pub use passes::{
    GraphTransformConfig, GraphTransformResult, rewrite_graph,
    PipelineConfig, PipelineResult, ProgramPipelineResult,
    analyze_function, analyze_program,
    FlatOp, FlattenedFunction, Label,
    AnnotationState, annotate_graph,
    ConcreteType, TypeResolutionState, resolve_types,
};
pub use patterns::{TracePattern, classify_from_graph, classify_from_pattern};

use serde::{Deserialize, Serialize};

/// Complete analysis result for an interpreter crate.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Opcode dispatch arms extracted from the main match
    pub opcodes: Vec<OpcodeArm>,
    /// Helper function classifications
    pub helpers: Vec<FunctionInfo>,
    /// Struct layouts (field names, types, offsets)
    pub type_layouts: Vec<TypeLayout>,
    /// Trait implementations found
    pub trait_impls: Vec<TraitImplInfo>,
    /// Named offset constants (e.g., INT_INTVAL_OFFSET = offset_of!(...))
    #[serde(default)]
    pub offset_constants: Vec<(String, String)>,
}

/// A single opcode match arm
#[derive(Debug, Serialize, Deserialize)]
pub struct OpcodeArm {
    /// Pattern string (e.g., "Instruction::BinaryOp { op }")
    pub pattern: String,
    /// Handler method calls found in the arm body
    pub handler_calls: Vec<String>,
    /// Resolved call chain (trait impl → concrete method → helpers)
    #[serde(default)]
    pub resolved_calls: Vec<ResolvedCall>,
    /// Trace pattern classification
    pub trace_pattern: Option<TracePattern>,
}

/// A resolved method/function call with source context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedCall {
    pub name: String,
    pub impl_type: Option<String>,
    pub trait_name: Option<String>,
    pub body_summary: String,
    /// Pre-built semantic graph (avoids re-parsing body_summary).
    #[serde(default)]
    pub graph: Option<graph::MajitGraph>,
}

/// Struct field layout information
#[derive(Debug, Serialize, Deserialize)]
pub struct TypeLayout {
    pub name: String,
    pub fields: Vec<FieldInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FieldInfo {
    pub name: String,
    pub ty: String,
    pub offset_expr: Option<String>,
}

/// Trait implementation info
#[derive(Debug, Serialize, Deserialize)]
pub struct TraitImplInfo {
    pub trait_name: String,
    pub for_type: String,
    pub methods: Vec<MethodInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MethodInfo {
    pub name: String,
    pub body_summary: String,
    /// Pre-built semantic graph (avoids re-parsing body_summary).
    #[serde(default)]
    pub graph: Option<graph::MajitGraph>,
}

/// Full analysis: parse + build graph + run pipeline + classify.
///
/// This is the recommended entry point. It runs both the legacy
/// string-based analysis (for backward compatibility) and the new
/// graph-based pipeline (for RPython-parity classification).
pub fn analyze_full(source: &str) -> (AnalysisResult, passes::ProgramPipelineResult) {
    let mut legacy = analyze(source);
    let parsed = parse::parse_source(source);
    let program = front::build_semantic_program(&parsed);
    let pipeline = passes::analyze_program(&program, &passes::PipelineConfig::default());

    // Enrich: if graph pipeline classified a function that matches an
    // unclassified opcode arm handler, propagate the graph classification.
    for arm in &mut legacy.opcodes {
        if arm.trace_pattern.is_some() {
            continue;
        }
        for call in &arm.resolved_calls {
            if let Some(ref graph) = call.graph {
                if let Some(pattern) = patterns::classify_from_graph(graph) {
                    arm.trace_pattern = Some(pattern);
                    break;
                }
            }
        }
    }

    (legacy, pipeline)
}

/// Analyze a single source file (legacy string-based path).
pub fn analyze(source: &str) -> AnalysisResult {
    analyze_multiple(&[source])
}

/// Analyze multiple source files as a unified program.
///
/// This is the main entry point — equivalent to RPython's annotation + rtyping.
/// Pass all relevant source files (opcode_step.rs, eval.rs, object types, etc.)
/// and the analyzer builds a cross-file view of the interpreter.
pub fn analyze_multiple(sources: &[&str]) -> AnalysisResult {
    let mut all_helpers = Vec::new();
    let mut all_types = Vec::new();
    let mut all_trait_impls = Vec::new();
    let mut all_functions = std::collections::HashMap::new();
    let mut all_function_graphs = std::collections::HashMap::new();
    let mut all_offset_constants = Vec::new();

    // Phase 1: Parse all files and collect items
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();

    for parsed in &parsed_files {
        all_helpers.extend(classify::classify_functions(parsed));
        all_types.extend(parse::extract_type_layouts(parsed));
        all_trait_impls.extend(parse::extract_trait_impls(parsed));
        all_offset_constants.extend(parse::extract_offset_constants(parsed));
        parse::collect_functions_with_graphs(parsed, &mut all_functions, &mut all_function_graphs);
    }

    // Phase 2: Extract opcode dispatch (needs cross-file trait resolution)
    let mut opcodes = Vec::new();
    for parsed in &parsed_files {
        let file_opcodes = parse::extract_opcode_dispatch(parsed, &all_trait_impls);
        if !file_opcodes.is_empty() {
            opcodes = file_opcodes;
            break; // Only one file has execute_opcode_step
        }
    }

    // Phase 3: Resolve call chains across files
    for arm in &mut opcodes {
        resolve_call_chain(arm, &all_trait_impls, &all_functions, &all_function_graphs);
    }

    AnalysisResult {
        opcodes,
        helpers: all_helpers,
        type_layouts: all_types,
        trait_impls: all_trait_impls,
        offset_constants: all_offset_constants,
    }
}

/// Resolve handler method calls through trait impls and function definitions.
fn resolve_call_chain(
    arm: &mut OpcodeArm,
    trait_impls: &[TraitImplInfo],
    functions: &std::collections::HashMap<String, String>,
    function_graphs: &std::collections::HashMap<String, graph::MajitGraph>,
) {
    let mut resolved_calls = Vec::new();

    for call_name in &arm.handler_calls {
        // Check trait impls first
        for impl_info in trait_impls {
            for method in &impl_info.methods {
                if &method.name == call_name {
                    resolved_calls.push(ResolvedCall {
                        name: call_name.clone(),
                        impl_type: Some(impl_info.for_type.clone()),
                        trait_name: Some(impl_info.trait_name.clone()),
                        body_summary: method.body_summary.clone(),
                        graph: method.graph.clone(),
                    });
                }
            }
        }

        // Check free functions (with pre-built graph)
        if let Some(body) = functions.get(call_name) {
            resolved_calls.push(ResolvedCall {
                name: call_name.clone(),
                impl_type: None,
                trait_name: None,
                body_summary: body.clone(),
                graph: function_graphs.get(call_name).cloned(),
            });
        }
    }

    arm.resolved_calls = resolved_calls;

    // PRIMARY: Graph-based classification.
    // Use pre-built semantic graphs from parse time (no body_summary re-parse).
    // This is the RPython-parity path — the flow graph is the canonical IR.
    if arm.trace_pattern.is_none() {
        for call in &arm.resolved_calls {
            if let Some(ref graph) = call.graph {
                if let Some(pattern) = patterns::classify_from_graph(graph) {
                    arm.trace_pattern = Some(pattern);
                    break;
                }
            }
        }
    }

    // SECONDARY: Method name heuristic (for cases where body parsing fails
    // or the graph classifier doesn't recognize the pattern yet).
    if arm.trace_pattern.is_none() {
        arm.trace_pattern = classify_from_method_names(&arm.resolved_calls, &arm.handler_calls);
    }

    // TERTIARY: Opcode pattern text (for arms without resolved calls).
    if arm.trace_pattern.is_none() {
        arm.trace_pattern = patterns::classify_from_pattern(&arm.pattern);
    }
}

/// Classify from method names only (no body_summary heuristic).
///
/// This is the secondary fallback after graph-based classification.
/// It only looks at the handler method name (e.g., "binary_op",
/// "load_fast") without parsing the body.
fn classify_from_method_names(
    resolved: &[ResolvedCall],
    handler_calls: &[String],
) -> Option<TracePattern> {
    // Try method name matching from resolved calls
    for call in resolved {
        match call.name.as_str() {
            "binary_op" => {
                return Some(TracePattern::UnboxIntBinop {
                    op_name: "dispatch".into(),
                    has_overflow_guard: true,
                });
            }
            "compare_op" => {
                return Some(TracePattern::UnboxIntCompare {
                    op_name: "dispatch".into(),
                });
            }
            "unary_negative" | "unary_invert" => {
                return Some(TracePattern::UnboxIntUnary {
                    op_name: call.name.clone(),
                });
            }
            "unary_not" => return Some(TracePattern::TruthCheck),
            "load_fast" | "load_fast_checked" | "load_fast_load_fast" => {
                return Some(TracePattern::LocalRead);
            }
            "store_fast" | "store_fast_checked" | "store_fast_store_fast"
            | "store_fast_load_fast" => {
                return Some(TracePattern::LocalWrite);
            }
            "load_const" | "load_small_int" => return Some(TracePattern::ConstLoad),
            "call" => return Some(TracePattern::FunctionCall),
            "for_iter" => return Some(TracePattern::RangeIterNext),
            "end_for" | "pop_iter" => return Some(TracePattern::IterCleanup),
            "pop_top" | "copy_value" | "swap" | "push_null" => {
                return Some(TracePattern::StackManip);
            }
            "jump_forward" | "jump_backward" => return Some(TracePattern::Jump),
            "pop_jump_if_false" | "pop_jump_if_true" => {
                return Some(TracePattern::ConditionalJump);
            }
            "return_value" => return Some(TracePattern::Return),
            "store_name" => {
                return Some(TracePattern::NamespaceAccess {
                    is_load: false,
                    is_global: false,
                });
            }
            "load_name" => {
                return Some(TracePattern::NamespaceAccess {
                    is_load: true,
                    is_global: false,
                });
            }
            "load_global" => {
                return Some(TracePattern::NamespaceAccess {
                    is_load: true,
                    is_global: true,
                });
            }
            "build_list" => {
                return Some(TracePattern::BuildCollection {
                    kind: "list".into(),
                });
            }
            "build_tuple" => {
                return Some(TracePattern::BuildCollection {
                    kind: "tuple".into(),
                });
            }
            "build_map" => {
                return Some(TracePattern::BuildCollection { kind: "map".into() });
            }
            "unpack_sequence" => return Some(TracePattern::UnpackSequence),
            "store_subscr" => return Some(TracePattern::SequenceSetitem),
            "list_append" => return Some(TracePattern::CollectionAppend),
            "get_iter" | "make_function" => {
                return Some(TracePattern::Residual {
                    helper_name: call.name.clone(),
                });
            }
            _ => {}
        }
    }

    // Also check handler_calls (from arm body analysis)
    for name in handler_calls {
        match name.as_str() {
            "binary_op" => {
                return Some(TracePattern::UnboxIntBinop {
                    op_name: "dispatch".into(),
                    has_overflow_guard: true,
                });
            }
            "load_fast" | "load_fast_checked" => return Some(TracePattern::LocalRead),
            "store_fast" | "store_fast_checked" => return Some(TracePattern::LocalWrite),
            _ => {}
        }
    }

    None
}

/// Generate tracing code from analysis results.
///
/// This is the main codegen entry point — equivalent to RPython's JitCode generation.
pub fn generate_trace_code(result: &AnalysisResult) -> String {
    codegen::generate(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_pyre_file(name: &str) -> String {
        let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../pyre/");
        std::fs::read_to_string(format!("{base}{name}"))
            .unwrap_or_else(|_| panic!("failed to read {name}"))
    }

    #[test]
    fn test_analyze_opcode_step() {
        let source = read_pyre_file("pyre-runtime/src/opcode_step.rs");
        let result = analyze(&source);

        assert!(
            result.opcodes.len() > 20,
            "expected >20 opcode arms, got {}",
            result.opcodes.len()
        );

        eprintln!("=== Single-file Analysis ===");
        eprintln!("Opcodes: {}", result.opcodes.len());
        for (i, arm) in result.opcodes.iter().enumerate() {
            eprintln!("  [{i}] {} → {:?}", arm.pattern, arm.handler_calls);
        }
    }

    #[test]
    fn test_multi_file_analysis() {
        let opcode_step = read_pyre_file("pyre-runtime/src/opcode_step.rs");
        let eval = read_pyre_file("pyre-interp/src/eval.rs");

        let result = analyze_multiple(&[&opcode_step, &eval]);

        eprintln!("=== Multi-file Analysis ===");
        eprintln!("Opcodes: {}", result.opcodes.len());
        eprintln!("Helpers: {}", result.helpers.len());
        eprintln!("Types: {}", result.type_layouts.len());
        eprintln!("Trait impls: {}", result.trait_impls.len());

        // Should have trait impls from eval.rs (PyFrame impls)
        let pyframe_impls: Vec<_> = result
            .trait_impls
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

        // Should have resolved opcode patterns
        eprintln!("\nOpcode patterns:");
        for arm in &result.opcodes {
            if let Some(ref pattern) = arm.trace_pattern {
                eprintln!("  {} → {:?}", arm.pattern, pattern);
            }
        }

        // Verify key patterns are detected
        let binary_op = result
            .opcodes
            .iter()
            .find(|a| a.pattern.contains("BinaryOp"));
        assert!(binary_op.is_some(), "BinaryOp arm not found");
        assert!(
            binary_op.unwrap().trace_pattern.is_some(),
            "BinaryOp should have a trace pattern"
        );

        // Verify classification coverage (39/40 — only `other` wildcard is unclassified)
        let classified_count = result
            .opcodes
            .iter()
            .filter(|a| a.trace_pattern.is_some())
            .count();
        assert!(
            classified_count >= 30,
            "expected >=30 classified, got {}",
            classified_count
        );

        // Verify new pattern categories are present
        let patterns_debug: Vec<String> = result
            .opcodes
            .iter()
            .filter_map(|a| a.trace_pattern.as_ref().map(|p| format!("{:?}", p)))
            .collect();
        let pattern_str = patterns_debug.join(" ");
        assert!(pattern_str.contains("Jump"), "missing Jump pattern");
        assert!(
            pattern_str.contains("ConditionalJump"),
            "missing ConditionalJump pattern"
        );
        assert!(pattern_str.contains("Return"), "missing Return pattern");
        assert!(
            pattern_str.contains("NamespaceAccess"),
            "missing NamespaceAccess pattern"
        );
        assert!(
            pattern_str.contains("IterCleanup"),
            "missing IterCleanup pattern"
        );
        assert!(pattern_str.contains("Noop"), "missing Noop pattern");
        assert!(
            pattern_str.contains("BuildCollection"),
            "missing BuildCollection pattern"
        );
        assert!(
            pattern_str.contains("UnpackSequence"),
            "missing UnpackSequence pattern"
        );
        assert!(
            pattern_str.contains("SequenceSetitem"),
            "missing SequenceSetitem pattern"
        );
        assert!(
            pattern_str.contains("CollectionAppend"),
            "missing CollectionAppend pattern"
        );
    }

    #[test]
    fn test_codegen_output() {
        let opcode_step = read_pyre_file("pyre-runtime/src/opcode_step.rs");
        let eval = read_pyre_file("pyre-interp/src/eval.rs");
        let result = analyze_multiple(&[&opcode_step, &eval]);
        let code = generate_trace_code(&result);

        // Should contain dispatch table
        assert!(code.contains("TRACE_PATTERNS"), "missing TRACE_PATTERNS");
        assert!(code.contains("UnboxIntBinop"), "missing UnboxIntBinop");
        assert!(code.contains("LocalRead"), "missing LocalRead");
        assert!(code.contains("FunctionCall"), "missing FunctionCall");
        assert!(code.contains("Jump"), "missing Jump");
        assert!(code.contains("ConditionalJump"), "missing ConditionalJump");
        assert!(code.contains("Return"), "missing Return");
        assert!(code.contains("NamespaceAccess"), "missing NamespaceAccess");
        assert!(code.contains("IterCleanup"), "missing IterCleanup");
        assert!(code.contains("Noop"), "missing Noop");
        assert!(code.contains("BuildCollection"), "missing BuildCollection");
        assert!(code.contains("SequenceSetitem"), "missing SequenceSetitem");

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
        assert_eq!(program.functions.len(), 2, "should have load_fast + store_fast");

        // Step 2: graph transform (with virtualizable config)
        let config = passes::GraphTransformConfig {
            vable_fields: vec![("next_instr".into(), 0)],
            vable_arrays: vec![("locals_w".into(), 0)],
            ..Default::default()
        };

        let load_fast_graph = &program.functions[0].graph;
        let result = passes::rewrite_graph(load_fast_graph, &config);
        assert!(
            result.vable_rewrites > 0,
            "load_fast should have vable rewrites, got notes: {:?}",
            result.notes
        );

        // Step 3: graph-based classification
        let pattern = patterns::classify_from_graph(load_fast_graph);
        eprintln!("load_fast graph ops: {:?}", load_fast_graph.block(load_fast_graph.entry).ops);
        eprintln!("load_fast classified as: {:?}", pattern);
        // load_fast reads a field + reads array → should be detectable
    }

    #[test]
    fn test_graph_pipeline_on_pyre_source() {
        // Run the full graph pipeline on actual pyre interpreter source.
        // This validates that the pipeline handles real-world Rust code.
        let source = read_pyre_file("pyre-runtime/src/opcode_step.rs");
        let parsed = parse::parse_source(&source);
        let program = front::build_semantic_program(&parsed);

        let config = passes::PipelineConfig::default();
        let result = passes::analyze_program(&program, &config);

        eprintln!("=== Graph Pipeline on pyre-runtime ===");
        eprintln!("Functions analyzed: {}", result.functions.len());
        eprintln!("Total blocks: {}", result.total_blocks);
        eprintln!("Total flat ops: {}", result.total_ops);

        // Should analyze many functions from the real source
        assert!(
            result.functions.len() >= 5,
            "expected >=5 functions from opcode_step.rs, got {}",
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
    fn test_analyze_full_runs_both_pipelines() {
        let source = read_pyre_file("pyre-runtime/src/opcode_step.rs");
        let (legacy, graph_result) = analyze_full(&source);

        // Legacy pipeline should produce opcodes
        assert!(legacy.opcodes.len() >= 30);

        // Graph pipeline should analyze functions
        assert!(graph_result.functions.len() >= 5);

        eprintln!(
            "analyze_full: legacy {} opcodes, graph {} functions / {} blocks / {} ops",
            legacy.opcodes.len(),
            graph_result.functions.len(),
            graph_result.total_blocks,
            graph_result.total_ops,
        );
    }
}

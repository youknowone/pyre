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

pub mod ast_transform;
mod classify;
mod codegen;
pub mod interp_extract;
pub mod mjit_codegen;
mod parse;
mod patterns;

pub use classify::{FunctionInfo, HelperClassification};
pub use parse::ParsedInterpreter;
pub use patterns::TracePattern;

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
}

/// Analyze a single source file.
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
    let mut all_offset_constants = Vec::new();

    // Phase 1: Parse all files and collect items
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();

    for parsed in &parsed_files {
        all_helpers.extend(classify::classify_functions(parsed));
        all_types.extend(parse::extract_type_layouts(parsed));
        all_trait_impls.extend(parse::extract_trait_impls(parsed));
        all_offset_constants.extend(parse::extract_offset_constants(parsed));
        parse::collect_functions(parsed, &mut all_functions);
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
        resolve_call_chain(arm, &all_trait_impls, &all_functions);
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
                    });
                }
            }
        }

        // Check free functions
        if let Some(body) = functions.get(call_name) {
            resolved_calls.push(ResolvedCall {
                name: call_name.clone(),
                impl_type: None,
                trait_name: None,
                body_summary: body.clone(),
            });
        }
    }

    arm.resolved_calls = resolved_calls;

    // Try to classify the trace pattern from resolved calls
    if arm.trace_pattern.is_none() {
        arm.trace_pattern = patterns::classify_from_resolved(&arm.resolved_calls);
    }

    // Fallback: classify from the opcode pattern text itself
    if arm.trace_pattern.is_none() {
        arm.trace_pattern = patterns::classify_from_pattern(&arm.pattern);
    }
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
}

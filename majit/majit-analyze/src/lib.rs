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

mod parse;
mod classify;
mod patterns;
mod codegen;

pub use parse::ParsedInterpreter;
pub use classify::{HelperClassification, FunctionInfo};
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
}

/// A single opcode match arm
#[derive(Debug, Serialize, Deserialize)]
pub struct OpcodeArm {
    /// Pattern string (e.g., "Instruction::BinaryOp { op }")
    pub pattern: String,
    /// Handler method calls found in the arm body
    pub handler_calls: Vec<String>,
    /// Trace pattern classification
    pub trace_pattern: Option<TracePattern>,
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

/// Analyze a bundled interpreter source file.
///
/// This is the main entry point — equivalent to RPython's annotation + rtyping.
pub fn analyze(source: &str) -> AnalysisResult {
    let parsed = parse::parse_source(source);
    let helpers = classify::classify_functions(&parsed);
    let type_layouts = parse::extract_type_layouts(&parsed);
    let trait_impls = parse::extract_trait_impls(&parsed);
    let opcodes = parse::extract_opcode_dispatch(&parsed, &trait_impls);

    AnalysisResult {
        opcodes,
        helpers,
        type_layouts,
        trait_impls,
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

    #[test]
    fn test_analyze_opcode_step() {
        let source = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../pyre/pyre-runtime/src/opcode_step.rs"),
        )
        .expect("failed to read opcode_step.rs");
        let result = analyze(&source);

        // Should find the execute_opcode_step match arms
        assert!(
            result.opcodes.len() > 20,
            "expected >20 opcode arms, got {}",
            result.opcodes.len()
        );

        // Print summary for manual inspection
        eprintln!("=== Analysis Results ===");
        eprintln!("Opcodes: {}", result.opcodes.len());
        eprintln!("Helpers: {}", result.helpers.len());
        eprintln!("Types: {}", result.type_layouts.len());
        eprintln!("Trait impls: {}", result.trait_impls.len());
        for (i, arm) in result.opcodes.iter().enumerate() {
            eprintln!("  [{i}] {} → {:?}", arm.pattern, arm.handler_calls);
        }
    }
}

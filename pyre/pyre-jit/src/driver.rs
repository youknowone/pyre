//! JIT driver for pyre — manages trace compilation and execution.
//!
//! This module is the Rust equivalent of RPython's jit/metainterp
//! integration layer. It connects the interpreter (pyre-interp) with
//! the JIT compiler (majit) through auto-generated tracing code.
//!
//! Key principle: pyre-interp has zero JIT dependencies.
//! All JIT logic lives here in pyre-jit.

// Re-export the auto-generated tracing functions
pub use crate::{
    TRACE_PATTERNS, trace_box_int, trace_int_binop, trace_int_binop_ovf, trace_int_compare,
    trace_unbox_int,
};

/// Check if an opcode pattern has been classified for JIT tracing.
pub fn is_traceable(pattern_name: &str) -> bool {
    TRACE_PATTERNS
        .iter()
        .any(|(_, cls)| *cls != "Unclassified" && cls.contains(pattern_name))
}

/// Get the trace classification for an opcode pattern.
pub fn get_trace_classification(opcode_pattern: &str) -> Option<&'static str> {
    TRACE_PATTERNS
        .iter()
        .find(|(pat, _)| *pat == opcode_pattern)
        .map(|(_, cls)| *cls)
        .filter(|cls| *cls != "Unclassified")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traceable_opcodes() {
        assert!(is_traceable("UnboxIntBinop"));
        assert!(is_traceable("LocalRead"));
        assert!(is_traceable("FunctionCall"));
        assert!(!is_traceable("NonExistent"));
    }

    #[test]
    fn test_classification_lookup() {
        // Find BinaryOp pattern
        let cls = TRACE_PATTERNS
            .iter()
            .find(|(pat, _)| pat.contains("BinaryOp"))
            .map(|(_, cls)| *cls);
        assert!(cls.is_some());
        assert!(cls.unwrap().contains("UnboxIntBinop"));
    }
}

//! pyre-mjit: Auto-generated JIT for pyre.
//!
//! This crate is the Rust equivalent of RPython's `rpython/jit/` —
//! it contains automatically generated tracing code produced by
//! `majit-analyze` during build time.
//!
//! The interpreter (`pyre-interp`) has zero JIT dependencies.
//! All JIT logic lives here, generated from static analysis of
//! the interpreter's source code.

// Include auto-generated trace dispatch table and helpers
include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generated_dispatch_table() {
        // Verify the auto-generated code compiled and has content
        assert!(
            TRACE_PATTERNS.len() > 20,
            "expected >20 patterns, got {}",
            TRACE_PATTERNS.len()
        );

        // Check key patterns exist
        let has_binary = TRACE_PATTERNS
            .iter()
            .any(|(_, p)| p.contains("UnboxIntBinop"));
        assert!(has_binary, "missing UnboxIntBinop pattern");

        let has_local_read = TRACE_PATTERNS
            .iter()
            .any(|(_, p)| p.contains("LocalRead"));
        assert!(has_local_read, "missing LocalRead pattern");

        let has_call = TRACE_PATTERNS
            .iter()
            .any(|(_, p)| p.contains("FunctionCall"));
        assert!(has_call, "missing FunctionCall pattern");

        eprintln!("Auto-generated {} trace patterns", TRACE_PATTERNS.len());
        for (pat, cls) in TRACE_PATTERNS {
            if *cls != "Unclassified" {
                eprintln!("  {} → {}", pat, cls);
            }
        }
    }
}

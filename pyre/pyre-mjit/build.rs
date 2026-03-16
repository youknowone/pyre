/// Build script for pyre-mjit: runs majit-analyze on interpreter source
/// to auto-generate tracing code. This is the Rust equivalent of
/// RPython's translation pipeline.
fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let pyre_base = format!("{manifest_dir}/..");

    // Read interpreter source files
    let opcode_step = std::fs::read_to_string(format!(
        "{pyre_base}/pyre-runtime/src/opcode_step.rs"
    ))
    .expect("failed to read opcode_step.rs");

    let eval = std::fs::read_to_string(format!(
        "{pyre_base}/pyre-interp/src/eval.rs"
    ))
    .expect("failed to read eval.rs");

    // Run analysis
    let result = majit_analyze::analyze_multiple(&[&opcode_step, &eval]);

    // Generate tracing code
    let code = majit_analyze::generate_trace_code(&result);

    // Write to OUT_DIR
    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out_dir}/jit_trace_gen.rs"), &code).unwrap();

    // Also generate JSON metadata for debugging
    let json = serde_json::to_string_pretty(&result).unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), &json).unwrap();

    // Report
    eprintln!(
        "[pyre-mjit build.rs] analyzed {} opcodes ({} classified), generated {} bytes",
        result.opcodes.len(),
        result.opcodes.iter().filter(|a| a.trace_pattern.is_some()).count(),
        code.len(),
    );

    // Rerun if source changes
    println!("cargo::rerun-if-changed={pyre_base}/pyre-runtime/src/opcode_step.rs");
    println!("cargo::rerun-if-changed={pyre_base}/pyre-interp/src/eval.rs");
}

/// Build script for pyre-mjit: runs majit-analyze on the ENTIRE pyre
/// interpreter to auto-generate tracing code. This is the Rust
/// equivalent of RPython's translation pipeline.
///
/// Analyzes all source files from:
/// - pyre-object (Python object types: W_IntObject, W_FloatObject, etc.)
/// - pyre-runtime (opcode step, shared handlers, runtime ops)
/// - pyre-interp (PyFrame, eval loop, JIT hints)
fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let pyre_base = format!("{manifest_dir}/..");

    // Collect ALL source files from the pyre interpreter crates
    let source_dirs = [
        format!("{pyre_base}/pyre-object/src"),
        format!("{pyre_base}/pyre-runtime/src"),
        format!("{pyre_base}/pyre-interp/src"),
    ];

    let mut sources = Vec::new();
    let mut source_paths = Vec::new();

    for dir in &source_dirs {
        collect_rs_files(dir, &mut sources, &mut source_paths);
    }

    eprintln!(
        "[pyre-mjit build.rs] reading {} source files from {} dirs: {:?}",
        sources.len(),
        source_dirs.len(),
        source_paths,
    );

    // Run analysis on ALL files
    let source_refs: Vec<&str> = sources.iter().map(|s| s.as_str()).collect();
    let result = majit_analyze::analyze_multiple(&source_refs);

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
        "[pyre-mjit build.rs] analyzed {} opcodes ({} classified), {} helpers, {} types, {} trait impls, generated {} bytes",
        result.opcodes.len(),
        result
            .opcodes
            .iter()
            .filter(|a| a.trace_pattern.is_some())
            .count(),
        result.helpers.len(),
        result.type_layouts.len(),
        result.trait_impls.len(),
        code.len(),
    );

    // Rerun if any source file changes
    for path in &source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
}

/// Recursively collect all .rs files from a directory.
fn collect_rs_files(dir: &str, sources: &mut Vec<String>, paths: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("[pyre-mjit build.rs] warning: cannot read {dir}");
        return;
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path.to_string_lossy(), sources, paths);
        } else if path.extension().map_or(false, |ext| ext == "rs") {
            let path_str = path.to_string_lossy().to_string();
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    paths.push(path_str);
                    sources.push(content);
                }
                Err(e) => {
                    eprintln!("[pyre-mjit build.rs] warning: cannot read {path_str}: {e}");
                }
            }
        }
    }
}

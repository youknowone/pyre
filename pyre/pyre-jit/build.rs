#[path = "src/jit/virtualizable_spec.rs"]
mod virtualizable_spec;

/// Build script for pyre-jit: runs majit-analyze on the ENTIRE pyre
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
        "[pyre-jit build.rs] reading {} source files from {} dirs: {:?}",
        sources.len(),
        source_dirs.len(),
        source_paths,
    );

    // Run analysis on ALL files with PyFrame virtualizable metadata.
    //
    // This keeps the generic analyzer closer to the proc-macro/runtime path:
    // graph rewrite can recognize `next_instr`, `valuestackdepth`, and
    // `locals_cells_stack_w[*]` as virtualizable accesses before legacy
    // TracePattern classification runs.
    let source_refs: Vec<&str> = sources.iter().map(|s| s.as_str()).collect();
    let result = majit_analyze::analyze_multiple_full_with_config(
        &source_refs,
        &majit_analyze::AnalyzeConfig {
            pipeline: majit_analyze::PipelineConfig {
                transform: majit_analyze::GraphTransformConfig {
                    vable_fields: virtualizable_spec::PYFRAME_VABLE_FIELDS
                        .iter()
                        .map(|(name, idx)| ((*name).to_string(), *idx))
                        .collect(),
                    vable_arrays: virtualizable_spec::PYFRAME_VABLE_ARRAYS
                        .iter()
                        .map(|(name, idx)| ((*name).to_string(), *idx))
                        .collect(),
                    ..Default::default()
                },
            },
        },
    );

    // Generate tracing code from the canonical graph-first analysis result.
    let code = majit_analyze::generate_trace_code_from_full(&result);

    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out_dir}/jit_trace_gen.rs"), &code).unwrap();

    // ── New path: extract opcode match from pyre-runtime and generate
    //    JIT mainloop via codewriter (like aheui-mjit). ──
    //    RPython equivalent: codewriter.transform_graph_to_jitcode()
    let opcode_step_path = format!("{pyre_base}/pyre-runtime/src/opcode_step.rs");
    if let Ok(opcode_src) = std::fs::read_to_string(&opcode_step_path) {
        if let Ok(file) = syn::parse_str::<syn::File>(&opcode_src) {
            use majit_analyze::interp_extract::{find_function, find_opcode_match};
            if let Some(func) = find_function(&file, "execute_opcode_step") {
                if let Some(opcode_match) = find_opcode_match(func) {
                    eprintln!(
                        "[pyre-jit build.rs] extracted opcode match: {} arms from execute_opcode_step",
                        opcode_match.arms.len()
                    );
                    // TODO: construct JitDriverConfig for pyre and call
                    // codewriter::generate_jitcode(opcode_match, &binops, &config)
                    // to generate jit_mainloop_gen.rs
                } else {
                    eprintln!(
                        "[pyre-jit build.rs] warning: no opcode match found in execute_opcode_step"
                    );
                }
            } else {
                eprintln!("[pyre-jit build.rs] warning: execute_opcode_step not found");
            }
        }
    }

    // JSON metadata for debugging
    let json = serde_json::to_string_pretty(&result).unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), &json).unwrap();

    // Report
    eprintln!(
        "[pyre-jit build.rs] analyzed {} opcodes ({} classified), {} helpers, {} types, {} trait impls, generated {} bytes",
        result.legacy.opcodes.len(),
        result
            .legacy
            .opcodes
            .iter()
            .filter(|a| a.trace_pattern.is_some())
            .count(),
        result.legacy.helpers.len(),
        result.legacy.type_layouts.len(),
        result.legacy.trait_impls.len(),
        code.len(),
    );

    // Rerun if any source file changes
    for path in &source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
    println!("cargo::rerun-if-changed=src/jit/virtualizable_spec.rs");
}

/// Recursively collect all .rs files from a directory.
fn collect_rs_files(dir: &str, sources: &mut Vec<String>, paths: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("[pyre-jit build.rs] warning: cannot read {dir}");
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
                    eprintln!("[pyre-jit build.rs] warning: cannot read {path_str}: {e}");
                }
            }
        }
    }
}

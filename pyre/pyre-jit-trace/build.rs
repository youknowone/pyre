#[path = "src/call_spec.rs"]
mod call_spec;
#[path = "src/virtualizable_spec.rs"]
mod virtualizable_spec;

/// Build script for pyre-jit: runs majit-codewriter on the active pyre
/// interpreter to auto-generate tracing code. This is the Rust
/// equivalent of RPython's translation pipeline.
///
/// Analyzes all source files from:
/// - pyre-object (Python object types: W_IntObject, W_FloatObject, etc.)
/// - pyre-interpreter (object space, bytecode dispatch, eval loop)
fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let pyre_base = format!("{manifest_dir}/..");

    // Collect ALL source files from the active interpreter crates.
    let source_dirs = [
        format!("{pyre_base}/pyre-object/src"),
        format!("{pyre_base}/pyre-interpreter/src"),
    ];

    let mut sources = Vec::new();
    let mut source_paths = Vec::new();

    for dir in &source_dirs {
        collect_rs_files(dir, &mut sources, &mut source_paths);
    }

    eprintln!(
        "[pyre-jit-trace build.rs] reading {} source files from {} dirs: {:?}",
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
    let pipeline = majit_codewriter::analyze_multiple_pipeline_with_config(
        &source_refs,
        &majit_codewriter::AnalyzeConfig {
            pipeline: majit_codewriter::PipelineConfig {
                transform: majit_codewriter::GraphTransformConfig {
                    vable_fields: virtualizable_spec::PYFRAME_VABLE_FIELDS
                        .iter()
                        .map(|(name, idx)| {
                            majit_codewriter::VirtualizableFieldDescriptor::new(
                                *name,
                                Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                                *idx,
                            )
                        })
                        .collect(),
                    vable_arrays: virtualizable_spec::PYFRAME_VABLE_ARRAYS
                        .iter()
                        .map(|(name, idx)| {
                            majit_codewriter::VirtualizableFieldDescriptor::new(
                                *name,
                                Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                                *idx,
                            )
                        })
                        .collect(),
                    call_effects: build_call_effect_overrides(),
                    ..Default::default()
                },
            },
        },
    );

    // Generate tracing code from the canonical graph-first analysis result.
    let code = majit_codewriter::generate_trace_code_from_pipeline(&pipeline);

    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out_dir}/jit_trace_gen.rs"), &code).unwrap();

    // Trait impls live in a separate file because lib.rs `include!`s
    // jit_trace_gen.rs twice (once at crate root and once inside
    // `pub mod generated`). Trait impls would conflict (E0119) under double
    // inclusion, so they live in their own file included only once.
    //
    // The source-of-truth is `src/opcode_handler_impls.template.rs` — a
    // hand-maintained transcription of what trace_opcode.rs used to contain.
    // build.rs copies it verbatim into OUT_DIR; `tests/trait_impls_snapshot.rs`
    // guards against drift. Future phases will replace the template with
    // codegen-derived content, but the include path stays the same.
    let template_path = format!("{manifest_dir}/src/opcode_handler_impls.template.rs");
    let trait_impls_code = std::fs::read_to_string(&template_path).unwrap_or_else(|e| {
        panic!("[pyre-jit-trace build.rs] cannot read {template_path}: {e}");
    });
    std::fs::write(
        format!("{out_dir}/jit_trace_trait_impls.rs"),
        &trait_impls_code,
    )
    .unwrap();
    println!("cargo::rerun-if-changed={template_path}");

    // JSON metadata for debugging
    let json = serde_json::to_string_pretty(&pipeline).unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), &json).unwrap();

    // ── Per-opcode JitCode artifact (Phase C of eval-loop plan) ──
    //
    // Each PipelineOpcodeArm has a `jitcode: Option<JitCode>` produced by
    // assembling its flattened SSARepr. We serialize the per-opcode jitcodes
    // as a separate JSON file so the runtime loader (`jitcode_table.rs`)
    // can `include_str!` it without pulling in the full pipeline metadata.
    //
    // Format: `[ {"selector": "...", "jitcode": {...}}, ... ]`
    // The JitCode payload follows the assembler's serde derive.
    #[derive(serde::Serialize)]
    struct OpcodeJitCodeEntry<'a> {
        selector: String,
        jitcode: &'a majit_codewriter::assembler::JitCode,
    }
    let opcode_jitcode_entries: Vec<OpcodeJitCodeEntry<'_>> = pipeline
        .opcode_dispatch
        .iter()
        .filter_map(|arm| {
            arm.jitcode.as_ref().map(|jc| OpcodeJitCodeEntry {
                selector: arm.selector.canonical_key(),
                jitcode: jc,
            })
        })
        .collect();
    let opcode_jitcodes_json = serde_json::to_string(&opcode_jitcode_entries).unwrap();
    std::fs::write(
        format!("{out_dir}/pyre_opcode_jitcodes.json"),
        &opcode_jitcodes_json,
    )
    .unwrap();
    let total_jitcode_bytes: usize = opcode_jitcode_entries
        .iter()
        .map(|e| e.jitcode.code.len())
        .sum();

    // Report
    eprintln!(
        "[pyre-jit-trace build.rs] canonical analysis: {} opcode arms ({} flattened, {} jitcoded, {} jitcode bytes), {} functions, {} blocks, {} flat ops, generated {} bytes",
        pipeline.opcode_dispatch.len(),
        pipeline
            .opcode_dispatch
            .iter()
            .filter(|arm| arm.flattened.is_some())
            .count(),
        opcode_jitcode_entries.len(),
        total_jitcode_bytes,
        pipeline.functions.len(),
        pipeline.total_blocks,
        pipeline.total_ops,
        code.len(),
    );

    // Rerun if any source file changes
    for path in &source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
    println!("cargo::rerun-if-changed=src/virtualizable_spec.rs");
    println!("cargo::rerun-if-changed=src/call_spec.rs");
}

fn build_call_effect_overrides() -> Vec<majit_codewriter::CallEffectOverride> {
    call_spec::PYFRAME_CALL_EFFECTS
        .iter()
        .map(|spec| {
            let target = match spec.target {
                call_spec::CallTargetSpec::Method {
                    name,
                    receiver_root,
                } => majit_codewriter::CallTarget::method(name, Some(receiver_root.to_string())),
                call_spec::CallTargetSpec::FunctionPath(segments) => {
                    majit_codewriter::CallTarget::function_path(segments.iter().copied())
                }
            };
            let effect = match spec.effect {
                call_spec::CallEffectKind::Elidable => majit_codewriter::CallEffectKind::Elidable,
                call_spec::CallEffectKind::Residual => majit_codewriter::CallEffectKind::Residual,
            };
            majit_codewriter::CallEffectOverride::new(target, effect)
        })
        .collect()
}

/// Recursively collect all .rs files from a directory.
fn collect_rs_files(dir: &str, sources: &mut Vec<String>, paths: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("[pyre-jit-trace build.rs] warning: cannot read {dir}");
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
                    eprintln!("[pyre-jit-trace build.rs] warning: cannot read {path_str}: {e}");
                }
            }
        }
    }
}

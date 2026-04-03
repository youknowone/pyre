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
                classify: build_classification_config(),
            },
        },
    );

    // Generate tracing code from the canonical graph-first analysis result.
    let code = majit_codewriter::generate_trace_code_from_pipeline(&pipeline);

    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out_dir}/jit_trace_gen.rs"), &code).unwrap();

    // JSON metadata for debugging
    let json = serde_json::to_string_pretty(&pipeline).unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), &json).unwrap();

    // Report
    eprintln!(
        "[pyre-jit-trace build.rs] canonical analysis: {} opcode arms ({} graph-classified), {} functions, {} blocks, {} flat ops, generated {} bytes",
        pipeline.opcode_dispatch.len(),
        pipeline
            .opcode_dispatch
            .iter()
            .filter(|arm| arm.classified_pattern.is_some())
            .count(),
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

fn build_classification_config() -> majit_codewriter::ClassificationConfig {
    majit_codewriter::ClassificationConfig {
        field_roles: virtualizable_spec::PYFRAME_FIELD_ROLES
            .iter()
            .map(|spec| {
                let role = match spec.role {
                    virtualizable_spec::FieldPatternRole::LocalArray => {
                        majit_codewriter::FieldPatternRole::LocalArray
                    }
                    virtualizable_spec::FieldPatternRole::InstructionPosition => {
                        majit_codewriter::FieldPatternRole::InstructionPosition
                    }
                    virtualizable_spec::FieldPatternRole::ConstantPool => {
                        majit_codewriter::FieldPatternRole::ConstantPool
                    }
                };
                majit_codewriter::FieldRoleDescriptor::new(
                    spec.name,
                    Some(spec.owner_root.to_string()),
                    role,
                )
            })
            .collect(),
        call_roles: call_spec::PYFRAME_CALL_EFFECTS
            .iter()
            .filter_map(|spec| {
                let role = spec.role?;
                let target = match spec.target {
                    call_spec::CallTargetSpec::Method {
                        name,
                        receiver_root,
                    } => {
                        majit_codewriter::CallTarget::method(name, Some(receiver_root.to_string()))
                    }
                    call_spec::CallTargetSpec::FunctionPath(segments) => {
                        majit_codewriter::CallTarget::function_path(segments.iter().copied())
                    }
                };
                let role = match role {
                    call_spec::CallPatternRole::IntArithmetic => {
                        majit_codewriter::CallPatternRole::IntArithmetic
                    }
                    call_spec::CallPatternRole::FloatArithmetic => {
                        majit_codewriter::CallPatternRole::FloatArithmetic
                    }
                    call_spec::CallPatternRole::LocalRead => {
                        majit_codewriter::CallPatternRole::LocalRead
                    }
                    call_spec::CallPatternRole::LocalWrite => {
                        majit_codewriter::CallPatternRole::LocalWrite
                    }
                    call_spec::CallPatternRole::FunctionCall => {
                        majit_codewriter::CallPatternRole::FunctionCall
                    }
                    call_spec::CallPatternRole::TruthCheck => {
                        majit_codewriter::CallPatternRole::TruthCheck
                    }
                    call_spec::CallPatternRole::StackManip => {
                        majit_codewriter::CallPatternRole::StackManip
                    }
                    call_spec::CallPatternRole::ConstLoad => {
                        majit_codewriter::CallPatternRole::ConstLoad
                    }
                    call_spec::CallPatternRole::NamespaceLoadLocal => {
                        majit_codewriter::CallPatternRole::NamespaceLoadLocal
                    }
                    call_spec::CallPatternRole::NamespaceLoadGlobal => {
                        majit_codewriter::CallPatternRole::NamespaceLoadGlobal
                    }
                    call_spec::CallPatternRole::NamespaceStoreLocal => {
                        majit_codewriter::CallPatternRole::NamespaceStoreLocal
                    }
                    call_spec::CallPatternRole::NamespaceStoreGlobal => {
                        majit_codewriter::CallPatternRole::NamespaceStoreGlobal
                    }
                    call_spec::CallPatternRole::RangeIterNext => {
                        majit_codewriter::CallPatternRole::RangeIterNext
                    }
                    call_spec::CallPatternRole::IterCleanup => {
                        majit_codewriter::CallPatternRole::IterCleanup
                    }
                    call_spec::CallPatternRole::Return => majit_codewriter::CallPatternRole::Return,
                    call_spec::CallPatternRole::BuildList => {
                        majit_codewriter::CallPatternRole::BuildList
                    }
                    call_spec::CallPatternRole::BuildTuple => {
                        majit_codewriter::CallPatternRole::BuildTuple
                    }
                    call_spec::CallPatternRole::UnpackSequence => {
                        majit_codewriter::CallPatternRole::UnpackSequence
                    }
                    call_spec::CallPatternRole::SequenceSetitem => {
                        majit_codewriter::CallPatternRole::SequenceSetitem
                    }
                    call_spec::CallPatternRole::CollectionAppend => {
                        majit_codewriter::CallPatternRole::CollectionAppend
                    }
                };
                Some(majit_codewriter::CallRoleDescriptor::new(target, role))
            })
            .collect(),
    }
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

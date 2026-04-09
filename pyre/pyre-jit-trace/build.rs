#[path = "src/call_spec.rs"]
mod call_spec;
#[path = "src/virtualizable_spec.rs"]
mod virtualizable_spec;

use std::collections::HashSet;

/// One entry in `pyre_opcode_dispatch_manifest.json`.
///
/// Phase C v2: maps a single Python `Instruction::*` variant to the index
/// of its assembled handler jitcode in `pipeline.jitcodes`.
///
/// Multi-pattern match arms in `execute_opcode_step` (`A | B`) expand into
/// one manifest entry per variant; both share the same `arm_id`,
/// `selector`, and `jitcode_index` but list `variant` separately so the
/// runtime can do an O(1) variant→index lookup.
#[derive(Debug, serde::Serialize)]
struct OpcodeDispatchManifestEntry {
    /// Stable arm id assigned by the parser (extract order).
    arm_id: usize,
    /// Display label for the whole arm (e.g. `"Instruction::LoadFast | Instruction::LoadFastBorrow"`).
    selector: String,
    /// One specific Python `Instruction::*` variant covered by this arm.
    variant: String,
    /// Index into `pipeline.jitcodes` where the assembled handler lives.
    jitcode_index: usize,
    /// Synthetic CallPath used to register the arm body in CallControl.
    debug_name: String,
}

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
    // Assembled from THREE pieces (Phase B of the eval-loop automation plan):
    //   1. `opcode_handler_impls_pre.template.rs` — header + variant
    //      `SharedOpcodeHandler` impl (transcription).
    //   2. `majit_codewriter::handler_spec::emit_simple_trait_impls()` —
    //      the 5 simple traits (Constant/Stack/Truth/Iter/Local), emitted
    //      from the spec table in majit-codewriter/src/handler_spec.rs.
    //   3. `opcode_handler_impls_post.template.rs` — remaining variant
    //      `ControlFlow/Branch/Namespace/Arithmetic` impls (transcription).
    //
    // `tests/trait_impls_snapshot.rs` guards against drift by comparing the
    // assembled output against a checked-in snapshot.
    let pre_path = format!("{manifest_dir}/src/opcode_handler_impls_pre.template.rs");
    let post_path = format!("{manifest_dir}/src/opcode_handler_impls_post.template.rs");
    let pre = std::fs::read_to_string(&pre_path).unwrap_or_else(|e| {
        panic!("[pyre-jit-trace build.rs] cannot read {pre_path}: {e}");
    });
    let post = std::fs::read_to_string(&post_path).unwrap_or_else(|e| {
        panic!("[pyre-jit-trace build.rs] cannot read {post_path}: {e}");
    });
    let simple = majit_codewriter::handler_spec::emit_simple_trait_impls();
    // pre ends with `}\n\n` (Shared close + blank), simple ends with `}\n`,
    // post starts with `\n` (blank). Concat = `...}\n\nimpl Constant...}\n\nimpl ControlFlow...`
    // which matches the original single-template structure byte-for-byte.
    let trait_impls_code = format!("{pre}{simple}{post}");
    std::fs::write(
        format!("{out_dir}/jit_trace_trait_impls.rs"),
        &trait_impls_code,
    )
    .unwrap();
    println!("cargo::rerun-if-changed={pre_path}");
    println!("cargo::rerun-if-changed={post_path}");

    // JSON metadata for debugging
    let json = serde_json::to_string_pretty(&pipeline).unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), &json).unwrap();

    // ── Phase C v2: opcode-dispatch manifest (RPython orthodox) ──
    //
    // RPython parity: PyPy generates one JitCode per opcode-handler method.
    // pyre dispatches inside one big match, so each arm body has been
    // registered as a synthetic graph in CallControl and pulled through
    // the orthodox `drain_pending_graphs` loop. The result lives at
    // `pipeline.jitcodes[arm.entry_jitcode_index]`.
    //
    // The manifest is the *only* runtime artifact; it maps Python
    // `Instruction::*` variants to the jitcode_index that holds the
    // assembled handler bytecode. raw `assembler::JitCode` blobs stay in
    // `jit_metadata.json` for debugging only.
    //
    // Multi-pattern match arms (`A | B`) expand into one manifest entry
    // per variant. If the same variant appears under multiple arms (Rust
    // match: first wins), only the first occurrence is recorded.
    let manifest_entries = build_dispatch_manifest(&pipeline);
    let manifest_json = serde_json::to_string_pretty(&manifest_entries).unwrap();
    std::fs::write(
        format!("{out_dir}/pyre_opcode_dispatch_manifest.json"),
        &manifest_json,
    )
    .unwrap();

    // Report
    let arms_with_jitcode = pipeline
        .opcode_dispatch
        .iter()
        .filter(|arm| arm.entry_jitcode_index.is_some())
        .count();
    eprintln!(
        "[pyre-jit-trace build.rs] canonical analysis: {} opcode arms ({} flattened, {} indexed), {} functions, {} blocks, {} flat ops, {} all_jitcodes, manifest: {} variants, generated {} bytes",
        pipeline.opcode_dispatch.len(),
        pipeline
            .opcode_dispatch
            .iter()
            .filter(|arm| arm.flattened.is_some())
            .count(),
        arms_with_jitcode,
        pipeline.functions.len(),
        pipeline.total_blocks,
        pipeline.total_ops,
        pipeline.jitcodes.len(),
        manifest_entries.len(),
        code.len(),
    );

    // Rerun if any source file changes
    for path in &source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
    println!("cargo::rerun-if-changed=src/virtualizable_spec.rs");
    println!("cargo::rerun-if-changed=src/call_spec.rs");
}

/// Build the per-variant dispatch manifest from the canonical pipeline result.
///
/// Walks `pipeline.opcode_dispatch` in arm order. For each arm whose body
/// has been registered as a synthetic graph and assembled into
/// `pipeline.jitcodes[entry_jitcode_index]`, expand its selector into the
/// individual `Instruction::*` variants it covers and emit one manifest
/// entry per variant.
///
/// Rust match semantics guarantee "first arm wins" on duplicates, so if
/// the same variant string appears in two different arms (which the parser
/// already rejects via `reject_duplicate_opcode_selectors`, but this guard
/// makes the property local) only the lower-`arm_id` entry is recorded.
fn build_dispatch_manifest(
    pipeline: &majit_codewriter::ProgramPipelineResult,
) -> Vec<OpcodeDispatchManifestEntry> {
    let mut entries = Vec::new();
    let mut seen_variants: HashSet<String> = HashSet::new();
    for arm in &pipeline.opcode_dispatch {
        let Some(jitcode_index) = arm.entry_jitcode_index else {
            continue;
        };
        let selector_label = arm.selector.canonical_key();
        let debug_name = format!("__opcode_dispatch__/{}#{}", selector_label, arm.arm_id);
        for variant in expand_selector_variants(&arm.selector) {
            if !seen_variants.insert(variant.clone()) {
                eprintln!(
                    "[pyre-jit-trace build.rs] manifest: duplicate variant {variant:?} \
                     under arm_id={} (selector {:?}); first occurrence wins",
                    arm.arm_id, selector_label,
                );
                continue;
            }
            entries.push(OpcodeDispatchManifestEntry {
                arm_id: arm.arm_id,
                selector: selector_label.clone(),
                variant,
                jitcode_index,
                debug_name: debug_name.clone(),
            });
        }
    }
    entries
}

/// Expand a `OpcodeDispatchSelector` into the individual variant strings it
/// covers. `Path` selectors yield one entry; `Or` selectors expand recursively;
/// `Wildcard` and `Unsupported` yield none (they cannot be addressed by
/// `Instruction::*` lookup at runtime).
fn expand_selector_variants(selector: &majit_codewriter::OpcodeDispatchSelector) -> Vec<String> {
    use majit_codewriter::OpcodeDispatchSelector;
    match selector {
        OpcodeDispatchSelector::Path(path) => vec![path.canonical_key()],
        OpcodeDispatchSelector::Or(cases) => {
            cases.iter().flat_map(expand_selector_variants).collect()
        }
        OpcodeDispatchSelector::Wildcard | OpcodeDispatchSelector::Unsupported => Vec::new(),
    }
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

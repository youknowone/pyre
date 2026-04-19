#[path = "src/call_spec.rs"]
mod call_spec;
#[path = "src/virtualizable_spec.rs"]
mod virtualizable_spec;

use walkdir::WalkDir;

/// Build script for pyre-jit: runs majit-translate on the active pyre
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
    let analyze_config = majit_translate::AnalyzeConfig {
        pipeline: majit_translate::PipelineConfig {
            transform: majit_translate::GraphTransformConfig {
                vable_fields: virtualizable_spec::PYFRAME_VABLE_FIELDS
                    .iter()
                    .map(|(name, idx)| {
                        majit_translate::VirtualizableFieldDescriptor::new(
                            *name,
                            Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                            *idx,
                        )
                    })
                    .collect(),
                vable_arrays: virtualizable_spec::PYFRAME_VABLE_ARRAYS
                    .iter()
                    .map(|(name, idx)| {
                        // virtualizable.py:58 — VirtualizableInfo.array_descrs[i] =
                        // cpu.arraydescrof(getattr(VTYPE, name).TO). Python frame
                        // locals are PyObjectRef pointers: itemsize=8, is_signed=false.
                        majit_translate::VirtualizableFieldDescriptor::new_with_arraydescr(
                            *name,
                            Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                            *idx,
                            8,     // itemsize: PyObjectRef is a pointer
                            false, // is_signed: pointers are unsigned
                        )
                    })
                    .collect(),
                call_effects: build_call_effect_overrides(),
                ..Default::default()
            },
            portal: None,
        },
    };
    // warmspot.py:516 `vinfos[VTYPEPTR] = VirtualizableInfo(self, VTYPEPTR)` —
    // pyre's runtime `VirtualizableInfo` constructor lives in the
    // `majit-metainterp` crate (`__build_virtualizable_info`) and runs
    // at `JitDriver::new` (jitdriver.rs:285) where the field offsets
    // resolved by `mem::offset_of!` are available.  build.rs cannot
    // import that crate (no metainterp build-dep, and the offsets are
    // a runtime fact), so the codewriter-side factory returns `None`
    // here; the codewriter slot stays empty until the runtime metainterp
    // setter overrides it.  PRE-EXISTING-ADAPTATION documented at
    // `CallControl::make_virtualizable_infos`.
    let vinfo_factory: &majit_translate::VirtualizableInfoFactory<'_> = &|_jd_idx, _vtype| None;
    let pipeline = majit_translate::analyze_multiple_pipeline_with_vinfo_factory(
        &source_refs,
        &analyze_config,
        None,
        vinfo_factory,
    );

    // Generate tracing code from the canonical graph-first analysis result.
    let code = majit_translate::generate_trace_code_from_pipeline(&pipeline);

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
    //   2. `majit_translate::handler_spec::emit_simple_trait_impls()` —
    //      the 5 simple traits (Constant/Stack/Truth/Iter/Local), emitted
    //      from the spec table in majit-translate/src/handler_spec.rs.
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
    let simple = majit_translate::handler_spec::emit_simple_trait_impls();
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

    // Report
    let arms_with_jitcode = pipeline
        .opcode_dispatch
        .iter()
        .filter(|arm| arm.entry_jitcode_index.is_some())
        .count();
    eprintln!(
        "[pyre-jit-trace build.rs] canonical analysis: {} opcode arms ({} flattened, {} indexed), {} functions, {} blocks, {} flat ops, {} all_jitcodes, generated {} bytes",
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
        code.len(),
    );

    // Rerun if any source file changes
    for path in &source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
    println!("cargo::rerun-if-changed=src/virtualizable_spec.rs");
    println!("cargo::rerun-if-changed=src/call_spec.rs");
}

fn build_call_effect_overrides() -> Vec<majit_translate::CallEffectOverride> {
    call_spec::PYFRAME_CALL_EFFECTS
        .iter()
        .map(|spec| {
            let target = match spec.target {
                call_spec::CallTargetSpec::Method {
                    name,
                    receiver_root,
                } => majit_translate::CallTarget::method(name, Some(receiver_root.to_string())),
                call_spec::CallTargetSpec::FunctionPath(segments) => {
                    majit_translate::CallTarget::function_path(segments.iter().copied())
                }
            };
            let effect = match spec.effect {
                call_spec::CallEffectKind::Elidable => majit_translate::CallEffectKind::Elidable,
                call_spec::CallEffectKind::Residual => majit_translate::CallEffectKind::Residual,
            };
            majit_translate::CallEffectOverride::new(target, effect)
        })
        .collect()
}

/// Collect all `.rs` files from a directory tree.
fn collect_rs_files(dir: &str, sources: &mut Vec<String>, paths: &mut Vec<String>) {
    for entry in WalkDir::new(dir) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        let path = entry.path();
        let path_str = path.to_string_lossy().to_string();
        match std::fs::read_to_string(path) {
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

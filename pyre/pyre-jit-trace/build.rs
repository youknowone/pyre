#[path = "src/call_spec.rs"]
mod call_spec;
#[path = "src/virtualizable_spec.rs"]
mod virtualizable_spec;

use walkdir::WalkDir;

const CODEGEN_CACHE_VERSION: &str = "pyre-jit-trace-codegen-cache-v2";
const CODEGEN_OUTPUTS: &[&str] = &[
    "jit_trace_gen.rs",
    "jit_metadata.json",
    "jitcodes.bin",
    "jit_drivers.bin",
    "insns.bin",
    "descrs.bin",
    "fnaddr_bindings.bin",
    "static_pytype_bindings.bin",
    "static_ref_bindings.bin",
];

/// Build script for pyre-jit: runs majit-translate on the active pyre
/// interpreter to auto-generate tracing code. This is the Rust
/// equivalent of RPython's translation pipeline.
///
/// Analyzes all source files from:
/// - pyre-object (Python object types: W_IntObject, W_FloatObject, etc.)
/// - pyre-interpreter (object space, bytecode dispatch, eval loop)
fn main() {
    println!("cargo::rerun-if-env-changed=MAJIT_LLBC_EXTRACTION");
    if std::env::var_os("MAJIT_LLBC_EXTRACTION").as_deref() == Some(std::ffi::OsStr::new("1")) {
        emit_llbc_extraction_placeholders();
        return;
    }
    // Fail fast with an actionable message when the Charon-extracted LLBC
    // artefacts the codegen consumes are absent.  Without this, the missing
    // set surfaces deep inside `real_main` as a worker-thread `panic!`
    // (`build-script worker panicked: Any { .. }`) printed *below* the
    // 150-line source-path dump, where it is easy to miss.
    preflight_llbc_or_fail();

    // The codegen worker (`real_main`) runs the RPythonTyper
    // specialization (`specialize_legacy_graph_with_registry` → annotator
    // `complete_pending_blocks` / rtyper `specialize_more_blocks`).  Its
    // visit order is keyed off the annotator/rtyper worklist maps
    // (`genpendingblocks`, `annotated`, `all_blocks`, …), which are
    // insertion-ordered `IndexMap`s, so the order in which the callee
    // specialization chain is walked is deterministic and independent of the
    // per-process SipHash seed.  A single in-process run suffices, matching
    // RPython's single-shot translator.  The 1 GiB thread stack is needed
    // for syn's recursive parse of ~150 files plus the rtyper chain
    // (on Windows the main thread's 1 MiB default would
    // STATUS_STACK_OVERFLOW).
    run_worker();
}

/// Break the LLBC bootstrap dependency cycle explicitly.
///
/// Charon compiling `pyre-jit` must compile its `pyre-jit-trace` dependency
/// before `pyre-jit.ullbc` exists. None of these artifacts execute during
/// extraction; they only need to satisfy `include!` / `include_bytes!` so
/// rustc can expose `pyre-jit`'s MIR. The next normal Cargo build observes
/// `MAJIT_LLBC_EXTRACTION` changing and replaces every placeholder.
fn emit_llbc_extraction_placeholders() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set");
    std::fs::write(
        format!("{out_dir}/jit_trace_gen.rs"),
        "pub const COMPILED_JIT_DRIVERS: &[(&str, usize)] = &[];\n\
         pub const CANONICAL_JITCODES: &[(&str, usize)] = &[];\n",
    )
    .unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), b"{}\n").unwrap();
    std::fs::write(
        format!("{out_dir}/jitcodes.bin"),
        bincode::serialize(&Vec::<std::sync::Arc<majit_translate::jitcode::JitCode>>::new())
            .unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/jit_drivers.bin"),
        bincode::serialize(&Vec::<majit_translate::CompiledJitDriver>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/insns.bin"),
        bincode::serialize(&std::collections::BTreeMap::<String, u8>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/descrs.bin"),
        bincode::serialize(&Vec::<majit_translate::jitcode::BhDescr>::new()).unwrap(),
    )
    .unwrap();
    for name in [
        "fnaddr_bindings.bin",
        "static_pytype_bindings.bin",
        "static_ref_bindings.bin",
    ] {
        std::fs::write(
            format!("{out_dir}/{name}"),
            bincode::serialize(&Vec::<(String, i64)>::new()).unwrap(),
        )
        .unwrap();
    }
}

/// Pre-flight the LLBC prerequisite, mirroring the resolution order in
/// `majit-translate` (`build_semantic_program_via_active_frontend`):
/// honour the `PYRE_MIR_FRONTEND_LLBC` override, else require the canonical
/// `build/llbc/{pyre-object,pyre-interpreter,pyre-jit}.ullbc` set. The third
/// artifact contains the exact `eval::eval_loop_jit` portal.
///
/// When neither resolves, emit a clean, copy-pasteable bootstrap message
/// and fail the build *before* the worker spawns — so the contributor
/// sees the exact steps to run instead of a worker-thread panic buried
/// under the source-file dump.  Auto-running the bootstrap from here is
/// deliberately avoided: `scripts/extract-llbc.py` shells out to a nested
/// `cargo build`, which would block on the outer build's target-directory
/// lock (deadlock), and a build script that downloads a toolchain breaks
/// hermetic / offline / CI builds.
fn preflight_llbc_or_fail() {
    // Explicit override: trust it and let the translator validate the
    // individual paths (its loader panics per-file with the bad path).
    if std::env::var_os("PYRE_MIR_FRONTEND_LLBC")
        .map(|v| std::env::split_paths(&v).any(|p| !p.as_os_str().is_empty()))
        .unwrap_or(false)
    {
        return;
    }

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let llbc_dir = repo_root.join("build").join("llbc");
    // Pyre production configures the exact `eval::eval_loop_jit` portal, so
    // unlike generic two-artifact consumers it requires pyre-jit.ullbc too.
    const REQUIRED: &[&str] = &[
        "pyre-object.ullbc",
        "pyre-interpreter.ullbc",
        "pyre-jit.ullbc",
    ];
    let missing: Vec<&str> = REQUIRED
        .iter()
        .copied()
        .filter(|name| !llbc_dir.join(name).exists())
        .collect();
    if missing.is_empty() {
        return;
    }

    let charon_present = repo_root
        .join("build")
        .join("charon")
        .join("charon")
        .exists();

    // `cargo::error=` lines (no embedded newlines) surface in Cargo's
    // error summary on modern Cargo and fail the build on their own; the
    // framed stderr block below stays readable on every toolchain, and the
    // explicit non-zero exit is the belt-and-suspenders stop for a Cargo
    // too old to recognise `cargo::error`.
    println!(
        "cargo::error=pyre-jit codegen needs the Charon-extracted LLBC artefacts, but build/llbc/ is missing: {}",
        missing.join(", "),
    );
    if !charon_present {
        println!("cargo::error=Install charon (one-time): scripts/install-charon.py");
    }
    println!("cargo::error=Extract the LLBC: scripts/extract-llbc.py");

    // The install step is only needed when the charon binary is absent;
    // with it present the fix is the single extract command.
    let install_line = if charon_present {
        String::new()
    } else {
        "   scripts/install-charon.py                            # one-time\n".to_string()
    };
    eprintln!(
        "\n\
========================================================================\n\
 pyre-jit-trace: JIT codegen prerequisite missing\n\
------------------------------------------------------------------------\n\
 The Charon-extracted LLBC artefacts are required but were not found:\n\
{}\n\
 Bootstrap (run from the repo root):\n\
{}\
   scripts/extract-llbc.py\n\
\n\
 …or point the build at existing artefacts:\n\
   export PYRE_MIR_FRONTEND_LLBC=/abs/pyre-object.ullbc:/abs/pyre-interpreter.ullbc:/abs/pyre-jit.ullbc\n\
========================================================================\n",
        missing
            .iter()
            .map(|name| format!("   build/llbc/{name}"))
            .collect::<Vec<_>>()
            .join("\n"),
        install_line,
    );

    std::process::exit(1);
}

/// Run the codegen worker on a large-stack thread, propagating any panic so
/// the build fails loudly instead of emitting partial output.
fn run_worker() {
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(real_main)
        .expect("spawn build-script worker")
        .join()
        .expect("build-script worker panicked");
}

fn real_main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let pyre_base = format!("{manifest_dir}/..");
    let repo_root = format!("{manifest_dir}/../..");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Collect ALL source file paths from the active interpreter crates.
    // Only the paths are consumed — module-path derivation below plus
    // `cargo::rerun-if-changed` tracking; the graph bodies come from the
    // Charon-extracted LLBC set, so the file contents are never read.
    let source_dirs = [
        format!("{pyre_base}/pyre-object/src"),
        format!("{pyre_base}/pyre-interpreter/src"),
    ];

    let mut source_paths = Vec::new();

    for dir in &source_dirs {
        collect_rs_files(dir, &mut source_paths);
    }

    // Include the canonical portal source so its module-qualified identity is
    // present in the analyzer manifest. Production configures the exact
    // `eval::eval_loop_jit` CallPath; there is no handler-level fallback.
    // Using the real portal preserves the dispatch loop that blackhole resume
    // needs to continue from a user bytecode PC.
    //
    // Single-file inclusion (not the whole `pyre-jit/src` tree)
    // because `eval_loop_jit` is the only function in pyre-jit that
    // belongs in the portal closure; the rest of pyre-jit
    // (codewriter, assembler, regalloc, etc.) is JIT infrastructure
    // that must NOT be analyzed as user code, and the orchestration
    // would inflate analysis time and risk pulling unrelated
    // helpers into `find_all_graphs(portal)` BFS.
    let eval_path = format!("{pyre_base}/pyre-jit/src/eval.rs");
    collect_single_file(&eval_path, &mut source_paths);

    // `materialize_virtual_from_rd` (`pyre-jit/src/eval.rs`) destructures
    // `majit_ir::RdVirtualInfo` enum variants whose named fields carry
    // primitive concretetypes (`kind: u8`, `descr_index: u32`, ...).
    // Without the enum declaration in the analyser source set the variant
    // arms surface as untyped, and the rtyper's exitswitch emission for
    // `match descr.kind { 0 => ..., 1 => ..., 2 => ..., }` falls back to
    // 'r' (Ref), which `flatten.rs:385` rejects as
    // `switch exitswitch must be int`.  Single-file inclusion mirrors the
    // `pyre-jit/src/eval.rs` carve-out above: only the resoperation enum
    // declarations belong in the analyser closure, not the rest of
    // majit-ir which would inflate analysis time and pull JIT
    // infrastructure into the user-code BFS.
    let resoperation_path = format!("{repo_root}/majit/majit-ir/src/resoperation.rs");
    collect_single_file(&resoperation_path, &mut source_paths);

    eprintln!(
        "[pyre-jit-trace build.rs] collected {} source paths from {} dirs (+ pyre-jit/src/eval.rs): {:?}",
        source_paths.len(),
        source_dirs.len(),
        source_paths,
    );

    // Run analysis on ALL files with PyFrame virtualizable metadata.
    //
    // This keeps the generic analyzer closer to the proc-macro/runtime path:
    // graph rewrite can recognize `next_instr`, `valuestackdepth`, and
    // `locals_cells_stack_w[*]` as virtualizable accesses before legacy
    // TracePattern classification runs.
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
            jit_drivers: vec![
                majit_translate::JitDriverSpec {
                    portal: majit_translate::CallPath::from_segments(["eval", "eval_loop_jit"]),
                    greens: vec![
                        "next_instr".to_string(),
                        "is_being_profiled".to_string(),
                        "pycode".to_string(),
                    ],
                    reds: vec!["frame".to_string(), "ec".to_string()],
                    autoreds: false,
                    virtualizables: vec!["frame".to_string()],
                    red_types: vec!["PyFrame".to_string(), "ExecutionContext".to_string()],
                },
                majit_translate::JitDriverSpec {
                    // pypy/interpreter/baseobjspace.py:1003 `_unpackiterable_unknown_length`;
                    // greens=['greenkey'], reds='auto' (baseobjspace.py:29-32).
                    portal: majit_translate::CallPath::from_segments([
                        "baseobjspace",
                        "_unpackiterable_unknown_length",
                    ]),
                    greens: vec!["greenkey".to_string()],
                    reds: vec![],
                    autoreds: true,
                    virtualizables: vec![],
                    red_types: vec![],
                },
            ],
            // pyre production registers no trait-dispatch families (#346).
            register_trait_families: Vec::new(),
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
    // setter overrides it.  TODO: documented at
    // `CallControl::make_virtualizable_infos`.
    let vinfo_factory: &majit_translate::VirtualizableInfoFactory<'_> = &|_jd_idx, _vtype| None;
    let fnaddr_bindings = pyre_interpreter::jit_trace_fnaddrs();
    // Prebuilt object-space singleton addresses (static `PyType` pointers
    // and dict-strategy refs).  `majit-translate` is the translation
    // layer and must not import `pyre-object`; the driver supplies these
    // across the translation boundary.  Resolved here in the same
    // build-script process the translator runs in, so the captured
    // addresses match a direct `&pyre_object::X` read at the codewriter
    // call site.
    let static_pytype_addrs = pyre_interpreter::jit_static_pytype_addrs();
    let static_ref_addrs = pyre_interpreter::jit_static_ref_addrs();
    let static_int_values = pyre_interpreter::jit_static_int_values();
    let static_addrs = majit_translate::HostStaticAddrs {
        pytypes: &static_pytype_addrs,
        refs: &static_ref_addrs,
        int_values: &static_int_values,
    };
    // Per-source crate-stripped module paths — the analyzer-side
    // metadata (`front::mir`) records
    // `struct_origins[bare_name] = module_path`.  Aligns with
    // the runtime's `build_object_descr_group_with_def_path` qualified
    // def-path slot in `gc_cache._cache_size` so a future
    // `path_hash(canonical_struct_name)` analyzer hash lands on the
    // same Arc the runtime publishes (PyPy `cache[STRUCT]` lltype-
    // object identity, descr.py:108-118).
    let module_paths: Vec<String> = source_paths
        .iter()
        .map(|p| module_path_from_source_file(p))
        .collect();
    let module_path_refs: Vec<&str> = module_paths.iter().map(|s| s.as_str()).collect();

    emit_rerun_directives(&repo_root, &source_paths);

    let cache_key = codegen_cache_key(manifest_dir, &repo_root, &source_paths);
    let cache_dir = codegen_cache_dir(&repo_root, &cache_key);
    if restore_codegen_cache(&cache_dir, &out_dir) {
        eprintln!(
            "[pyre-jit-trace build.rs] restored generated JIT trace artifacts from cache {}",
            cache_key
        );
        return;
    }

    let pipeline = majit_translate::analyze_multiple_pipeline_with_modules(
        &module_path_refs,
        &analyze_config,
        None,
        vinfo_factory,
        &fnaddr_bindings,
        static_addrs,
    );

    // Generate tracing code from the canonical graph-first analysis result.
    let code = majit_translate::generate_trace_code_from_pipeline(&pipeline);

    std::fs::write(format!("{out_dir}/jit_trace_gen.rs"), &code).unwrap();

    // JSON metadata for debugging
    let json = serde_json::to_string_pretty(&pipeline).unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), &json).unwrap();

    // Persist `pipeline.jitcodes` (RPython `all_jitcodes` from
    // codewriter.py:89) as a
    // single bincode artifact. Runtime deserializes this once into the shared
    // MetaInterpStaticData jitcodes store — same single-store model as
    // RPython `warmspot.py:281-282` `self.metainterp_sd.jitcodes =
    // codewriter.make_jitcodes()`.
    let jitcodes_bin = bincode::serialize(&pipeline.jitcodes).unwrap();
    std::fs::write(format!("{out_dir}/jitcodes.bin"), &jitcodes_bin).unwrap();

    // Persist the explicit portal → main-JitCode mapping. Runtime consumes
    // this directly instead of rediscovering the portal through name or flag
    // scans.
    let jit_drivers_bin = bincode::serialize(&pipeline.jit_drivers).unwrap();
    std::fs::write(format!("{out_dir}/jit_drivers.bin"), &jit_drivers_bin).unwrap();

    // Persist the runtime opname → u8 table so
    // `JitCode.code` (assembler-local mapping) decodes back to the
    // canonical `(opname, argcodes)` shape at runtime (shadow dispatch,
    // IR diffing).  RPython equivalent: the table handed to
    // `BlackholeInterpBuilder::setup_insns` at metainterp startup
    // (`pyjitpl.py:2227-2243`).
    //
    // RPython parity (`assembler.py:220 self.insns.setdefault(key,
    // len(self.insns))`): the table is the assembler's emission-driven
    // dict, populated by `write_insn` calls during graph flattening.
    // Pyre's analog is `pipeline.insns`, snapshotted from
    // `codewriter.assembler.insns()` after `make_jitcodes` finishes
    // (`majit-translate/src/lib.rs:910`).  Each distinct key gets a
    // fresh byte; the forward map is injective.  `blackhole.py:913`
    // aliases the bhimpl handler under two Python attribute names
    // (`bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not`) but
    // does NOT register a second opname in `Assembler.insns`; the
    // alias is at the dispatch-function-name level only.  Pyre
    // therefore registers exactly one opname per byte; the runtime
    // inverse (`byte → opname`) is 1:1 and panics on duplicate-byte
    // collisions (`jitcode_runtime.rs:INSNS_BYTE_TO_OPNAME`).
    //
    // Serialize through a `BTreeMap` view so the byte output is stable
    // across processes (Rust's `HashMap` SipHash makes raw iteration
    // non-deterministic; RPython's Python dict is insertion-ordered).
    let insns_sorted: std::collections::BTreeMap<&String, &u8> = pipeline.insns.iter().collect();
    let insns_bin = bincode::serialize(&insns_sorted).unwrap();
    std::fs::write(format!("{out_dir}/insns.bin"), &insns_bin).unwrap();

    // RPython `blackhole.py:59 self.setup_descrs(asm.descrs)` + `:102-103
    // def setup_descrs(self, descrs): self.descrs = descrs`. Persists the
    // build-time assembler's shared descr pool so that 'd'/'j' argcodes
    // in `JitCode.code` resolve at runtime via
    // `BlackholeInterpBuilder::setup_descrs(...)` — the single-store
    // model (same list consumed by every `BlackholeInterpreter` produced
    // by `acquire_interp`).
    let descrs_bin = bincode::serialize(&pipeline.descrs).unwrap();
    std::fs::write(format!("{out_dir}/descrs.bin"), &descrs_bin).unwrap();

    // RPython's translator AOT-compiles every helper into a single binary, so
    // `JitCode.fnaddr` / `constants_i` funcptrs are linker-resolved and stable
    // at runtime.  Pyre's `majit-translate` runs in `build.rs` — a separate
    // process from `pyre-dynasm` — so every fnaddr captured here is the
    // build-script process's address, which ASLR (and the divergent executable
    // layouts) invalidates at runtime.  Persist the `(path, build_fnaddr)`
    // table the codewriter consumed so the runtime patcher
    // (`runtime_fnaddr_patch::patch_constants_i_fnaddrs`) can pair each build
    // address with the matching runtime address from
    // `pyre_interpreter::jit_trace_fnaddrs()` and overwrite stale constants
    // before the walker invokes them.
    let fnaddr_bindings_owned: Vec<(String, i64)> = fnaddr_bindings
        .iter()
        .map(|(p, a)| ((*p).to_string(), *a))
        .collect();
    let fnaddr_bindings_bin = bincode::serialize(&fnaddr_bindings_owned).unwrap();
    std::fs::write(
        format!("{out_dir}/fnaddr_bindings.bin"),
        &fnaddr_bindings_bin,
    )
    .unwrap();

    // Same ASLR hazard for the static-data addresses the codewriter baked
    // into `constants_i` (host `PyType` singletons and prebuilt refs supplied
    // via `HostStaticAddrs`): the build-script process's `&pyre_object::X`
    // address does not survive into the runtime executable.  Persist the
    // `(name, build_addr)` tables so `runtime_fnaddr_patch::
    // patch_constants_i_static_addrs` can re-pair them with the runtime
    // addresses from `jit_static_pytype_addrs` / `jit_static_ref_addrs`.
    let pytype_bindings_owned: Vec<(String, i64)> = static_pytype_addrs
        .iter()
        .map(|(n, a)| ((*n).to_string(), *a))
        .collect();
    std::fs::write(
        format!("{out_dir}/static_pytype_bindings.bin"),
        bincode::serialize(&pytype_bindings_owned).unwrap(),
    )
    .unwrap();
    let ref_bindings_owned: Vec<(String, i64)> = static_ref_addrs
        .iter()
        .map(|(n, a)| ((*n).to_string(), *a))
        .collect();
    std::fs::write(
        format!("{out_dir}/static_ref_bindings.bin"),
        bincode::serialize(&ref_bindings_owned).unwrap(),
    )
    .unwrap();

    // Report
    eprintln!(
        "[pyre-jit-trace build.rs] canonical analysis: {} JIT drivers, {} functions, {} blocks, {} flat ops, {} all_jitcodes ({} bytes bincode), generated {} bytes",
        pipeline.jit_drivers.len(),
        pipeline.functions.len(),
        pipeline.total_blocks,
        pipeline.total_ops,
        pipeline.jitcodes.len(),
        jitcodes_bin.len(),
        code.len(),
    );

    if let Err(e) = store_codegen_cache(&cache_dir, &out_dir) {
        eprintln!(
            "[pyre-jit-trace build.rs] warning: could not store generated JIT trace cache {}: {e}",
            cache_key
        );
    }
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

fn emit_rerun_directives(repo_root: &str, source_paths: &[String]) {
    for path in source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
    emit_rerun_if_changed_recursive(&format!("{repo_root}/majit/majit-translate/src"));
    println!("cargo::rerun-if-changed=src/virtualizable_spec.rs");
    println!("cargo::rerun-if-changed=src/call_spec.rs");
    // The mir-frontend analysis derives `jit_trace_gen.rs` from
    // the workspace LLBC artefacts or the `PYRE_MIR_FRONTEND_LLBC`
    // override. Track both so re-extracting LLBC or repointing the override
    // invalidates Cargo's build-script cache and our content cache key.
    println!("cargo::rerun-if-env-changed=PYRE_MIR_FRONTEND_LLBC");
    if let Some(paths) = std::env::var_os("PYRE_MIR_FRONTEND_LLBC") {
        for path in std::env::split_paths(&paths) {
            if !path.as_os_str().is_empty() {
                println!("cargo::rerun-if-changed={}", path.display());
            }
        }
    }
    for llbc in [
        "pyre-object.ullbc",
        "pyre-interpreter.ullbc",
        "pyre-jit.ullbc",
    ] {
        println!("cargo::rerun-if-changed={repo_root}/build/llbc/{llbc}");
    }
}

fn codegen_cache_dir(repo_root: &str, cache_key: &str) -> std::path::PathBuf {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(repo_root).join("target"));
    target_dir
        .join("pyre-jit-trace-cache")
        .join(CODEGEN_CACHE_VERSION)
        .join(cache_key)
}

fn restore_codegen_cache(cache_dir: &std::path::Path, out_dir: &str) -> bool {
    if !CODEGEN_OUTPUTS
        .iter()
        .all(|name| cache_dir.join(name).is_file())
    {
        return false;
    }
    for name in CODEGEN_OUTPUTS {
        let src = cache_dir.join(name);
        let dst = std::path::Path::new(out_dir).join(name);
        if let Err(e) = std::fs::copy(&src, &dst) {
            eprintln!(
                "[pyre-jit-trace build.rs] warning: cache restore failed for {}: {e}",
                src.display()
            );
            return false;
        }
    }
    true
}

fn store_codegen_cache(cache_dir: &std::path::Path, out_dir: &str) -> std::io::Result<()> {
    if CODEGEN_OUTPUTS
        .iter()
        .all(|name| cache_dir.join(name).is_file())
    {
        return Ok(());
    }
    let Some(parent) = cache_dir.parent() else {
        return Ok(());
    };
    std::fs::create_dir_all(parent)?;
    let tmp_dir = parent.join(format!(
        ".{}.tmp-{}",
        cache_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("cache"),
        std::process::id()
    ));
    if tmp_dir.exists() {
        std::fs::remove_dir_all(&tmp_dir)?;
    }
    std::fs::create_dir_all(&tmp_dir)?;
    for name in CODEGEN_OUTPUTS {
        let src = std::path::Path::new(out_dir).join(name);
        let dst = tmp_dir.join(name);
        std::fs::copy(src, dst)?;
    }
    match std::fs::rename(&tmp_dir, cache_dir) {
        Ok(()) => Ok(()),
        Err(e) if cache_dir.exists() => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            eprintln!("[pyre-jit-trace build.rs] cache already stored by another process: {e}");
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            Err(e)
        }
    }
}

fn codegen_cache_key(manifest_dir: &str, repo_root: &str, source_paths: &[String]) -> String {
    let mut h = CacheHasher::new();
    h.write_str(CODEGEN_CACHE_VERSION);
    for key in ["HOST", "TARGET", "PROFILE", "OPT_LEVEL"] {
        h.write_str(key);
        h.write_os(std::env::var_os(key));
    }
    let mut cargo_env: Vec<(String, String)> = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("CARGO_FEATURE_") || key.starts_with("CARGO_CFG_TARGET_")
        })
        .collect();
    cargo_env.sort();
    for (key, value) in cargo_env {
        h.write_str(&key);
        h.write_str(&value);
    }
    h.write_os(std::env::var_os("PYRE_MIR_FRONTEND_LLBC"));

    // The codegen output also depends on every crate linked into this
    // build-script binary — `majit-translate`'s own dependencies
    // (`majit-ir`, `majit-charon-reader`, `rustpython-compiler-core`, …)
    // and their serde wire formats — whose sources are not hashed below
    // (only `majit-translate/src` and `resoperation.rs` are). Cargo
    // recompiles and reruns the build script whenever any of them changes,
    // but the content key would otherwise stay identical and restore a
    // stale snapshot (e.g. `*.bin` written under an older `majit-ir`
    // bincode layout). Folding the build-script executable's own bytes into
    // the key rekeys the cache on any transitive code change.
    match std::env::current_exe() {
        Ok(exe) => hash_file_content(&mut h, &exe),
        Err(e) => {
            h.write_str("current-exe-error");
            h.write_str(&e.to_string());
        }
    }

    hash_file_content(&mut h, &std::path::Path::new(manifest_dir).join("build.rs"));
    hash_file_content(
        &mut h,
        &std::path::Path::new(manifest_dir).join("src/virtualizable_spec.rs"),
    );
    hash_file_content(
        &mut h,
        &std::path::Path::new(manifest_dir).join("src/call_spec.rs"),
    );

    for path in source_paths {
        hash_file_content(&mut h, std::path::Path::new(path));
    }
    hash_rs_dir_content(
        &mut h,
        &std::path::Path::new(repo_root).join("majit/majit-translate/src"),
    );
    hash_llbc_inputs(&mut h, repo_root);

    format!("{:016x}", h.finish())
}

fn hash_llbc_inputs(h: &mut CacheHasher, repo_root: &str) {
    // Hash the LLBC by content, not by (len, mtime) signature. The
    // analysis (`analyze_multiple_pipeline_with_modules`) derives every
    // generated artefact from these graph bodies, so a content change that
    // happens to preserve size and mtime — `git checkout`, a cache restore
    // that keeps timestamps, an in-place rewrite of equal length — must
    // still rekey the cache. A signature would let `restore_codegen_cache`
    // serve stale output and skip re-analysis. The `.ullbc` set is a few
    // MB; the read is negligible next to the analysis it gates.
    if let Some(paths) = std::env::var_os("PYRE_MIR_FRONTEND_LLBC") {
        for path in std::env::split_paths(&paths) {
            if !path.as_os_str().is_empty() {
                hash_file_content(h, &path);
            }
        }
        return;
    }
    for llbc in [
        "pyre-object.ullbc",
        "pyre-interpreter.ullbc",
        "pyre-jit.ullbc",
    ] {
        hash_file_content(
            h,
            &std::path::Path::new(repo_root)
                .join("build")
                .join("llbc")
                .join(llbc),
        );
    }
}

fn hash_rs_dir_content(h: &mut CacheHasher, dir: &std::path::Path) {
    let mut paths = Vec::new();
    for entry in WalkDir::new(dir) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        paths.push(entry.path().to_path_buf());
    }
    paths.sort();
    for path in paths {
        hash_file_content(h, &path);
    }
}

fn hash_file_content(h: &mut CacheHasher, path: &std::path::Path) {
    h.write_path(path);
    let Ok(mut file) = std::fs::File::open(path) else {
        h.write_str("missing");
        return;
    };
    h.write_str("content");
    let mut buf = [0_u8; 64 * 1024];
    loop {
        match std::io::Read::read(&mut file, &mut buf) {
            Ok(0) => break,
            Ok(n) => h.write_bytes(&buf[..n]),
            Err(e) => {
                h.write_str("read-error");
                h.write_str(&e.to_string());
                break;
            }
        }
    }
}

/// Length-prefixed hashing wrapper over the std hasher.
///
/// `DefaultHasher` is fixed-key SipHash: deterministic within a given Rust
/// toolchain (no per-process seed), which is all the cache needs — a key
/// produced by one build matches the same build's stored entry. std does
/// not promise the algorithm is stable across Rust releases, so a toolchain
/// upgrade changes every key; that is fine here (a miss just regenerates,
/// and the build-script executable is already in the key, so a toolchain
/// bump rekeys regardless). Inputs are length-prefixed so adjacent fields
/// cannot run together: `("ab", "c")` and `("a", "bc")` hash differently.
struct CacheHasher(std::collections::hash_map::DefaultHasher);

impl CacheHasher {
    fn new() -> Self {
        Self(std::collections::hash_map::DefaultHasher::new())
    }

    fn finish(&self) -> u64 {
        std::hash::Hasher::finish(&self.0)
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        // Fixed-width length prefix, then the payload — the prefix frames
        // the byte run so concatenation stays unambiguous.
        std::hash::Hasher::write_u64(&mut self.0, bytes.len() as u64);
        std::hash::Hasher::write(&mut self.0, bytes);
    }

    fn write_str(&mut self, value: &str) {
        self.write_bytes(value.as_bytes());
    }

    fn write_os(&mut self, value: Option<std::ffi::OsString>) {
        match value {
            Some(value) => self.write_str(&value.to_string_lossy()),
            None => self.write_str("<unset>"),
        }
    }

    fn write_path(&mut self, path: &std::path::Path) {
        self.write_str(&path.to_string_lossy());
    }
}

/// Collect a single `.rs` file by absolute path, mirroring
/// `collect_rs_files`'s read-into-vecs convention.  Used to thread
/// `pyre-jit/src/eval.rs` (the portal canonical)
/// into the analysis without including the rest of pyre-jit's JIT
/// infrastructure (codewriter, assembler, regalloc, ...).
/// Crate-stripped module path for a source file at `path`.
///
/// Strips the crate root (`/.../<crate>/src/` prefix) and the `.rs`
/// suffix, then converts `/` to `::` for nested files.  Matches the
/// runtime `module_path!()` macro output after the leading crate
/// segment is dropped — both sides hash the same string so
/// `gc_cache._cache_size[LLType::Struct(path_hash(path))]` slots
/// align (PyPy descr.py:108-118 `cache[STRUCT]` identity).
///
/// Examples (input → output):
/// - `"pyre/pyre-object/src/intobject.rs"` → `"intobject"`
/// - `"pyre/pyre-interpreter/src/pyframe.rs"` → `"pyframe"`
/// - `"pyre/pyre-interpreter/src/foo/bar.rs"` → `"foo::bar"`
/// - `"pyre/pyre-interpreter/src/lib.rs"` → `""` (crate root, no qualifier)
///
/// Returns `""` when the path does not contain `/src/` — callers
/// outside the canonical layout (synthesized files, fixtures) keep
/// the simple-name registration.
fn module_path_from_source_file(path: &str) -> String {
    // Windows `WalkDir` yields native paths with `\` separators; the marker
    // search + `/lib` / `/mod` suffix strips + final `/` → `::` rewrite
    // below all assume forward slashes, so an unnormalised Windows path
    // falls into the `rfind` `None` branch and every source file ends up
    // with an empty `module_path`.  Empty module paths skip
    // `register_struct_origins` (`lib.rs:374-382`), which breaks
    // classdef-keyed method resolution downstream and silently drops
    // graphs from the analyzer — surfacing later as missing opcodes in
    // `pipeline.insns` (e.g. `setfield_vable_i/rid`).
    let normalized_path = path.replace('\\', "/");
    let path = normalized_path.as_str();
    let marker = "/src/";
    let Some(idx) = path.rfind(marker) else {
        return String::new();
    };
    let rest = &path[idx + marker.len()..];
    let stem = rest.strip_suffix(".rs").unwrap_or(rest);
    let normalized = stem
        .strip_suffix("/lib")
        .or_else(|| stem.strip_suffix("/mod"))
        .unwrap_or(stem);
    if normalized == "lib" || normalized == "mod" {
        return String::new();
    }
    normalized.replace('/', "::")
}

fn collect_single_file(path: &str, paths: &mut Vec<String>) {
    match std::fs::metadata(path) {
        Ok(_) => paths.push(path.to_string()),
        Err(e) => {
            eprintln!("[pyre-jit-trace build.rs] warning: cannot read {path}: {e}");
        }
    }
}

/// Collect all `.rs` files from a directory tree.
///
/// Sorts entries by path so the collected source order is stable
/// across platforms.  Without this, `WalkDir` yields entries in the
/// filesystem's native `readdir` order — APFS (macOS) and ext4
/// (Linux) and NTFS (Windows) return different sequences, which
/// causes the analyzer to encounter type/method definitions in a
/// different order and exposes platform-divergent classdef-less
/// SomeInstance failures (PR 91 CI: Ubuntu/Windows fail with
/// `SomeBuiltin.call(): no analyser registered for std.ptr.null_mut`
/// and `SomeInstance.getattr on classdef-less instance` while macOS
/// passes).  Stable lexicographic order makes the build reproducible
/// and lets one fix cover every platform.
fn collect_rs_files(dir: &str, paths: &mut Vec<String>) {
    for entry in WalkDir::new(dir).sort_by_file_name() {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        paths.push(entry.path().to_string_lossy().to_string());
    }
}

fn emit_rerun_if_changed_recursive(dir: &str) {
    for entry in WalkDir::new(dir) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        println!("cargo::rerun-if-changed={}", entry.path().display());
    }
}

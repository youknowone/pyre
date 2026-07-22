//! majit-translate: RPython translation pipeline.
//!
//! Upstream counterparts:
//! - `codewriter/` ← `rpython/jit/codewriter/`
//! - `flowspace/`, `annotator/`, `rtyper/`, `translator/` — line-by-line
//!   ports of `rpython/{flowspace,annotator,rtyper,translator}/`
//! - `translator/rtyper/legacy_{annotator,resolve,pipeline}.rs` —
//!   transitional pyre-only adapters with no upstream counterpart;
//!   slated for retirement once Skip categories close
//!   (see [`crate::translator::rtyper::cutover`])
//!
//! Everything under one crate because upstream has circular imports
//! (flowspace/operation.py ↔ annotator, annotator/annrpython.py ↔
//! rtyper/normalizecalls.py) that Cargo's DAG crate boundary cannot model.

pub mod annotator;
pub mod codewriter;
pub mod config;
pub mod flowspace;
pub mod tool;
pub use codewriter::{
    assembler, call, flatten, format, insns, jitcode, jtransform, liveness, policy, support,
};
pub use tool::algo::regalloc;

mod codegen;
pub mod front;
// TODO(pyre): pyre-interpreter handler JitCode registry
// (Rust source → FunctionGraph bridge with no RPython counterpart;
// upstream assumes rtyper-produced `translator.graphs` is already in
// memory at codewriter entry).
pub mod generated;
pub mod hints;
pub mod inline;
pub mod layout;
mod local_crates;
pub mod model;
pub mod model_ssa;
mod parse;
pub mod pipeline;
#[cfg(test)]
mod test_support;
// `translator/` is the RPython-orthodox port home — see
// `translator/mod.rs` for the contract.  Currently hosts
// `translator/rtyper/{rclass.rs, rpbc.rs}`, the `rpython/rtyper/` 1:1
// port, alongside the transitional `legacy_{annotator,resolve,pipeline}.rs`
// adapters that the cutover (`translator/rtyper/cutover.rs`) consumes
// until the real-rtyper path types every production graph end-to-end.
pub mod translator;

pub use call::{CallDescriptor, StructFieldLayout, StructLayout};
pub use codewriter::type_state::ConcreteType;
pub use flatten::{FlatOp, GraphFlattener, Label, RegKind, SSARepr, flatten_graph};
pub use front::{AstGraphOptions, SemanticFunction, SemanticProgram};
pub use jtransform::{
    CallEffectKind, CallEffectOverride, GraphTransformConfig, GraphTransformResult,
    VirtualizableFieldDescriptor, transform_graph,
};
pub use layout::{HeuristicLayoutProvider, LayoutProvider};
pub use model::{Block, BlockId, CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueType};
pub use parse::CallPath;
pub use pipeline::{
    CompiledJitDriver, JitDriverSpec, PipelineConfig, PipelineResult, ProgramPipelineResult,
};

use serde::{Deserialize, Serialize};

/// Configuration for the canonical graph/pipeline analyzer.
///
/// Consumers supply graph-rewrite metadata such as virtualizable
/// field/array mappings before the codewriter-style passes run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeConfig {
    pub pipeline: PipelineConfig,
}

/// Trait implementation info
#[derive(Debug, Serialize, Deserialize)]
pub struct TraitImplInfo {
    pub trait_name: String,
    pub for_type: String,
    #[serde(default)]
    pub self_ty_root: Option<String>,
    pub methods: Vec<MethodInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MethodInfo {
    pub name: String,
    /// Canonical semantic graph for this method when available.
    #[serde(skip, default)]
    pub graph: Option<model::FunctionGraph>,
    /// RPython: op.result.concretetype — return type for array identity.
    #[serde(default)]
    pub return_type: Option<String>,
    /// RPython: function-level JIT hints (elidable, close_stack, etc.).
    #[serde(default)]
    pub hints: Vec<String>,
}

/// Feature-gated SemanticProgram builder.
///
/// When the `mir-frontend` feature is enabled, route the build
/// through [`front::mir::build_semantic_program_from_llbcs`] using
/// LLBC artefacts discovered via, in priority order:
///
/// 1. Paths supplied by an explicit-LLBC analyzer entry point. These are the
///    complete input set and disable every fallback below.
/// 2. `PYRE_MIR_FRONTEND_LLBC` env-var (OS path-list: `;`-separated on
///    Windows, `:`-separated elsewhere). Explicit override for CI /
///    test fixtures targeting a specific LLBC set.
/// 3. Auto-discovery at `<workspace>/build/llbc/<expected>.ullbc`,
///    where `scripts/extract-llbc.py` writes. If every expected file
///    exists, the MIR front-end engages automatically.
///
/// Panics when neither source resolves: the MIR front-end is the only
/// graph builder, so a missing LLBC set (Charon not installed, or
/// `scripts/extract-llbc.py` not run) is a fatal misconfiguration
/// rather than a fallback to another path.
fn build_semantic_program_via_active_frontend(
    module_paths: &[&str],
    static_addrs: HostStaticAddrs<'_>,
    explicit_llbc_paths: Option<&[&str]>,
) -> front::SemanticProgram {
    #[cfg(feature = "mir-frontend")]
    {
        // Accept an OS path-list so production can pass the canonical
        // pyre LLBC set in one env-var.
        // `std::env::split_paths` uses the platform separator (`;` on
        // Windows, `:` elsewhere) so a Windows drive letter like `Z:`
        // is not mistaken for a separator.  The single-path form also
        // works.
        //
        // If the env-var is unset, auto-discover the canonical
        // workspace LLBC artefacts before failing loud.
        let resolved_paths: Option<Vec<String>> = explicit_llbc_paths
            .filter(|paths| !paths.is_empty())
            .map(|paths| paths.iter().map(|path| (*path).to_string()).collect())
            .or_else(|| {
                std::env::var_os("PYRE_MIR_FRONTEND_LLBC")
                    .map(|v| {
                        std::env::split_paths(&v)
                            .filter(|p| !p.as_os_str().is_empty())
                            .map(|p| p.to_string_lossy().into_owned())
                            .collect()
                    })
                    // A present-but-blank override (`PYRE_MIR_FRONTEND_LLBC=`)
                    // collects to an empty vec; treat it as unset so workspace
                    // auto-discovery still runs instead of feeding an empty LLBC
                    // set into the frontend.
                    .filter(|paths: &Vec<String>| !paths.is_empty())
            })
            .or_else(|| auto_discover_workspace_llbc_paths(module_paths));
        if let Some(paths) = resolved_paths {
            let llbcs: Vec<majit_charon_reader::Llbc> = paths
                .iter()
                .map(|p| {
                    majit_charon_reader::Llbc::load(p)
                        .unwrap_or_else(|e| panic!("Step 4.4 cutover: load {p}: {e}"))
                })
                .collect();
            // Seed the local-crate alias roots from the loaded set so
            // `free_function_alias_paths` and the registry's canonical
            // dedup treat every extracted crate name as an alias root
            // (`local_crates.rs`); non-pyre consumers (aheui) resolve
            // their crate-qualified cross-crate callsites through this.
            crate::local_crates::register_local_crate_roots(
                llbcs.iter().map(|l| l.crate_name().to_string()),
            );
            let mut program =
                front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
                    &llbcs,
                    static_addrs,
                    module_paths,
                )
                .unwrap_or_else(|e| panic!("Step 4.4 cutover: lower llbcs {paths:?}: {e}"));
            // JIT-hint pass.  pyre's proc-macro attributes
            // (`#[majit_macros::elidable*]` / `dont_look_inside` /
            // `loop_invariant` / `unroll_safe`) are consumed by the
            // proc-macro at expansion time and do not survive in
            // Charon's `attr_info`, so the macros leave `#[doc(hidden)]`
            // marker consts (`_elidable_function_<NAME>`, …) next to each
            // annotated fn.  Charon extracts those into `global_decls`;
            // `front::llbc_hints` reads them back and the hints merge into
            // the MIR-driven SemanticProgram by qualified path — the analog of
            // RPython's translator reading `func._elidable_function_` off
            // the function object.
            merge_hints_from_llbcs(&mut program, &llbcs);
            // Re-source the unsafe-fn stub carrier from Charon: walk the
            // full LLBC set for every local `unsafe fn` / unsafe
            // impl-method projecting to a unit/bool return.  The consumer
            // at `call_control.unsafe_fn_stubs` (lib.rs) reads this carrier.
            program.unsafe_fn_stubs = llbcs
                .iter()
                .flat_map(front::mir::collect_unsafe_fn_stubs_from_llbc)
                .collect();
            // Foreign opaque-ADT methods (`<BigInt as Add>::add`, …) that
            // `impl_method_owner` routes through `CallTarget::FunctionPath`
            // for an opaque owner.  Declared external here so the residual
            // `FunctionPath` lookup resolves rather than panicking
            // `SomeInstance.getattr` on the classdef-less receiver.
            program.foreign_opaque_method_externals = llbcs
                .iter()
                .flat_map(front::mir::collect_foreign_opaque_method_externals)
                .collect();
            // Whole-program type metadata (`known_struct_names`,
            // `known_trait_names`, `struct_fields`) comes from the MIR
            // builder's `derive_program_metadata` walk over Charon's
            // `type_decls` / `trait_decls`; struct field-type strings are
            // resolved by `tyref_to_ast_string` (Charon-resolved types,
            // e.g. `*mut PyObject`, `Vec<u8>`, `i64`) rather than the syn
            // re-parse.
            return program;
        }
    }
    let _ = module_paths; // silence unused warning when the feature is off
    // The MIR front-end is the only graph builder.  Reaching this
    // point means neither `PYRE_MIR_FRONTEND_LLBC` nor the workspace
    // auto-discover located an LLBC source — surface the
    // misconfiguration immediately.
    panic!(
        "no LLBC source resolved.\n\
         Run `scripts/extract-llbc.py` to produce \
         `build/llbc/{{pyre-object,pyre-interpreter,pyre-jit}}.ullbc`, \
         or set `PYRE_MIR_FRONTEND_LLBC` to an OS path-list \
         (`;`-separated on Windows, `:` elsewhere) explicitly."
    );
}

/// Locate the workspace's `build/llbc/` directory and return paths to
/// the canonical pyre LLBC artefacts when every expected file is
/// present *and* the caller looks like a production build (not a test
/// fixture).
///
/// Returns `None` when:
///   - no source carries a `module_path` (test fixtures pass empty
///     module paths; production passes per-file crate-stripped paths),
///   - the caller passed fewer than `PROD_SOURCE_FILES_FLOOR`
///     module paths (single-source diagnostic),
///   - a mandatory artefact (`pyre-object.ullbc` /
///     `pyre-interpreter.ullbc`) is missing (contributor without
///     Charon installed), or
///   - the workspace anchoring fails.
///
/// `pyre-jit.ullbc` is included when present. Pyre's production
/// `eval::eval_loop_jit` driver requires it; the extraction bootstrap no
/// longer relies on a different temporary portal and instead bypasses this
/// analysis explicitly with `MAJIT_LLBC_EXTRACTION=1`. Returning the mandatory
/// pair when it is absent remains useful only to consumers whose configured
/// portal is contained in those artifacts; exact driver resolution rejects an
/// unavailable portal later in the pipeline.
///
/// The two gates together match the production fingerprint:
/// `pyre-jit-trace/build.rs` calls
/// `analyze_multiple_pipeline_with_modules` with ≈100 per-file
/// `module_path`s.  Non-production callers (test fixtures and the
/// 5-module `generated::PYRE_JIT_GRAPH_MODULES` manifest) stay below
/// the floor, so auto-discovery does not silently swap their
/// front-end.
///
/// The workspace root is anchored at compile time via
/// `env!("CARGO_MANIFEST_DIR")` — `<workspace>/majit/majit-translate`
/// resolves up to `<workspace>` via two `..` segments.  The
/// `scripts/extract-llbc.py` script writes to the same
/// `<workspace>/build/llbc/` directory by convention, so the two
/// halves stay in sync.
#[cfg(feature = "mir-frontend")]
fn auto_discover_workspace_llbc_paths(module_paths: &[&str]) -> Option<Vec<String>> {
    const PROD_SOURCE_FILES_FLOOR: usize = 50;
    if module_paths.len() < PROD_SOURCE_FILES_FLOOR {
        return None;
    }
    if !module_paths.iter().any(|mp| !mp.is_empty()) {
        return None;
    }
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let llbc_dir = workspace_root.join("build").join("llbc");
    // Canonical production set.  `pyre-module.ullbc` is intentionally
    // omitted — it is empty in current builds and adds nothing.
    // `corpus.ullbc` is the Charon fixture, not production.
    //
    // The set is fixed at exactly these crates so the generated
    // `all_jitcodes` table is environment-invariant: every real build
    // (`cargo test`, `pyre/check.py`) consumes the same three `.ullbc`
    // inputs, so a local tree and CI produce byte-identical codegen.
    // `pyre-object.ullbc` and `pyre-interpreter.ullbc` are mandatory.
    //
    // `pyre-jit.ullbc` hosts Pyre's exact `eval::eval_loop_jit` portal. The
    // extraction driver sets `MAJIT_LLBC_EXTRACTION=1` while producing that
    // artifact, so `pyre-jit-trace/build.rs` emits compile-only placeholders
    // and never enters this analysis during the dependency bootstrap. A normal
    // Pyre build requires all three artifacts; generic consumers whose portal
    // lives in the mandatory pair may still use the two-artifact result.
    const MANDATORY: &[&str] = &["pyre-object.ullbc", "pyre-interpreter.ullbc"];
    let mut paths = Vec::with_capacity(3);
    for name in MANDATORY {
        let p = llbc_dir.join(name);
        if !p.exists() {
            return None;
        }
        paths.push(p.to_string_lossy().into_owned());
    }
    let pyre_jit = llbc_dir.join("pyre-jit.ullbc");
    if pyre_jit.exists() {
        paths.push(pyre_jit.to_string_lossy().into_owned());
    } else {
        eprintln!(
            "[majit-translate] pyre-jit.ullbc absent — using the 2-crate \
             front-end. A configured eval::eval_loop_jit driver will fail \
             exact portal resolution; run scripts/extract-llbc.py first."
        );
    }
    Some(paths)
}

/// Merge JIT-hint markers harvested from the ullbc surrogate consts
/// into a MIR-driven SemanticProgram.
///
/// `front::llbc_hints::harvest_hints_from_llbcs` reads the
/// `#[doc(hidden)]` marker consts the `majit_macros` proc-macros emit
/// (`_elidable_function_<NAME>`, `_jit_elidable_cannot_raise_<NAME>`,
/// `_jit_look_inside_<NAME>`, …) out of Charon's `global_decls`,
/// keyed by the crate-stripped function path.  Each `SemanticFunction`
/// is matched by its `{module_path}::{name}` path so same-named helpers
/// in different modules cannot inherit each other's hints.
#[cfg(feature = "mir-frontend")]
fn merge_hints_from_llbcs(
    program: &mut front::SemanticProgram,
    llbcs: &[majit_charon_reader::Llbc],
) {
    let hints_by_path = front::llbc_hints::harvest_hints_from_llbcs(llbcs);
    for f in &mut program.functions {
        let path = if f.module_path.is_empty() {
            f.name.clone()
        } else {
            format!("{}::{}", f.module_path, f.name)
        };
        if let Some(h) = hints_by_path.get(&path) {
            f.hints.clone_from(h);
            // A `dont_look_inside` callee returning `*mut PyObject`
            // (`SemanticFunction::returns_objectptr`, set structurally by
            // `front::mir::output_type_is_objectptr`) residualizes as an
            // opaque call.  The MIR driver leaves `return_type` `None`,
            // which the cutover residual prefill maps `None`→`Void` — a
            // miscompile for a callee the caller reads as a pointer.
            // Stamp the object-pointer marker so the residual reports a
            // `Ref` result instead.  Gated on the hint so non-opaque
            // object-pointer-returning fns keep `return_type == None`
            // (the call-signature validator's TyRef-label-misclassify
            // safeguard, `front::mir`).
            if f.return_type.is_none()
                && f.returns_objectptr
                && h.iter().any(|hint| hint == "dont_look_inside")
            {
                f.return_type =
                    Some(translator::rtyper::cutover::OBJECTPTR_RETURN_TYPE.to_string());
            }
        }
    }
}

/// `make_virtualizable_infos` constructor closure type — mirrors the
/// upstream `VirtualizableInfo(self, VTYPEPTR)` call (warmspot.py:543).
/// `(jd_idx, vtypeptr_token) -> Option<handle>`.  Hosts that own a
/// runtime `VirtualizableInfo` impl supply the constructor here.
pub type VirtualizableInfoFactory<'a> =
    dyn Fn(usize, &str) -> Option<std::sync::Arc<dyn call::VirtualizableInfoHandle>> + 'a;

/// Optional binding table from macro-generated helper path
/// (`module_path!()::helper_name`) to the compiled trace-call address.
///
/// `#[jit_module]::__majit_helper_trace_fnaddrs()` produces this shape.
/// `analyze_pipeline_from_module_paths` strips the crate-name prefix and binds
/// both canonical aliases (`helpers::foo` and `crate::helpers::foo`) on
/// `CallControl` before `get_jitcode()` / `jtransform` query fnaddrs.
pub type FnAddrBindings<'a> = [(&'a str, i64)];

/// Structured binding table for impl-method helpers.  Each entry is
/// `(module_path_with_crate, impl_type_as_written, method_name, fnaddr)`.
/// The codewriter applies the
/// `CallControl::register_macro_impl_helper_trace_fnaddr` qualification
/// rule — bare types get the module prefix (minus crate
/// name) prepended, qualified types are kept verbatim — before storing
/// the canonical `[impl_type_joined, method]` 2-segment CallPath.
///
/// `#[jit_module]::__majit_helper_impl_trace_fnaddrs()` produces this
/// shape and `analyze_pipeline_from_module_paths` feeds it through
/// `CallControl::register_macro_impl_helper_trace_fnaddr`.
pub type ImplFnAddrBindings<'a> = [(&'a str, &'a str, &'a str, i64)];

/// Host-supplied addresses of prebuilt object-space singletons that pyre
/// source carries through the flowgraph as opaque `LOAD_GLOBAL`
/// constants (the static `PyType` pointers and dict-strategy refs).
///
/// `majit-translate` is the `rpython/` translation layer and must not
/// import the `pyre-object` object space; the driver
/// (`pyre-jit-trace/build.rs`, via `pyre_interpreter::jit_static_*_addrs()`)
/// supplies these prebuilt-instance addresses across the translation
/// boundary, exactly as `rpython/jit` receives `Constant(GCREF)` from the
/// host instead of importing `pypy/objspace`.  Resolved in the same
/// build-script process that runs the translator, so the addresses match
/// a direct `&pyre_object::X` read at the codewriter call site.
///
/// `pytypes` are recorded as `ValueType::Int`, `refs` as
/// `ValueType::Ref(None)`, matching the front-end `KnownStaticsCatalogue`
/// classification.  Empty (the `Default`) for test / legacy entry points
/// that lower fixtures not referencing these singletons.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostStaticAddrs<'a> {
    pub pytypes: &'a [(&'a str, i64)],
    pub refs: &'a [(&'a str, i64)],
    /// Immutable size `const`s baked as their compile-time value
    /// (`ValueType::Int` `ConstInt`), keyed by crate-stripped
    /// `module::NAME`.  Distinct from `refs`: the value, not the address.
    pub int_values: &'a [(&'a str, i64)],
}

/// Multi-file analysis with explicit per-source module paths.
///
/// `module_paths[i]` is the crate-stripped module path of the i-th
/// analyzed source file (e.g. `"intobject"` for
/// `pyre_object/src/intobject.rs`).  The graph surface itself comes
/// from the Charon-extracted LLBC set; the `module_paths` slice drives
/// the workspace LLBC auto-discovery production fingerprint and stays
/// available for per-file lexical resolution.  Source text is not
/// consumed — callers pass paths only.
///
/// An empty `module_paths[i]` keeps simple-name registration only —
/// runtime convergence is then handled solely by the
/// `build_object_descr_group_with_def_path` dual-publish.
pub fn analyze_multiple_pipeline_with_modules(
    module_paths: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
    static_addrs: HostStaticAddrs<'_>,
) -> pipeline::ProgramPipelineResult {
    analyze_pipeline_from_module_paths(
        module_paths,
        None,
        config,
        layout_provider,
        vinfo_factory,
        fnaddr_bindings,
        &[],
        static_addrs,
    )
}

/// Multi-file analysis with an explicit, ordered LLBC input set.
///
/// Consumer build scripts should prefer this entry point over the legacy
/// `PYRE_MIR_FRONTEND_LLBC` compatibility environment variable. The supplied
/// paths are the complete translation input and never fall back to Pyre's
/// workspace auto-discovery.
pub fn analyze_multiple_pipeline_from_llbc_with_modules(
    llbc_paths: &[&str],
    module_paths: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
    static_addrs: HostStaticAddrs<'_>,
) -> pipeline::ProgramPipelineResult {
    assert!(
        !llbc_paths.is_empty(),
        "explicit LLBC analysis requires at least one input artifact"
    );
    analyze_pipeline_from_module_paths(
        module_paths,
        Some(llbc_paths),
        config,
        layout_provider,
        vinfo_factory,
        fnaddr_bindings,
        &[],
        static_addrs,
    )
}

/// Register a free-function graph under one alias path.  Panics if the
/// same alias is already mapped to a different `func.name` — this is
/// the parity guard against silent cross-crate name-tail collisions.
/// Same-name re-registration (e.g. a function reachable through more
/// than one well-known crate alias) is treated as idempotent.
fn register_function_graph_alias(
    graphs: &mut std::collections::HashMap<crate::parse::CallPath, crate::model::FunctionGraph>,
    sources: &mut std::collections::HashMap<crate::parse::CallPath, String>,
    path: crate::parse::CallPath,
    source_name: &str,
    graph: &crate::model::FunctionGraph,
) {
    if let Some(prev) = sources.get(&path) {
        assert!(
            prev == source_name,
            "function-graph alias collision at {}: previously registered by {prev:?}, now {source_name:?}; \
             cross-crate name-tail aliasing must not silently route to a different graph",
            path.canonical_key(),
        );
        return;
    }
    sources.insert(path.clone(), source_name.to_string());
    graphs.insert(path, graph.clone());
}

/// Compute the full alias spelling set for a free function lifted
/// from a Rust source.  Mirrors the graph-alias loop in
/// [`analyze_pipeline_from_module_paths`] so call-site lookups that key on
/// these spellings (function_graphs,
/// elidable/loopinvariant/cannot_collect/oopspec targets) all see
/// the same FunctionPath set.  Without this, a module-qualified call
/// site finds the graph alias but misses the elidable/loopinvariant
/// hint registered only under the bare name.
///
/// `name` is the `SemanticFunction.name` (already module-prefixed
/// when the function lives inside a `mod foo { fn bar() }` block);
/// `source_module` is the file's crate-stripped path (populated by
/// `front::mir` from the module portion of Charon's `name_path()`).
fn free_function_alias_paths(name: &str, source_module: &str) -> Vec<crate::parse::CallPath> {
    let segments: Vec<&str> = name.split("::").collect();
    let mut paths = Vec::new();
    paths.push(crate::parse::CallPath::from_segments(
        segments.iter().copied(),
    ));
    let mut crate_segs = vec!["crate"];
    crate_segs.extend(segments.iter().copied());
    paths.push(crate::parse::CallPath::from_segments(crate_segs));
    let crate_roots = crate::local_crates::local_crate_roots();
    for crate_alias in &crate_roots {
        let mut alias_segs = vec![crate_alias.as_str()];
        alias_segs.extend(segments.iter().copied());
        paths.push(crate::parse::CallPath::from_segments(alias_segs));
    }
    let name_segs_prefix: Vec<&str> = name.split("::").collect();
    let mod_segs_prefix: Vec<&str> = source_module.split("::").collect();
    // The starts_with check is meant to skip the module-qualified loop
    // when `name` already carries the module prefix (e.g. a nested
    // `mod foo { fn bar() }` whose `sf.name` is set to "foo::bar"
    // by `front::mir`).  Without
    // the length-strict guard, a function whose bare leaf happens to
    // equal its containing module's name (`pyre-interpreter/src/
    // stack_check.rs` `pub fn stack_check`) collides — its single-
    // segment name "stack_check" `starts_with` ["stack_check"] is
    // true, so the loop is skipped and the
    // `["module", "name"]` / `["crate", "module", "name"]` / aliased
    // spellings never get registered, leaving every call site that
    // writes `crate::stack_check::stack_check` looking for an
    // unregistered path.  Require `name` to be STRICTLY LONGER than
    // `module` to count as already-prefixed.
    let name_already_prefixed = name_segs_prefix.len() > mod_segs_prefix.len()
        && name_segs_prefix.starts_with(&mod_segs_prefix);
    if !source_module.is_empty() && !name_already_prefixed {
        let module_segs: Vec<&str> = source_module.split("::").collect();
        let mut module_qualified_segs: Vec<&str> = module_segs.clone();
        module_qualified_segs.extend(segments.iter().copied());
        paths.push(crate::parse::CallPath::from_segments(
            module_qualified_segs.iter().copied(),
        ));
        let mut crate_module_segs = vec!["crate"];
        crate_module_segs.extend(module_qualified_segs.iter().copied());
        paths.push(crate::parse::CallPath::from_segments(
            crate_module_segs.iter().copied(),
        ));
        for crate_alias in &crate_roots {
            let mut alias_segs = vec![crate_alias.as_str()];
            alias_segs.extend(module_qualified_segs.iter().copied());
            paths.push(crate::parse::CallPath::from_segments(
                alias_segs.iter().copied(),
            ));
        }
    }
    paths
}

fn analyze_pipeline_from_module_paths(
    module_paths: &[&str],
    explicit_llbc_paths: Option<&[&str]>,
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
    impl_fnaddr_bindings: &ImplFnAddrBindings<'_>,
    static_addrs: HostStaticAddrs<'_>,
) -> pipeline::ProgramPipelineResult {
    let profile = std::env::var_os("PYRE_PROFILE_PIPELINE").is_some();
    let phase_start = std::time::Instant::now();
    let mut last = phase_start;
    macro_rules! mark_phase {
        ($name:literal) => {{
            #[allow(unused_assignments)]
            if profile {
                let now = std::time::Instant::now();
                eprintln!(
                    "[PYRE_PROFILE_PIPELINE] {:>9.3}s  {:>9.3}s  {}",
                    (now - phase_start).as_secs_f64(),
                    (now - last).as_secs_f64(),
                    $name,
                );
                last = now;
            }
        }};
    }
    mark_phase!("entry");
    // `FORCE_ATTRIBUTES_INTO_CLASSES` is seeded from the LLBC-sourced
    // `program.struct_field_attrs` further below, once `program` is
    // built.
    // RPython `translator/translator.py:55 buildflowgraph` — FlowingError
    // propagates out and translation halts.  Pyre's top-level analyzer
    // requires a complete program; a FlowingError here means a user-
    // facing source file contains a construct we cannot yet lower, and
    // the correct response is to abort loudly so the coverage audit
    // surfaces the unsupported expression rather than silently dropping
    // a graph.
    // When the `mir-frontend` feature is enabled, route the production
    // SemanticProgram build through the MIR-driven
    // `front::mir::build_semantic_program_from_llbcs` path.  The LLBC
    // source is a Charon-extracted .ullbc snapshot (produced by
    // `scripts/extract-llbc.py`), located via `PYRE_MIR_FRONTEND_LLBC`
    // or workspace auto-discovery.
    mark_phase!("known_statics + struct_field_attrs populated");
    let program =
        build_semantic_program_via_active_frontend(module_paths, static_addrs, explicit_llbc_paths);
    // Publish the `(bare struct leaf → defining crate-relative module
    // path)` map into the process-global `STRUCT_ORIGIN_REGISTRY` so the
    // later `codewriter` `canonical_struct_name` `path_hash` sites
    // resolve bare struct tokens to their qualified canonical form (PyPy
    // `bookkeeper.getdesc(TYPE)` analog).  Derived from the LLBC
    // `iter_type_decls()` name paths in
    // `front::mir::derive_program_metadata`; any leaf absent from the map
    // still resolves through the runtime's simple-name dual-publish slot.
    majit_ir::descr::register_struct_origins(program.struct_origins.clone());
    // Seed the name → StructId resolver so the layout consumers that only
    // hold a string (`llmemory::FieldOffset`'s `st._name`, a nested
    // field's rendered type) can reach the identity-keyed layout maps.
    majit_ir::descr::register_struct_ids(program.struct_ids.clone());
    // Tier-3: seed `FORCE_ATTRIBUTES_INTO_CLASSES` (classdesc.py:957-961)
    // from the LLBC-sourced `program.struct_field_attrs` so
    // `ClassDesc::_init_classdef` pre-fills `ClassDef.attrs` before the
    // annotator's `attrs_populated` narrowing gate
    // (`flowspace_adapter.rs::derive_subject_inputcells`).  Replaces the
    // syn `pre_register_struct_fields_from_file` walk.
    // `struct_field_attrs` is the Charon `derive_program_metadata`
    // projection, keyed by the crate-stripped qualified item path
    // (`intobject::W_IntObject`).  Register each entry under that single
    // canonical key only.  The consumer reads by `cls.qualname()` (the
    // bare struct leaf) and canonicalises it through the
    // `STRUCT_ORIGIN_REGISTRY` (`canonical_struct_name`) before lookup,
    // so no bare-leaf alias is needed — dropping it keeps a leaf shared
    // by distinct modules (e.g. two `FrameBlock`s with different field
    // shapes) from clobbering each other.  Iterate in sorted key order
    // for build determinism.
    {
        let mut entries: Vec<_> = program.struct_field_attrs.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (qualified, fields) in entries {
            crate::annotator::classdesc::register_struct_fields(qualified, fields);
        }
    }
    mark_phase!("build_semantic_program_from_parsed_files");
    let mut canonical_trait_impls = Vec::new();
    let mut canonical_inherent_methods = Vec::new();
    // `(trait_leaf, trait_qualified, method_name, owner, return_type,
    // hints)` for every concrete trait-impl method — input to the
    // single-impl devirtualization pass below.  `trait_qualified` is
    // the trait's full `name_path()`; the leaf keys the
    // direct-path-binding bookkeeping (matching `canonical_trait_impls`
    // spelling), the qualified path keys the unique-impl map.
    #[allow(clippy::type_complexity)]
    let mut concrete_trait_methods: Vec<(
        String,
        String,
        String,
        String,
        Option<String>,
        Vec<String>,
        crate::model::FunctionGraph,
    )> = Vec::new();
    let mut canonical_function_graphs = std::collections::HashMap::new();
    // `bookkeeper.py:353-409 getdesc` / `newfuncdesc` keys on the host
    // function-object identity, so two unrelated `crate_a::helper` and
    // `crate_b::helper` resolve to distinct `FunctionDesc` instances.
    // Pyre's call registry is keyed on `CallPath` segment strings, and
    // the alias expansion below intentionally registers each free
    // function under every well-known pyre-crate prefix
    // (`pyre_interpreter` / `pyre_object` / `pyre_jit`).  Without a
    // collision check the second registration silently overwrites the
    // first when two source crates happen to define a function with
    // the same tail segments, so the call resolver may then route to
    // the wrong graph.  `canonical_function_alias_source` records the
    // canonical `func.name` that won each alias slot so a later
    // mismatched registration panics with a diagnostic instead of
    // silently aliasing across crates.
    let mut canonical_function_alias_source: std::collections::HashMap<
        crate::parse::CallPath,
        String,
    > = std::collections::HashMap::new();

    // Source impl-method registration from the MIR-lowered
    // `program.functions`.  Each `SemanticFunction` carries
    // `self_ty_root` / `trait_root` populated to the exact registration
    // keys the downstream loops below consume, so those loops are
    // agnostic to where the `canonical_*` vectors come from.
    //
    // `program.functions` is a superset of every graph-carrying
    // method the source surface defines.  Trait methods with no
    // graph are intentional non-JIT targets — `#[cfg(test)]` impls,
    // `into_py` on primitive receivers Charon cannot key to an ADT,
    // residual `PyreBlackholeAllocator::*` allocator hooks, std-trait
    // impls (`Drop`/`From`/`PartialEq`/…), and `BoxEnv` default bodies
    // MIR does not lower.  They carry no graph (so the BFS never
    // reaches them) and their `return_types` are redundant with the
    // CFG-scan result kind (`graph_result_kind` = `getkind(FUNC.RESULT)`,
    // `codewriter.rs:656`) — the calldescr builder reads the declared
    // type only as an `i↔r` tiebreak and `debug_assert`s it against the
    // CFG kind; no impl method diverges, so omitting the side-table
    // entries is a no-op for call-descriptor typing.
    //
    // `mir_graph_lookup` is consulted by the registration loops below
    // (trait-default vs concrete-impl graph fetch).
    let mir_graph_lookup = front::semantic::MirGraphLookup::from_program(&program);
    // `classdesc.py:749 lookup` MRO walk for generic trait dispatch:
    // a concrete override shadows the trait default.  `front::mir`
    // lowers `CallKind::Trait` (a call through a generic `<E: Trait>`
    // receiver) to the direct path `[<Trait>, <method>]`, and the
    // registration loop below binds that path for BFS discovery and
    // `get_jitcode` resolution.  Collect each trait's concrete impl
    // types and per-method overrides so the direct-path registration
    // can prefer the unique override: with exactly one concrete impl
    // in the analyzed program the annotator binds the receiver to
    // that instantiation, so the MRO finds the override before the
    // default body.  With several concrete impl types the receiver
    // stays ambiguous and the default-body registration is kept
    // (indirect-call family territory).
    let mut trait_concrete_impl_types: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    let mut trait_method_overrides: std::collections::HashMap<
        (&str, &str),
        (&str, &front::semantic::SemanticFunction),
    > = std::collections::HashMap::new();
    for func in &program.functions {
        match (&func.self_ty_root, &func.trait_root) {
            // Concrete trait-impl method: `impl Trait for Type { fn m }`.
            // Registration-wise it behaves exactly like an inherent
            // method (`[Owner, method]` graph + hints below); routing it
            // through `register_trait_method` instead would also seed
            // `method_to_impl_types`, flipping `resolve_method`'s
            // name-based lookup for every same-named method.  The trait
            // identity feeds two complementary direct-path consumers:
            // the `concrete_trait_methods` loop below binds the unique
            // impl when the trait method has NO default body, and the
            // `trait_method_overrides` lookup inside the default-body
            // registration prefers the unique override over the default
            // (`classdesc.py:749` — a concrete override shadows the
            // trait default in the MRO walk).
            (Some(owner), Some(trait_leaf)) => {
                concrete_trait_methods.push((
                    trait_leaf.clone(),
                    func.trait_qualified
                        .clone()
                        .unwrap_or_else(|| trait_leaf.clone()),
                    func.name.clone(),
                    owner.clone(),
                    func.return_type.clone(),
                    func.hints.clone(),
                    func.graph.clone(),
                ));
                let types = trait_concrete_impl_types
                    .entry(trait_leaf.as_str())
                    .or_default();
                if !types.contains(&owner.as_str()) {
                    types.push(owner.as_str());
                }
                trait_method_overrides.insert(
                    (trait_leaf.as_str(), func.name.as_str()),
                    (owner.as_str(), func),
                );
                canonical_inherent_methods.push(parse::InherentMethodInfo {
                    for_type: owner.clone(),
                    self_ty_root: Some(owner.clone()),
                    name: func.name.clone(),
                    graph: func.graph.clone(),
                    return_type: func.return_type.clone(),
                    hints: func.hints.clone(),
                });
            }
            // Trait default-body: `trait T { fn m { … } }`.  The pseudo
            // `for_type` matches the extractor's sentinel so the loop's
            // `is_default` branch and the `call.rs` resolve-method
            // filters keep distinguishing default from concrete impl.
            (None, Some(trait_leaf)) => {
                canonical_trait_impls.push(TraitImplInfo {
                    trait_name: trait_leaf.clone(),
                    for_type: format!("<default methods of {}>", trait_leaf),
                    self_ty_root: None,
                    methods: vec![MethodInfo {
                        name: func.name.clone(),
                        graph: Some(func.graph.clone()),
                        return_type: func.return_type.clone(),
                        hints: func.hints.clone(),
                    }],
                });
            }
            // Inherent method: `impl Type { fn m }`.
            (Some(owner), None) => {
                canonical_inherent_methods.push(parse::InherentMethodInfo {
                    for_type: owner.clone(),
                    self_ty_root: Some(owner.clone()),
                    name: func.name.clone(),
                    graph: func.graph.clone(),
                    return_type: func.return_type.clone(),
                    hints: func.hints.clone(),
                });
            }
            // Free function — registered by the dedicated loop below.
            (None, None) => {}
        }
    }
    // RPython: use the rtyped graphs (with concretetype info) for all analysis.
    // Use program.functions' graphs which were built with full struct_fields
    // context, NOT re-parsed graphs (which lose array_type_id etc.).
    for func in &program.functions {
        if func.self_ty_root.is_none() {
            // Stamp the source return type onto the graph so the JIT
            // codewriter signature validator reads `FUNC.RESULT`
            // directly off the callee graph (RPython
            // `funcptr._obj.TO.RESULT`).
            let graph = match &func.return_type {
                Some(rt) => func.graph.clone().with_return_type(rt),
                None => func.graph.clone(),
            };
            // Free function: register under every canonical alias
            // spelling computed by `free_function_alias_paths` — bare
            // segments, `crate::` prefix, three pyre-crate prefixes,
            // and the same set re-prefixed with `func.module_path`
            // (the file's crate-stripped module path).  The hint
            // registration loop below uses the same helper so call
            // sites that hit a module-qualified spelling find both the
            // graph and its elidable/loopinvariant/oopspec hints.
            for path in free_function_alias_paths(&func.name, &func.module_path) {
                register_function_graph_alias(
                    &mut canonical_function_graphs,
                    &mut canonical_function_alias_source,
                    path,
                    &func.name,
                    &graph,
                );
            }
        }
    }

    // ── Build CallControl (RPython call.py) ──
    // Populate with all discovered function graphs and trait impl methods.
    let mut call_control = call::CallControl::new();
    // RPython: known struct types for get_type_flag(ARRAY.OF) → FLAG_STRUCT.
    call_control.set_known_struct_names(program.known_struct_names.clone());
    // RPython: struct field types for op.args[0].concretetype resolution.
    call_control.set_struct_fields(program.struct_fields.clone());
    // Enum `discriminant → variant` tables for the `__discriminant`
    // getattr's discriminant→variant narrowing knowntypedata.
    call_control.set_enum_variant_by_discriminant(program.enum_variant_by_discriminant.clone());
    // RPython: symbolic.get_field_token / get_size — resolve struct layouts
    // through the LayoutProvider. If no provider is given, use the heuristic
    // (type-string-based approximation of #[repr(C)] layout).
    let heuristic;
    let provider: &dyn layout::LayoutProvider = match layout_provider {
        Some(p) => p,
        None => {
            heuristic = layout::HeuristicLayoutProvider::from_struct_fields(
                &program.struct_fields.fields,
                &program.known_struct_names,
                &program.immutable_fields,
            );
            &heuristic
        }
    };
    // RPython: per-class `_immutable_fields_` declaration. Drives
    // `FieldDescr.is_pure` for the heuristic-fallback path inside
    // `all_interiorfielddescrs` (the layout-provider path already carries
    // `is_immutable` on `StructFieldLayout`).
    call_control.immutable_fields_by_struct = program.immutable_fields.clone();
    // `descr.py:364 ARRAY_INSIDE._immutable_field(None)` parity.
    // Summarise `field[*]` annotations into the array-type-keyed set so
    // `arraydescrof_concrete` can fold field-level immutability into the
    // shared per-ARRAY descr's `is_pure` flag.
    call_control.recompute_immutable_array_types();
    // The `unsafe_fn_stubs` carrier lets the codewriter's
    // `dual_gate_registry` register every `unsafe fn` / unsafe
    // impl-method as a stub-pygraph entry in PyreCallRegistry, covering
    // the bulk of the "not registered in PyreCallRegistry" Skip cluster
    // dominated by `pyre_object::is_*` predicates whose body lowering is
    // intentionally rejected (raw-pointer access the flowspace adapter
    // does not model — only a typed signature stub is registered).
    // Sourced from Charon via
    // `front::mir::collect_unsafe_fn_stubs_from_llbc`, populated on the
    // SemanticProgram in `build_semantic_program_via_active_frontend`.
    call_control.unsafe_fn_stubs = program.unsafe_fn_stubs.clone();
    // Parallel carrier for foreign opaque-ADT method externals
    // (`<BigInt as Add>::add`, …) — `populate_call_registry_from_call_graphs`
    // registers each as an opaque external so the residual `FunctionPath`
    // form resolves; see `cutover::register_foreign_opaque_method_externals`.
    call_control.foreign_opaque_method_externals = program.foreign_opaque_method_externals.clone();
    // Populate CallControl with layouts from the provider.  Where Charon
    // resolved an exact layout (`program.exact_layouts`), use the true Rust
    // offsets and total size — `#[repr(Rust)]` reorders/repacks fields, so
    // the heuristic's `#[repr(C)]` approximation can disagree with the real
    // allocation. For enum-variant keys the exact offsets are tag-inclusive,
    // so the variant payload resolves at its real position. Per-field type
    // classification, immutability rank, and size stay as the provider
    // computed them; only the byte offsets and the struct size are corrected.
    for struct_name in program.struct_fields.fields.keys() {
        // Resolve the (possibly multi-spelled) field-registry key to its
        // identity token; an unresolved / ambiguous bare leaf has no
        // identity to key a layout on, so skip it.
        let Some(sid) = majit_ir::descr::struct_id_for_name(struct_name) else {
            continue;
        };
        let layout = match program.exact_layouts.get(&sid) {
            Some(exact) => {
                provider.get_struct_layout_exact(struct_name, &exact.field_offsets, exact.size)
            }
            None => provider.get_struct_layout(struct_name),
        };
        if let Some(layout) = layout {
            call_control.set_struct_layout(sid, layout);
        }
    }
    // Register graphs collected above (free functions only — trait
    // methods are handled separately via register_trait_method).
    for (path, graph) in &canonical_function_graphs {
        call_control.register_function_graph(path.clone(), graph.clone());
    }
    // Re-register free functions with their RPython-equivalent hints
    // (`elidable`, `loop_invariant`, `unroll_safe`, `jit_look_inside`)
    // so `JitPolicy::look_inside_graph` sees the same metadata RPython
    // reads off `func._jit_*_` / `_elidable_function_`.
    //
    // RPython parity: hints live on `graph.func` and survive alias
    // routing because the call path resolves to a single function
    // object identity (`policy.py:48` / `call.py:126`).  Pyre keys
    // hints on `CallPath` segments, so every spelling the graph alias
    // loop registered above (`free_function_alias_paths`: bare +
    // `crate::` + `pyre_interpreter|pyre_object|pyre_jit::` + the
    // `source_module`-prefixed variants thereof) must carry the same
    // hint set.  Missing the source-module-qualified aliases silently
    // disables `_jit_look_inside_` etc. for module-qualified callers,
    // which `CallControl::find_all_graphs` looks up by callee path.
    for func in &program.functions {
        if !func.self_ty_root.is_none() || func.hints.is_empty() {
            continue;
        }
        let graph = match &func.return_type {
            Some(rt) => func.graph.clone().with_return_type(rt),
            None => func.graph.clone(),
        };
        for path in free_function_alias_paths(&func.name, &func.module_path) {
            call_control.register_function_graph_with_hints(
                path,
                graph.clone(),
                func.hints.clone(),
            );
        }
    }
    // The registration loop below prefers the graph from
    // `mir_graph_lookup` over the one already carried in
    // `canonical_trait_impls` / `canonical_inherent_methods`.  Both
    // come from `program.functions`, so the lookup is a defensive
    // re-fetch keyed on the registration path — cheap, and it keeps a
    // single source of truth for the trait-default vs concrete-impl
    // distinction.
    for impl_info in &canonical_trait_impls {
        let impl_type = impl_info
            .self_ty_root
            .as_deref()
            .unwrap_or(&impl_info.for_type);
        // RPython parity: `trait_root=Some(trait_name)` for real trait impls,
        // `None` for inherent impls (impl SomeType { ... } without `for Trait`).
        // `parse.rs:237` always writes `trait_name`; a sentinel empty string
        // from the inherent branch (see parse.rs:357-389) needs special-casing.
        let trait_root = if impl_info.trait_name.is_empty() {
            None
        } else {
            Some(impl_info.trait_name.as_str())
        };
        let is_default = impl_info.for_type.starts_with("<default methods of ");
        for method in &impl_info.methods {
            // `classdesc.py:749 lookup` MRO: on the generic-dispatch
            // direct path `[<Trait>, <method>]`, the unique concrete
            // override shadows the trait default body (pre-pass
            // above).  `None` for concrete-impl entries, defaults
            // without an override, and traits with several concrete
            // impl types (the receiver stays ambiguous —
            // indirect-call family territory keeps the default).
            //
            // Staged scope: whole-trait shadowing is blocked on the
            // classdef-hints-before-BFS annotator work — registering
            // objspace-heavy overrides (load_attr / call_callable /
            // binary_op …) pulls their graph closure into the BFS,
            // where the annotator fails on classdef-less SomeInstance
            // attr reads and on runtime statics no build-time table
            // can resolve (`JIT_DRIVER`).  Grown deliberately, one
            // fail-loud resolution at a time; the motivating members are
            // the exception-handler pair whose empty defaults broke
            // generic-dispatch resolution.
            const DEFAULT_SHADOW_DEVIRT_SCOPE: &[&str] = &["push_exc_info", "pop_except"];
            let devirt: Option<(&str, &front::semantic::SemanticFunction)> =
                if is_default && DEFAULT_SHADOW_DEVIRT_SCOPE.contains(&method.name.as_str()) {
                    trait_method_overrides
                        .get(&(impl_info.trait_name.as_str(), method.name.as_str()))
                        .filter(|_| {
                            trait_concrete_impl_types
                                .get(impl_info.trait_name.as_str())
                                .is_some_and(|types| types.len() == 1)
                        })
                        .copied()
                } else {
                    None
                };
            // Hints for the direct path follow the graph registered
            // there (RPython binds hints to graph identity).
            let direct_hints: &Vec<String> = match devirt {
                Some((_, override_info)) => &override_info.hints,
                None => &method.hints,
            };
            // Read the MIR-built graph from `program.functions`.
            // `method.graph` (`Option`) remains as a residual fallback
            // for the handful of MIR-uncovered entries, though every
            // method registered above carries a graph so the fallback
            // is effectively unreached.
            let mir_graph: Option<&model::FunctionGraph> = if is_default {
                mir_graph_lookup.lookup_trait_default(&impl_info.trait_name, &method.name)
            } else {
                mir_graph_lookup.lookup_impl_method(impl_type, &method.name)
            };
            let graph_source: Option<model::FunctionGraph> =
                mir_graph.cloned().or_else(|| method.graph.clone());
            if let Some(graph) = graph_source {
                // Stamp the source return type onto the graph itself so
                // the JIT codewriter signature validator reads
                // `FUNC.RESULT` directly off the callee graph
                // (RPython `funcptr._obj.TO.RESULT`).
                let graph = match &method.return_type {
                    Some(rt) => graph.with_return_type(rt),
                    None => graph,
                };
                call_control.register_trait_method(&method.name, trait_root, impl_type, graph);
                // Parity with upstream `rpython/annotator/classdesc.py:749
                // lookup` MRO walk: a trait default body is the
                // "base-class method" for every impl that does not
                // override it. Rust-idiomatic call sites emit the call
                // as `<Trait>::<method>(receiver, ...)` —
                // `front::mir` lowers that into
                // `CallTarget::FunctionPath { segments: [<Trait>,
                // <method>] }`. The upstream-equivalent registration key
                // is therefore `[<Trait>, <method>]`. The pseudo-type
                // path `[<default methods of Trait>, <method>]` set by
                // `register_trait_method` is retained so the filter logic
                // at `call.rs:1921,1970 resolve_method*` and
                // `lib.rs:935 push_matching_trait_methods` can continue
                // to distinguish "trait default" from "concrete impl".
                if is_default {
                    let direct_path = crate::parse::CallPath::from_segments([
                        impl_info.trait_name.as_str(),
                        method.name.as_str(),
                    ]);
                    // Prefer the MIR graph for the direct_path
                    // registration too; fall back to the carried
                    // graph when the lookup has no entry.  `devirt`
                    // (hoisted above) swaps in the unique concrete
                    // override's graph and return type.
                    let (direct_source, direct_return_type) = match devirt {
                        Some((impl_type, override_info)) => (
                            mir_graph_lookup
                                .lookup_impl_method(impl_type, &method.name)
                                .cloned()
                                .or_else(|| Some(override_info.graph.clone())),
                            override_info.return_type.as_ref(),
                        ),
                        None => (
                            mir_graph.cloned().or_else(|| method.graph.clone()),
                            method.return_type.as_ref(),
                        ),
                    };
                    if let Some(g) = direct_source {
                        let direct_graph = match direct_return_type {
                            Some(rt) => g.with_return_type(rt),
                            None => g,
                        };
                        call_control.register_function_graph(direct_path, direct_graph);
                    }
                }
            }
            let path = crate::parse::CallPath::for_impl_method(impl_type, method.name.as_str());
            // Mirror RPython `func._elidable_function_` / `func._jit_*_`:
            // `register_trait_method` registers the graph without hints, so
            // the BFS would see hint-less SemanticFunctions for trait methods
            // and inline elidable methods that should remain residual.
            // `register_function_hints_for` stamps the hints onto the
            // already-registered graph (`graph.hints`) keyed on the same
            // `[impl_type, method_name]` path the BFS reads.
            if !method.hints.is_empty() {
                call_control.register_function_hints_for(path.clone(), method.hints.clone());
            }
            // Default-method bodies also register under `[trait_name,
            // method_name]` (see the `register_function_graph(direct_path,
            // direct_graph)` branch above); mirror the hint registration
            // so the BFS reaches the same `_reject_function("elidable")`
            // verdict regardless of which path it walks.  `direct_hints`
            // follows the graph registered there — the override's when
            // the direct path was devirtualized.
            if is_default && !direct_hints.is_empty() {
                let direct_path = crate::parse::CallPath::from_segments([
                    impl_info.trait_name.as_str(),
                    method.name.as_str(),
                ]);
                call_control.register_function_hints_for(direct_path, direct_hints.clone());
            }
            // RPython: hints bound to graph identity.
            for hint in &method.hints {
                match hint.as_str() {
                    "elidable" => call_control.mark_elidable(path.clone()),
                    "elidable_cannot_raise" => {
                        call_control.mark_cannot_raise_assertion(path.clone())
                    }
                    "elidable_or_memerror" => {
                        call_control.mark_memerror_only_assertion(path.clone())
                    }
                    "loopinvariant" => call_control.mark_loopinvariant(path.clone()),
                    "close_stack" => call_control.mark_close_stack(path.clone()),
                    "cannot_collect" => call_control.mark_cannot_collect(path.clone()),
                    "gc_effects" => call_control.mark_external_gc_effects(path.clone()),
                    _ => {}
                }
            }
            if is_default {
                let dp = crate::parse::CallPath::from_segments([
                    impl_info.trait_name.as_str(),
                    method.name.as_str(),
                ]);
                for hint in direct_hints {
                    match hint.as_str() {
                        "elidable" => call_control.mark_elidable(dp.clone()),
                        "elidable_cannot_raise" => {
                            call_control.mark_cannot_raise_assertion(dp.clone())
                        }
                        "elidable_or_memerror" => {
                            call_control.mark_memerror_only_assertion(dp.clone())
                        }
                        "loopinvariant" => call_control.mark_loopinvariant(dp.clone()),
                        _ => {}
                    }
                }
            }
        }
    }
    // Single-impl devirtualization for REQUIRED trait methods.  A call
    // site `<Trait>::<method>(receiver, …)` lowers to
    // `CallTarget::FunctionPath { segments: [<Trait>, <method>] }`
    // (`front/mir.rs` `call_target_segments` `CallKind::Trait` arm).
    // Default bodies registered that direct path in the loop above; a
    // required method (declaration only, no default body) left it
    // unregistered, so the registry lift failed with "not registered
    // in PyreCallRegistry" and poisoned every caller.  RPython
    // resolves the call on the receiver's class
    // (`classdesc.py:749 lookup` MRO walk); when exactly one class
    // implements the method in the closed LLBC world, that walk has a
    // single possible answer — bind the direct path to it.  Two or
    // more impls (or a default body, already covered) stay off this
    // path so ambiguous dispatch keeps failing loud until
    // receiver-driven resolution lands.
    let mut default_trait_methods: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    for impl_info in &canonical_trait_impls {
        if impl_info.trait_name.is_empty()
            || !impl_info.for_type.starts_with("<default methods of ")
        {
            continue;
        }
        for method in &impl_info.methods {
            default_trait_methods.insert((impl_info.trait_name.clone(), method.name.clone()));
        }
    }
    let mut concrete_impl_counts: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();
    for (trait_leaf, _, method_name, _, _, _, _) in &concrete_trait_methods {
        *concrete_impl_counts
            .entry((trait_leaf.clone(), method_name.clone()))
            .or_insert(0) += 1;
    }
    // Trait → owner of its only concrete impl, for the dual-gate
    // bookkeeper's generic-receiver classdef seeding
    // (`derive_subject_inputcells` resolves a bound-trait `class_root`
    // through this map).  Unlike the direct-path registration below,
    // this is annotation-only metadata — it pulls no graph bodies into
    // callers.
    let mut trait_impl_owners: std::collections::HashMap<
        String,
        std::collections::BTreeSet<String>,
    > = std::collections::HashMap::new();
    for (_, trait_qualified, _, owner, _, _, _) in &concrete_trait_methods {
        // Keyed by the trait's qualified `name_path()` — two distinct
        // traits sharing a leaf name must not pool their impl owners
        // (`tyref_generic_trait_bound_root` resolves bound-trait
        // receivers through this map by the same qualified path).
        trait_impl_owners
            .entry(trait_qualified.clone())
            .or_default()
            .insert(owner.clone());
    }
    // Leaf names shared by more than one qualified struct in the
    // field registry: a unique-impl entry whose owner collapses to
    // such a leaf could seed the receiver with the OTHER same-named
    // struct's classdef, so those entries are dropped (fail-safe: the
    // receiver keeps the classdef-less shell and the block stays
    // census-visible).
    let mut struct_leaf_counts: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    for key in program.struct_fields.fields.keys() {
        if let Some((_, leaf)) = key.rsplit_once("::") {
            *struct_leaf_counts.entry(leaf).or_default() += 1;
        }
    }
    // Opt-in receiver-driven dispatch families (issue #346): a consumer
    // (e.g. the aheui census) names `>=2`-impl trait qualified paths
    // whose `dyn Trait` receivers should annotate to a base ClassDef
    // linking the impl subclasses, so a method getattr on the receiver
    // resolves the impl MethodDesc family.  Built from `trait_impl_owners`
    // (before it is consumed below) and forwarded through `CallControl`
    // to `dual_gate_registry`, which mints each family before the
    // class-method seeding.
    //
    // Per-family builder shared by the config and the gated auto-population
    // paths.  Skips a family whose impl owners collapse to one leaf under
    // the leaf-keyed (`canonical_struct_name`) family interning + method
    // seeding — that would silently drop a member; leaving the trait's
    // classdef-less / fail-loud disposition is the fail-safe.  Mirrors the
    // `struct_leaf_counts` bail on the single-impl path below.
    let make_registration = |trait_qualified: &str,
                             owners: &std::collections::BTreeSet<String>|
     -> Option<call::TraitFamilyRegistration> {
        let impl_roots: Vec<String> = owners
            .iter()
            .map(|owner| owner.rsplit("::").next().unwrap_or(owner).to_string())
            .collect();
        // Within-family leaf collision: two impls of THIS trait whose
        // qualified owners share a leaf across modules.
        let mut seen = std::collections::HashSet::new();
        if let Some(dup) = impl_roots.iter().find(|leaf| !seen.insert(leaf.as_str())) {
            eprintln!(
                "register_trait_families: {trait_qualified:?} has impls sharing \
                 leaf {dup:?} across modules ({owners:?}); skipping — leaf-keyed \
                 seeding cannot disambiguate them"
            );
            return None;
        }
        Some(call::TraitFamilyRegistration {
            base_root: trait_qualified.to_string(),
            impl_roots,
        })
    };
    // Config-named families first (empty in pyre production).  Preserves
    // the prior spelling-mismatch diagnostic on the `None` arm.
    let mut trait_family_registrations: Vec<call::TraitFamilyRegistration> = config
        .pipeline
        .register_trait_families
        .iter()
        .filter_map(
            |trait_qualified| match trait_impl_owners.get(trait_qualified) {
                Some(owners) => make_registration(trait_qualified, owners),
                None => {
                    // Config validation: a named family that matches no
                    // harvested trait is almost always a spelling mismatch
                    // (the key is the trait's full `name_path()`).  List the
                    // available multi-impl trait paths so the consumer can
                    // correct it.
                    let candidates: Vec<&String> = trait_impl_owners
                        .iter()
                        .filter(|(_, owners)| owners.len() >= 2)
                        .map(|(path, _)| path)
                        .collect();
                    eprintln!(
                        "register_trait_families: no harvested trait matches {trait_qualified:?}; \
                 known multi-impl trait paths: {candidates:?}"
                    );
                    None
                }
            },
        )
        .collect();
    // Gated auto-population (issue #346): with `PYRE_DYN_INDIRECT`
    // on, register EVERY `>=2`-impl trait so its inline `dyn Trait`
    // receiver (lowered to `CallTarget::Indirect`) narrows to the family
    // base ClassDef.  Union with the config list (dedup by base_root),
    // never replacing it, to stay forward-safe with the aheui census.
    // MUST stay behind the gate: minting base/impl subclass classdefs
    // perturbs `pyre_struct_root_names` → `ensure_session` inheritance-id
    // numbering even off-path, so gate-OFF keeps the config-only (empty in
    // prod) set byte-identical.  Applies the same within-family `dup_leaf`
    // guard as the config path plus the cross-registry `struct_leaf_counts`
    // bail: an impl leaf that also names a DIFFERENT registered struct
    // would mis-seed the family, so skip it (fail-safe classdef-less).
    if front::mir::dyn_indirect_enabled() {
        let already: std::collections::HashSet<&str> = trait_family_registrations
            .iter()
            .map(|r| r.base_root.as_str())
            .collect();
        let mut auto: Vec<call::TraitFamilyRegistration> =
            trait_impl_owners
                .iter()
                .filter(|(trait_qualified, owners)| {
                    owners.len() >= 2 && !already.contains(trait_qualified.as_str())
                })
                .filter_map(|(trait_qualified, owners)| {
                    let reg = make_registration(trait_qualified, owners)?;
                    // Cross-registry leaf collision: an impl leaf shared by
                    // another qualified struct in the field registry would
                    // collapse the subclass onto that struct's classdef.
                    if let Some(dup) = reg.impl_roots.iter().find(|leaf| {
                        struct_leaf_counts.get(leaf.as_str()).copied().unwrap_or(0) > 1
                    }) {
                        eprintln!(
                            "register_trait_families: auto {trait_qualified:?} impl leaf {dup:?} \
                         collides with another registered struct; skipping — leaf-keyed \
                         seeding cannot disambiguate them"
                        );
                        return None;
                    }
                    Some(reg)
                })
                .collect();
        // Deterministic mint order (BTreeMap iteration on `trait_impl_owners`
        // is already sorted, but sort defensively since a HashMap seeded it).
        auto.sort_by(|a, b| a.base_root.cmp(&b.base_root));
        trait_family_registrations.extend(auto);
    }
    call_control.set_trait_family_registrations(trait_family_registrations);
    let trait_unique_impls: std::collections::HashMap<String, String> = trait_impl_owners
        .into_iter()
        .filter_map(|(trait_qualified, owners)| {
            if owners.len() != 1 {
                return None;
            }
            let owner = owners.into_iter().next().unwrap();
            // `self_ty_root` may be module-qualified
            // (`pyframe::PyFrame`); the struct-field registry and
            // `getuniqueclassdef_for_struct_root` key on the leaf.
            let leaf = owner.rsplit("::").next().unwrap_or(&owner).to_string();
            if struct_leaf_counts.get(leaf.as_str()).copied().unwrap_or(0) > 1 {
                return None;
            }
            Some((trait_qualified, leaf))
        })
        .collect();
    call_control.set_trait_unique_impls(trait_unique_impls);
    for (trait_leaf, _, method_name, _owner, return_type, hints, graph) in &concrete_trait_methods {
        let key = (trait_leaf.clone(), method_name.clone());
        if concrete_impl_counts.get(&key) != Some(&1) || default_trait_methods.contains(&key) {
            continue;
        }
        // Use this trait-impl method's own graph rather than
        // `lookup_impl_method(owner, method_name)`: when the impl owner
        // also has a same-named *inherent* method (e.g. `PyFrame` has both
        // the inherent `peek_at` and `<PyFrame as SharedOpcodeHandler>::
        // peek_at`), the two collide on the `(owner, name)` key and the
        // lookup resolves to `Err(())` (ambiguous), silently dropping the
        // single-impl devirtualization.  The carried graph is unambiguous.
        let graph = match return_type {
            Some(rt) => graph.clone().with_return_type(rt),
            None => graph.clone(),
        };
        let direct_path =
            crate::parse::CallPath::from_segments([trait_leaf.as_str(), method_name.as_str()]);
        call_control.register_function_graph(direct_path.clone(), graph);
        // Mirror the default-direct hint registration above so the
        // BFS reaches the same `_reject_function` verdict on this
        // path as on `[impl_type, method]`.
        if !hints.is_empty() {
            call_control.register_function_hints_for(direct_path.clone(), hints.clone());
        }
        for hint in hints {
            match hint.as_str() {
                "elidable" => call_control.mark_elidable(direct_path.clone()),
                "elidable_cannot_raise" => {
                    call_control.mark_cannot_raise_assertion(direct_path.clone())
                }
                "elidable_or_memerror" => {
                    call_control.mark_memerror_only_assertion(direct_path.clone())
                }
                "loopinvariant" => call_control.mark_loopinvariant(direct_path.clone()),
                _ => {}
            }
        }
    }
    // RPython: direct_call → funcobj.graph — inherent methods are resolved
    // by direct graph linkage, not name-based trait method lookup.
    // Register only in function_graphs under qualified path (Type::method).
    for method_info in &canonical_inherent_methods {
        let impl_type = method_info
            .self_ty_root
            .as_deref()
            .unwrap_or(&method_info.for_type);
        let path = crate::parse::CallPath::for_impl_method(impl_type, method_info.name.as_str());
        // Read the MIR-built graph; `method_info.graph` is the
        // residual fallback for the handful of MIR-uncovered entries,
        // effectively unreached because every inherent method
        // registered above carries a graph.
        let graph: model::FunctionGraph = mir_graph_lookup
            .lookup_impl_method(impl_type, &method_info.name)
            .map(|g| g.clone())
            .unwrap_or_else(|| method_info.graph.clone());
        // Pair the graph with the method's hints so the BFS-driven
        // `look_inside_graph` synthesises a `SemanticFunction` whose
        // `_reject_function("elidable")` mirrors RPython's
        // `getattr(func, "_elidable_function_", False)`.  Without this
        // the BFS sees impl methods as hint-less and inlines elidable
        // methods (e.g. `PyFrame::nlocals`) that should remain residual.
        // Stamp the source return type onto the graph so the JIT
        // codewriter signature validator reads `FUNC.RESULT` directly off
        // the callee graph (`funcptr._obj.TO.RESULT`), matching the
        // free-function and trait-method registration paths above.
        let graph = match &method_info.return_type {
            Some(rt) => graph.with_return_type(rt),
            None => graph,
        };
        if method_info.hints.is_empty() {
            call_control.register_function_graph(path.clone(), graph);
        } else {
            call_control.register_function_graph_with_hints(
                path.clone(),
                graph,
                method_info.hints.clone(),
            );
        }
        // RPython: hints bound to graph identity.
        for hint in &method_info.hints {
            match hint.as_str() {
                "elidable" => call_control.mark_elidable(path.clone()),
                "elidable_cannot_raise" => call_control.mark_cannot_raise_assertion(path.clone()),
                "elidable_or_memerror" => call_control.mark_memerror_only_assertion(path.clone()),
                "loopinvariant" => call_control.mark_loopinvariant(path.clone()),
                "close_stack" => call_control.mark_close_stack(path.clone()),
                "cannot_collect" => call_control.mark_cannot_collect(path.clone()),
                "gc_effects" => call_control.mark_external_gc_effects(path.clone()),
                _ => {}
            }
        }
    }
    for &(full_path, fnaddr) in fnaddr_bindings {
        call_control.register_macro_helper_trace_fnaddr(full_path, fnaddr);
    }
    for &(module_path_with_crate, impl_type_as_written, method, fnaddr) in impl_fnaddr_bindings {
        call_control.register_macro_impl_helper_trace_fnaddr(
            module_path_with_crate,
            impl_type_as_written,
            method,
            fnaddr,
        );
    }
    // RPython: GC transformer sets _gctransformer_hint_close_stack_,
    // _gctransformer_hint_cannot_collect_ on functions, and
    // random_effects_on_gcobjs on external function objects.
    // In majit, detect these from #[jit_close_stack], #[jit_cannot_collect],
    // #[jit_gc_effects] attributes on functions in the parsed source.
    // RPython: hints are bound to the function/graph object, not the name.
    // Register under ALL canonical paths so any call-site lookup finds them.
    for func in &program.functions {
        if func.hints.is_empty() {
            continue;
        }
        // Build all canonical paths for this function.  Free functions
        // must use the same alias spelling set as the graph-alias loop
        // above so a module-qualified call site finds both the graph
        // AND the hints (elidable/loopinvariant/cannot_collect/oopspec).
        let paths = if let Some(ref owner) = func.self_ty_root {
            // impl method: ["owner", "method"]
            vec![crate::parse::CallPath::from_segments([
                owner.as_str(),
                func.name.as_str(),
            ])]
        } else {
            // Same alias-spelling set as the graph-registration loop
            // above so every spelling the call-site might canonicalise
            // to also finds the `#[oopspec]` / `#[loop_invariant]` /
            // `#[cannot_collect]` / elidable hint.
            free_function_alias_paths(&func.name, &func.module_path)
        };
        for hint in &func.hints {
            for p in &paths {
                // rlib/jit.py:250 — `@oopspec(spec)` registers func.oopspec = spec.
                if let Some(spec) = hint.strip_prefix("oopspec:") {
                    call_control.mark_oopspec(p.clone(), spec.to_string());
                    continue;
                }
                // `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]`
                // — companion hint emitted by `front::llbc_hints::harvest_hints_from_llbcs`
                // when `#[oopspec(...)]` is paired with a function signature.
                // Threads the declaration-order parameter names into
                // `CallControl::oopspec_argnames` so `parse_oopspec`
                // (`support.py:701-715` port) can resolve identifier
                // slots in the spec's `(...)` pattern.
                if let Some(names) = hint.strip_prefix("oopspec_argnames:") {
                    let argnames: Vec<String> = names
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if !argnames.is_empty() {
                        call_control.mark_oopspec_argnames(p.clone(), argnames);
                    }
                    continue;
                }
                match hint.as_str() {
                    "elidable" => call_control.mark_elidable(p.clone()),
                    "elidable_cannot_raise" => call_control.mark_cannot_raise_assertion(p.clone()),
                    "elidable_or_memerror" => call_control.mark_memerror_only_assertion(p.clone()),
                    "loopinvariant" => call_control.mark_loopinvariant(p.clone()),
                    "close_stack" => call_control.mark_close_stack(p.clone()),
                    "cannot_collect" => call_control.mark_cannot_collect(p.clone()),
                    // rlib/jit.py:260 — @not_in_trace sets func.oopspec = "jit.not_in_trace()"
                    "not_in_trace" => {
                        call_control.mark_oopspec(p.clone(), "jit.not_in_trace".to_string());
                    }
                    // RPython: random_effects_on_gcobjs is on external funcobj only.
                    // Only register for paths WITHOUT a graph (external functions).
                    "gc_effects" => {
                        if !call_control.function_graphs().contains_key(p) {
                            call_control.mark_external_gc_effects(p.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // RPython: setup_jitdriver(jitdriver_sd) — register every explicit
    // portal and its green/red layout. Portal binding and graph discovery
    // are independent of any interpreter dispatch representation.
    register_configured_jitdrivers(&mut call_control, &config.pipeline.jit_drivers);
    // warmspot.py:515-545 WarmRunnerDesc.make_virtualizable_infos —
    // assigns each registered driver's virtualizable metadata only after the
    // complete driver set exists.
    call_control
        .make_virtualizable_infos(|jd_idx, vtypeptr_token| vinfo_factory(jd_idx, vtypeptr_token));
    // Register oopspecs for jit.* builtin functions.
    // rlib/jit.py: these functions carry @oopspec("jit.*") decorators;
    // the codewriter converts calls to them into dedicated opcodes.
    for (func_name, spec) in &[
        // rlib/jit.py:269-292 — @oopspec("jit.*") decorated functions
        ("isconstant", "jit.isconstant"),
        ("isvirtual", "jit.isvirtual"),
        ("current_trace_length", "jit.current_trace_length"),
        ("jit_debug", "jit.debug"),
        ("assert_green", "jit.assert_green"),
        // NOTE: conditional_call!/conditional_call_elidable!/record_known_result!
        // are handled by jitcode_lower (proc-macro level), not here.
        // They are macro_rules! that the codewriter AST parser does not expand.
    ] {
        // Register under common call-site path patterns.
        for path in [
            parse::CallPath::from_segments(["jit", func_name]),
            parse::CallPath::from_segments(["crate", "jit", func_name]),
            parse::CallPath::from_segments(["majit_metainterp", "jit", func_name]),
        ] {
            call_control.mark_oopspec(path, spec.to_string());
        }
    }
    let mut policy = policy::DefaultJitPolicy::new();
    call_control.find_all_graphs(&mut policy);

    // The canonical jitcode emitter below is the production analysis
    // path; `ProgramPipelineResult.functions` / `total_*` remain diagnostic fields.
    let mut pipeline = pipeline::ProgramPipelineResult {
        functions: Vec::new(),
        jit_drivers: Vec::new(),
        jitcodes: Vec::new(),
        jitcodes_by_path: indexmap::IndexMap::new(),
        insns: indexmap::IndexMap::new(),
        descrs: Vec::new(),
        all_liveness: Vec::new(),
        total_blocks: 0,
        total_ops: 0,
        total_vable_rewrites: 0,
    };

    mark_phase!("call_control + canonical_trait_impls + register graphs");
    let (jitcodes, insns, descrs, all_liveness) =
        make_jitcodes(&config.pipeline, &mut call_control);
    mark_phase!("make_jitcodes");
    pipeline.jit_drivers = call_control
        .jitdrivers_sd()
        .iter()
        .map(|driver| pipeline::CompiledJitDriver {
            portal: driver.portal_graph.clone(),
            main_jitcode_index: driver
                .mainjitcode
                .as_ref()
                .expect("configured JIT driver must have a main JitCode")
                .index(),
        })
        .collect();
    pipeline.jitcodes = jitcodes;
    // Mirror of `CallControl::jitcodes` (RPython `call.py:87 self.jitcodes`)
    // captured before `call_control` is dropped. Needed because consumers
    // that look up a JitCode by graph identity cannot reconstruct the key
    // from the alloc-ordered `pipeline.jitcodes` vector alone.
    pipeline.jitcodes_by_path = call_control.jitcodes().clone();
    pipeline.insns = insns;
    pipeline.descrs = descrs;
    // Snapshotted in `make_jitcodes` from the assembler's shared `all_liveness`
    // table (the target of `pyjitpl.py:2264 self.liveness_info =
    // "".join(asm.all_liveness)`) so the runtime can resolve the `BC_LIVE`
    // offsets baked into `JitCode.code`.
    pipeline.all_liveness = all_liveness;

    pipeline
}

/// RPython `CodeWriter.setup_jitdriver` validation and registration.
fn register_configured_jitdrivers(
    call_control: &mut call::CallControl,
    specs: &[pipeline::JitDriverSpec],
) {
    assert!(
        !specs.is_empty(),
        "full JitCode analysis requires at least one explicit JIT driver"
    );
    for (index, spec) in specs.iter().enumerate() {
        assert!(
            !spec.portal.segments.is_empty(),
            "JIT driver {index} has an empty portal CallPath"
        );
        assert!(
            spec.red_types.is_empty() || spec.red_types.len() == spec.reds.len(),
            "JIT driver `{}` has {} red types for {} reds",
            spec.portal.canonical_key(),
            spec.red_types.len(),
            spec.reds.len(),
        );
        let portal_graph = call_control.function_graphs().get(&spec.portal);
        assert!(
            portal_graph.is_some(),
            "configured JIT driver portal `{}` does not resolve to an exact graph; \
             configure its qualified CallPath rather than a leaf-name alias",
            spec.portal.canonical_key(),
        );
        let portal_graph = portal_graph.unwrap();
        assert!(
            !specs[..index].iter().any(|previous| {
                call_control
                    .function_graphs()
                    .get(&previous.portal)
                    .is_some_and(|previous_graph| std::ptr::eq(previous_graph, portal_graph))
            }),
            "duplicate JIT driver portal graph for `{}`; aliases of one graph \
             must share one JitDriverStaticData",
            spec.portal.canonical_key(),
        );
        call_control.setup_jitdriver(
            spec.portal.clone(),
            spec.greens.clone(),
            spec.reds.clone(),
            spec.autoreds,
            spec.virtualizables.clone(),
            spec.red_types.clone(),
        );
    }
}

/// Produce JitCodes for the graph closure reachable from configured portals.
///
/// RPython parity (`rpython/jit/codewriter/codewriter.py:74-89` `make_jitcodes`):
///
/// ```python
/// def make_jitcodes(self, verbose=False):
///     self.callcontrol.grab_initial_jitcodes()
///     all_jitcodes = []
///     for graph, jitcode in self.callcontrol.enum_pending_graphs():
///         self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
///         all_jitcodes.append(jitcode)
///     self.assembler.finished(self.callcontrol.callinfocollection)
///     return all_jitcodes
/// ```
fn make_jitcodes(
    pipeline_config: &pipeline::PipelineConfig,
    call_control: &mut call::CallControl,
) -> (
    Vec<std::sync::Arc<jitcode::JitCode>>,
    indexmap::IndexMap<String, u8>,
    Vec<jitcode::BhDescr>,
    Vec<u8>,
) {
    // RPython codewriter.py:74-89: make_jitcodes().
    //
    // `Arc<JitCode>` shells live in `CallControl::jitcodes`; the drain loop
    // commits each shell's body via `JitCode::set_body`. After all phases,
    // `collect_jitcodes_in_alloc_order` materialises the `all_jitcodes[]`
    // vector with `all_jitcodes[i].index == i` (RPython codewriter.py:80
    // invariant).
    let mut codewriter = codewriter::CodeWriter::new();

    // `warmspot.py:262-264` `vrefinfo = VirtualRefInfo(self);
    //  self.codewriter.setup_vrefinfo(vrefinfo)` — installs the
    // virtualref descr-index carrier on the codewriter's callcontrol
    // BEFORE `make_jitcodes` runs (line 281 in warmspot).  Pyre uses
    // a const-backed `DefaultVirtualRefInfoHandle` so majit-translate
    // can perform the install without depending on majit-metainterp
    // (the concrete `VirtualRefInfo` lives there).  The runtime path
    // through `MetaInterpStaticData::finish_setup` reads the handle
    // back at `pyjitpl.py:2267 self.virtualref_info = codewriter.
    // callcontrol.virtualref_info` parity, so this site is the
    // codewrite-time anchor for that read.
    codewriter.setup_vrefinfo(
        call_control,
        std::sync::Arc::new(call::DefaultVirtualRefInfoHandle),
    );

    // RPython grab_initial_jitcodes + drain portals and their callees.
    // RPython call.py:145-148.
    call_control.grab_initial_jitcodes();
    // Two-phase rtyper prepass (production default; `PYRE_TWO_PHASE_RTYPE=0`
    // opts out): annotate-all → rtype-all over the portal closure before the
    // drain publishes any covered graph, so a shared callee (e.g. `type_error`)
    // is unioned across all its callers before being rtyped.
    codewriter.run_two_phase_prepass_if_enabled(call_control);
    codewriter.drain_pending_graphs(call_control, &pipeline_config.transform);

    // RPython codewriter.py:85: self.assembler.finished(callinfocollection).
    codewriter
        .assembler
        .finished(&call_control.callinfocollection);

    // Materialise `all_jitcodes[]` from the completed jitcodes. Each
    // jitcode receives its dense index when appended, matching RPython
    // `make_jitcodes()`.
    let jitcodes = call_control.collect_jitcodes_in_alloc_order();

    // RPython codewriter.py + assembler.py: `Assembler.insns` grows as
    // `write_insn` encounters new keys.  We snapshot the final table
    // here so the runtime can map `JitCode.code[i]` bytes back to
    // opnames — the key consumed by `BlackholeInterpBuilder::setup_insns`.
    let insns = codewriter.assembler.insns().clone();

    // RPython blackhole.py:59 `self.setup_descrs(asm.descrs)` — the
    // shared descr table every 'd'/'j' argcode indexes into at runtime.
    // Snapshotted here so the build artifact carries it alongside
    // `insns`, mirroring RPython's single-store model.
    let descrs: Vec<jitcode::BhDescr> = codewriter.assembler.snapshot_descrs();

    // RPython `pyjitpl.py:2264 self.liveness_info = "".join(asm.all_liveness)`.
    // Snapshot the assembler's shared `all_liveness` byte stream so the runtime
    // can resolve the `BC_LIVE` offsets baked into `JitCode.code`.
    let all_liveness = codewriter.assembler.all_liveness().to_vec();

    (jitcodes, insns, descrs, all_liveness)
}

/// Generate tracing code directly from the canonical pipeline result.
pub fn generate_trace_code_from_pipeline(result: &pipeline::ProgramPipelineResult) -> String {
    codegen::generate_from_pipeline(result)
}

/// Like [`generate_trace_code_from_pipeline`] but takes a
/// [`codegen::CodegenFlavor`]. The flavor is currently a no-op (the
/// pyre-specific helpers it used to gate now live in
/// `pyre-jit-trace/src/trace_helpers.rs`); retained for external callers
/// such as `aheui-jit`.
pub fn generate_trace_code_from_pipeline_with_flavor(
    result: &pipeline::ProgramPipelineResult,
    flavor: codegen::CodegenFlavor,
) -> String {
    codegen::generate_from_pipeline_with_flavor(result, flavor)
}

pub use codegen::CodegenFlavor;

/// Produce a recognition report for canonical driver and JitCode output.
pub fn recognition_report(result: &pipeline::ProgramPipelineResult) -> codegen::RecognitionReport {
    codegen::recognition_report(result)
}
pub use codegen::{JitCodeRecognition, RecognitionReport};

/// Generate code from graph pipeline results.

/// `rlib` — Rust port of `rpython/rlib/` helpers pulled in on demand.
/// Currently only the pieces required by the annotator port are
/// present (rarithmetic subset for `compute_restype`).
pub mod rlib;

#[cfg(test)]
mod portal_driver_tests {
    use super::*;

    fn driver(portal: CallPath) -> pipeline::JitDriverSpec {
        pipeline::JitDriverSpec {
            portal,
            greens: Vec::new(),
            reds: Vec::new(),
            autoreds: false,
            virtualizables: Vec::new(),
            red_types: Vec::new(),
        }
    }

    fn return_graph(name: &str) -> FunctionGraph {
        let mut graph = FunctionGraph::new(name);
        graph.set_return(graph.startblock, None);
        graph
    }

    #[test]
    #[should_panic(expected = "duplicate JIT driver portal graph")]
    fn rejects_two_aliases_of_one_portal_graph() {
        let mut call_control = call::CallControl::new();
        let first = CallPath::from_segments(["first_alias"]);
        let second = CallPath::from_segments(["second_alias"]);
        // GraphStore intentionally interns aliases of one source graph. The
        // driver layer must preserve that identity instead of allocating two
        // JitDriverStaticData records for it.
        call_control.register_function_graph(first.clone(), return_graph("portal"));
        call_control.register_function_graph(second.clone(), return_graph("portal"));

        register_configured_jitdrivers(&mut call_control, &[driver(first), driver(second)]);
    }

    #[test]
    fn make_jitcodes_compiles_a_registered_portal_graph() {
        let mut call_control = call::CallControl::new();
        let portal = CallPath::from_segments(["fixture", "portal"]);
        call_control.register_function_graph(portal.clone(), return_graph("portal"));
        let config = pipeline::PipelineConfig {
            transform: GraphTransformConfig::default(),
            jit_drivers: vec![driver(portal.clone())],
            register_trait_families: Vec::new(),
        };
        register_configured_jitdrivers(&mut call_control, &config.jit_drivers);
        let mut policy = policy::DefaultJitPolicy::new();
        call_control.find_all_graphs(&mut policy);

        let (jitcodes, _, _, _) = make_jitcodes(&config, &mut call_control);
        assert_eq!(jitcodes.len(), 1);
        assert_eq!(jitcodes[0].index(), 0);
        assert!(call_control.jitcodes().contains_key(&portal));
    }

    #[test]
    fn make_jitcodes_keeps_explicit_driver_reds_from_marker_callsite() {
        use crate::codewriter::flatten::FlatOp;
        use crate::codewriter::type_state::ConcreteType;

        let mut call_control = call::CallControl::new();
        let portal = CallPath::from_segments(["fixture", "eval_loop_jit"]);
        let mut graph = FunctionGraph::new("eval_loop_jit");
        let receiver = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let next_instr = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let is_being_profiled = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let pycode = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let frame = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let ec = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        graph.blocks[graph.startblock.0].inputargs = vec![
            receiver.clone(),
            next_instr.clone(),
            is_being_profiled.clone(),
            pycode.clone(),
            frame.clone(),
            ec.clone(),
        ];
        graph.blocks[graph.startblock.0]
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::method("jit_merge_point", Some("PyPyJitDriver".into())),
                    args: vec![
                        receiver,
                        next_instr,
                        is_being_profiled,
                        pycode,
                        frame.clone(),
                        ec.clone(),
                    ],
                    result_ty: ValueType::Void,
                },
            });
        graph.set_return(graph.startblock, None);
        call_control.register_function_graph(portal.clone(), graph);

        let config = pipeline::PipelineConfig {
            transform: GraphTransformConfig::default(),
            jit_drivers: vec![pipeline::JitDriverSpec {
                portal: portal.clone(),
                greens: vec![
                    "next_instr".into(),
                    "is_being_profiled".into(),
                    "pycode".into(),
                ],
                reds: vec!["frame".into(), "ec".into()],
                autoreds: false,
                virtualizables: vec!["frame".into()],
                red_types: vec!["PyFrame".into(), "ExecutionContext".into()],
            }],
            register_trait_families: Vec::new(),
        };
        register_configured_jitdrivers(&mut call_control, &config.jit_drivers);
        let mut policy = policy::DefaultJitPolicy::new();
        call_control.find_all_graphs(&mut policy);

        let (jitcodes, _, _, _) = make_jitcodes(&config, &mut call_control);
        let merge = jitcodes[0]
            .body()
            ._ssarepr
            .as_ref()
            .expect("assembled portal keeps its SSA representation")
            .insns
            .iter()
            .find_map(|flat_op| match flat_op {
                FlatOp::Op(SpaceOperation {
                    kind: OpKind::JitMergePoint { reds_r, .. },
                    ..
                }) => Some(reds_r),
                _ => None,
            })
            .expect("portal emits a jit_merge_point");
        assert_eq!(merge, &vec![frame, ec]);
        let jd0 = &call_control.jitdrivers_sd()[0];
        assert!(!jd0.autoreds);
        assert_eq!(jd0.numreds, Some(2));
    }
}

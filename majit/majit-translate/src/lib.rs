//! majit-translate: RPython translation pipeline.
//!
//! Upstream counterparts:
//! - `jit_codewriter/` ← `rpython/jit/codewriter/`
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
pub mod config;
pub mod flowspace;
pub mod jit_codewriter;
pub mod tool;
pub use jit_codewriter::{
    assembler, call, codewriter, flatten, format, insns, jitcode, jtransform, liveness, policy,
    regalloc, support,
};

mod codegen;
pub mod front;
// TODO(pyre): pyre-interpreter handler JitCode registry
// (Rust source → FunctionGraph bridge with no RPython counterpart;
// upstream assumes rtyper-produced `translator.graphs` is already in
// memory at codewriter entry).
pub mod generated;
pub mod handler_spec;
pub mod hints;
pub mod inline;
pub mod layout;
pub mod model;
pub mod opcode_dispatch;
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
pub use flatten::{FlatOp, GraphFlattener, Label, RegKind, SSARepr, flatten, flatten_graph};
pub use front::{
    AstGraphOptions, SemanticFunction, SemanticProgram, build_semantic_program,
    build_semantic_program_from_parsed_files,
};
pub use jit_codewriter::type_state::ConcreteType;
pub use jtransform::{
    CallEffectKind, CallEffectOverride, GraphTransformConfig, GraphTransformResult,
    VirtualizableFieldDescriptor, rewrite_graph,
};
pub use layout::{HeuristicLayoutProvider, LayoutProvider};
pub use model::{Block, BlockId, CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueType};
pub use opcode_dispatch::PipelineOpcodeArm;
pub use parse::{
    CallPath, ExtractedHandlerCall, ExtractedOpcodeArm, OpcodeDispatchSelector, ParsedInterpreter,
    ReceiverTraitBindings, extract_inherent_impl_methods, extract_opcode_dispatch_arms,
    extract_opcode_dispatch_receiver_traits, extract_trait_impls, find_opcode_dispatch_match,
    parse_source,
};
pub use pipeline::{PipelineConfig, PipelineResult, PortalSpec, ProgramPipelineResult};

use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::translator::rtyper::legacy_annotator as annotate;
#[cfg(test)]
use crate::translator::rtyper::legacy_resolve as rtype;

/// Configuration for the canonical graph/pipeline analyzer.
///
/// Consumers supply graph-rewrite metadata such as virtualizable
/// field/array mappings before the codewriter-style passes run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

/// Canonical single-file analysis entry point.
pub fn analyze_pipeline(source: &str) -> pipeline::ProgramPipelineResult {
    analyze_pipeline_with_config(source, &AnalyzeConfig::default())
}

/// Issue #97 Step 4.4 — feature-gated SemanticProgram builder cutover.
///
/// When the `mir-frontend` feature is enabled, route the build
/// through [`front::mir::build_semantic_program_from_llbcs`] using
/// LLBC artefacts discovered via, in priority order:
///
/// 1. `PYRE_MIR_FRONTEND_LLBC` env-var (colon-separated paths).
///    Explicit override for CI / test fixtures targeting a specific
///    LLBC set.
/// 2. Auto-discovery at `<workspace>/build/llbc/<expected>.ullbc`
///    (Step 6.B 2026-05-25).  `scripts/extract-llbc.sh` writes here.
///    If every expected file exists, MIR cutover engages
///    automatically.
///
/// Falls back to the syn-AST builder when neither source resolves
/// (Charon not installed, contributor on stable Rust without
/// LLBC extraction).
fn build_semantic_program_via_active_frontend(
    parsed_files: &[parse::ParsedInterpreter],
) -> front::SemanticProgram {
    #[cfg(feature = "mir-frontend")]
    {
        // Step 4.6 multi-LLBC cutover: accept colon-separated paths
        // (matching unix PATH convention) so production can pass
        // both `pyre-object.ullbc` and `pyre-interpreter.ullbc`
        // (and any future per-crate ullbc) in one env-var.  The
        // single-path form continues to work — it parses as a
        // length-one slice.
        //
        // Step 6.B: if the env-var is unset, auto-discover the
        // canonical workspace LLBC artefacts before falling through
        // to AST.
        let resolved_paths: Option<Vec<String>> = std::env::var("PYRE_MIR_FRONTEND_LLBC")
            .ok()
            .map(|v| {
                v.split(':')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect()
            })
            .or_else(|| auto_discover_workspace_llbc_paths(parsed_files));
        if let Some(paths) = resolved_paths {
            let llbcs: Vec<majit_charon_reader::Llbc> = paths
                .iter()
                .map(|p| {
                    majit_charon_reader::Llbc::load(p)
                        .unwrap_or_else(|e| panic!("Step 4.4 cutover: load {p}: {e}"))
                })
                .collect();
            let mut program = front::mir::build_semantic_program_from_llbcs(&llbcs)
                .unwrap_or_else(|e| panic!("Step 4.4 cutover: lower llbcs {paths:?}: {e}"));
            // Step 4.5.b: hybrid hint pass. Charon does not yet
            // serialize pyre's proc-macro attributes
            // (`#[majit_macros::elidable*]` / `oopspec` / `portal`),
            // but those attributes survive in the parsed_files syn::File
            // trees the AST front-end already had to parse for
            // compatibility. Walk the same source, build a
            // {fn_path → hints} table, and merge into the
            // MIR-driven SemanticProgram by name.
            merge_hints_from_parsed_files(&mut program, parsed_files);
            // Step 4.5.c: hybrid metadata pass. The MIR builder
            // cannot populate `fn_return_types` until
            // `Step 4.3.c.ext` (Charon dedup-table widening) resolves
            // `TyRef::Deduplicated{id}` to its primitive name.
            // Meanwhile `extract_inherent_impl_methods` / the AST
            // re-lowering path in `analyze_pipeline_from_parsed`
            // consults `fn_return_types` by bare name to type-check
            // `!is_str(x)`-style boolean expressions and
            // typed-method receivers.  Mirror Step 4.5.b: parse the
            // same parsed_files we already hold, collect
            // `(name → return_type_string)` from the AST pre-walk,
            // and merge into the MIR-built `SemanticProgram` so the
            // bare-leaf lookups land on the syn-source-derived
            // return-type strings.  This is the same hybrid surface
            // model: AST stays the source of truth for metadata
            // Charon does not yet expose; MIR replaces only the
            // function bodies the codewriter consumes.
            merge_fn_return_types_from_parsed_files(&mut program, parsed_files);
            return program;
        }
    }
    let _ = parsed_files; // silence unused warning when only AST is reachable
    front::build_semantic_program_from_parsed_files(parsed_files)
        .expect("pyre-interpreter source must lower without FlowingError")
}

/// Step 6.E audit helper — count how many methods produced by the
/// AST-side `parse::extract_trait_impls` /
/// `parse::extract_inherent_impl_methods` are reachable through
/// `program.functions`'s `self_ty_root` + `trait_root` indexing.
///
/// Prints a summary of the form `[ast-extract-audit] trait_methods=X
/// covered=Y missing=N inherent=A covered=B missing=N` plus the
/// first few missing entries.  Side-effect-only; the registration
/// loop still consumes the existing `canonical_*` vectors.
///
/// Goal: prove `program.functions` is a superset of the extractors'
/// output before the registration loop is re-routed to walk
/// `program.functions` directly and the extractors retire.
fn audit_ast_extract_coverage(
    program: &front::SemanticProgram,
    canonical_trait_impls: &[TraitImplInfo],
    canonical_inherent_methods: &[parse::InherentMethodInfo],
) {
    use std::collections::HashSet;
    // Index program.functions two ways:
    //   - `impl_methods`: keyed by (self_ty_root, name) — every
    //     inherent + trait-impl method's owner leaf.
    //   - `trait_defaults`: keyed by (trait_root, name) — every
    //     trait-default body (penultimate Ident matches a known
    //     trait leaf).
    let mut impl_methods: HashSet<(String, String)> = HashSet::new();
    let mut trait_defaults: HashSet<(String, String)> = HashSet::new();
    for f in &program.functions {
        if let Some(owner) = &f.self_ty_root {
            impl_methods.insert((owner.clone(), f.name.clone()));
        }
        if let Some(tr) = &f.trait_root {
            if f.self_ty_root.is_none() {
                trait_defaults.insert((tr.clone(), f.name.clone()));
            }
        }
    }
    let mut trait_total = 0usize;
    let mut trait_missing: Vec<String> = Vec::new();
    for ti in canonical_trait_impls {
        // Concrete trait-impl methods (impl Trait for Type { … }) have
        // `self_ty_root = Some("Type")` and look up via `impl_methods`.
        // Trait-default bodies (collect_trait_impls_from_items's
        // `Item::Trait` arm) have `self_ty_root = None` and
        // `for_type = "<default methods of <Trait>>"`; map them to
        // the `trait_defaults` index keyed by trait_name.
        let is_default = ti.for_type.starts_with("<default methods of ");
        for method in &ti.methods {
            trait_total += 1;
            let hit = if is_default {
                trait_defaults.contains(&(ti.trait_name.clone(), method.name.clone()))
            } else {
                let impl_type = ti.self_ty_root.as_deref().unwrap_or(&ti.for_type);
                impl_methods.contains(&(impl_type.to_string(), method.name.clone()))
            };
            if !hit {
                let key_kind = if is_default { "default" } else { "impl" };
                trait_missing.push(format!(
                    "{}::{} ({key_kind})",
                    ti.self_ty_root.as_deref().unwrap_or(&ti.for_type),
                    method.name
                ));
            }
        }
    }
    let mut inherent_total = 0usize;
    let mut inherent_missing: Vec<String> = Vec::new();
    for mi in canonical_inherent_methods {
        inherent_total += 1;
        let impl_type = mi.self_ty_root.as_deref().unwrap_or(&mi.for_type);
        if !impl_methods.contains(&(impl_type.to_string(), mi.name.clone())) {
            inherent_missing.push(format!("{}::{}", impl_type, mi.name));
        }
    }
    eprintln!(
        "[ast-extract-audit] trait_methods={trait_total} \
         covered={} missing={} inherent={} covered={} missing={}",
        trait_total - trait_missing.len(),
        trait_missing.len(),
        inherent_total,
        inherent_total - inherent_missing.len(),
        inherent_missing.len(),
    );
    for s in trait_missing.iter().take(10) {
        eprintln!("  trait_missing: {s}");
    }
    for s in inherent_missing.iter().take(10) {
        eprintln!("  inherent_missing: {s}");
    }
}

/// Step 6.B: locate the workspace's `build/llbc/` directory and
/// return paths to the canonical pyre LLBC artefacts when every
/// expected file is present *and* the caller looks like a
/// production build (not a test fixture).
///
/// Returns `None` when:
///   - no parsed_file carries a `module_path` (test fixtures use
///     `parse::parse_source` which leaves `module_path` empty;
///     production uses `parse::parse_source_with_module`),
///   - the caller passed fewer than `PROD_PARSED_FILES_FLOOR`
///     parsed_files (single-source diagnostic),
///   - any expected artefact is missing (contributor without
///     Charon installed), or
///   - the workspace anchoring fails.
///
/// The two gates together match the production fingerprint:
/// `pyre-jit-trace/build.rs:157` calls
/// `analyze_multiple_pipeline_with_modules` with ≈100 files and a
/// per-file `module_path`.  Tests via
/// `analyze_multiple_pipeline_with_config` parse without module
/// paths and stay below the floor, so auto-discovery does not
/// silently swap their front-end.
///
/// The workspace root is anchored at compile time via
/// `env!("CARGO_MANIFEST_DIR")` — `<workspace>/majit/majit-translate`
/// resolves up to `<workspace>` via two `..` segments.  The
/// `scripts/extract-llbc.sh` script writes to the same
/// `<workspace>/build/llbc/` directory by convention, so the two
/// halves stay in sync.
#[cfg(feature = "mir-frontend")]
fn auto_discover_workspace_llbc_paths(
    parsed_files: &[parse::ParsedInterpreter],
) -> Option<Vec<String>> {
    const PROD_PARSED_FILES_FLOOR: usize = 50;
    if parsed_files.len() < PROD_PARSED_FILES_FLOOR {
        return None;
    }
    if !parsed_files.iter().any(|p| !p.module_path.is_empty()) {
        return None;
    }
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let llbc_dir = workspace_root.join("build").join("llbc");
    // Canonical production set.  `pyre-module.ullbc` is intentionally
    // omitted — it is empty in current builds and adds nothing.
    // `corpus.ullbc` is the charon-spike fixture, not production.
    const REQUIRED: &[&str] = &["pyre-object.ullbc", "pyre-interpreter.ullbc"];
    let mut paths = Vec::with_capacity(REQUIRED.len());
    for name in REQUIRED {
        let p = llbc_dir.join(name);
        if !p.exists() {
            return None;
        }
        paths.push(p.to_string_lossy().into_owned());
    }
    Some(paths)
}

/// Step 4.5.c helper — populate the MIR program's
/// `fn_return_types` / `struct_fields` / `known_struct_names` /
/// `known_trait_names` / `immutable_fields` from the syn-AST
/// pre-walk of the same parsed_files.  Closes the
/// `extract_inherent_impl_methods` / annotator lookup gap that
/// blocks every consumer with non-empty test fixtures (multi-crate
/// loads, inline-source unit tests) under the cutover.  Entries
/// that the MIR builder already populated win; AST adds only
/// what MIR left empty.
#[cfg(feature = "mir-frontend")]
fn merge_fn_return_types_from_parsed_files(
    program: &mut front::SemanticProgram,
    parsed_files: &[parse::ParsedInterpreter],
) {
    let ast_metadata = front::ast::collect_program_metadata_pub(parsed_files);
    for (name, ty) in ast_metadata.fn_return_types {
        program.fn_return_types.entry(name).or_insert(ty);
    }
    for name in ast_metadata.known_struct_names {
        program.known_struct_names.insert(name);
    }
    for name in ast_metadata.known_trait_names {
        program.known_trait_names.insert(name);
    }
    for (owner, fields) in ast_metadata.struct_fields.fields {
        // AST field-type strings (`*mut PyFrame`, `Vec<u8>`, `&str`)
        // are the format downstream consumers expect; MIR's
        // `TyRef::label()` (`ty#170`, `pyre_interpreter::PyFrame`)
        // is a placeholder until Step 4.3.c.ext dedup-table widening
        // resolves primitives.  Override MIR's entries when the AST
        // produces a real type string for the same owner so
        // `receiver_type_root` / `lookup_method_return_type` see the
        // AST-orthodox shape.
        program.struct_fields.fields.insert(owner, fields);
    }
}

/// Step 4.5.b helper — merge syn-AST proc-macro hints into a
/// MIR-driven SemanticProgram.
///
/// Walks `parsed_files` for every top-level / impl-method `ItemFn`
/// and records its `#[majit_macros::*]` attribute hints under both
/// the bare leaf name and the module-qualified path. Then matches
/// each `SemanticFunction` in `program` by the trailing segment of
/// its `name` (Charon's `name_path` form, e.g.
/// `pyre_interpreter::frame::Frame::pop`) and copies the hints in.
///
/// Matches by leaf because Charon paths use the crate name as the
/// first segment (e.g. `pyre_interpreter::…`) while the AST parser
/// is module-rooted; a full-path match would always miss.
#[cfg(feature = "mir-frontend")]
fn merge_hints_from_parsed_files(
    program: &mut front::SemanticProgram,
    parsed_files: &[parse::ParsedInterpreter],
) {
    use std::collections::HashMap;
    let mut hints_by_leaf: HashMap<String, Vec<String>> = HashMap::new();
    for parsed in parsed_files {
        collect_hints_walk(&parsed.file.items, &mut hints_by_leaf);
    }
    for f in &mut program.functions {
        let leaf = f.name.rsplit("::").next().unwrap_or(&f.name);
        if let Some(h) = hints_by_leaf.get(leaf) {
            f.hints.clone_from(h);
        }
    }
}

#[cfg(feature = "mir-frontend")]
fn collect_hints_walk(
    items: &[syn::Item],
    out: &mut std::collections::HashMap<String, Vec<String>>,
) {
    for item in items {
        match item {
            syn::Item::Fn(item_fn) => {
                let hints = front::ast::collect_jit_hints_with_sig(&item_fn.attrs, &item_fn.sig);
                if !hints.is_empty() {
                    out.insert(item_fn.sig.ident.to_string(), hints);
                }
            }
            syn::Item::Impl(item_impl) => {
                for sub in &item_impl.items {
                    if let syn::ImplItem::Fn(impl_fn) = sub {
                        let hints = front::ast::collect_jit_hints_with_sig(
                            &impl_fn.attrs,
                            &impl_fn.sig,
                        );
                        if !hints.is_empty() {
                            out.insert(impl_fn.sig.ident.to_string(), hints);
                        }
                    }
                }
            }
            syn::Item::Mod(item_mod) => {
                if let Some((_, sub_items)) = &item_mod.content {
                    collect_hints_walk(sub_items, out);
                }
            }
            _ => {}
        }
    }
}

/// Configurable canonical single-file analysis entry point.
pub fn analyze_pipeline_with_config(
    source: &str,
    config: &AnalyzeConfig,
) -> pipeline::ProgramPipelineResult {
    analyze_multiple_pipeline_with_config(&[source], config)
}

/// Canonical multi-file analysis entry point.
///
/// This returns only the graph/pipeline result and is the preferred API for
/// RPython-like translator consumers.
pub fn analyze_multiple_pipeline(sources: &[&str]) -> pipeline::ProgramPipelineResult {
    analyze_multiple_pipeline_with_config(sources, &AnalyzeConfig::default())
}

/// Configurable canonical multi-file analysis entry point.
///
/// This is the canonical graph/pipeline translator entry point.
/// Uses `HeuristicLayoutProvider` for struct layouts (type-string approximation).
pub fn analyze_multiple_pipeline_with_config(
    sources: &[&str],
    config: &AnalyzeConfig,
) -> pipeline::ProgramPipelineResult {
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();
    analyze_pipeline_from_parsed(
        &parsed_files,
        config,
        None,
        &|_, _| None,
        &[],
        &[],
        HostStaticAddrs::default(),
    )
}

/// Multi-file analysis with explicit layout provider.
///
/// RPython equivalent: the translator resolves struct layouts via
/// `symbolic.get_field_token()` / `symbolic.get_size()`. The layout
/// provider supplies these values. Pass `None` to use the heuristic default.
pub fn analyze_multiple_pipeline_with_layout(
    sources: &[&str],
    config: &AnalyzeConfig,
    layout_provider: &dyn layout::LayoutProvider,
) -> pipeline::ProgramPipelineResult {
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();
    analyze_pipeline_from_parsed(
        &parsed_files,
        config,
        Some(layout_provider),
        &|_, _| None,
        &[],
        &[],
        HostStaticAddrs::default(),
    )
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
/// `analyze_pipeline_from_parsed` strips the crate-name prefix and binds
/// both canonical aliases (`helpers::foo` and `crate::helpers::foo`) on
/// `CallControl` before `get_jitcode()` / `jtransform` query fnaddrs.
pub type FnAddrBindings<'a> = [(&'a str, i64)];

/// Structured binding table for impl-method helpers.  Each entry is
/// `(module_path_with_crate, impl_type_as_written, method_name, fnaddr)`.
/// The codewriter applies the parser's `qualify_type_name` rule
/// (front/ast.rs:106) — bare types get the module prefix (minus crate
/// name) prepended, qualified types are kept verbatim — before storing
/// the canonical `[impl_type_joined, method]` 2-segment CallPath
/// (lib.rs:406-433).
///
/// `#[jit_module]::__majit_helper_impl_trace_fnaddrs()` produces this
/// shape and `analyze_pipeline_from_parsed` feeds it through
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
}

/// Multi-file analysis with explicit layout provider AND a
/// `VirtualizableInfo` factory wired into
/// `CallControl::make_virtualizable_infos` (warmspot.py:516).  The
/// factory delegates the runtime constructor call from the codewriter
/// (which sits below metainterp in the crate graph) back to the host.
pub fn analyze_multiple_pipeline_with_vinfo_factory(
    sources: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
) -> pipeline::ProgramPipelineResult {
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();
    analyze_pipeline_from_parsed(
        &parsed_files,
        config,
        layout_provider,
        vinfo_factory,
        &[],
        &[],
        HostStaticAddrs::default(),
    )
}

/// Multi-file analysis with explicit layout provider, optional
/// `VirtualizableInfo` factory, and host-supplied compiled helper
/// addresses.
///
/// This is the line-by-line `getfunctionptr(graph)` adapter for source-only
/// codewriter consumers: pass the output of
/// `#[jit_module]::__majit_helper_trace_fnaddrs()` here so `JitCode.fnaddr`
/// and residual-call lowering use the real helper surface instead of the
/// symbolic fallback.
pub fn analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings(
    sources: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
) -> pipeline::ProgramPipelineResult {
    analyze_multiple_pipeline_with_vinfo_and_all_fnaddr_bindings(
        sources,
        config,
        layout_provider,
        vinfo_factory,
        fnaddr_bindings,
        &[],
    )
}

/// Multi-file analysis with explicit per-source module paths.
///
/// `sources` and `module_paths` are parallel slices of equal length:
/// `module_paths[i]` is the crate-stripped module path of `sources[i]`
/// (e.g. `"intobject"` for `pyre_object/src/intobject.rs`).  Each file
/// is parsed via [`parse::parse_source_with_module`], populating
/// `ParsedInterpreter.{module_path, use_imports}` so the metadata
/// collectors can record `struct_origins[bare_name] = module_path`
/// and `qualify_to_canonical_struct` resolves cross-module references
/// through the use-import table.
///
/// An empty `module_paths[i]` keeps the simple-name registration of
/// the bare `analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings`
/// entry — runtime convergence is then handled solely by the
/// `build_object_descr_group_with_def_path` dual-publish.
pub fn analyze_multiple_pipeline_with_modules(
    sources: &[&str],
    module_paths: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
    static_addrs: HostStaticAddrs<'_>,
) -> pipeline::ProgramPipelineResult {
    assert_eq!(
        sources.len(),
        module_paths.len(),
        "analyze_multiple_pipeline_with_modules: parallel slices must have equal length",
    );
    let parsed_files: Vec<_> = sources
        .iter()
        .zip(module_paths.iter())
        .map(|(s, mp)| parse::parse_source_with_module(s, mp))
        .collect();
    analyze_pipeline_from_parsed(
        &parsed_files,
        config,
        layout_provider,
        vinfo_factory,
        fnaddr_bindings,
        &[],
        static_addrs,
    )
}

/// Like `analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings` but
/// additionally accepts an `impl_fnaddr_bindings` table produced by the
/// macro's `__majit_helper_impl_trace_fnaddrs()` registry. Entries bind
/// impl-method helpers via `register_macro_impl_helper_trace_fnaddr`,
/// resolving the structural `[impl_type_joined, method]` CallPath that
/// the string-split helper entry point cannot express.
pub fn analyze_multiple_pipeline_with_vinfo_and_all_fnaddr_bindings(
    sources: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
    impl_fnaddr_bindings: &ImplFnAddrBindings<'_>,
) -> pipeline::ProgramPipelineResult {
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();
    analyze_pipeline_from_parsed(
        &parsed_files,
        config,
        layout_provider,
        vinfo_factory,
        fnaddr_bindings,
        impl_fnaddr_bindings,
        HostStaticAddrs::default(),
    )
}

/// Multi-file analysis with compiled helper fnaddr bindings but without a
/// virtualizable-info factory.
pub fn analyze_multiple_pipeline_with_fnaddr_bindings(
    sources: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    fnaddr_bindings: &FnAddrBindings<'_>,
) -> pipeline::ProgramPipelineResult {
    analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings(
        sources,
        config,
        layout_provider,
        &|_, _| None,
        fnaddr_bindings,
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
/// [`analyze_pipeline_from_parsed`] so call-site lookups that key on
/// these spellings (function_graphs,
/// elidable/loopinvariant/cannot_collect/oopspec targets) all see
/// the same FunctionPath set.  Without this, a module-qualified call
/// site finds the graph alias but misses the elidable/loopinvariant
/// hint registered only under the bare name.
///
/// `name` is the `SemanticFunction.name` (already module-prefixed
/// when the function lives inside a `mod foo { fn bar() }` block);
/// `source_module` is the file's crate-stripped path (populated by
/// `front::ast::build_semantic_program_with_options` from
/// `parsed.module_path`).
fn free_function_alias_paths(name: &str, source_module: &str) -> Vec<crate::parse::CallPath> {
    let segments: Vec<&str> = name.split("::").collect();
    let mut paths = Vec::new();
    paths.push(crate::parse::CallPath::from_segments(
        segments.iter().copied(),
    ));
    let mut crate_segs = vec!["crate"];
    crate_segs.extend(segments.iter().copied());
    paths.push(crate::parse::CallPath::from_segments(crate_segs));
    for crate_alias in &["pyre_interpreter", "pyre_object", "pyre_jit"] {
        let mut alias_segs = vec![*crate_alias];
        alias_segs.extend(segments.iter().copied());
        paths.push(crate::parse::CallPath::from_segments(alias_segs));
    }
    let name_segs_prefix: Vec<&str> = name.split("::").collect();
    let mod_segs_prefix: Vec<&str> = source_module.split("::").collect();
    // The starts_with check is meant to skip the module-qualified loop
    // when `name` already carries the module prefix (e.g. a nested
    // `mod foo { fn bar() }` whose `sf.name` is set to "foo::bar"
    // before the module stamp at `front/ast.rs:1669-1676`).  Without
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
        for crate_alias in &["pyre_interpreter", "pyre_object", "pyre_jit"] {
            let mut alias_segs = vec![*crate_alias];
            alias_segs.extend(module_qualified_segs.iter().copied());
            paths.push(crate::parse::CallPath::from_segments(
                alias_segs.iter().copied(),
            ));
        }
    }
    paths
}

fn analyze_pipeline_from_parsed(
    parsed_files: &[parse::ParsedInterpreter],
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
    // Install the cross-file type-alias floor before any walker pass
    // runs.  The floor is the union of every parsed file's top-level
    // `type T = U;` declarations; it stays visible across the per-file
    // walker calls that happen later in the pipeline (lazy
    // `Translation::from_rust_*` creation).  Mirrors PyPy
    // `Bookkeeper`'s whole-program import-resolution map — a struct
    // field declared as `field: PyObjectRef` in `pyframe.rs` resolves
    // through the alias declared in `pyobject.rs` regardless of
    // walker iteration order.
    let _walker_alias_floor =
        crate::flowspace::rust_source::register::WalkerAliasFloorGuard::install(
            parsed_files.iter().map(|p| &p.file),
        );
    // Use-import resolver: harvest `(bare_name → defining_module_path)`
    // from every `ParsedInterpreter.module_path` non-empty entry, then
    // publish into the `majit_ir::descr::STRUCT_ORIGIN_REGISTRY` global
    // so subsequent `canonical_struct_name` lookups at `path_hash`
    // sites resolve bare struct tokens to their qualified canonical
    // form (PyPy `bookkeeper.getdesc(TYPE)` analog).  Empty
    // `module_path` files skip registration; their bare-name hashes
    // still resolve via the runtime's simple-name dual-publish slot.
    let mut struct_origins: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for parsed in parsed_files {
        if !parsed.module_path.is_empty() {
            front::ast::collect_struct_origins(
                &parsed.file.items,
                &parsed.module_path,
                &mut struct_origins,
            );
        }
    }
    majit_ir::descr::register_struct_origins(struct_origins);
    // Codewriter-side static catalogue.  Collected here so it can be
    // installed on `CallControl.static_decls` further down for the
    // adapter consumer.  The front-end `Expr::Path` arm receives the
    // same data through `KnownStaticsCatalogue` constructed inside
    // `build_semantic_program_from_parsed_files_with_options`.
    let early_static_decls: Vec<(
        Vec<String>,
        crate::model::ValueType,
        Option<crate::flowspace::model::ConstValue>,
    )> = parsed_files
        .iter()
        .flat_map(|parsed| {
            crate::flowspace::rust_source::register::extract_static_decls(
                &parsed.file,
                &parsed.module_path,
            )
        })
        .collect();
    // `use <path>::*` glob roots are expanded into explicit
    // `use_imports` entries inside
    // `build_semantic_program_*_with_options` so the front-end
    // `Expr::Path` arm resolves glob-imported bare names through the
    // primary `use_imports` lookup without a separate fallback.
    // Diagnostic pre-pass — populate
    // `FORCE_ATTRIBUTES_INTO_CLASSES` (classdesc.py:957-961) from each
    // `parsed_files` entry's top-level structs so
    // `ClassDesc::_init_classdef` can pre-fill `ClassDef.attrs` *before*
    // the annotator's narrowing gate at
    // `flowspace_adapter.rs::derive_subject_inputcells` checks
    // `attrs_populated`.  Production never drives the walker (only
    // `extract_static_decls` and `extract_unsafe_fn_stubs` are called
    // from `register`), which left the dict empty for parsed-only
    // structs and forced every impl-method `self` to carry
    // `SomeInstance(classdef=None)`.  Empty `module_path` files (test
    // fixtures) skip; their structs are registered through the bare-
    // leaf walker path when the fixture explicitly calls
    // `register_rust_module_at_with_source`.
    for parsed in parsed_files {
        if !parsed.module_path.is_empty() {
            crate::flowspace::rust_source::register::pre_register_struct_fields_from_file(
                &parsed.file,
                "",
            );
        }
    }
    // RPython `translator/translator.py:55 buildflowgraph` — FlowingError
    // propagates out and translation halts.  Pyre's top-level analyzer
    // requires a complete program; a FlowingError here means a user-
    // facing source file contains a construct we cannot yet lower, and
    // the correct response is to abort loudly so the coverage audit
    // surfaces the unsupported expression rather than silently dropping
    // a graph.
    // Issue #97 Step 4.4: feature-gated cutover. When the
    // `mir-frontend` feature is enabled AND `PYRE_MIR_FRONTEND_LLBC`
    // is set, route the production SemanticProgram build through the
    // MIR-driven `front::mir::build_semantic_program_from_llbc` path
    // instead of the syn-AST builder. The env-var carries the path to
    // a Charon-extracted .ullbc snapshot (produced by
    // `scripts/extract-llbc.sh`). The flag without the env-var still
    // takes the AST path so partial enablement stays a no-op.
    mark_phase!("known_statics + struct_origins + struct_field_attrs populated");
    let program = build_semantic_program_via_active_frontend(parsed_files);
    mark_phase!("build_semantic_program_from_parsed_files");
    let mut canonical_trait_impls = Vec::new();
    let mut canonical_inherent_methods = Vec::new();
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

    for parsed in parsed_files {
        canonical_trait_impls.extend(
            parse::extract_trait_impls(
                parsed,
                &program.struct_fields,
                &program.fn_return_types,
                &program.known_struct_names,
            )
            .expect("trait impls must lower without FlowingError"),
        );
        canonical_inherent_methods.extend(
            parse::extract_inherent_impl_methods(
                parsed,
                &program.struct_fields,
                &program.fn_return_types,
                &program.known_struct_names,
            )
            .expect("inherent methods must lower without FlowingError"),
        );
    }
    // Step 6.E audit (env-gated): measure how much of the AST-side
    // `canonical_trait_impls` / `canonical_inherent_methods` surface
    // is already covered by `program.functions`'s impl-method
    // entries.  Goal — confirm we can route every registration the
    // AST extractors do today through `program.functions` and retire
    // the extractors.  Set `PYRE_AST_EXTRACT_COVERAGE_AUDIT=1` to
    // emit a per-method diff to stderr.
    if std::env::var("PYRE_AST_EXTRACT_COVERAGE_AUDIT").is_ok() {
        audit_ast_extract_coverage(
            &program,
            &canonical_trait_impls,
            &canonical_inherent_methods,
        );
    }
    // RPython: use the rtyped graphs (with concretetype info) for all analysis.
    // Use program.functions' graphs which were built with full struct_fields
    // context, NOT re-parsed graphs (which lose array_type_id etc.).
    // Build the `pub use <src>::*` re-export index:
    // `globbed_source_path -> [importing_module_path, ...]`.  For each
    // file that does `pub use crate::M::*;`, M (as `::`-joined string)
    // maps to that file's `module_path`, so a function defined in M
    // also becomes callable under the importing module's namespace
    // (and through the full set of crate-alias spellings the alias
    // generator emits).  Mirrors Rust's resolution of `crate::
    // ImportingMod::name` through the re-export; without this fan-out
    // the registry would only carry the original `M::name` aliases
    // and `crate::ImportingMod::name` would fail to resolve.
    let mut glob_reexports: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for parsed in parsed_files {
        if parsed.module_path.is_empty() || parsed.pub_use_globs.is_empty() {
            continue;
        }
        for source_segments in &parsed.pub_use_globs {
            let source_key = source_segments.join("::");
            glob_reexports
                .entry(source_key)
                .or_default()
                .push(parsed.module_path.clone());
        }
    }

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
            // Additional alias spellings for `pub use crate::<func
            // module>::*;` re-exports — without this, a caller that
            // writes `crate::ImportingMod::func` (resolved through the
            // Rust-side glob re-export) finds no registered graph
            // because `free_function_alias_paths` only fans out under
            // the function's own module.
            if let Some(importing_modules) = glob_reexports.get(&func.module_path) {
                // Use just the function's leaf name (without module
                // prefix) so the re-export aliases mirror what the
                // alias generator would emit for a function natively
                // defined in `importing_module`.
                let leaf = func
                    .name
                    .rsplit("::")
                    .next()
                    .unwrap_or(&func.name)
                    .to_string();
                for importing_module in importing_modules {
                    let synthetic_name = if importing_module.is_empty() {
                        leaf.clone()
                    } else {
                        format!("{importing_module}::{leaf}")
                    };
                    for path in free_function_alias_paths(&synthetic_name, importing_module) {
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
        }
    }

    // ── Build CallControl (RPython call.py) ──
    // Populate with all discovered function graphs and trait impl methods.
    let mut call_control = call::CallControl::new();
    // RPython: known struct types for get_type_flag(ARRAY.OF) → FLAG_STRUCT.
    call_control.set_known_struct_names(program.known_struct_names.clone());
    // RPython: struct field types for op.args[0].concretetype resolution.
    call_control.set_struct_fields(program.struct_fields.clone());
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
    // Thread per-source-file `parsed.module_path` + `use_imports`
    // into CallControl as data carriers (orthodox PyPy
    // `bookkeeper.position` + `frame.f_globals` lexical-resolution
    // entry points, see [[orthodox-6item-2026-05-17]] item 2.3/2.4).
    // Today's consumers normalise at the runtime path_hash boundary
    // via `STRUCT_ORIGIN_REGISTRY` + `canonical_struct_name`; the
    // carriers here let a future per-graph lexical resolver land
    // without re-plumbing the parsed-source ingress.
    call_control.parsed_module_paths = parsed_files.iter().map(|p| p.module_path.clone()).collect();
    // `use_imports` aggregated across all parsed files —
    // `parse::collect_use_imports` populates per-file map at
    // `parse_source_with_module`; here we re-collect from the
    // `ParsedInterpreter` slice the analyzer entry received.
    let mut use_imports_agg: std::collections::HashMap<(String, String), String> =
        std::collections::HashMap::new();
    for parsed in parsed_files {
        for (alias, full) in &parsed.use_imports {
            use_imports_agg
                .entry((parsed.module_path.clone(), alias.clone()))
                .or_insert_with(|| full.clone());
        }
    }
    call_control.use_imports = use_imports_agg;
    // Z2.5 Path C — populate the metadata-only `unsafe_fn_stubs`
    // carrier so the codewriter's `dual_gate_registry` can register
    // every `unsafe fn` / unsafe impl-method as a stub-pygraph entry
    // in PyreCallRegistry.  Walks each parsed source file under its
    // crate-stripped `module_path` prefix, dropping unsafe fns whose
    // return type the slice 3a projection cannot represent (see
    // `flowspace::rust_source::register::simple_return_type_to_lltype`).
    // Closes the bulk of the "not registered in PyreCallRegistry"
    // Skip cluster (218 events at 2026-05-22 measurement) dominated
    // by `pyre_object::is_*` predicates whose body lowering is
    // intentionally rejected at `build_flow.rs:215`.
    let mut unsafe_stubs: Vec<(
        Vec<String>,
        crate::flowspace::argument::Signature,
        crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
    )> = Vec::new();
    for parsed in parsed_files {
        unsafe_stubs.extend(
            crate::flowspace::rust_source::register::extract_unsafe_fn_stubs(
                &parsed.file,
                &parsed.module_path,
            ),
        );
    }
    call_control.unsafe_fn_stubs = unsafe_stubs;
    // Codewriter-side mirror of the static catalogue.  Same
    // `(segments, ty)` shape that
    // [`KnownStaticsCatalogue::from_parsed_files`] feeds to the
    // front-end's `Expr::Path` lookup; reusing the
    // `early_static_decls` walk avoids a second pass.
    call_control.static_decls = early_static_decls;
    // Populate CallControl with layouts from the provider.
    for struct_name in program.struct_fields.fields.keys() {
        if let Some(layout) = provider.get_struct_layout(struct_name) {
            call_control.set_struct_layout(struct_name.clone(), layout);
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
    // Step 6.E Slice 3.B — index `program.functions` so the registration
    // loop below can substitute the MIR-built graph for the AST-built
    // one carried in `canonical_trait_impls` / `canonical_inherent_methods`.
    //
    // Impl methods (trait-impl + inherent) live under `self_ty_root`;
    // trait-default bodies live under `trait_root` with `self_ty_root =
    // None`.  The lookup mirrors `audit_ast_extract_coverage`'s indexing
    // shape so a covered entry guarantees a graph substitution path
    // exists.  AST graphs remain the fallback for entries MIR did not
    // build (CurrentFrameGuard::drop, PyFrame::__init__, etc.).
    let program_impl_graphs: std::collections::HashMap<(String, String), &model::FunctionGraph> =
        program
            .functions
            .iter()
            .filter_map(|f| {
                let owner = f.self_ty_root.as_deref()?;
                Some(((owner.to_string(), f.name.clone()), &f.graph))
            })
            .collect();
    let program_default_graphs: std::collections::HashMap<
        (String, String),
        &model::FunctionGraph,
    > = program
        .functions
        .iter()
        .filter_map(|f| {
            if f.self_ty_root.is_some() {
                return None;
            }
            let tr = f.trait_root.as_deref()?;
            Some(((tr.to_string(), f.name.clone()), &f.graph))
        })
        .collect();
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
            // Slice 3.B: prefer the MIR-built graph when program.functions
            // covers this entry; fall back to the AST-built graph.
            let mir_graph: Option<&model::FunctionGraph> = if is_default {
                program_default_graphs
                    .get(&(impl_info.trait_name.clone(), method.name.clone()))
                    .copied()
            } else {
                program_impl_graphs
                    .get(&(impl_type.to_string(), method.name.clone()))
                    .copied()
            };
            let graph_source: Option<model::FunctionGraph> = mir_graph
                .cloned()
                .or_else(|| method.graph.clone());
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
                // `front/ast.rs::canonical_call_target` turns that into
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
                    // Slice 3.B: prefer MIR's graph for the direct_path
                    // registration too; fall back to AST when MIR has
                    // no entry.
                    let direct_source: Option<model::FunctionGraph> = mir_graph
                        .cloned()
                        .or_else(|| method.graph.clone());
                    if let Some(g) = direct_source {
                        let direct_graph = match &method.return_type {
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
                // Default-method bodies also register under `[trait_name,
                // method_name]` (see the `register_function_graph(direct_path,
                // direct_graph)` branch above); mirror the hint registration
                // so the BFS reaches the same `_reject_function("elidable")`
                // verdict regardless of which path it walks.
                if impl_info.for_type.starts_with("<default methods of ") {
                    let direct_path = crate::parse::CallPath::from_segments([
                        impl_info.trait_name.as_str(),
                        method.name.as_str(),
                    ]);
                    call_control.register_function_hints_for(direct_path, method.hints.clone());
                }
            }
            // RPython: hints bound to graph identity.
            let default_direct_path = if impl_info.for_type.starts_with("<default methods of ") {
                Some(crate::parse::CallPath::from_segments([
                    impl_info.trait_name.as_str(),
                    method.name.as_str(),
                ]))
            } else {
                None
            };
            for hint in &method.hints {
                match hint.as_str() {
                    "elidable" => {
                        call_control.mark_elidable(path.clone());
                        if let Some(ref dp) = default_direct_path {
                            call_control.mark_elidable(dp.clone());
                        }
                    }
                    "elidable_cannot_raise" => {
                        call_control.mark_cannot_raise_assertion(path.clone());
                        if let Some(ref dp) = default_direct_path {
                            call_control.mark_cannot_raise_assertion(dp.clone());
                        }
                    }
                    "elidable_or_memerror" => {
                        call_control.mark_memerror_only_assertion(path.clone());
                        if let Some(ref dp) = default_direct_path {
                            call_control.mark_memerror_only_assertion(dp.clone());
                        }
                    }
                    "loopinvariant" => {
                        call_control.mark_loopinvariant(path.clone());
                        if let Some(ref dp) = default_direct_path {
                            call_control.mark_loopinvariant(dp.clone());
                        }
                    }
                    "close_stack" => call_control.mark_close_stack(path.clone()),
                    "cannot_collect" => call_control.mark_cannot_collect(path.clone()),
                    "gc_effects" => call_control.mark_external_gc_effects(path.clone()),
                    _ => {}
                }
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
        // Slice 3.B: prefer the MIR-built graph; fall back to the
        // AST-built graph carried in InherentMethodInfo.
        let graph: model::FunctionGraph = program_impl_graphs
            .get(&(impl_type.to_string(), method_info.name.clone()))
            .map(|g| (*g).clone())
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
            Some(rt) => method_info.graph.clone().with_return_type(rt),
            None => method_info.graph.clone(),
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
                // — companion hint emitted by `front::ast::collect_jit_hints`
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

    // RPython: setup_jitdriver(jitdriver_sd) — register portal + green/red layout.
    // PyPy interp_jit.py: greens = ['next_instr', 'is_being_profiled', 'pycode'],
    //                      reds = ['frame', 'ec'], virtualizables = ['frame']
    // Callers can override the portal binding via `PipelineConfig::portal`.
    //
    // Default portal identity: prefer `eval_loop_jit`
    // (pyre-jit/src/eval.rs:1187), the pyre analogue of upstream's
    // `warmspot.py::portal_runner` and the single graph seeded into
    // `find_all_graphs(portal, policy)` at `call.py:57`. When the graph
    // set does not include pyre-jit/src/eval.rs (e.g. compact test
    // inputs whose PYRE_JIT_GRAPH_SOURCES stops at pyre-interpreter),
    // fall back to `execute_opcode_step` so those tests retain a portal
    // target. `execute_opcode_step` itself is a handler reached from the
    // real portal's match arm, so seeding BFS from it treats a handler
    // as an entry point — tolerable only for the legacy test
    // configurations that have no `eval_loop_jit` at all; once those
    // tests feed the full Phase D0 source set the fallback is never
    // exercised and the eval_loop_jit-only identity locks in.
    let default_portal_name = {
        // Tolerate module-qualified registrations: `eval_loop_jit` may
        // land under `["eval", "eval_loop_jit"]` when its file
        // (`pyre-jit/src/eval.rs`) was parsed with `module_path =
        // "eval"`.  Suffix-match on the leaf segment covers both shapes.
        let has_leaf = |leaf: &str| {
            call_control
                .function_graphs()
                .keys()
                .any(|k| k.segments.last().map(|s| s == leaf).unwrap_or(false))
        };
        if has_leaf("eval_loop_jit") {
            "eval_loop_jit"
        } else {
            "execute_opcode_step"
        }
    };
    let (portal_name, portal_greens, portal_reds, portal_virtualizables, portal_red_types) =
        match &config.pipeline.portal {
            Some(spec) => (
                spec.name.clone(),
                spec.greens.clone(),
                spec.reds.clone(),
                spec.virtualizables.clone(),
                spec.red_types.clone(),
            ),
            None => (
                default_portal_name.to_string(),
                vec![
                    "next_instr".to_string(),
                    "is_being_profiled".to_string(),
                    "pycode".to_string(),
                ],
                vec!["frame".to_string(), "ec".to_string()],
                // PyPy interp_jit.py: virtualizables = ['frame'] —
                // jitdriver.virtualizables drives warmspot.py:527-545
                // make_virtualizable_infos selection.
                vec!["frame".to_string()],
                // jit.py / interp_jit.py: red types parallel to reds.
                // 'frame' is the virtualizable PyFrame; 'ec' is the
                // ExecutionContext (the wrapping struct in pyre).
                vec!["PyFrame".to_string(), "ExecutionContext".to_string()],
            ),
        };
    // Resolve the portal `CallPath` tolerantly so the lookup hits
    // regardless of how `build_graphs_from_items` qualifies the
    // function's name.  Free functions registered with a non-empty
    // `parsed.module_path` land under `["module", "name"]` (e.g.
    // `["pyopcode", "execute_opcode_step"]`); a bare-leaf alias
    // (`["execute_opcode_step"]`) is also published for cross-module
    // bare callsites (`lib.rs:432-448`).  Prefer the module-qualified
    // canonical key over the bare alias so production parity with
    // `module_path!()` semantics — and downstream tests asserting
    // `by_path.contains_key(&["module", "name"])` — match the JitCode
    // registration shape.  Selection order:
    //  1. module-qualified key whose leaf matches the portal name and
    //     is NOT `crate`-prefixed (multi-segment, source-of-truth);
    //  2. bare leaf (legacy empty-`module_path` fixtures);
    //  3. any remaining suffix-match (last-resort fallback).
    let portal = {
        let bare = parse::CallPath::from_segments([portal_name.as_str()]);
        let leaf_matches = |k: &&parse::CallPath| {
            k.segments
                .last()
                .map(|s| s == portal_name.as_str())
                .unwrap_or(false)
        };
        // Prefer the shortest qualified path among aliases — the
        // canonical `[module, name]` shape (length 2) over the
        // `[crate_alias, module, name]` shape (length 3+).  HashMap
        // iteration order is non-deterministic, so `find()` without
        // tie-breaking would silently flip between aliases across
        // builds, making downstream tests (e.g.
        // `generated::tests::eval_loop_jit_portal_*`) flake on the
        // by_path key shape.  Choosing the shortest path mirrors
        // `lib.rs:465-471 register_function_graph_alias` order: the
        // `[module, name]` form is registered FIRST and is the
        // source-of-truth canonical name for the graph; the longer
        // forms are crate-prefixed aliases for cross-crate callsites.
        //
        // Tie-break among same-length candidates by deprioritising
        // keys whose first segment is a crate alias
        // (`pyre_interpreter`, `pyre_object`, `pyre_jit`), then by
        // lexicographic order.  Without this, two length-2 keys like
        // `["eval", "eval_loop_jit"]` (the module-qualified form) and
        // `["pyre_object", "eval_loop_jit"]` (a crate-alias form
        // emitted by `free_function_alias_paths`) tie on length, and
        // the winner depends on HashMap iteration order — the source
        // of the eval_loop_jit_portal_* flake.
        let is_crate_alias =
            |seg: &str| matches!(seg, "pyre_interpreter" | "pyre_object" | "pyre_jit");
        let qualified = call_control
            .function_graphs()
            .keys()
            .filter(|k| {
                leaf_matches(k)
                    && k.segments.len() > 1
                    && k.segments.first().map(|s| s != "crate").unwrap_or(false)
            })
            .min_by_key(|k| {
                (
                    k.segments.len(),
                    k.segments
                        .first()
                        .map(|s| is_crate_alias(s.as_str()))
                        .unwrap_or(false),
                    k.segments.clone(),
                )
            })
            .cloned();
        if let Some(qualified) = qualified {
            qualified
        } else if call_control.function_graphs().contains_key(&bare) {
            bare
        } else {
            call_control
                .function_graphs()
                .keys()
                .find(leaf_matches)
                .cloned()
                .unwrap_or(bare)
        }
    };
    if call_control.function_graphs().contains_key(&portal) {
        call_control.setup_jitdriver(
            portal,
            portal_greens,
            portal_reds,
            portal_virtualizables,
            portal_red_types,
        );
        // warmspot.py:515-545 WarmRunnerDesc.make_virtualizable_infos —
        // assigns jd.index_of_virtualizable, builds GreenFieldInfo
        // when any green name contains '.', and delegates the upstream
        // `VirtualizableInfo(self, VTYPEPTR)` constructor call
        // (warmspot.py:543) to `vinfo_factory`.  Hosts entering through
        // `analyze_multiple_pipeline_with_vinfo_factory` supply a real
        // closure that builds the metainterp-side `VirtualizableInfo`
        // (e.g. via `majit_metainterp::virtualizable::VirtualizableInfo::
        // build_for(VTYPEPTR)`); the default entry points pass
        // `|_, _| None` so the codewriter slot stays empty until
        // `MetaInterp::set_virtualizable_info` (jitdriver.rs:285) wires
        // it at runtime.
        call_control.make_virtualizable_infos(|jd_idx, vtypeptr_token| {
            vinfo_factory(jd_idx, vtypeptr_token)
        });
    }
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
    // path; `ProgramPipelineResult.functions` / `total_*` are populated
    // only with their default values because every consumer reads
    // `opcode_dispatch` / `jitcodes` / `insns` / `descrs` instead.
    let mut pipeline = pipeline::ProgramPipelineResult {
        functions: Vec::new(),
        opcode_dispatch: Vec::new(),
        jitcodes: Vec::new(),
        jitcodes_by_path: indexmap::IndexMap::new(),
        insns: majit_ir::vec_assoc::VecAssoc::new(),
        descrs: Vec::new(),
        total_blocks: 0,
        total_ops: 0,
        total_vable_rewrites: 0,
    };

    mark_phase!("call_control + canonical_trait_impls + register graphs");
    let (opcode_dispatch, jitcodes, insns, descrs) = build_canonical_opcode_dispatch(
        parsed_files,
        &program.fn_return_types,
        &config.pipeline,
        &mut call_control,
    );
    mark_phase!("build_canonical_opcode_dispatch");
    pipeline.opcode_dispatch = opcode_dispatch;
    pipeline.jitcodes = jitcodes;
    // Mirror of `CallControl::jitcodes` (RPython `call.py:87 self.jitcodes`)
    // captured before `call_control` is dropped. Needed because consumers
    // that look up a JitCode by graph identity cannot reconstruct the key
    // from the alloc-ordered `pipeline.jitcodes` vector alone.
    pipeline.jitcodes_by_path = call_control.jitcodes().clone();
    pipeline.insns = insns;
    pipeline.descrs = descrs;

    pipeline
}

/// Build opcode dispatch arms and produce JitCodes for discovered callee graphs.
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
///
/// PyPy registers each opcode handler as its own Python method, so PyPy's
/// codewriter naturally gets one jitcode per opcode via the discovery loop.
/// pyre's interpreter dispatches inside one big match instead of separate
/// methods, so we register each match arm body as a synthetic graph here
/// (`CallPath::["__opcode_dispatch__", "<selector>#<arm_id>"]`) and the
/// orthodox `drain_pending_graphs` loop picks them up exactly the same way
/// it picks up callee graphs discovered during jtransform.
fn build_canonical_opcode_dispatch(
    parsed_files: &[parse::ParsedInterpreter],
    fn_return_types: &std::collections::HashMap<String, String>,
    pipeline_config: &pipeline::PipelineConfig,
    call_control: &mut call::CallControl,
) -> (
    Vec<opcode_dispatch::PipelineOpcodeArm>,
    Vec<std::sync::Arc<jitcode::JitCode>>,
    majit_ir::vec_assoc::VecAssoc<String, u8>,
    Vec<jitcode::BhDescr>,
) {
    let mut opcode_arms = Vec::new();

    for parsed in parsed_files {
        let file_opcodes = parse::extract_opcode_dispatch_arms(parsed, fn_return_types);
        if !file_opcodes.is_empty() {
            opcode_arms = file_opcodes;
            break;
        }
    }

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

    // Phase 1: RPython grab_initial_jitcodes + drain portal + callees.
    // RPython call.py:145-148.
    call_control.grab_initial_jitcodes();
    codewriter.drain_pending_graphs(call_control, &pipeline_config.transform);

    // Phase 2: register each opcode arm body as a synthetic graph.
    //
    // PyPy's interpreter has one Python method per opcode and `find_all_graphs`
    // discovers them naturally; pyre dispatches inside one match, so we walk
    // the parser-extracted arms and call `register_function_graph` +
    // `get_jitcode` ourselves. Each arm gets a stable `arm_id` (extract order)
    // and a synthetic `CallPath::["__opcode_dispatch__", "<selector>#<arm_id>"]`
    // which decouples display label (selector) from identity (path/index).
    //
    // `arm.flattened` is set after `drain_pending_graphs` from the
    // assembled jitcode's `body._ssarepr`; the previous eager
    // dual_gate→lower_indirect_calls→jtransform→merge_synth_kinds→
    // regalloc→flatten chain at this site duplicated the work that
    // `transform_graph_to_jitcode` does inside the drain loop, so we
    // register the arm bodies here and let the canonical pipeline
    // produce the SSARepr we then read back below.
    let mut dispatch: Vec<opcode_dispatch::PipelineOpcodeArm> =
        Vec::with_capacity(opcode_arms.len());
    let mut dispatch_paths: Vec<Option<parse::CallPath>> = Vec::with_capacity(opcode_arms.len());
    for (arm_id, arm) in opcode_arms.into_iter().enumerate() {
        // Register the arm body in CallControl. RPython call.py:155-172
        // `get_jitcode(graph)` returns the callee object; the final
        // `jitcode.index` is assigned only after assembly completes.
        let entry_jitcode_path = arm.body_graph.map(|body_graph| {
            let synthetic_path = synthetic_opcode_arm_path(&arm.selector, arm_id);
            call_control.register_function_graph(synthetic_path.clone(), body_graph);
            let _ = call_control.get_jitcode(&synthetic_path);
            synthetic_path
        });

        dispatch.push(opcode_dispatch::PipelineOpcodeArm {
            arm_id,
            selector: arm.selector,
            entry_jitcode_index: None,
            flattened: None,
        });
        dispatch_paths.push(entry_jitcode_path);
    }

    // Phase 3: Drain pending graphs.
    //
    // RPython codewriter.py:79-84: `for graph, jitcode in enum_pending_graphs()`.
    // After Phase 2 every opcode arm body is on `unfinished_graphs`, plus any
    // callees discovered during the parser-level transform pass above. Each
    // gets transformed → assembled in turn, and any *new* callees they reach
    // are added to the queue and picked up by the same loop.
    codewriter.drain_pending_graphs(call_control, &pipeline_config.transform);

    // RPython codewriter.py:85: self.assembler.finished(callinfocollection).
    codewriter
        .assembler
        .finished(&call_control.callinfocollection);

    for (arm, path) in dispatch.iter_mut().zip(dispatch_paths.into_iter()) {
        if let Some(path) = path {
            let jitcode = call_control
                .jitcode_handle(&path)
                .expect("opcode arm jitcode handle must exist after registration");
            arm.entry_jitcode_index = Some(jitcode.index());
            arm.flattened = jitcode.try_body().and_then(|body| body._ssarepr.clone());
        }
    }

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

    (dispatch, jitcodes, insns, descrs)
}

/// Synthetic CallPath for an opcode-dispatch arm body.
///
/// PyPy uses `graph.name` (which is the Python method name) as the
/// debug label and `graph` object identity as the dict key in
/// `CallControl.jitcodes`. pyre uses `CallPath` as the dict key, so we
/// build a 2-segment synthetic path:
///   `["__opcode_dispatch__", "<selector_canonical>#<arm_id>"]`
/// The `#<arm_id>` suffix guarantees collision-free keys even if two
/// arms shared a selector string (parser already rejects that, but the
/// suffix makes the invariant local to this function).
fn synthetic_opcode_arm_path(
    selector: &parse::OpcodeDispatchSelector,
    arm_id: usize,
) -> parse::CallPath {
    parse::CallPath::from_segments([
        "__opcode_dispatch__".to_string(),
        format!("{}#{}", selector.canonical_key(), arm_id),
    ])
}

/// Generate tracing code directly from the canonical pipeline result.
pub fn generate_trace_code_from_pipeline(result: &pipeline::ProgramPipelineResult) -> String {
    codegen::generate_from_pipeline(result)
}

/// Like [`generate_trace_code_from_pipeline`] but respects the selected
/// [`codegen::CodegenFlavor`]. Callers outside of pyre (e.g. `aheui-jit`)
/// opt for [`codegen::CodegenFlavor::Minimal`] to skip emission of
/// pyre-specific helpers.
pub fn generate_trace_code_from_pipeline_with_flavor(
    result: &pipeline::ProgramPipelineResult,
    flavor: codegen::CodegenFlavor,
) -> String {
    codegen::generate_from_pipeline_with_flavor(result, flavor)
}

pub use codegen::CodegenFlavor;

/// Produce a recognition report: how much the pipeline understands.
pub fn recognition_report(result: &pipeline::ProgramPipelineResult) -> codegen::RecognitionReport {
    codegen::recognition_report(result)
}
pub use codegen::{OpcodeRecognition, RecognitionReport};

/// Generate code from graph pipeline results.
#[cfg(test)]
pub fn generate_graph_code(result: &pipeline::ProgramPipelineResult) -> String {
    codegen::generate_from_graph(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use walkdir::WalkDir;

    fn read_pyre_file(name: &str) -> String {
        let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../pyre/");
        std::fs::read_to_string(format!("{base}{name}"))
            .unwrap_or_else(|_| panic!("failed to read {name}"))
    }

    fn collect_rs_files(dir: &Path, sources: &mut Vec<String>) {
        for entry in WalkDir::new(dir) {
            let entry = entry.unwrap_or_else(|_| panic!("failed to walk dir {}", dir.display()));
            let path = entry.path();
            if entry.file_type().is_file() && path.extension().is_some_and(|ext| ext == "rs") {
                sources.push(
                    std::fs::read_to_string(path)
                        .unwrap_or_else(|_| panic!("failed to read {}", path.display())),
                );
            }
        }
    }

    /// Collect `(source, crate-stripped module_path)` per file under `dir`.
    /// The module_path matches what `module_path!()` would emit at runtime
    /// minus the leading crate-name segment — `lib.rs` → `""`,
    /// `baseobjspace.rs` → `"baseobjspace"`, `module/inner.rs` → `"module::inner"`.
    /// Feeds `parse::parse_source_with_module` so call-site segments
    /// emitted by `canonical_call_target` for `crate::module::name` paths
    /// hit the same `module::name` keys the registry collects.
    fn collect_rs_files_with_modules(
        dir: &Path,
        sources: &mut Vec<String>,
        module_paths: &mut Vec<String>,
    ) {
        for entry in WalkDir::new(dir) {
            let entry = entry.unwrap_or_else(|_| panic!("failed to walk dir {}", dir.display()));
            let path = entry.path();
            if entry.file_type().is_file() && path.extension().is_some_and(|ext| ext == "rs") {
                let source = std::fs::read_to_string(path)
                    .unwrap_or_else(|_| panic!("failed to read {}", path.display()));
                let relative = path.strip_prefix(dir).unwrap_or(path).with_extension("");
                let mut segments: Vec<String> = relative
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().to_string())
                    .collect();
                // lib.rs / main.rs / mod.rs occupy the parent module path
                // (no leaf segment), so strip them and let the remaining
                // ancestor chain stand as the module path.  Empty path
                // (`lib.rs` at the crate root) registers under the empty
                // prefix, matching today's behaviour.
                if matches!(
                    segments.last().map(String::as_str),
                    Some("lib" | "main" | "mod")
                ) {
                    segments.pop();
                }
                sources.push(source);
                module_paths.push(segments.join("::"));
            }
        }
    }

    fn read_all_pyre_sources() -> Vec<String> {
        let (sources, _module_paths) = read_all_pyre_sources_with_modules();
        sources
    }

    fn read_all_pyre_sources_with_modules() -> (Vec<String>, Vec<String>) {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pyre");
        let mut sources = Vec::new();
        let mut module_paths = Vec::new();
        for dir in [
            base.join("pyre-object/src"),
            base.join("pyre-interpreter/src"),
        ] {
            collect_rs_files_with_modules(&dir, &mut sources, &mut module_paths);
        }
        (sources, module_paths)
    }

    #[test]
    fn test_analyze_pyopcode() {
        let source = read_pyre_file("pyre-interpreter/src/pyopcode.rs");
        let result = analyze_multiple_pipeline_with_config(
            &[&source],
            &crate::test_support::pyre_analyze_config(),
        );

        assert!(
            result.opcode_dispatch.len() > 20,
            "expected >20 opcode arms, got {}",
            result.opcode_dispatch.len()
        );

        eprintln!("=== Single-file Analysis ===");
        eprintln!("Opcodes: {}", result.opcode_dispatch.len());
        for (i, arm) in result.opcode_dispatch.iter().enumerate() {
            eprintln!(
                "  [{i}] {} → {:?}",
                arm.selector.canonical_key(),
                arm.flattened.as_ref().map(|f| f.insns.len())
            );
        }
    }

    /// Whole-program pipeline fixture: parse + lower the entire pyre
    /// interpreter exactly once, then run every whole-program assertion
    /// block against the shared result. `ProgramPipelineResult` holds
    /// `Rc`-based IR (`!Send`/`!Sync`), so it cannot be shared across
    /// cargo's parallel test threads via a `static`/`LazyLock`; merging
    /// the formerly-separate per-block tests into one is the only way to
    /// pay the (interpreter-size-linear) lowering cost a single time.
    #[test]
    fn test_full_pipeline_analysis() {
        let sources = read_all_pyre_sources();
        let source_refs: Vec<&str> = sources.iter().map(String::as_str).collect();
        let parsed_files: Vec<_> = source_refs
            .iter()
            .map(|source| parse::parse_source(source))
            .collect();
        let config = crate::test_support::pyre_analyze_config();
        // Single whole-program lowering shared by every block below.
        let result = analyze_pipeline_from_parsed(
            &parsed_files,
            &config,
            None,
            &|_, _| None,
            &[],
            &[],
            HostStaticAddrs::default(),
        );
        // Walker-populated metadata mirrors the production
        // `analyze_pipeline_from_parsed` path: `extract_trait_impls`
        // lowers method bodies against this registry so the
        // `expr_unary_not_operand_kind` classifier can resolve
        // cross-module bool calls. Empty registries previously masked
        // unsupported `!x` patterns through the UNARY_NOT bool-fork
        // fall-through; with the fail-loud restoration, the test
        // fixture must populate the registries the same way production
        // does.
        let metadata = crate::front::ast::collect_program_metadata_pub(&parsed_files);
        let trait_impls: Vec<TraitImplInfo> = parsed_files
            .iter()
            .flat_map(|p| {
                parse::extract_trait_impls(
                    p,
                    &metadata.struct_fields,
                    &metadata.fn_return_types,
                    &metadata.known_struct_names,
                )
                .expect("trait impls must lower")
            })
            .collect();

        assert_multi_file_analysis(&result, &trait_impls);
        assert_codegen_output(&result);
        assert_recognition_report(&result);
    }

    fn assert_multi_file_analysis(
        result: &pipeline::ProgramPipelineResult,
        trait_impls: &[TraitImplInfo],
    ) {
        eprintln!("=== Multi-file Analysis ===");
        eprintln!("Opcodes: {}", result.opcode_dispatch.len());
        eprintln!("Functions: {}", result.functions.len());
        eprintln!("Trait impls: {}", trait_impls.len());

        // Should have trait impls from eval.rs (PyFrame impls)
        let pyframe_impls: Vec<_> = trait_impls
            .iter()
            .filter(|i| i.for_type.contains("PyFrame"))
            .collect();
        eprintln!("\nPyFrame trait impls: {}", pyframe_impls.len());
        for impl_info in &pyframe_impls {
            eprintln!(
                "  impl {} for PyFrame — {} methods",
                impl_info.trait_name,
                impl_info.methods.len()
            );
            for m in &impl_info.methods {
                eprintln!("    {}", m.name);
            }
        }

        // Should have resolved opcode patterns (flattened op counts)
        eprintln!("\nOpcode patterns:");
        for arm in &result.opcode_dispatch {
            if let Some(ref flat) = arm.flattened {
                eprintln!(
                    "  {} → {} flat ops",
                    arm.selector.canonical_key(),
                    flat.insns.len()
                );
            }
        }

        // Report flattened (inline→jtransform→flatten) stats
        let flattened_count = result
            .opcode_dispatch
            .iter()
            .filter(|a| a.flattened.is_some())
            .count();
        eprintln!(
            "\nFlattened (inline pipeline): {flattened_count}/{}",
            result.opcode_dispatch.len()
        );
        for arm in &result.opcode_dispatch {
            if let Some(ref flat) = arm.flattened {
                eprintln!(
                    "  {} → {} flat ops",
                    arm.selector.canonical_key(),
                    flat.insns.len()
                );
            }
        }

        // Verify canonical graph/pipeline dispatch flattens a useful subset.
        let flattened_dispatch_count = result
            .opcode_dispatch
            .iter()
            .filter(|a| a.flattened.is_some())
            .count();
        assert!(
            flattened_dispatch_count >= 10,
            "expected >=10 flattened opcode arms, got {}",
            flattened_dispatch_count
        );

        // Verify flattened arms produce non-empty op sequences.
        assert!(
            result
                .opcode_dispatch
                .iter()
                .filter_map(|arm| arm.flattened.as_ref())
                .all(|f| f.insns.len() > 0),
            "all flattened arms should have non-empty op sequences"
        );

        // RPython: CodeWriter.make_jitcodes() produces JitCode for each graph.
        // Verify the full pipeline (regalloc + liveness + assemble) runs.
        //
        // `collect_jitcodes_in_alloc_order` preserves the dense invariant
        // `all_jitcodes[i].index == i` (call.rs:1620-1633, matching RPython
        // codewriter.py:80), and by construction every slot in the vec is
        // a legitimate shell produced by `get_jitcode`. Shells whose body
        // was never committed (e.g. graph registered by a caller but never
        // drained because the test harness doesn't wire an fnaddr binding)
        // round-trip as body-less entries — `try_body()` is the documented
        // probe for distinguishing them.
        eprintln!("\nJitCodes: {}", result.jitcodes.len());
        let mut bodied = 0usize;
        for (i, jitcode) in result.jitcodes.iter().enumerate() {
            match jitcode.try_body() {
                Some(body) => {
                    eprintln!(
                        "  [{}] {} → {} bytes, regs i={} r={} f={}",
                        i,
                        jitcode.name,
                        body.code.len(),
                        body.c_num_regs_i,
                        body.c_num_regs_r,
                        body.c_num_regs_f,
                    );
                    bodied += 1;
                }
                None => eprintln!("  [{}] {} → <shell: body not committed>", i, jitcode.name),
            }
        }
        assert!(
            !result.jitcodes.is_empty(),
            "CodeWriter should produce JitCodes from opcode arms"
        );
        assert!(
            bodied > 0,
            "at least one JitCode should have a committed body"
        );
        assert!(
            result
                .jitcodes
                .iter()
                .filter_map(|jc| jc.try_body())
                .all(|body| !body.code.is_empty()),
            "every committed JitCode body must have non-empty bytecode"
        );
    }

    fn assert_codegen_output(result: &pipeline::ProgramPipelineResult) {
        let code = generate_trace_code_from_pipeline(result);
        let flattened_arms: Vec<_> = result
            .opcode_dispatch
            .iter()
            .filter(|arm| arm.flattened.is_some())
            .collect();

        // Should contain canonical dispatch table
        assert!(
            code.contains("CANONICAL_TRACE_PATTERNS"),
            "missing CANONICAL_TRACE_PATTERNS"
        );
        assert!(
            !code.contains("pub const TRACE_PATTERNS"),
            "canonical output should not emit legacy TRACE_PATTERNS alias"
        );
        assert!(
            code.contains("Canonical analysis summary:"),
            "missing canonical summary"
        );
        assert!(!flattened_arms.is_empty(), "expected flattened opcode arms");

        eprintln!("=== Generated Code ({} bytes) ===", code.len());
        // Print first 50 lines
        for (i, line) in code.lines().enumerate().take(50) {
            eprintln!("{:3}: {}", i + 1, line);
        }
    }

    fn assert_recognition_report(result: &pipeline::ProgramPipelineResult) {
        let report = recognition_report(result);

        eprintln!("=== Recognition Report ===");
        eprintln!(
            "Total opcodes: {}, Flattened: {} ({:.0}%)",
            report.total_opcodes,
            report.flattened,
            if report.total_opcodes > 0 {
                report.flattened as f64 / report.total_opcodes as f64 * 100.0
            } else {
                0.0
            }
        );
        eprintln!(
            "Total flat ops: {}, Unknown: {}, Unresolved calls: {}",
            report.total_flat_ops, report.unknown_ops, report.unresolved_calls
        );
        eprintln!("\nPer-opcode:");
        for opc in &report.per_opcode {
            let status = if opc.flat_ops > 0 {
                format!(
                    "{} ops ({}U {}C)",
                    opc.flat_ops, opc.unknowns, opc.unresolved
                )
            } else {
                "unflattened".to_string()
            };
            eprintln!("  {:40} {}", opc.selector, status);
        }

        // Scoreboard assertions
        assert!(
            report.total_opcodes > 20,
            "expected >20 opcodes, got {}",
            report.total_opcodes
        );
        assert!(
            report.flattened >= 10,
            "expected >=10 flattened, got {}",
            report.flattened
        );
    }

    #[test]
    fn test_graph_pipeline_e2e() {
        // E2E test: source → AST front-end → semantic graph → graph transform → classify
        let parsed = parse::parse_source(
            r#"
            struct Frame { next_instr: usize, locals_w: Vec<i64> }
            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
                fn store_fast(&mut self, val: i64) {
                    let idx = self.next_instr;
                    self.locals_w[idx] = val;
                }
            }
        "#,
        );

        // Step 1: AST → semantic graph
        let program = front::build_semantic_program(&parsed).expect("source must lower");
        assert_eq!(
            program.functions.len(),
            2,
            "should have load_fast + store_fast"
        );

        // Step 2: graph transform (with virtualizable config)
        let config = GraphTransformConfig {
            vable_fields: vec![VirtualizableFieldDescriptor::new(
                "next_instr",
                Some("Frame".into()),
                0,
            )],
            vable_arrays: vec![VirtualizableFieldDescriptor::new(
                "locals_w",
                Some("Frame".into()),
                0,
            )],
            ..Default::default()
        };

        let load_fast_graph = &program.functions[0].graph;
        let result = rewrite_graph(load_fast_graph, &config);
        assert!(
            result.vable_rewrites > 0,
            "load_fast should have vable rewrites, got notes: {:?}",
            result.notes
        );

        // Step 3: flatten the rewritten graph
        // `resolve_types` commits per-value `concretetype` cells on
        // each backing Variable as it builds, so downstream consumers
        // can read kinds via `FunctionGraph::concretetype_of(&v)`
        // without a separate publish step.
        annotate::annotate(&result.graph);
        rtype::resolve_types(&result.graph);
        let mut result = result;
        crate::regalloc::augment_canonical_exceptblock_on_graph(&mut result.graph);
        let mut regallocs = crate::regalloc::perform_all_register_allocations(&result.graph);
        let flattened = flatten::flatten_graph(&result.graph, &mut regallocs);
        eprintln!(
            "load_fast graph ops: {:?}",
            load_fast_graph.block(load_fast_graph.startblock).operations
        );
        eprintln!("load_fast flattened: {} ops", flattened.insns.len());
        assert!(
            flattened.insns.len() > 0,
            "load_fast should produce flat ops"
        );
    }

    #[test]
    fn test_analyze_pipeline_runs_canonical_graph_path() {
        let source = read_pyre_file("pyre-interpreter/src/pyopcode.rs");
        let graph_result = analyze_pipeline(&source);

        // Canonical pipeline should produce per-opcode dispatch arms.
        assert!(
            graph_result.opcode_dispatch.len() >= 5,
            "expected >=5 opcode dispatch arms, got {}",
            graph_result.opcode_dispatch.len(),
        );
        assert!(
            !graph_result.jitcodes.is_empty(),
            "canonical pipeline should produce jitcodes"
        );
    }

    /// Parity tripwire: the `Instruction::LoadFast` dispatch arm must
    /// inline `load_fast` and the inlined body must have rewritten the
    /// virtualizable `self.next_instr` field read and `self.locals_w[idx]`
    /// array read to `getfield_vable` / `getarrayitem_vable`
    /// ([`OpKind::VableFieldRead`] / [`OpKind::VableArrayRead`]).  A plain
    /// `getfield` / `getarrayitem` (vable rewrite silently skipped) leaves
    /// these kinds absent and must fail the test, not pass it — guarding
    /// against the tripwire degrading to a mere "non-empty flattening"
    /// check.
    fn assert_load_fast_rewrites_vable_accesses(arm: &opcode_dispatch::PipelineOpcodeArm) {
        use crate::jit_codewriter::flatten::FlatOp;
        use crate::model::OpKind;

        let flattened = arm
            .flattened
            .as_ref()
            .expect("LoadFast arm should be flattened");
        let inlined = flattened
            .insns
            .iter()
            .find_map(|insn| match insn {
                FlatOp::Op(op) => match &op.kind {
                    OpKind::InlineCall { jitcode, .. } => jitcode.body()._ssarepr.as_ref(),
                    _ => None,
                },
                _ => None,
            })
            .expect("LoadFast dispatch should inline the load_fast method body");
        let body_has = |pred: &dyn Fn(&OpKind) -> bool| {
            inlined
                .insns
                .iter()
                .any(|insn| matches!(insn, FlatOp::Op(op) if pred(&op.kind)))
        };
        assert!(
            body_has(&|k| matches!(k, OpKind::VableFieldRead { .. })),
            "self.next_instr should rewrite to a vable field read"
        );
        assert!(
            body_has(&|k| matches!(k, OpKind::VableArrayRead { .. })),
            "self.locals_w[idx] should rewrite to a vable array read"
        );
    }

    #[test]
    fn test_analyze_multiple_with_config_rewrites_virtualizable_graphs() {
        let source = r#"
            enum Instruction { LoadFast }

            struct Frame {
                next_instr: usize,
                locals_w: Vec<i64>,
            }

            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
            }

            fn execute_opcode_step(frame: &mut Frame, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => {
                        let _ = frame.load_fast();
                    }
                }
            }
        "#;

        let result = analyze_multiple_pipeline_with_config(
            &[source],
            &AnalyzeConfig {
                pipeline: PipelineConfig {
                    transform: GraphTransformConfig {
                        vable_fields: vec![VirtualizableFieldDescriptor::new(
                            "next_instr",
                            Some("Frame".into()),
                            0,
                        )],
                        vable_arrays: vec![VirtualizableFieldDescriptor::new(
                            "locals_w",
                            Some("Frame".into()),
                            0,
                        )],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
        );

        let load_fast = result
            .opcode_dispatch
            .iter()
            .find(|arm| arm.selector.canonical_key() == "Instruction::LoadFast")
            .expect("LoadFast opcode arm");
        assert!(
            load_fast.flattened.is_some(),
            "LoadFast should be flattened"
        );
        assert!(
            load_fast.flattened.as_ref().unwrap().insns.len() > 0,
            "LoadFast flattened should have ops"
        );
        assert_load_fast_rewrites_vable_accesses(load_fast);
    }

    #[test]
    fn test_analyze_multiple_pipeline_with_config_produces_canonical_vable_dispatch() {
        let source = r#"
            enum Instruction { LoadFast }

            struct Frame {
                next_instr: usize,
                locals_w: Vec<i64>,
            }

            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
            }

            fn execute_opcode_step(frame: &mut Frame, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => {
                        let _ = frame.load_fast();
                    }
                }
            }
        "#;

        let result = analyze_multiple_pipeline_with_config(
            &[source],
            &AnalyzeConfig {
                pipeline: PipelineConfig {
                    transform: GraphTransformConfig {
                        vable_fields: vec![VirtualizableFieldDescriptor::new(
                            "next_instr",
                            Some("Frame".into()),
                            0,
                        )],
                        vable_arrays: vec![VirtualizableFieldDescriptor::new(
                            "locals_w",
                            Some("Frame".into()),
                            0,
                        )],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
        );
        let canonical_load_fast = result
            .opcode_dispatch
            .iter()
            .find(|arm| arm.selector.canonical_key() == "Instruction::LoadFast")
            .expect("canonical LoadFast opcode arm");
        assert!(
            canonical_load_fast.flattened.is_some(),
            "canonical LoadFast should be flattened"
        );
        assert!(
            canonical_load_fast.flattened.as_ref().unwrap().insns.len() > 0,
            "canonical LoadFast flattened should have ops"
        );
        assert_load_fast_rewrites_vable_accesses(canonical_load_fast);
    }

    #[test]
    fn test_analyze_multiple_pipeline_with_fnaddr_bindings_stamps_real_jitcode_fnaddr() {
        let source = r#"
            fn helper_opaque(a: i64, b: i64) -> i64 {
                a + b
            }

            fn execute_opcode_step() -> i64 {
                helper_opaque(2, 3)
            }
        "#;

        let result = analyze_multiple_pipeline_with_fnaddr_bindings(
            &[source],
            &AnalyzeConfig::default(),
            None,
            &[("testcrate::helper_opaque", 0x1234_5678)],
        );

        let helper = result
            .jitcodes
            .iter()
            .find(|jitcode| jitcode.name == "helper_opaque")
            .expect("helper_opaque jitcode");
        assert_eq!(helper.fnaddr, 0x1234_5678);
    }

    #[test]
    fn test_opcode_dispatch_uses_trait_bound_default_method_graphs() {
        let source = r#"
            enum Instruction { LoadFast }

            trait OpcodeStepExecutor {
                fn load_fast_checked(&mut self, idx: usize) {
                    let _ = idx;
                }
            }

            fn execute_opcode_step<E: OpcodeStepExecutor>(executor: &mut E, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => executor.load_fast_checked(0),
                }
            }
        "#;

        let result = analyze_multiple_pipeline(&[source]);
        let arm = result
            .opcode_dispatch
            .iter()
            .find(|arm| arm.selector.canonical_key() == "Instruction::LoadFast")
            .expect("LoadFast opcode arm");
        assert!(
            arm.flattened.is_some(),
            "trait-bound default method should produce a flattened result"
        );
    }

    /// Integration test: CallControl + inline on real pyre sources.
    ///
    /// Verifies that the inline pass produces graphs with low-level ops
    /// (FieldRead, ArrayRead) from inlined handler method bodies.
    #[test]
    fn test_inline_pipeline_integration() {
        let sources = read_all_pyre_sources();
        let source_refs: Vec<&str> = sources.iter().map(String::as_str).collect();
        let parsed_files: Vec<_> = source_refs.iter().map(|s| parse::parse_source(s)).collect();

        // Build CallControl from parsed sources
        let mut call_control = call::CallControl::new();
        let mut function_graphs = std::collections::HashMap::new();
        let metadata = crate::front::ast::collect_program_metadata_pub(&parsed_files);
        for parsed in &parsed_files {
            parse::collect_function_graphs(parsed, &metadata, &mut function_graphs)
                .expect("collect_function_graphs: FlowingError must propagate");
        }
        for (path, graph) in &function_graphs {
            call_control.register_function_graph(path.clone(), graph.clone());
        }
        let trait_impls: Vec<TraitImplInfo> = parsed_files
            .iter()
            .flat_map(|p| {
                parse::extract_trait_impls(
                    p,
                    &metadata.struct_fields,
                    &metadata.fn_return_types,
                    &metadata.known_struct_names,
                )
                .expect("trait impls must lower")
            })
            .collect();
        for impl_info in &trait_impls {
            let impl_type = impl_info
                .self_ty_root
                .as_deref()
                .unwrap_or(&impl_info.for_type);
            let trait_root = if impl_info.trait_name.is_empty() {
                None
            } else {
                Some(impl_info.trait_name.as_str())
            };
            for method in &impl_info.methods {
                if let Some(graph) = &method.graph {
                    call_control.register_trait_method(
                        &method.name,
                        trait_root,
                        impl_type,
                        graph.clone(),
                    );
                }
            }
        }
        call_control.find_all_graphs_for_tests();

        // Get opcode_load_fast_checked graph and inline it
        let path = parse::CallPath::from_segments(["opcode_load_fast_checked"]);
        let graph = function_graphs.get(&path);
        assert!(
            graph.is_some(),
            "opcode_load_fast_checked should exist in function_graphs"
        );
        let mut graph = graph.unwrap().clone();

        let pre_inline_blocks = graph.blocks.len();
        let inlined = inline::inline_graph(&mut graph, &call_control, 3);

        eprintln!("=== Inline Integration Test ===");
        eprintln!(
            "  opcode_load_fast_checked: {pre_inline_blocks} blocks → {} blocks, {inlined} call sites inlined",
            graph.blocks.len()
        );
        for block in &graph.blocks {
            for op in &block.operations {
                eprintln!("    {:?}", op.kind);
            }
        }

        // inline.rs is a graph utility, NOT part of the RPython-orthodox
        // pipeline. Method calls are now correctly Residual (not auto-Regular),
        // so fewer call sites may be inlined. This is expected.
        eprintln!("  inlined count: {inlined}");

        // Check if any low-level ops emerged from inlining FunctionPath calls
        let all_ops: Vec<_> = graph.blocks.iter().flat_map(|b| &b.operations).collect();
        let has_low_level = all_ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { .. }
                    | OpKind::ArrayRead { .. }
                    | OpKind::ArrayWrite { .. }
                    | OpKind::FieldWrite { .. }
            )
        });
        eprintln!("  has low-level ops after inline: {has_low_level}");
    }
}

/// `rlib` — Rust port of `rpython/rlib/` helpers pulled in on demand.
/// Currently only the pieces required by the annotator port are
/// present (rarithmetic subset for `compute_restype`).
pub mod rlib;

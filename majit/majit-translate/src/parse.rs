//! Source parsing: extract opcode dispatch and trait impls.

use crate::front::opcode_wrapper::{RaiseStubReturn, WrappedUnitVariantReturn};
use crate::{MethodInfo, TraitImplInfo};
use serde::{Deserialize, Serialize};
use syn::{ExprMatch, File, Item, ItemFn, Pat, Path, visit::Visit};

/// Raw opcode-dispatch arm extracted from the interpreter match.
///
/// This is the canonical parse/front-end view of opcode dispatch before
/// graph/pipeline classification is attached.
#[derive(Debug, Clone)]
pub struct ExtractedOpcodeArm {
    pub selector: OpcodeDispatchSelector,
    pub handler_calls: Vec<ExtractedHandlerCall>,
    /// Semantic graph of the match arm body.
    /// This is the handler's own graph — the primary input for
    /// jtransform/flatten. handler_calls are metadata only.
    pub body_graph: Option<crate::model::FunctionGraph>,
    /// Set when the arm body is a single tail-call to a lifted
    /// per-opcode handler free fn (`execute_<op>(dispatcher params)`).
    /// In that case `body_graph` is the mechanically synthesized
    /// dispatcher-shaped wrapper, and this records the handler's
    /// [`CallPath`] — the seam that lets the JIT resolve the Charon/MIR
    /// handler graph by name instead of re-lowering the arm body.
    pub mir_handler_path: Option<CallPath>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallPath {
    pub segments: Vec<String>,
}

impl CallPath {
    pub fn from_segments<I, S>(segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            segments: segments.into_iter().map(Into::into).collect(),
        }
    }

    /// Build the canonical CallPath for an inherent / trait-impl method.
    ///
    /// `impl_type_joined` may be a single segment (`"Foo"`) or a
    /// `::`-joined type path (`"a::Foo"`, `"mod::Outer::Inner"`). The
    /// impl_type is split into its individual segments and concatenated
    /// with the method name so that the resulting CallPath is uniform
    /// with free-fn paths (`["a", "b", "f"]`) — both the
    /// type-qualified prefix and the method name live at the same
    /// segment granularity. Previously impl methods were stored as
    /// 2-segment `[impl_type_joined, method]`, which diverged from
    /// free-fn shape and forced macro-side heuristics; this form
    /// restores uniformity (RPython parity: `getfunctionptr(graph)` is
    /// string-free and does not distinguish the two shapes
    /// `rpython/jit/codewriter/call.py:174-187`).
    ///
    // Structural adaptation: Rust `::` ↔ PyPy `.` path separator.
    // `impl_type_joined` may arrive in either spelling — Rust extraction
    // emits `module::Type`, while `ClassDef.name` mirrors classdesc.py
    // `cls.__module__ + '.' + cls.__name__` (a `.`-joined `module.Class`).
    // Split on both so the segment granularity is independent of which
    // caller minted the string: callers in lib.rs / call.rs /
    // codewriter.rs do not all route through a `.`→`::` normalization
    // boundary, so accepting both keeps the invariant statically true.
    pub fn for_impl_method(impl_type_joined: &str, method: &str) -> Self {
        let mut segments: Vec<String> = impl_type_joined
            .split("::")
            .flat_map(|s| s.split('.'))
            .filter(|seg| !seg.is_empty())
            .map(|seg| seg.to_string())
            .collect();
        segments.push(method.to_string());
        Self { segments }
    }

    pub fn last_segment(&self) -> Option<&str> {
        self.segments.last().map(String::as_str)
    }

    pub fn canonical_key(&self) -> String {
        self.segments.join("::")
    }

    /// For a path built by `for_impl_method`, extract the impl type
    /// portion (all segments except the trailing method name).
    pub fn impl_type_prefix(&self) -> String {
        if self.segments.len() >= 2 {
            self.segments[..self.segments.len() - 1].join("::")
        } else {
            self.segments.join("::")
        }
    }
}

/// Strip the module prefix and return the trailing identifier.
///
/// Accepts both spellings: a `::`-joined Rust path and the `.`-joined
/// `ClassDef.name` form (classdesc.py `cls.__module__ + '.' +
/// cls.__name__`). A plain `rsplit('.')` misses Rust-rooted values and a
/// plain `rsplit("::")` misses Python-rooted values, so strip the longer
/// `::` prefix first and then any residual `.` prefix — the final
/// identifier is recovered regardless of which separator the caller used.
pub fn canonical_leaf(name: &str) -> &str {
    let after_colon = name.rsplit("::").next().unwrap_or(name);
    after_colon.rsplit('.').next().unwrap_or(after_colon)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpcodeDispatchSelector {
    Path(CallPath),
    Wildcard,
    Or(Vec<OpcodeDispatchSelector>),
    Unsupported,
}

impl OpcodeDispatchSelector {
    pub fn canonical_key(&self) -> String {
        match self {
            Self::Path(path) => path.canonical_key(),
            Self::Wildcard => "_".into(),
            Self::Or(cases) => cases
                .iter()
                .map(Self::canonical_key)
                .collect::<Vec<_>>()
                .join(" | "),
            Self::Unsupported => "<unsupported>".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractedHandlerCall {
    Method {
        name: String,
        receiver_root: Option<String>,
    },
    FunctionPath(CallPath),
    UnsupportedFunctionExpr,
}

#[derive(Debug, Clone)]
pub struct InherentMethodInfo {
    pub for_type: String,
    pub self_ty_root: Option<String>,
    pub name: String,
    pub graph: crate::model::FunctionGraph,
    /// RPython: op.result.concretetype — return type for array identity.
    pub return_type: Option<String>,
    /// RPython: function-level JIT hints (elidable, close_stack, etc.).
    pub hints: Vec<String>,
}

/// Parsed representation of an interpreter source file.
pub struct ParsedInterpreter {
    pub file: File,
    /// Crate-stripped module path of this source file
    /// (e.g. `"intobject"` for `pyre_object/src/intobject.rs`).
    /// Empty when the caller did not supply one — top-level items
    /// remain at simple-name registration.
    pub module_path: String,
    /// `use` declarations resolved into an alias → fully-qualified-path
    /// table, populated by [`collect_use_imports`].  Mirrors PyPy's
    /// `annotator.bookkeeper` import-resolution step: when the AST
    /// references a bare type name `Foo` that this file pulled in via
    /// `use other_mod::Foo;` (or `use other_mod::Foo as Q;`), the
    /// canonical fully-qualified path lives under the in-scope alias
    /// here so `qualify_to_canonical_struct` can resolve cross-module
    /// type identity without re-walking the source tree.
    pub use_imports: std::collections::HashMap<String, String>,
    /// `pub use <path>::*` glob re-exports at the file root.  Each
    /// entry is the source-path segments (after `crate::` /
    /// `self::` / `super::` are stripped), without the trailing
    /// glob.  Example: `pub use crate::objspace::descroperation::*;`
    /// in `baseobjspace.rs` (module path `"baseobjspace"`) records
    /// `["objspace", "descroperation"]`.  Consumed at
    /// `lib.rs::analyze_pipeline_from_parsed` to emit extra alias
    /// paths so callers writing `crate::baseobjspace::pos(...)`
    /// (Rust-resolved through the re-export) resolve to the same
    /// function graph the walker registered under
    /// `crate::objspace::descroperation::pos`.  Non-pub `use` globs
    /// AND pub uses of non-`crate::` paths (external crates, stdlib)
    /// are skipped.
    pub pub_use_globs: Vec<Vec<String>>,
    /// Non-`pub` `use <path>::*` glob imports at the file root.  Each
    /// entry is the source-path segments after `crate::` / `self::` /
    /// pyre-internal-crate-alias roots are stripped, without the
    /// trailing glob.  Example: `use crate::pyobject::*;` in
    /// `excobject.rs` records `["pyobject"]`.  Consumed at semantic
    /// build time by
    /// [`crate::front::ast::build_semantic_program_from_parsed_files_with_options`]
    /// which expands each glob into explicit `(alias → full_path)`
    /// entries on the per-file `use_imports` map — mirroring Python's
    /// import-resolution step that binds glob-imported names into the
    /// importing module's namespace at module-load time.  The
    /// front-end `Expr::Path` arm then resolves bare names through
    /// the primary `use_imports` lookup without a separate glob
    /// fallback.
    pub use_globs: Vec<Vec<String>>,
}

pub fn parse_source(source: &str) -> ParsedInterpreter {
    let file = syn::parse_file(source).expect("failed to parse bundled source");
    let use_imports = collect_use_imports(&file.items);
    let pub_use_globs = collect_pub_use_globs(&file.items);
    let use_globs = collect_use_globs(&file.items);
    ParsedInterpreter {
        file,
        module_path: String::new(),
        use_imports,
        pub_use_globs,
        use_globs,
    }
}

/// Parse a bundled Rust source file with its crate-stripped module
/// path.  e.g. `parse_source_with_module(src, "intobject")` for
/// `pyre_object/src/intobject.rs` — aligns analyzer-side
/// `path_hash(canonical_struct_name)` with the runtime's
/// dual-published `path_hash(strip_crate(module_path!())::Name)` slot
/// in `gc_cache._cache_size` (PyPy `cache[STRUCT]` lltype-object
/// identity, descr.py:108-118).
pub fn parse_source_with_module(source: &str, module_path: &str) -> ParsedInterpreter {
    let file = syn::parse_file(source).expect("failed to parse bundled source");
    let use_imports = collect_use_imports(&file.items);
    let pub_use_globs = collect_pub_use_globs(&file.items);
    let use_globs = collect_use_globs(&file.items);
    ParsedInterpreter {
        file,
        module_path: module_path.to_string(),
        use_imports,
        pub_use_globs,
        use_globs,
    }
}

/// Walk every `Item::Use` at the file root and recursively expand the
/// use tree into an `{alias → full_path}` table.
///
/// Handles `UseTree::Path`, `UseTree::Name`, `UseTree::Rename` (`use X
/// as Y`), and `UseTree::Group` (`use X::{A, B}`).  `UseTree::Glob`
/// (`use X::*`) is recorded as a no-op: pyre cannot resolve glob
/// exports without re-parsing the target module, so glob-imported
/// bare names fall back to the same-module-default qualification.
///
/// Restricted to file-root use statements: PyPy's resolver also only
/// honours module-level imports (`annrpython.py` bookkeeper); function-
/// local `use` clauses are out of scope.
pub(crate) fn collect_use_imports(items: &[Item]) -> std::collections::HashMap<String, String> {
    let mut imports = std::collections::HashMap::new();
    for item in items {
        if let Item::Use(u) = item {
            walk_use_tree(&u.tree, &mut Vec::new(), &mut imports);
        }
    }
    imports
}

/// Walk every file-root `pub use <path>::*;` and collect the source
/// path segments (after `crate::` / `self::` are stripped, with
/// internal pyre-crate roots also stripped to mirror the rest of the
/// analyzer's namespace normalisation).  Non-`pub` use globs are
/// excluded — those are private imports only the file itself can
/// resolve through, never visible to external callers.
///
/// `pub use foo::*` (no `crate::`) and globs rooted in external
/// crates (`std`, `core`, `alloc`, well-known external workspace
/// crates) are also skipped: pyre cannot synthesise sensible aliases
/// for them without re-walking the target module, and the function-
/// registry only carries pyre-source paths in the first place.
pub(crate) fn collect_pub_use_globs(items: &[Item]) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for item in items {
        let Item::Use(u) = item else {
            continue;
        };
        if !matches!(u.vis, syn::Visibility::Public(_)) {
            continue;
        }
        walk_pub_use_for_globs(&u.tree, &mut Vec::new(), &mut out);
    }
    out
}

/// Walk every file-root `use <path>::*` (non-`pub`) statement and
/// collect the source-path segments (after `crate::` / `self::` /
/// pyre-internal-crate-alias roots are stripped).  Mirrors
/// [`collect_pub_use_globs`] but for plain `use` instead of `pub use`.
/// The result is stored on [`ParsedInterpreter::use_globs`] and
/// consumed by `front::ast::build_semantic_program_*_with_options`,
/// which expands each glob root into explicit `use_imports` entries
/// at semantic build time.
pub(crate) fn collect_use_globs(items: &[Item]) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for item in items {
        let Item::Use(u) = item else {
            continue;
        };
        if matches!(u.vis, syn::Visibility::Public(_)) {
            continue;
        }
        walk_pub_use_for_globs(&u.tree, &mut Vec::new(), &mut out);
    }
    out
}

fn walk_pub_use_for_globs(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    out: &mut Vec<Vec<String>>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            walk_pub_use_for_globs(&p.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Glob(_) => {
            let stripped = strip_glob_root(prefix);
            if !stripped.is_empty() {
                out.push(stripped);
            }
        }
        syn::UseTree::Group(g) => {
            for sub in &g.items {
                walk_pub_use_for_globs(sub, prefix, out);
            }
        }
        syn::UseTree::Name(_) | syn::UseTree::Rename(_) => {
            // Named re-exports (`pub use crate::M::name`) are handled
            // by the regular `walk_use_tree` namespace machinery via
            // `use_imports` — only Glob entries need the per-source
            // alias-fan-out at registration time.
        }
    }
}

/// Strip the leading namespace-root token (`crate`, `self`, or a
/// well-known pyre crate alias) so the stored segments match the
/// `func.module_path` shape used downstream (`pyre-interpreter/src/
/// foo.rs` → module path `"foo"`, segments `["foo"]`).
fn strip_glob_root(prefix: &[String]) -> Vec<String> {
    if prefix.is_empty() {
        return Vec::new();
    }
    let first = prefix[0].as_str();
    let is_local_root = matches!(first, "crate" | "self");
    let is_pyre_alias = PYRE_INTERNAL_CRATES.contains(&first);
    if is_local_root || is_pyre_alias {
        prefix[1..].iter().cloned().collect()
    } else {
        // External crates / stdlib — skip.  Returning empty signals
        // the caller to drop this entry.
        Vec::new()
    }
}

fn walk_use_tree(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    imports: &mut std::collections::HashMap<String, String>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            walk_use_tree(&p.tree, prefix, imports);
            prefix.pop();
        }
        syn::UseTree::Name(n) => {
            let alias = n.ident.to_string();
            prefix.push(alias.clone());
            imports.insert(alias, joined_use_path(prefix));
            prefix.pop();
        }
        syn::UseTree::Rename(r) => {
            prefix.push(r.ident.to_string());
            imports.insert(r.rename.to_string(), joined_use_path(prefix));
            prefix.pop();
        }
        syn::UseTree::Glob(_) => {
            // No exposed names — caller falls back to local-module qualification.
        }
        syn::UseTree::Group(g) => {
            for sub in &g.items {
                walk_use_tree(sub, prefix, imports);
            }
        }
    }
}

/// Crate-root names that the analyzer treats as the local namespace —
/// stripped from `use` paths the same way `crate::` is stripped.  This
/// list aligns the analyzer's `path_hash(canonical_struct_name)`
/// namespace with the runtime's `module_path!()`-stripped namespace and
/// keeps cross-crate impl-method receiver spelling identical to the
/// crate-stripped `module_path_from_source_file` form used by the
/// production `analyze_multiple_pipeline_with_modules` entries.
pub(crate) const PYRE_INTERNAL_CRATES: &[&str] = &[
    "pyre_interpreter",
    "pyre_jit",
    "pyre_jit_trace",
    "pyre_object",
    "majit_ir",
    "majit_metainterp",
    "majit_translate",
    "majit_gc",
    "majit_backend_dynasm",
    "majit_backend_cranelift",
];

/// Join the accumulated `use` path segments and drop the leading
/// `crate::` keyword (or any analyzer-internal crate root in
/// [`PYRE_INTERNAL_CRATES`]) when present.  Runtime `#[jit_struct]`
/// hashes types through `majit_ir::descr::path_hash_stripped_crate`,
/// which strips the leading `module_path!()` segment (the crate root)
/// before hashing.  Analyzer-side `path_hash` must see the same
/// namespace, so the `crate::` syntactic marker (and the equivalent
/// crate-root segment for cross-crate `use foo_crate::bar::T` imports
/// inside the analyzer's source set) is dropped here at collection
/// time rather than at every consumer.  `use other_crate::Foo` paths
/// from crates outside the analyzer's source set are kept verbatim.
fn joined_use_path(segments: &[String]) -> String {
    if let Some(first) = segments.first().map(String::as_str) {
        if first == "crate" || PYRE_INTERNAL_CRATES.contains(&first) {
            return segments[1..].join("::");
        }
    }
    segments.join("::")
}

/// Find a top-level function by exact name in the parsed source.
pub(crate) fn find_function<'a>(parsed: &'a ParsedInterpreter, name: &str) -> Option<&'a ItemFn> {
    find_function_in_file(&parsed.file, name)
}

/// Find a top-level function by exact name in a parsed file.
pub(crate) fn find_function_in_file<'a>(file: &'a File, name: &str) -> Option<&'a ItemFn> {
    file.items.iter().find_map(|item| {
        if let Item::Fn(func) = item {
            (func.sig.ident == name).then_some(func)
        } else {
            None
        }
    })
}

/// Find an opcode-dispatch `match` expression within a function.
fn find_opcode_match(func: &ItemFn) -> Option<&ExprMatch> {
    struct Finder<'a> {
        result: Option<&'a ExprMatch>,
    }

    impl<'ast> Visit<'ast> for Finder<'ast> {
        fn visit_expr_match(&mut self, node: &'ast ExprMatch) {
            if self.result.is_none() && node.arms.first().is_some_and(is_opcode_pattern) {
                self.result = Some(node);
                return;
            }
            syn::visit::visit_expr_match(self, node);
        }
    }

    let mut finder = Finder { result: None };
    finder.visit_item_fn(func);
    finder.result
}

/// Extract trait implementations AND trait default methods from the parsed source.
/// Recurses into `Item::Mod` for whole-program visibility (RPython parity).
///
/// RPython `flowspace/objspace.py:49` + `flowcontext.py:417` —
/// `build_flow()` / `buildflowgraph()` re-raise `FlowingError`, and the
/// translator observes unsupported constructs as a hard failure.  The
/// extractor mirrors that: if any trait method's body hits an
/// unsupported construct, the whole extraction aborts with Err rather
/// than silently recording a `MethodInfo.graph = None` that dispatch
/// could later route through without a semantic graph.
pub fn extract_trait_impls(
    parsed: &ParsedInterpreter,
    struct_fields: &crate::front::StructFieldRegistry,
    known_struct_names: &std::collections::HashSet<String>,
    mir_graphs: Option<&crate::front::semantic::MirGraphLookup>,
) -> Result<Vec<TraitImplInfo>, crate::front::semantic::FlowingError> {
    let mut impls = Vec::new();
    let mut known_trait_names = std::collections::HashSet::new();
    collect_trait_names(&parsed.file.items, "", &mut known_trait_names);
    collect_trait_impls_from_items(
        &parsed.file.items,
        "",
        struct_fields,
        &parsed.use_imports,
        known_struct_names,
        &known_trait_names,
        mir_graphs,
        &mut impls,
    )?;
    Ok(impls)
}

fn collect_trait_impls_from_items(
    items: &[Item],
    prefix: &str,
    struct_fields: &crate::front::StructFieldRegistry,
    use_imports: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
    mir_graphs: Option<&crate::front::semantic::MirGraphLookup>,
    impls: &mut Vec<TraitImplInfo>,
) -> Result<(), crate::front::semantic::FlowingError> {
    for item in items {
        match item {
            // Concrete trait impls (impl Trait for Type)
            Item::Impl(impl_block) => {
                if let Some((_, trait_path, _)) = &impl_block.trait_ {
                    let trait_name =
                        canonical_trait_path_name(trait_path, prefix, known_trait_names);
                    let self_ty = &impl_block.self_ty;
                    let for_type = canonical_type_name(self_ty);
                    // Qualify bare type name with module prefix (RPython: unique type identity).
                    // Route through `qualify_type_name_with_imports` with the same
                    // `parsed.use_imports` map graph build threads into
                    // `GraphBuildContext` so trait-impl registration keys align
                    // with use-site lookups when the receiver type is referenced
                    // via a `use <path> as alias` form.
                    let self_ty_root = type_root_ident(self_ty).map(|t| {
                        crate::front::semantic::qualify_type_name_with_imports(
                            &t,
                            prefix,
                            use_imports,
                        )
                    });
                    let mut methods: Vec<MethodInfo> = Vec::new();
                    for item in &impl_block.items {
                        if let syn::ImplItem::Fn(method) = item {
                            let fake_fn = syn::ItemFn {
                                attrs: method.attrs.clone(),
                                vis: syn::Visibility::Inherited,
                                sig: method.sig.clone(),
                                block: Box::new(method.block.clone()),
                            };
                            // jit.py:184-201 — `@elidable_promote` on a
                            // trait-impl method installs two callables
                            // (orig + wrapper); `synthesize_or_passthrough`
                            // makes this lowering pass see both.  Method
                            // signature parity (return type, hints) is
                            // taken from the synthesized `ItemFn`, not the
                            // original `method`, so the orig's
                            // `_orig_<NAME>_unlikely_name` identity
                            // survives downstream.
                            // `?` propagates `FlowingError` out of the
                            // extractor (RPython re-raise at
                            // `flowspace/flowcontext.py:417`).  The
                            // qualified `self_ty_root` is threaded into
                            // synthesis so the wrapper's tail call uses
                            // the same `<ImplType>::_orig_<name>_unlikely_name`
                            // path that `lib.rs:531-537` registers via
                            // `CallPath::for_impl_method`.
                            for synth in crate::front::syn_metadata::synthesize_or_passthrough(
                                fake_fn,
                                self_ty_root.as_deref(),
                            ) {
                                let method_name = synth.sig.ident.to_string();
                                let return_type = match &synth.sig.output {
                                    syn::ReturnType::Type(_, ty) => {
                                        crate::front::syn_metadata::qualified_full_type_string(
                                            ty,
                                            prefix,
                                            known_struct_names,
                                            known_trait_names,
                                        )
                                    }
                                    syn::ReturnType::Default => Some("()".to_string()),
                                };
                                // Slice 3.C/3.F/5: take MIR's graph when
                                // the lookup hits; otherwise emit
                                // `graph: None`.  The AST graph builder
                                // fallback is retired (Slice 5).
                                let mir_hit = mir_graphs.and_then(|m| {
                                    self_ty_root
                                        .as_deref()
                                        .and_then(|owner| m.lookup_impl_method(owner, &method_name))
                                });
                                let hints = crate::front::syn_metadata::collect_jit_hints_with_sig(
                                    &synth.attrs,
                                    &synth.sig,
                                );
                                let graph = mir_hit.cloned();
                                methods.push(MethodInfo {
                                    name: method_name,
                                    graph,
                                    return_type,
                                    hints,
                                });
                            }
                        }
                    }
                    impls.push(TraitImplInfo {
                        trait_name,
                        for_type,
                        self_ty_root,
                        methods,
                    });
                }
            }
            // Trait definitions with default methods
            Item::Trait(trait_def) => {
                let trait_name = crate::front::syn_metadata::qualify_known_trait_name(
                    &trait_def.ident.to_string(),
                    prefix,
                    known_trait_names,
                );
                let mut methods: Vec<MethodInfo> = Vec::new();
                for item in &trait_def.items {
                    if let syn::TraitItem::Fn(method) = item {
                        if let Some(block) = &method.default {
                            let fake_fn = syn::ItemFn {
                                attrs: method.attrs.clone(),
                                vis: syn::Visibility::Inherited,
                                sig: method.sig.clone(),
                                block: Box::new(block.clone()),
                            };
                            // jit.py:184-201 — trait default methods get
                            // the same wrapper/orig synthesis.  The
                            // concrete `Self` type is not known until
                            // a `for <T>` impl resolves the trait, so
                            // the synthesizer emits a bare-path tail
                            // call (`None`); a downstream `impl Trait
                            // for S` block would re-emit the method
                            // with `self_ty_root = "S"` and lower the
                            // wrapper through this same path.
                            for synth in
                                crate::front::syn_metadata::synthesize_or_passthrough(fake_fn, None)
                            {
                                let method_name = synth.sig.ident.to_string();
                                let return_type = match &synth.sig.output {
                                    syn::ReturnType::Type(_, ty) => {
                                        crate::front::syn_metadata::qualified_full_type_string(
                                            ty,
                                            prefix,
                                            known_struct_names,
                                            known_trait_names,
                                        )
                                    }
                                    syn::ReturnType::Default => Some("()".to_string()),
                                };
                                // Slice 3.C/3.F/5: take MIR's trait-default
                                // graph; emit `graph: None` on miss.  The
                                // AST graph builder fallback is retired.
                                let mir_hit = mir_graphs.and_then(|m| {
                                    m.lookup_trait_default(&trait_name, &method_name)
                                });
                                let hints = crate::front::syn_metadata::collect_jit_hints_with_sig(
                                    &synth.attrs,
                                    &synth.sig,
                                );
                                let graph = mir_hit.cloned();
                                methods.push(MethodInfo {
                                    name: method_name,
                                    graph,
                                    return_type,
                                    hints,
                                });
                            }
                        }
                    }
                }
                if !methods.is_empty() {
                    impls.push(TraitImplInfo {
                        trait_name: trait_name.clone(),
                        for_type: format!("<default methods of {}>", trait_name),
                        self_ty_root: None,
                        methods,
                    });
                }
            }
            // Recurse into module blocks with qualified prefix.
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_trait_impls_from_items(
                        sub_items,
                        &mod_prefix,
                        struct_fields,
                        use_imports,
                        known_struct_names,
                        known_trait_names,
                        mir_graphs,
                        impls,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn collect_trait_names(
    items: &[Item],
    prefix: &str,
    known_trait_names: &mut std::collections::HashSet<String>,
) {
    for item in items {
        match item {
            Item::Trait(trait_def) => {
                let bare_name = trait_def.ident.to_string();
                known_trait_names.insert(bare_name.clone());
                if !prefix.is_empty() {
                    known_trait_names.insert(format!("{}::{}", prefix, bare_name));
                }
            }
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_trait_names(sub_items, &mod_prefix, known_trait_names);
                }
            }
            _ => {}
        }
    }
}

/// Extract inherent impl methods (impl Type { ... }) as canonical call targets.
/// Recurses into `Item::Mod` for whole-program visibility (RPython parity).
///
/// RPython `flowspace/objspace.py:49` — `build_flow()` re-raises
/// `FlowingError` rather than silently skipping a function.  Pyre's
/// inherent-method extractor mirrors that: `FlowingError` in any
/// method body aborts the whole extraction so dispatch resolution
/// never sees a silently-dropped method.
pub fn extract_inherent_impl_methods(
    parsed: &ParsedInterpreter,
    struct_fields: &crate::front::StructFieldRegistry,
    known_struct_names: &std::collections::HashSet<String>,
    mir_graphs: Option<&crate::front::semantic::MirGraphLookup>,
) -> Result<Vec<InherentMethodInfo>, crate::front::semantic::FlowingError> {
    let mut methods = Vec::new();
    // Feed `parsed.module_path` so the inherent-impl receiver-root
    // qualification agrees with the caller-side
    // `qualify_type_name_with_imports` result, which `analyze_pipeline_from_parsed`
    // routes through `STRUCT_ORIGIN_REGISTRY` populated by
    // `collect_struct_origins` over the same module path.  Empty
    // `module_path` (fixtures using `parse_source`) falls through to
    // the bare-name registration path.
    collect_inherent_methods_from_items(
        &parsed.file.items,
        &parsed.module_path,
        struct_fields,
        &parsed.use_imports,
        known_struct_names,
        mir_graphs,
        &mut methods,
    )?;
    Ok(methods)
}

fn collect_inherent_methods_from_items(
    items: &[Item],
    prefix: &str,
    struct_fields: &crate::front::StructFieldRegistry,
    use_imports: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    mir_graphs: Option<&crate::front::semantic::MirGraphLookup>,
    methods: &mut Vec<InherentMethodInfo>,
) -> Result<(), crate::front::semantic::FlowingError> {
    for item in items {
        match item {
            Item::Impl(impl_block) => {
                if impl_block.trait_.is_some() {
                    continue;
                }
                let for_type = canonical_type_name(&impl_block.self_ty);
                // Two `self_ty_root` forms run side by side:
                //  - `self_ty_root_bare` keeps the raw `impl` type ident
                //    (e.g. `"PyFrame"`) and is fed into the graph build so
                //    inside-the-graph `self.field` accesses carry the
                //    same `owner_root` spelling the virtualizable spec
                //    (`virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT`,
                //    plain `"PyFrame"`) matches against
                //    (`jit_codewriter/jtransform.rs::VirtualizableFieldDescriptor::matches`).
                //  - `self_ty_root_qualified` runs the bare name through
                //    `qualify_type_name_with_imports` so the inherent-method
                //    registration `CallPath::for_impl_method` agrees with
                //    the caller-side receiver-type spelling
                //    (`receiver_type_root` → `local_type_roots`, also fed
                //    through `qualify_type_name_with_imports`).  PyPy
                //    `bookkeeper.getdesc(value).graph` single-source
                //    identity uses the same lookup at both ends.
                let self_ty_root_bare = type_root_ident(&impl_block.self_ty);
                let self_ty_root_qualified = self_ty_root_bare.as_ref().map(|t| {
                    crate::front::semantic::qualify_type_name_with_imports(t, prefix, use_imports)
                });
                for sub in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = sub {
                        let fake_fn = syn::ItemFn {
                            attrs: method.attrs.clone(),
                            vis: syn::Visibility::Inherited,
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        // jit.py:184-201 — inherent-impl methods get the
                        // same wrapper/orig synthesis as free fns;
                        // `?` propagates `FlowingError` per
                        // `flowspace/flowcontext.py:417`.  `synthesize_or_passthrough`
                        // and `build_function_graph_with_self_ty_pub`
                        // both receive the bare spelling so the wrapper's
                        // self-typed tail call and the body's `self.field`
                        // accesses share the same `owner_root` spelling
                        // the vable spec asserts on.
                        for synth in crate::front::syn_metadata::synthesize_or_passthrough(
                            fake_fn,
                            self_ty_root_bare.as_deref(),
                        ) {
                            let method_name = synth.sig.ident.to_string();
                            let return_type = match &synth.sig.output {
                                syn::ReturnType::Type(_, ty) => {
                                    crate::front::syn_metadata::qualified_full_type_string(
                                        ty,
                                        prefix,
                                        known_struct_names,
                                        &std::collections::HashSet::new(),
                                    )
                                }
                                syn::ReturnType::Default => Some("()".to_string()),
                            };
                            // Slice 3.C/3.F/5: take MIR's inherent-method
                            // graph; skip the entry on miss (the AST
                            // graph builder fallback is retired, and
                            // InherentMethodInfo.graph is non-Option).
                            let mir_hit = mir_graphs.and_then(|m| {
                                self_ty_root_qualified
                                    .as_deref()
                                    .and_then(|owner| m.lookup_impl_method(owner, &method_name))
                            });
                            let graph = match mir_hit {
                                Some(g) => g.clone(),
                                None => continue,
                            };
                            let hints = crate::front::syn_metadata::collect_jit_hints_with_sig(
                                &synth.attrs,
                                &synth.sig,
                            );
                            methods.push(InherentMethodInfo {
                                for_type: for_type.clone(),
                                self_ty_root: self_ty_root_qualified.clone(),
                                name: method_name,
                                graph,
                                return_type,
                                hints,
                            });
                        }
                    }
                }
            }
            // Recurse into module blocks with qualified prefix.
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_inherent_methods_from_items(
                        sub_items,
                        &mod_prefix,
                        struct_fields,
                        use_imports,
                        known_struct_names,
                        mir_graphs,
                        methods,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Extract canonical opcode dispatch arms from `execute_opcode_step`.
///
/// This preserves source-level match structure and handler calls so canonical
/// graph/pipeline consumers can resolve and classify these arms directly.
///
/// Duplicate opcode selectors are rejected. Silently keeping the first arm
/// would hide dispatch drift in the interpreter source.
pub fn extract_opcode_dispatch_arms(parsed: &ParsedInterpreter) -> Vec<ExtractedOpcodeArm> {
    let Some(func) = find_function(parsed, "execute_opcode_step") else {
        return Vec::new();
    };
    let Some(opcode_match) = find_opcode_match(func) else {
        return Vec::new();
    };
    reject_duplicate_opcode_selectors(extract_match_arms(opcode_match, &func.sig))
}

/// Extract opcode-dispatch arms from the canonical dispatch match only.
///
/// RPython `flowspace/objspace.py:49` + `flowcontext.py:417` —
/// `FlowingError` propagates out of `build_flow()`, making unsupported
/// constructs a hard failure.  Pyre's dispatch extractor mirrors that:
/// an arm body that hits an unsupported construct aborts the walk with
/// a panic rather than silently dropping the arm's graph.  Silently
/// dropping would let a `PipelineOpcodeArm` reach the codewriter
/// without the semantic graph the downstream jitcode path depends on.
fn extract_match_arms(expr: &ExprMatch, sig: &syn::Signature) -> Vec<ExtractedOpcodeArm> {
    expr.arms
        .iter()
        .map(|arm| {
            let handler_calls = extract_handler_calls(&arm.body);
            let selector = extract_opcode_dispatch_selector(&arm.pat);
            let name = selector.canonical_key();
            // Seam: an arm whose body is a single tail-call to a lifted
            // per-opcode handler free fn (`execute_<op>(dispatcher
            // params)`) gets a mechanically synthesized
            // dispatcher-shaped wrapper instead of a syn-AST lowering, so
            // the JIT resolves the Charon/MIR handler graph by name.
            if let Some((handler_path, forwarded)) = detect_single_tail_call(&arm.body) {
                let graph = synthesize_tail_call_wrapper(&name, sig, &handler_path, &forwarded);
                return ExtractedOpcodeArm {
                    selector,
                    handler_calls,
                    body_graph: Some(graph),
                    mir_handler_path: Some(handler_path),
                };
            }
            // Seam: a constant-return arm (`Ok(StepResult::Continue)`, …) gets
            // a mechanically synthesized const-return wrapper instead of a
            // syn-AST lowering — same two-`Call` ctor chain, no graph builder.
            if let Some(ret) = detect_wrapped_unit_variant_return(&arm.body) {
                let graph = synthesize_wrapped_unit_variant_wrapper(&name, sig, &ret);
                return ExtractedOpcodeArm {
                    selector,
                    handler_calls,
                    body_graph: Some(graph),
                    mir_handler_path: None,
                };
            }
            // Seam: a raise-stub arm (`Err(PyError::type_error("…").into())`,
            // the async stubs) gets a mechanically synthesized abort+ctor
            // chain instead of a syn-AST lowering.
            if let Some(stub) = detect_raise_stub_return(&arm.body) {
                let graph = synthesize_raise_stub_wrapper(&name, sig, &stub);
                return ExtractedOpcodeArm {
                    selector,
                    handler_calls,
                    body_graph: Some(graph),
                    mir_handler_path: None,
                };
            }
            // No synthesizer seam matched this arm body (tail-call /
            // const-return / raise-stub).  With the syn-AST lowerer retired
            // there is no body graph to build, so the arm carries no JIT entry
            // and is dispatched by the interpreter.  Every real
            // `execute_opcode_step` arm matches a seam, so `body_graph: None`
            // arises only for synthetic inputs and is handled by the dispatch
            // registrar (`lib.rs` `arm.body_graph.map(..)`).
            ExtractedOpcodeArm {
                selector,
                handler_calls,
                body_graph: None,
                mir_handler_path: None,
            }
        })
        .collect()
}

/// Recognize an arm body of the form `execute_<op>(p0, p1, …)` where
/// every argument is a bare parameter identifier (a forwarded
/// dispatcher param).  Returns the handler `CallPath` and the forwarded
/// parameter names in call order.  A leading single-statement block
/// (`{ execute_<op>(…) }`) is unwrapped.  Anything else (method calls,
/// multi-statement bodies, non-`execute_` callees, non-ident args)
/// returns `None` and falls back to syn-AST lowering.
fn detect_single_tail_call(body: &syn::Expr) -> Option<(CallPath, Vec<String>)> {
    let expr = match body {
        syn::Expr::Block(block) if block.block.stmts.len() == 1 => match &block.block.stmts[0] {
            syn::Stmt::Expr(inner, _) => inner,
            _ => return None,
        },
        other => other,
    };
    let call = match expr {
        syn::Expr::Call(call) => call,
        _ => return None,
    };
    let func_path = match &*call.func {
        syn::Expr::Path(path) if path.qself.is_none() => &path.path,
        _ => return None,
    };
    let leaf = func_path.segments.last()?.ident.to_string();
    if !leaf.starts_with("execute_") {
        return None;
    }
    let mut forwarded = Vec::with_capacity(call.args.len());
    for arg in &call.args {
        match arg {
            syn::Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
                forwarded.push(path.path.segments[0].ident.to_string());
            }
            _ => return None,
        }
    }
    let segments = func_path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    Some((CallPath { segments }, forwarded))
}

fn path_owner_leaf(segments: &[String]) -> (Vec<String>, String) {
    let (leaf, owner) = segments
        .split_last()
        .expect("synthetic ctor path is non-empty");
    (owner.to_vec(), leaf.clone())
}

/// Recognize an arm body of the form `Wrapper(Owner::Variant)` where
/// `Wrapper` is a transparent result/option ctor (`Ok`/`Err`/`Some`)
/// and the single argument is a synthetic unit-variant path
/// (`StepResult::Continue`, …).  A leading single-statement block
/// (`{ Ok(StepResult::Continue) }`) is unwrapped.  Returns the
/// owner/leaf split of both ctors so the synthesizer can build the
/// two-`Call` chain directly.  Anything else (other call shapes, method
/// calls, non-allowlisted paths, multi-arg wrappers) returns `None`.
fn detect_wrapped_unit_variant_return(body: &syn::Expr) -> Option<WrappedUnitVariantReturn> {
    use crate::front::syn_metadata::{
        is_synthetic_result_option_wrapper_path, is_synthetic_unit_variant_path,
    };
    let expr = match body {
        syn::Expr::Block(block) if block.block.stmts.len() == 1 => match &block.block.stmts[0] {
            syn::Stmt::Expr(inner, _) => inner,
            _ => return None,
        },
        other => other,
    };
    let call = match expr {
        syn::Expr::Call(call) => call,
        _ => return None,
    };
    let wrapper_path = match &*call.func {
        syn::Expr::Path(path) if path.qself.is_none() => &path.path,
        _ => return None,
    };
    let wrapper_segments: Vec<String> = wrapper_path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    if !is_synthetic_result_option_wrapper_path(&wrapper_segments) || call.args.len() != 1 {
        return None;
    }
    let inner_path = match &call.args[0] {
        syn::Expr::Path(path) if path.qself.is_none() => &path.path,
        _ => return None,
    };
    let inner_segments: Vec<String> = inner_path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    if !is_synthetic_unit_variant_path(&inner_segments) {
        return None;
    }
    let (wrapper_owner, wrapper_name) = path_owner_leaf(&wrapper_segments);
    let (inner_owner, inner_name) = path_owner_leaf(&inner_segments);
    Some(WrappedUnitVariantReturn {
        wrapper_owner,
        wrapper_name,
        inner_owner,
        inner_name,
    })
}

/// Build the dispatcher-shaped wrapper graph for a single-tail-call arm
/// mechanically, with no syn-AST graph builder.  Shape: one
/// `OpKind::Input` per dispatcher parameter (in signature order, typed
/// via `classify_fn_arg_ty`) as the startblock inputargs, then one
/// `OpKind::Call` to `handler_path` forwarding the mapped inputarg vars
/// with `result_ty: Unknown`, then a return of the call result.  The
/// runtime seeds the dispatch entry from the full dispatcher register
/// layout, so the wrapper MUST keep every dispatcher param as an
/// inputarg even when the handler forwards only a subset.
/// Extract the dispatcher parameter prologue `(name, ValueType)` from a
/// syn signature, in signature order (`self` receiver typed via its
/// reference type, named params via [`canonical_pat_name`] +
/// [`classify_fn_arg_ty`]).  This is the only syn-touching step of arm
/// wrapper synthesis; the resulting prologue feeds the syn-free builders
/// in [`crate::front::opcode_wrapper`] that the MIR dispatch extractor
/// shares.
fn dispatcher_params_from_sig(sig: &syn::Signature) -> Vec<(String, crate::model::ValueType)> {
    sig.inputs
        .iter()
        .map(|param| match param {
            syn::FnArg::Receiver(recv) => (
                "self".to_string(),
                crate::front::syn_metadata::classify_fn_arg_ty(&recv.ty),
            ),
            syn::FnArg::Typed(pat_type) => (
                crate::front::syn_metadata::canonical_pat_name(&pat_type.pat),
                crate::front::syn_metadata::classify_fn_arg_ty(&pat_type.ty),
            ),
        })
        .collect()
}

fn synthesize_tail_call_wrapper(
    name: &str,
    sig: &syn::Signature,
    handler_path: &CallPath,
    forwarded: &[String],
) -> crate::model::FunctionGraph {
    crate::front::opcode_wrapper::build_tail_call_wrapper(
        name,
        &dispatcher_params_from_sig(sig),
        handler_path,
        forwarded,
    )
}

fn synthesize_wrapped_unit_variant_wrapper(
    name: &str,
    sig: &syn::Signature,
    ret: &WrappedUnitVariantReturn,
) -> crate::model::FunctionGraph {
    crate::front::opcode_wrapper::build_wrapped_unit_variant_wrapper(
        name,
        &dispatcher_params_from_sig(sig),
        ret,
    )
}

/// Recognize an arm body of the form `Wrapper(FnPath(<lit>).method())`
/// where `Wrapper` is a transparent result/option ctor (`Ok`/`Err`/`Some`)
/// and the single argument is a no-arg method call on a one-argument
/// function call whose argument is a string literal.  A leading
/// single-statement block is unwrapped.  This captures the async-stub
/// family `Err(crate::PyError::type_error("...").into())`; anything else
/// returns `None`.
fn detect_raise_stub_return(body: &syn::Expr) -> Option<RaiseStubReturn> {
    use crate::front::syn_metadata::is_synthetic_result_option_wrapper_path;
    use crate::model::UnsupportedLiteralKind;
    let expr = match body {
        syn::Expr::Block(block) if block.block.stmts.len() == 1 => match &block.block.stmts[0] {
            syn::Stmt::Expr(inner, _) => inner,
            _ => return None,
        },
        other => other,
    };
    let call = match expr {
        syn::Expr::Call(call) => call,
        _ => return None,
    };
    let wrapper_path = match &*call.func {
        syn::Expr::Path(path) if path.qself.is_none() => &path.path,
        _ => return None,
    };
    let wrapper_segments: Vec<String> = wrapper_path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    if !is_synthetic_result_option_wrapper_path(&wrapper_segments) || call.args.len() != 1 {
        return None;
    }
    // The wrapped argument must be `<func>(<lit>).<method>()` — a no-arg
    // method call on a single-argument function call.
    let method_call = match &call.args[0] {
        syn::Expr::MethodCall(mc) if mc.args.is_empty() => mc,
        _ => return None,
    };
    let inner_call = match &*method_call.receiver {
        syn::Expr::Call(c) => c,
        _ => return None,
    };
    let fn_path = match &*inner_call.func {
        syn::Expr::Path(path) if path.qself.is_none() => &path.path,
        _ => return None,
    };
    if inner_call.args.len() != 1 {
        return None;
    }
    let literal = match &inner_call.args[0] {
        syn::Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Str(_) => UnsupportedLiteralKind::Str,
            _ => return None,
        },
        _ => return None,
    };
    let fn_segments = fn_path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    let (wrapper_owner, wrapper_name) = path_owner_leaf(&wrapper_segments);
    Some(RaiseStubReturn {
        fn_segments,
        method_name: method_call.method.to_string(),
        wrapper_owner,
        wrapper_name,
        literal,
    })
}

fn synthesize_raise_stub_wrapper(
    name: &str,
    sig: &syn::Signature,
    stub: &RaiseStubReturn,
) -> crate::model::FunctionGraph {
    crate::front::opcode_wrapper::build_raise_stub_wrapper(
        name,
        &dispatcher_params_from_sig(sig),
        stub,
    )
}

fn reject_duplicate_opcode_selectors(arms: Vec<ExtractedOpcodeArm>) -> Vec<ExtractedOpcodeArm> {
    let mut seen = std::collections::HashMap::new();
    for (idx, arm) in arms.iter().enumerate() {
        let key = arm.selector.canonical_key();
        if let Some(first_idx) = seen.insert(key.clone(), idx) {
            panic!(
                "duplicate opcode dispatch selector `{key}` at arm {} and arm {}",
                first_idx + 1,
                idx + 1
            );
        }
    }
    arms
}

fn is_opcode_pattern(arm: &syn::Arm) -> bool {
    pattern_is_opcode_dispatch(&arm.pat)
}

fn pattern_is_opcode_dispatch(pat: &Pat) -> bool {
    match pat {
        Pat::Ident(pat) => pat.ident.to_string().starts_with("OP_"),
        Pat::Path(path) => path_is_opcode_dispatch(&path.path),
        Pat::Struct(pat) => path_is_opcode_dispatch(&pat.path),
        Pat::TupleStruct(pat) => path_is_opcode_dispatch(&pat.path),
        Pat::Tuple(pat) => pat.elems.iter().any(pattern_is_opcode_dispatch),
        Pat::Or(pat) => pat.cases.iter().any(pattern_is_opcode_dispatch),
        _ => false,
    }
}

fn path_is_opcode_dispatch(path: &Path) -> bool {
    let last = path
        .segments
        .last()
        .map(|segment| segment.ident.to_string());
    if let Some(last) = last {
        if last.starts_with("OP_") {
            return true;
        }
    }
    path.segments
        .iter()
        .any(|segment| segment.ident == "Instruction")
}

fn extract_opcode_dispatch_selector(pat: &Pat) -> OpcodeDispatchSelector {
    match pat {
        Pat::Ident(pat) => {
            OpcodeDispatchSelector::Path(CallPath::from_segments([pat.ident.to_string()]))
        }
        Pat::Path(path) => OpcodeDispatchSelector::Path(CallPath::from_segments(
            path.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        Pat::Struct(pat) => OpcodeDispatchSelector::Path(CallPath::from_segments(
            pat.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        Pat::TupleStruct(pat) => OpcodeDispatchSelector::Path(CallPath::from_segments(
            pat.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        Pat::Or(pat) => OpcodeDispatchSelector::Or(
            pat.cases
                .iter()
                .map(extract_opcode_dispatch_selector)
                .collect(),
        ),
        Pat::Wild(_) => OpcodeDispatchSelector::Wildcard,
        _ => OpcodeDispatchSelector::Unsupported,
    }
}

/// Extract handler call identities from an expression.
fn extract_handler_calls(expr: &syn::Expr) -> Vec<ExtractedHandlerCall> {
    let mut calls = Vec::new();
    let mut collector = CallCollector { calls: &mut calls };
    syn::visit::visit_expr(&mut collector, expr);
    calls
}

struct CallCollector<'a> {
    calls: &'a mut Vec<ExtractedHandlerCall>,
}

impl<'ast, 'a> Visit<'ast> for CallCollector<'a> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.calls.push(ExtractedHandlerCall::Method {
            name: call.method.to_string(),
            receiver_root: expr_root_ident(&call.receiver),
        });
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.calls.push(extract_function_call_identity(&call.func));
        syn::visit::visit_expr_call(self, call);
    }
}

fn extract_function_call_identity(expr: &syn::Expr) -> ExtractedHandlerCall {
    match expr {
        syn::Expr::Path(path) => ExtractedHandlerCall::FunctionPath(CallPath::from_segments(
            path.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        _ => ExtractedHandlerCall::UnsupportedFunctionExpr,
    }
}

fn expr_root_ident(expr: &syn::Expr) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path.path.get_ident().map(|ident| ident.to_string()),
        syn::Expr::Reference(r) => expr_root_ident(&r.expr),
        syn::Expr::Paren(p) => expr_root_ident(&p.expr),
        syn::Expr::Field(field) => expr_root_ident(&field.base),
        syn::Expr::Index(index) => expr_root_ident(&index.expr),
        _ => None,
    }
}

/// Returns full type path — all segments joined by "::".
/// RPython: lltype.Struct has globally unique identity.
fn type_root_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            let segments: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if segments.is_empty() {
                None
            } else {
                Some(segments.join("::"))
            }
        }
        syn::Type::Reference(reference) => type_root_ident(&reference.elem),
        syn::Type::Paren(paren) => type_root_ident(&paren.elem),
        syn::Type::Group(group) => type_root_ident(&group.elem),
        _ => None,
    }
}

fn canonical_path_name(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn canonical_trait_path_name(
    path: &syn::Path,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> String {
    let canonical = canonical_path_name(path);
    if path.segments.len() == 1 {
        crate::front::syn_metadata::qualify_known_trait_name(&canonical, prefix, known_trait_names)
    } else {
        canonical
    }
}

fn canonical_type_name(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(path) => canonical_path_name(&path.path),
        syn::Type::Reference(reference) => canonical_type_name(&reference.elem),
        syn::Type::Paren(paren) => canonical_type_name(&paren.elem),
        syn::Type::Group(group) => canonical_type_name(&group.elem),
        syn::Type::Ptr(ptr) => canonical_type_name(&ptr.elem),
        syn::Type::Slice(slice) => format!("[{}]", canonical_type_name(&slice.elem)),
        syn::Type::Array(array) => format!("[{}]", canonical_type_name(&array.elem)),
        _ => "<unsupported-type>".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_tail_call_arm_synthesizes_dispatcher_shaped_wrapper() {
        let body: syn::Expr = syn::parse_str("execute_pop_top(executor)").unwrap();
        let (path, forwarded) =
            detect_single_tail_call(&body).expect("single tail-call should be detected");
        assert_eq!(path.segments, vec!["execute_pop_top".to_string()]);
        assert_eq!(forwarded, vec!["executor".to_string()]);

        let sig: syn::Signature = syn::parse_str(
            "fn execute_opcode_step<E>(executor: &mut E, code: &CodeObject, instruction: Instruction, op_arg: OpArg, next_instr: usize) -> Result<StepResult, PyError>",
        )
        .unwrap();
        let graph = synthesize_tail_call_wrapper("PopTop", &sig, &path, &forwarded);

        let start = graph.block(graph.startblock);
        // All 5 dispatcher params are inputargs even though only
        // `executor` is forwarded (runtime seeds the full layout).
        assert_eq!(start.inputargs.len(), 5);
        // 5 Input ops + 1 Call op.
        assert_eq!(start.operations.len(), 6);
        match &start.operations[5].kind {
            crate::model::OpKind::Call {
                target,
                args,
                result_ty,
            } => {
                assert_eq!(*result_ty, crate::model::ValueType::Unknown);
                assert_eq!(args.len(), 1, "only the forwarded `executor` arg");
                match target {
                    crate::model::CallTarget::FunctionPath { segments } => {
                        assert_eq!(segments, &vec!["execute_pop_top".to_string()]);
                    }
                    other => panic!("expected FunctionPath, got {other:?}"),
                }
            }
            other => panic!("expected Call op as last operation, got {other:?}"),
        }

        // Non-matching shapes fall back to syn-AST lowering.
        let method_call: syn::Expr = syn::parse_str("executor.pop_top()").unwrap();
        assert!(detect_single_tail_call(&method_call).is_none());
        let multi_stmt: syn::Expr =
            syn::parse_str("{ let x = 1; execute_pop_top(executor) }").unwrap();
        assert!(detect_single_tail_call(&multi_stmt).is_none());
        let non_execute: syn::Expr = syn::parse_str("handle_pop_top(executor)").unwrap();
        assert!(detect_single_tail_call(&non_execute).is_none());
    }

    #[test]
    fn wrapped_unit_variant_arm_synthesizes_const_return_wrapper() {
        let body: syn::Expr = syn::parse_str("Ok(StepResult::Continue)").unwrap();
        let ret = detect_wrapped_unit_variant_return(&body)
            .expect("Ok(StepResult::Continue) should be detected");
        assert_eq!(ret.wrapper_name, "Ok");
        assert!(ret.wrapper_owner.is_empty());
        assert_eq!(ret.inner_name, "Continue");
        assert_eq!(ret.inner_owner, vec!["StepResult".to_string()]);

        let sig: syn::Signature = syn::parse_str(
            "fn execute_opcode_step<E>(executor: &mut E, code: &CodeObject, instruction: Instruction, op_arg: OpArg, next_instr: usize) -> Result<StepResult, PyError>",
        )
        .unwrap();
        let graph = synthesize_wrapped_unit_variant_wrapper("NoOp", &sig, &ret);

        let start = graph.block(graph.startblock);
        // All 5 dispatcher params are inputargs (runtime seeds the full
        // layout) even though the constant return reads none of them.
        assert_eq!(start.inputargs.len(), 5);
        // 5 Input ops + inner unit-variant ctor + outer wrapper ctor.
        assert_eq!(start.operations.len(), 7);
        // operations[5] = inner `StepResult::Continue` ctor (0 args).
        match &start.operations[5].kind {
            crate::model::OpKind::Call {
                target,
                args,
                result_ty,
            } => {
                assert_eq!(*result_ty, crate::model::ValueType::Unknown);
                assert!(args.is_empty(), "unit-variant ctor takes no args");
                match target {
                    crate::model::CallTarget::SyntheticTransparentCtor { name, owner_path } => {
                        assert_eq!(name, "Continue");
                        assert_eq!(owner_path, &vec!["StepResult".to_string()]);
                    }
                    other => panic!("expected SyntheticTransparentCtor, got {other:?}"),
                }
            }
            other => panic!("expected inner Call op, got {other:?}"),
        }
        // operations[6] = outer `Ok` wrapper ctor (1 arg).
        match &start.operations[6].kind {
            crate::model::OpKind::Call { target, args, .. } => {
                assert_eq!(args.len(), 1, "wrapper takes the inner ctor result");
                match target {
                    crate::model::CallTarget::SyntheticTransparentCtor { name, owner_path } => {
                        assert_eq!(name, "Ok");
                        assert!(owner_path.is_empty());
                    }
                    other => panic!("expected SyntheticTransparentCtor, got {other:?}"),
                }
            }
            other => panic!("expected outer Call op, got {other:?}"),
        }

        // A single-statement block body is unwrapped.
        let block_body: syn::Expr = syn::parse_str("{ Ok(StepResult::Continue) }").unwrap();
        assert!(detect_wrapped_unit_variant_return(&block_body).is_some());

        // Non-matching shapes (async Err-with-call, method calls,
        // non-allowlisted wrappers/variants) fall back to syn-AST lowering.
        let async_body: syn::Expr =
            syn::parse_str(r#"Err(crate::PyError::type_error("x").into())"#).unwrap();
        assert!(detect_wrapped_unit_variant_return(&async_body).is_none());
        let method_body: syn::Expr = syn::parse_str("executor.unsupported(&other)").unwrap();
        assert!(detect_wrapped_unit_variant_return(&method_body).is_none());
        let non_variant: syn::Expr = syn::parse_str("Ok(some_local)").unwrap();
        assert!(detect_wrapped_unit_variant_return(&non_variant).is_none());
    }

    #[test]
    fn raise_stub_arm_detection_and_synthesis() {
        let body: syn::Expr = syn::parse_str(
            r#"Err(crate::PyError::type_error("async not yet implemented").into())"#,
        )
        .unwrap();
        let stub = detect_raise_stub_return(&body).expect("async stub should be detected");
        assert_eq!(
            stub.fn_segments,
            vec![
                "crate".to_string(),
                "PyError".to_string(),
                "type_error".to_string()
            ]
        );
        assert_eq!(stub.method_name, "into");
        assert_eq!(stub.wrapper_name, "Err");
        assert!(stub.wrapper_owner.is_empty());

        let sig: syn::Signature = syn::parse_str(
            "fn execute_opcode_step<E>(executor: &mut E, code: &CodeObject, instruction: Instruction, op_arg: OpArg, next_instr: usize) -> Result<StepResult, PyError>",
        )
        .unwrap();
        let synth_graph = synthesize_raise_stub_wrapper("Async", &sig, &stub);
        // 5 Input ops + Abort + 3 Call ops (type_error / into / Err).
        let start = synth_graph.block(synth_graph.startblock);
        assert_eq!(start.inputargs.len(), 5);
        assert_eq!(start.operations.len(), 9);

        // Non-matching shapes fall through.
        let const_body: syn::Expr = syn::parse_str("Ok(StepResult::Continue)").unwrap();
        assert!(detect_raise_stub_return(&const_body).is_none());
        let method_body: syn::Expr = syn::parse_str("executor.unsupported(&other)").unwrap();
        assert!(detect_raise_stub_return(&method_body).is_none());
    }

    #[test]
    fn extract_function_call_identity_preserves_path_segments() {
        let expr: syn::Expr = syn::parse_quote!(crate::runtime::exec_build_list(frame, 1));
        let call = match expr {
            syn::Expr::Call(call) => call,
            _ => panic!("expected call expr"),
        };

        let identity = extract_function_call_identity(&call.func);
        assert_eq!(
            identity,
            ExtractedHandlerCall::FunctionPath(CallPath::from_segments([
                "crate",
                "runtime",
                "exec_build_list",
            ]))
        );
    }

    #[test]
    fn extract_opcode_dispatch_selector_uses_exact_variant_path() {
        let pat: syn::Pat = syn::parse_quote!(Instruction::LoadFast { var_num });
        let selector = extract_opcode_dispatch_selector(&pat);
        assert_eq!(
            selector,
            OpcodeDispatchSelector::Path(CallPath::from_segments(["Instruction", "LoadFast",]))
        );
        assert_eq!(selector.canonical_key(), "Instruction::LoadFast");
    }

    #[test]
    fn extract_opcode_dispatch_selector_preserves_or_cases() {
        let pat: syn::Pat = syn::parse_quote!(Instruction::LoadFast | Instruction::StoreFast);
        let selector = extract_opcode_dispatch_selector(&pat);
        assert_eq!(
            selector.canonical_key(),
            "Instruction::LoadFast | Instruction::StoreFast"
        );
    }

    #[test]
    fn find_function_uses_canonical_parse_surface() {
        let parsed = parse_source(
            r#"
            fn helper() {}
            fn mainloop() {}
        "#,
        );
        let func = find_function(&parsed, "mainloop").expect("mainloop");
        assert_eq!(func.sig.ident, "mainloop");
    }

    #[test]
    fn find_opcode_match_uses_canonical_parse_surface() {
        let parsed = parse_source(
            r#"
            fn mainloop() {
                match op {
                    OP_ADD => {},
                    _ => {},
                }
            }
        "#,
        );
        let func = find_function(&parsed, "mainloop").expect("mainloop");
        let opcode_match = find_opcode_match(func).expect("opcode match");
        assert_eq!(opcode_match.arms.len(), 2);
    }

    #[test]
    fn find_opcode_match_finds_nested_dispatch() {
        let parsed = parse_source(
            r#"
            fn mainloop() {
                loop {
                    match op {
                        OP_ADD => {},
                        OP_SUB => {},
                        _ => {},
                    }
                }
            }
        "#,
        );
        let func = find_function(&parsed, "mainloop").expect("mainloop");
        let opcode_match = find_opcode_match(func).expect("opcode match");
        assert_eq!(opcode_match.arms.len(), 3);
    }

    #[test]
    fn find_opcode_match_accepts_instruction_enum_dispatch() {
        let parsed = parse_source(
            r#"
            fn execute_opcode_step(inst: Instruction) {
                match inst {
                    Instruction::LoadConst { idx } => {}
                    Instruction::Add => {}
                    _ => {}
                }
            }
        "#,
        );
        let func = find_function(&parsed, "execute_opcode_step").expect("execute_opcode_step");
        let opcode_match = find_opcode_match(func).expect("opcode match");
        assert_eq!(opcode_match.arms.len(), 3);
    }

    #[test]
    fn extract_opcode_dispatch_arms_ignores_nested_matches_in_arm_bodies() {
        let parsed = parse_source(
            r#"
            fn execute_opcode_step(inst: Instruction, x: i64) {
                match inst {
                    Instruction::Copy => {
                        match x {
                            Instruction::Copy => {}
                            _ => {}
                        }
                    }
                    Instruction::YieldValue => {}
                    _ => {}
                }
            }
        "#,
        );
        let arms = extract_opcode_dispatch_arms(&parsed);
        let selectors: Vec<String> = arms
            .iter()
            .map(|arm| arm.selector.canonical_key())
            .collect();
        assert_eq!(
            selectors,
            vec![
                "Instruction::Copy".to_string(),
                "Instruction::YieldValue".to_string(),
                "_".to_string(),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "duplicate opcode dispatch selector `Instruction::Copy`")]
    fn extract_opcode_dispatch_arms_rejects_duplicate_selectors() {
        let parsed = parse_source(
            r#"
            fn execute_opcode_step(inst: Instruction) {
                match inst {
                    Instruction::Copy => {}
                    Instruction::Copy => {}
                    _ => {}
                }
            }
        "#,
        );
        let _ = extract_opcode_dispatch_arms(&parsed);
    }
}

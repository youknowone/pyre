//! Source parsing: extract opcode dispatch and trait impls.

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
}

#[derive(Debug, Clone, Default)]
pub struct ReceiverTraitBindings {
    pub traits_by_receiver: std::collections::HashMap<String, Vec<String>>,
    pub type_root_by_receiver: std::collections::HashMap<String, String>,
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
    pub fn for_impl_method(impl_type_joined: &str, method: &str) -> Self {
        let mut segments: Vec<String> = impl_type_joined
            .split("::")
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
}

pub fn parse_source(source: &str) -> ParsedInterpreter {
    let file = syn::parse_file(source).expect("failed to parse bundled source");
    let use_imports = collect_use_imports(&file.items);
    ParsedInterpreter {
        file,
        module_path: String::new(),
        use_imports,
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
    ParsedInterpreter {
        file,
        module_path: module_path.to_string(),
        use_imports,
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

/// Find the canonical opcode-dispatch match in a parsed interpreter source.
///
/// This is the public parse/front-end helper for consumers that still need the
/// raw `match` AST rather than the extracted `ExtractedOpcodeArm` view.
pub fn find_opcode_dispatch_match(parsed: &ParsedInterpreter) -> Option<&ExprMatch> {
    find_function(parsed, "mainloop").and_then(find_opcode_match)
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
    fn_return_types: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
) -> Result<Vec<TraitImplInfo>, crate::front::ast::FlowingError> {
    let mut impls = Vec::new();
    let mut known_trait_names = std::collections::HashSet::new();
    collect_trait_names(&parsed.file.items, "", &mut known_trait_names);
    collect_trait_impls_from_items(
        &parsed.file.items,
        "",
        struct_fields,
        fn_return_types,
        &parsed.use_imports,
        known_struct_names,
        &known_trait_names,
        &mut impls,
    )?;
    Ok(impls)
}

fn collect_trait_impls_from_items(
    items: &[Item],
    prefix: &str,
    struct_fields: &crate::front::StructFieldRegistry,
    fn_return_types: &std::collections::HashMap<String, String>,
    use_imports: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
    impls: &mut Vec<TraitImplInfo>,
) -> Result<(), crate::front::ast::FlowingError> {
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
                        crate::front::ast::qualify_type_name_with_imports(&t, prefix, use_imports)
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
                            for synth in crate::front::ast::synthesize_or_passthrough(
                                fake_fn,
                                self_ty_root.as_deref(),
                            ) {
                                let sf = crate::front::ast::build_function_graph_with_self_ty_pub(
                                    &synth,
                                    self_ty_root.clone(),
                                    struct_fields,
                                    fn_return_types,
                                    prefix,
                                    use_imports,
                                    known_struct_names,
                                    known_trait_names,
                                )?;
                                let return_type = match &synth.sig.output {
                                    syn::ReturnType::Type(_, ty) => {
                                        crate::front::ast::qualified_full_type_string(
                                            ty,
                                            prefix,
                                            known_struct_names,
                                            known_trait_names,
                                        )
                                    }
                                    syn::ReturnType::Default => Some("()".to_string()),
                                };
                                methods.push(MethodInfo {
                                    name: synth.sig.ident.to_string(),
                                    graph: Some(sf.graph),
                                    return_type,
                                    hints: sf.hints,
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
                let trait_name = qualify_known_trait_name(
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
                            for synth in crate::front::ast::synthesize_or_passthrough(fake_fn, None)
                            {
                                let sf = crate::front::ast::build_function_graph_with_self_ty_pub(
                                    &synth,
                                    None,
                                    struct_fields,
                                    fn_return_types,
                                    prefix,
                                    use_imports,
                                    known_struct_names,
                                    known_trait_names,
                                )?;
                                let return_type = match &synth.sig.output {
                                    syn::ReturnType::Type(_, ty) => {
                                        crate::front::ast::qualified_full_type_string(
                                            ty,
                                            prefix,
                                            known_struct_names,
                                            known_trait_names,
                                        )
                                    }
                                    syn::ReturnType::Default => Some("()".to_string()),
                                };
                                methods.push(MethodInfo {
                                    name: synth.sig.ident.to_string(),
                                    graph: Some(sf.graph),
                                    return_type,
                                    hints: sf.hints,
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
                        fn_return_types,
                        use_imports,
                        known_struct_names,
                        known_trait_names,
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
    fn_return_types: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
) -> Result<Vec<InherentMethodInfo>, crate::front::ast::FlowingError> {
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
        fn_return_types,
        &parsed.use_imports,
        known_struct_names,
        &mut methods,
    )?;
    Ok(methods)
}

fn collect_inherent_methods_from_items(
    items: &[Item],
    prefix: &str,
    struct_fields: &crate::front::StructFieldRegistry,
    fn_return_types: &std::collections::HashMap<String, String>,
    use_imports: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    methods: &mut Vec<InherentMethodInfo>,
) -> Result<(), crate::front::ast::FlowingError> {
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
                    crate::front::ast::qualify_type_name_with_imports(t, prefix, use_imports)
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
                        for synth in crate::front::ast::synthesize_or_passthrough(
                            fake_fn,
                            self_ty_root_bare.as_deref(),
                        ) {
                            let sf = crate::front::ast::build_function_graph_with_self_ty_pub(
                                &synth,
                                self_ty_root_bare.clone(),
                                struct_fields,
                                fn_return_types,
                                prefix,
                                use_imports,
                                known_struct_names,
                                &std::collections::HashSet::new(),
                            )?;
                            let return_type = match &synth.sig.output {
                                syn::ReturnType::Type(_, ty) => {
                                    crate::front::ast::qualified_full_type_string(
                                        ty,
                                        prefix,
                                        known_struct_names,
                                        &std::collections::HashSet::new(),
                                    )
                                }
                                syn::ReturnType::Default => Some("()".to_string()),
                            };
                            methods.push(InherentMethodInfo {
                                for_type: for_type.clone(),
                                self_ty_root: self_ty_root_qualified.clone(),
                                name: synth.sig.ident.to_string(),
                                graph: sf.graph,
                                return_type,
                                hints: sf.hints,
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
                        fn_return_types,
                        use_imports,
                        known_struct_names,
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

/// Extract receiver -> trait bounds for `execute_opcode_step`.
///
/// This lets canonical dispatch resolution follow generic receiver methods
/// through the trait that actually defines their default bodies.
pub fn extract_opcode_dispatch_receiver_traits(
    parsed: &ParsedInterpreter,
) -> ReceiverTraitBindings {
    for item in &parsed.file.items {
        if let Item::Fn(func) = item {
            if func.sig.ident == "execute_opcode_step" {
                return extract_receiver_trait_bindings(func);
            }
        }
    }
    ReceiverTraitBindings::default()
}

/// Collect canonical function names and graphs for the active pipeline path.
///
/// Test-only helper.  RPython `flowspace/objspace.py:49` re-raise
/// semantics — `FlowingError` propagates out rather than silently
/// dropping the graph.
///
/// `metadata` carries the whole-program registries
/// (`struct_fields` / `fn_return_types` / `known_struct_names` /
/// `known_trait_names`) the per-function `build_function_graph_*`
/// call needs; callers build it once across all parsed files so a
/// callsite in one file can resolve a free function defined in
/// another (RPython `annrpython.py:103-150 build_types` is a single
/// whole-program pass before per-function graph build).
#[cfg(test)]
pub fn collect_function_graphs(
    parsed: &ParsedInterpreter,
    metadata: &crate::front::ast::ProgramMetadata,
    graphs: &mut std::collections::HashMap<CallPath, crate::model::FunctionGraph>,
) -> Result<(), crate::front::ast::FlowingError> {
    for item in &parsed.file.items {
        if let Item::Fn(func) = item {
            let name = func.sig.ident.to_string();
            let sf = crate::front::ast::build_function_graph_with_self_ty_pub(
                func,
                None,
                &metadata.struct_fields,
                &metadata.fn_return_types,
                "",
                &parsed.use_imports,
                &metadata.known_struct_names,
                &metadata.known_trait_names,
            )?;
            graphs.insert(CallPath::from_segments([name.clone()]), sf.graph.clone());
            graphs.insert(CallPath::from_segments(["crate", name.as_str()]), sf.graph);
        }
    }
    Ok(())
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
            let mut graph = crate::model::FunctionGraph::new(name.clone());
            // Pre-bind `execute_opcode_step`'s formal parameters as
            // startblock inputargs so the arm body's `Expr::Path`
            // references (e.g. `frame`, `instruction`, `executor`)
            // resolve to those inputargs instead of falling through
            // to the naked body-`Input` emit that the flowspace
            // adapter rejects as "adapter cross-block body Input"
            // (the dominant Cat 2.1 Skip family).  PyPy/RPython parity:
            // each per-opcode handler method receives the
            // dispatcher's parameters in its formal signature.
            crate::front::ast::lower_expr_into_graph_with_signature(
                &mut graph,
                &arm.body,
                Some(sig),
            )
            .unwrap_or_else(|e| {
                panic!("opcode dispatch arm `{name}` must lower without FlowingError: {e:?}")
            });
            ExtractedOpcodeArm {
                selector,
                handler_calls,
                body_graph: Some(graph),
            }
        })
        .collect()
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

fn extract_receiver_trait_bindings(func: &ItemFn) -> ReceiverTraitBindings {
    let mut generic_bounds: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    let push_bounds = |generic_bounds: &mut std::collections::HashMap<String, Vec<String>>,
                       name: String,
                       bounds: Vec<String>| {
        if bounds.is_empty() {
            return;
        }
        let entry = generic_bounds.entry(name).or_default();
        for b in bounds {
            if !entry.iter().any(|existing| existing == &b) {
                entry.push(b);
            }
        }
    };

    let collect_trait_names =
        |bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>| {
            bounds
                .iter()
                .filter_map(|bound| match bound {
                    syn::TypeParamBound::Trait(trait_bound) => {
                        Some(canonical_path_name(&trait_bound.path))
                    }
                    _ => None,
                })
                .filter(|name| !name.is_empty())
                .collect::<Vec<_>>()
        };

    for param in &func.sig.generics.params {
        if let syn::GenericParam::Type(ty) = param {
            let bounds = collect_trait_names(&ty.bounds);
            push_bounds(&mut generic_bounds, ty.ident.to_string(), bounds);
        }
    }

    if let Some(where_clause) = &func.sig.generics.where_clause {
        for predicate in &where_clause.predicates {
            let syn::WherePredicate::Type(pred) = predicate else {
                continue;
            };
            let Some(name) = type_root_ident(&pred.bounded_ty) else {
                continue;
            };
            let bounds = collect_trait_names(&pred.bounds);
            push_bounds(&mut generic_bounds, name, bounds);
        }
    }

    let mut traits_by_receiver = std::collections::HashMap::new();
    let mut type_root_by_receiver = std::collections::HashMap::new();
    for arg in &func.sig.inputs {
        let syn::FnArg::Typed(arg) = arg else {
            continue;
        };
        let syn::Pat::Ident(pat_ident) = &*arg.pat else {
            continue;
        };
        if let Some(type_name) = type_root_ident(&arg.ty) {
            type_root_by_receiver.insert(pat_ident.ident.to_string(), type_name.clone());
            if let Some(bounds) = generic_bounds.get(&type_name) {
                traits_by_receiver.insert(pat_ident.ident.to_string(), bounds.clone());
            }
        }
    }

    ReceiverTraitBindings {
        traits_by_receiver,
        type_root_by_receiver,
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

fn qualify_known_trait_name(
    bare: &str,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> String {
    let qualified = if prefix.is_empty() || bare.contains("::") {
        None
    } else {
        Some(format!("{}::{}", prefix, bare))
    };
    if let Some(qualified) = qualified {
        if known_trait_names.contains(&qualified) {
            qualified
        } else {
            bare.to_string()
        }
    } else {
        bare.to_string()
    }
}

fn canonical_trait_path_name(
    path: &syn::Path,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> String {
    let canonical = canonical_path_name(path);
    if path.segments.len() == 1 {
        qualify_known_trait_name(&canonical, prefix, known_trait_names)
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
    fn extract_receiver_trait_bindings_from_execute_opcode_step() {
        let parsed = parse_source(
            r#"
            pub trait OpcodeStepExecutor { fn load_fast_checked(&mut self, idx: usize); }
            pub fn execute_opcode_step<E: OpcodeStepExecutor>(executor: &mut E, idx: usize) {
                executor.load_fast_checked(idx);
            }
        "#,
        );

        let bindings = extract_opcode_dispatch_receiver_traits(&parsed);
        assert_eq!(
            bindings.traits_by_receiver.get("executor"),
            Some(&vec!["OpcodeStepExecutor".to_string()])
        );
    }

    /// The pyre-interpreter `execute_opcode_step` signature uses a single
    /// direct bound (`<E: OpcodeStepExecutor>`) plus a `where E: Trait + Trait
    /// + ...` clause listing every handler trait. The extractor must collect
    /// both sources so downstream resolution (callcontrol) can map trait
    /// method calls to impl graphs.
    #[test]
    fn extract_receiver_trait_bindings_handles_where_clause_bounds() {
        let parsed = parse_source(
            r#"
            pub trait OpcodeStepExecutor {}
            pub trait SharedOpcodeHandler {}
            pub trait ConstantOpcodeHandler {}
            pub trait LocalOpcodeHandler {}
            pub trait NamespaceOpcodeHandler {}
            pub trait StackOpcodeHandler {}
            pub trait IterOpcodeHandler {}
            pub trait TruthOpcodeHandler {}
            pub trait ControlFlowOpcodeHandler {}
            pub trait BranchOpcodeHandler {}
            pub trait ArithmeticOpcodeHandler {}

            pub fn execute_opcode_step<E: OpcodeStepExecutor>(
                executor: &mut E,
            ) -> ()
            where
                E: SharedOpcodeHandler
                    + ConstantOpcodeHandler
                    + LocalOpcodeHandler
                    + NamespaceOpcodeHandler
                    + StackOpcodeHandler
                    + IterOpcodeHandler
                    + TruthOpcodeHandler
                    + ControlFlowOpcodeHandler
                    + BranchOpcodeHandler
                    + ArithmeticOpcodeHandler,
            {
                let _ = executor;
            }
        "#,
        );

        let bindings = extract_opcode_dispatch_receiver_traits(&parsed);
        let traits = bindings
            .traits_by_receiver
            .get("executor")
            .expect("executor receiver binding");
        let expected = [
            "OpcodeStepExecutor",
            "SharedOpcodeHandler",
            "ConstantOpcodeHandler",
            "LocalOpcodeHandler",
            "NamespaceOpcodeHandler",
            "StackOpcodeHandler",
            "IterOpcodeHandler",
            "TruthOpcodeHandler",
            "ControlFlowOpcodeHandler",
            "BranchOpcodeHandler",
            "ArithmeticOpcodeHandler",
        ];
        for name in expected {
            assert!(
                traits.iter().any(|t| t == name),
                "expected trait `{}` in executor bindings, got {:?}",
                name,
                traits
            );
        }
    }

    #[test]
    fn extract_trait_impls_qualify_nested_trait_names() {
        let parsed = parse_source(
            r#"
            mod a {
                pub trait Handler {
                    fn run(&mut self) {}
                }
                pub struct A;
                impl Handler for A {}
            }
            mod b {
                pub trait Handler {
                    fn run(&mut self) {}
                }
                pub struct B;
                impl Handler for B {}
            }
        "#,
        );
        let impls = extract_trait_impls(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("trait impls must lower");
        let trait_names: std::collections::HashSet<&str> =
            impls.iter().map(|imp| imp.trait_name.as_str()).collect();
        assert!(trait_names.contains("a::Handler"));
        assert!(trait_names.contains("b::Handler"));
    }

    #[test]
    fn extract_trait_default_methods_include_graphs() {
        let parsed = parse_source(
            r#"
            trait Foo {
                fn helper(&mut self, x: i64) -> i64 {
                    x + 1
                }
            }
        "#,
        );
        let impls = extract_trait_impls(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("trait impls must lower");
        let helper = impls[0]
            .methods
            .iter()
            .find(|m| m.name == "helper")
            .expect("helper method");
        assert!(
            helper.graph.is_some(),
            "trait default method should carry graph"
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

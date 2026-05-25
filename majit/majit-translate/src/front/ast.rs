//! AST front-end: build semantic graphs from Rust source.
//!
//! RPython equivalent: flowspace/ — converts source to Block/Link/Variable/SpaceOperation.
//! This module lowers syn AST nodes into FunctionGraph ops with proper data flow (Variable linking).

use std::collections::HashMap;
use syn::{Item, ItemFn};

use crate::ParsedInterpreter;
use crate::flowspace::model::ConstValue;
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FrameState, FunctionGraph, ImmutableRank, Link,
    LinkArg, OpKind, UnknownKind, UnsupportedExprKind, UnsupportedLiteralKind, ValueType,
    exception_exitcase,
};

pub use crate::front::semantic::{
    AstGraphOptions, FlowingError, LoweringAbort, ProgramMetadata, SemanticFunction,
    SemanticProgram, StructFieldRegistry, qualify_module_path, qualify_type_name_with_imports,
};

/// Result of lowering one expression or statement-list tail.
///
/// `path_closed` tracks the RPython `FlowSignal` state-machine.  When a
/// sub-expression raises `Return` / `Raise` / `Break` / `Continue`, the
/// block where the signal fires is closed with the appropriate
/// terminator and `path_closed` becomes `true`; parent walkers stop
/// lowering into that block but continue their sibling walks.
#[derive(Debug, Clone)]
pub struct Lowered {
    pub value: Option<crate::flowspace::model::Variable>,
    pub path_closed: bool,
}

impl Lowered {
    pub fn no_value() -> Self {
        Lowered {
            value: None,
            path_closed: false,
        }
    }
    pub fn path_closed() -> Self {
        Lowered {
            value: None,
            path_closed: true,
        }
    }

    /// Clone [`Self::value`] for call sites that need an owned
    /// `Option<Variable>`.  `graph` is retained in the signature for
    /// caller-side symmetry with [`Self::from_value_var`] (both take
    /// the graph handle even though the carrier change made the
    /// projection unnecessary); future cleanup can drop the param at
    /// both sides together.
    pub fn value_var(&self, _graph: &FunctionGraph) -> Option<crate::flowspace::model::Variable> {
        self.value.clone()
    }

    /// Construct a `Lowered` whose `value` is the supplied `Variable`
    /// handle (cloned in).  `graph` is retained for caller-side
    /// symmetry with [`Self::value_var`]; future cleanup can drop the
    /// param at both sides together.
    pub fn from_value_var(_graph: &FunctionGraph, var: &crate::flowspace::model::Variable) -> Self {
        Lowered {
            value: Some(var.clone()),
            path_closed: false,
        }
    }
}

/// Propagate `path_closed` up the call chain, or unwrap the inner
/// upstream `Variable` if the child produced one.  Used in
/// expression contexts that REQUIRE a value from the sub-expression
/// — if the sub-expr returned `None` with the path still open, that
/// is a FlowingError (well-typed Rust does not produce such a state).
/// Projects [`Lowered::value`] through [`Lowered::value_var`] so the
/// caller receives the backing `Variable` directly.
macro_rules! get_value_var {
    ($lowered:expr, $graph:expr) => {{
        let __l = $lowered;
        if __l.path_closed {
            return Ok(Lowered::path_closed());
        }
        match __l.value_var($graph) {
            Some(v) => v,
            None => {
                return Err(FlowingError::Unsupported {
                    kind: UnknownKind::UnsupportedExpr {
                        variant: UnsupportedExprKind::OtherExpr,
                    },
                });
            }
        }
    }};
}


/// RPython `annrpython.py:103-150 build_types` whole-program walk —
/// runs `collect_struct_names` + `collect_trait_names` +
/// `collect_fields_and_returns` over the items of every parsed
/// file in `parsed_files`.  Public counterpart of the per-pipeline
/// collectors at
/// `build_semantic_program_from_parsed_files_with_options:744-764`,
/// for test-only entry points (`parse::collect_function_graphs`)
/// that need the same registries before invoking
/// `build_function_graph_with_self_ty_pub`.  Accepts a slice so a
/// callsite in one file can resolve a free function defined in
/// another file — single-file metadata leaves cross-file calls
/// (`!crate::is_str(...)` from dictobject.rs against `is_str`
/// defined in strobject.rs) unclassified.
pub fn collect_program_metadata_pub(parsed_files: &[ParsedInterpreter]) -> ProgramMetadata {
    let mut known_struct_names = std::collections::HashSet::new();
    let mut known_trait_names = std::collections::HashSet::new();
    let mut struct_fields = StructFieldRegistry::default();
    let mut fn_return_types: HashMap<String, String> = HashMap::new();
    let mut immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>> = HashMap::new();
    let mut struct_origins: HashMap<String, String> = HashMap::new();
    let mut use_imports: HashMap<(String, String), String> = HashMap::new();
    let mut module_statics: HashMap<(String, String), crate::parse::ModuleStaticDecl> =
        HashMap::new();
    for parsed in parsed_files {
        collect_struct_names(&parsed.file.items, "", &mut known_struct_names);
        collect_trait_names(&parsed.file.items, "", &mut known_trait_names);
        // PyPy `annrpython.py` bookkeeper: every newly-seen STRUCT
        // gets cached under its lltype-object identity, which is the
        // defining-module path.  Pyre carries names as strings — record
        // `bare_name → module_path` so cross-file references can
        // resolve to the canonical hash slot the runtime publishes.
        // Empty `module_path` (legacy `parse_source` entry) skips the
        // record; consumers fall back to dual-publish convergence.
        if !parsed.module_path.is_empty() {
            collect_struct_origins(&parsed.file.items, &parsed.module_path, &mut struct_origins);
        }
        // Mirror the per-file `ParsedInterpreter.use_imports` into the
        // program-wide `(module_path, alias) → fully_qualified_path`
        // registry.  Caller may pass the same alias across multiple
        // files; the per-file key disambiguates.
        for (alias, full) in &parsed.use_imports {
            use_imports.insert((parsed.module_path.clone(), alias.clone()), full.clone());
        }
        for ((nested, name), decl) in &parsed.module_statics {
            let module = qualify_module_path(&parsed.module_path, nested);
            module_statics.insert((module, name.clone()), decl.clone());
        }
    }
    for parsed in parsed_files {
        collect_fields_and_returns(
            &parsed.file.items,
            "",
            &parsed.use_imports,
            &known_struct_names,
            &known_trait_names,
            &mut struct_fields,
            &mut fn_return_types,
            &mut immutable_fields,
        );
    }
    ProgramMetadata {
        known_struct_names,
        known_trait_names,
        struct_fields,
        fn_return_types,
        struct_origins,
        use_imports,
        module_statics,
    }
}

/// Walk all top-level (and nested `mod`) `Item::Struct` declarations in
/// `items` and record each struct's bare name → defining module path.
/// Mirrors PyPy `bookkeeper.getdesc(TYPE)` resolution: every observed
/// lltype STRUCT identity has a canonical home module; pyre carries
/// names as strings so this map serves the same role.
///
/// Nested `mod foo { struct Bar; }` extends the prefix to `outer::foo`
/// so the registered origin matches what `path_hash(canonical)` would
/// produce for the qualified key.
pub(crate) fn collect_struct_origins(
    items: &[Item],
    module_prefix: &str,
    origins: &mut HashMap<String, String>,
) {
    for item in items {
        match item {
            Item::Struct(s) => {
                let bare = s.ident.to_string();
                // First-write-wins: if two files defined the same bare
                // name, keep the first-seen.  Callers can disambiguate
                // via use-import alias; the program-wide map only
                // serves the most-common-case "bare name resolves to
                // single module" path.
                origins
                    .entry(bare)
                    .or_insert_with(|| module_prefix.to_string());
            }
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let nested = if module_prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", module_prefix, m.ident)
                    };
                    collect_struct_origins(sub_items, &nested, origins);
                }
            }
            _ => {}
        }
    }
}


/// Read `#[jit_immutable_fields("a", "b?", "c[*]", "d?[*]")]` attributes
/// off a struct declaration and return the declared field names paired
/// with their `ImmutableRank`.  Bare idents (`#[jit_immutable_fields(a, b)]`)
/// remain accepted as `ImmutableRank::Immutable` for backward compatibility.
///
/// Multiple attributes accumulate; non-recognised tokens are silently
/// skipped (matching `syn::Meta::parse` looseness).  Rank suffix encoding
/// follows RPython `rpython/rtyper/rclass.py:644-678 _parse_field_list`.
fn collect_immutable_field_attrs(attrs: &[syn::Attribute]) -> Vec<(String, ImmutableRank)> {
    use syn::punctuated::Punctuated;
    use syn::{Expr, ExprLit, ExprPath, Lit, Token};

    let mut specs = Vec::new();
    for attr in attrs {
        let Some(ident) = attr.path().get_ident() else {
            continue;
        };
        if ident != "jit_immutable_fields" {
            continue;
        }
        // Accept a comma-separated list of string literals and/or bare
        // idents:  `#[jit_immutable_fields("foo?", bar)]`.  String form
        // carries the RPython suffix; bare ident form is
        // `ImmutableRank::Immutable`.
        let parsed = attr.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated);
        let Ok(items) = parsed else {
            continue;
        };
        for item in items {
            match item {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => {
                    specs.push(ImmutableRank::parse(&s.value()));
                }
                Expr::Path(ExprPath { path, .. }) => {
                    if let Some(id) = path.get_ident() {
                        specs.push((id.to_string(), ImmutableRank::Immutable));
                    }
                }
                _ => {}
            }
        }
    }
    specs
}

/// Pass 1a: collect all struct names (bare + qualified) recursively.
fn collect_struct_names(
    items: &[Item],
    prefix: &str,
    known_struct_names: &mut std::collections::HashSet<String>,
) {
    for item in items {
        match item {
            Item::Struct(s) => {
                let bare_name = s.ident.to_string();
                known_struct_names.insert(bare_name.clone());
                if !prefix.is_empty() {
                    known_struct_names.insert(format!("{}::{}", prefix, bare_name));
                }
            }
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_struct_names(sub_items, &mod_prefix, known_struct_names);
                }
            }
            _ => {}
        }
    }
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

/// Pass 1b: collect field types + fn return types using known_struct_names.
///
/// `use_imports` is the per-source `use <path> as alias` table — same
/// map that lowering reads through `GraphBuildContext.use_imports`, so
/// struct field type / fn return type metadata strings produced here
/// land in the same name namespace as the parameter / local-binding
/// type strings later mint by `qualify_type_name_with_imports`.  Without
/// this thread-through, `bookkeeper.getdesc`-style alias resolution
/// would diverge between metadata + lowering (PyPy single-frame
/// `f_globals` walk: `rpython/annotator/bookkeeper.py:353`).
fn collect_fields_and_returns(
    items: &[Item],
    prefix: &str,
    use_imports: &HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
    struct_fields: &mut StructFieldRegistry,
    fn_return_types: &mut HashMap<String, String>,
    immutable_fields: &mut HashMap<String, Vec<(String, ImmutableRank)>>,
) {
    for item in items {
        match item {
            Item::Struct(s) => {
                let bare_name = s.ident.to_string();
                // RPython: T.TO gives the actual lltype object.
                // qualified_full_type_string uses known_struct_names to identify
                // which inner types are user structs (not heuristic).
                let fields: Vec<(String, String)> = s
                    .fields
                    .iter()
                    .filter_map(|f| {
                        let field_name = f.ident.as_ref()?.to_string();
                        let field_type = qualified_full_type_string_with_imports(
                            &f.ty,
                            prefix,
                            use_imports,
                            known_struct_names,
                            known_trait_names,
                        )?;
                        Some((field_name, field_type))
                    })
                    .collect();
                // RPython: `_immutable_fields_ = ['a', 'b']` on the class
                // body. We accept `#[jit_immutable_fields(a, b)]` on the
                // struct declaration (proc-macro pass-through in
                // `majit_macros::jit_immutable_fields`). Multiple
                // attributes accumulate.
                let immutables = collect_immutable_field_attrs(&s.attrs);
                if !immutables.is_empty() {
                    if prefix.is_empty() {
                        immutable_fields
                            .entry(bare_name.clone())
                            .or_default()
                            .extend(immutables.iter().cloned());
                        // Canonical defining-module alias: `field_immutability`
                        // callers route owner through `qualify_type_name`
                        // non-empty-prefix canonical path; mirror the
                        // `struct_fields` dual-publish below so the lookup
                        // hits exactly under both spellings.
                        let canonical = majit_ir::descr::canonical_struct_name(&bare_name);
                        if canonical != bare_name {
                            immutable_fields
                                .entry(canonical)
                                .or_default()
                                .extend(immutables.iter().cloned());
                        }
                    } else {
                        let qualified = format!("{}::{}", prefix, bare_name);
                        immutable_fields
                            .entry(qualified)
                            .or_default()
                            .extend(immutables.iter().cloned());
                        immutable_fields
                            .entry(bare_name.clone())
                            .or_default()
                            .extend(immutables.iter().cloned());
                    }
                }
                if prefix.is_empty() {
                    // Dual-publish under bare name + canonical defining-module
                    // form when `STRUCT_ORIGIN_REGISTRY` (populated by
                    // `analyze_pipeline_from_parsed:372`) supplies an
                    // origin.  Use-site lookups now route through
                    // `qualify_type_name` non-empty-prefix canonical path
                    // (PyPy `bookkeeper.getdesc` analog) and land on the
                    // canonical key directly without falling back to the
                    // `unique_suffix_owner_key` shim.
                    let canonical = majit_ir::descr::canonical_struct_name(&bare_name);
                    struct_fields
                        .fields
                        .insert(bare_name.clone(), fields.clone());
                    if canonical != bare_name {
                        struct_fields.fields.insert(canonical, fields);
                    }
                } else {
                    let qualified = format!("{}::{}", prefix, bare_name);
                    struct_fields.fields.insert(qualified, fields);
                }
            }
            Item::Fn(func) => {
                // RPython: op.result.concretetype — module-qualified return type.
                let ret_ty = match &func.sig.output {
                    syn::ReturnType::Type(_, ty) => qualified_full_type_string_with_imports(
                        ty,
                        prefix,
                        use_imports,
                        known_struct_names,
                        known_trait_names,
                    ),
                    syn::ReturnType::Default => Some("()".to_string()),
                };
                if let Some(ret_ty) = ret_ty {
                    let key = if prefix.is_empty() {
                        func.sig.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, func.sig.ident)
                    };
                    fn_return_types.insert(key, ret_ty);
                }
                // Nested `fn`s declared as `Stmt::Item(Item::Fn(_))`
                // inside this fn's body.  Rust's lexical scoping makes
                // them callable only from within the parent body, but
                // the classifier (`expr_unary_not_operand_kind`) sees
                // them as bare-name `Expr::Call` paths and needs the
                // signature to disambiguate `!nested_pred(arg)` between
                // UNARY_NOT and UNARY_INVERT.  RPython parity:
                // `bookkeeper.getdesc(value)` resolves any callable in
                // scope by host-identity (`annrpython.py` callee
                // resolution); pyre's static walker substitutes by
                // registering the nested signature under the bare ident.
                collect_nested_fn_returns(
                    &func.block.stmts,
                    prefix,
                    use_imports,
                    known_struct_names,
                    known_trait_names,
                    fn_return_types,
                );
            }
            // RPython has no `Item::Const` analogue — Python module-level
            // constants reach `flowcontext.py` as `LOAD_GLOBAL(name)`
            // followed by the bookkeeper's PBC table lookup
            // (`bookkeeper.py:329-340 immutablevalue`).  Pyre's walker
            // doesn't model PBCs, so consts surface as plain
            // `Expr::Path` identifier references; the classifier needs
            // a typed-name entry to resolve `!FOO`-shape uses.  Reuse
            // `fn_return_types` keyed by ident — Rust convention
            // (SCREAMING_SNAKE_CASE consts vs snake_case fns) keeps the
            // namespaces separate in pyre's source.
            Item::Const(c) => {
                if let Some(ty) = qualified_full_type_string_with_imports(
                    &c.ty,
                    prefix,
                    use_imports,
                    known_struct_names,
                    known_trait_names,
                ) {
                    let key = if prefix.is_empty() {
                        c.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, c.ident)
                    };
                    fn_return_types.insert(key, ty);
                }
            }
            Item::Impl(impl_block) => {
                let self_ty_root = type_root_ident(&impl_block.self_ty);
                for sub in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = sub {
                        let ret_ty = match &method.sig.output {
                            syn::ReturnType::Type(_, ty) => {
                                qualified_full_type_string_with_imports(
                                    ty,
                                    prefix,
                                    use_imports,
                                    known_struct_names,
                                    known_trait_names,
                                )
                            }
                            syn::ReturnType::Default => Some("()".to_string()),
                        };
                        if let Some(ret_ty) = ret_ty {
                            if let Some(ref ty_root) = self_ty_root {
                                let qualified_ty =
                                    qualify_type_name_with_imports(ty_root, prefix, use_imports);
                                fn_return_types.insert(
                                    format!("{}::{}", qualified_ty, method.sig.ident),
                                    ret_ty,
                                );
                            }
                        }
                    }
                    // Impl-block associated consts — `impl Foo { const
                    // CONST_BIT: u32 = 1 << 31; }`. RPython peer:
                    // `bookkeeper.getdesc(value)` resolves class-level
                    // descriptors (`bookkeeper.py:329-340 immutablevalue`)
                    // by host-identity; pyre's static walker registers
                    // them under `Type::CONST_NAME` so `Type::CONST` /
                    // `Self::CONST` references can resolve via the
                    // last-two-segments fallback in
                    // `lookup_function_return_type` /
                    // `expr_unary_not_operand_kind`.
                    if let syn::ImplItem::Const(item_const) = sub {
                        if let Some(ty) = qualified_full_type_string_with_imports(
                            &item_const.ty,
                            prefix,
                            use_imports,
                            known_struct_names,
                            known_trait_names,
                        ) && let Some(ref ty_root) = self_ty_root
                        {
                            let qualified_ty =
                                qualify_type_name_with_imports(ty_root, prefix, use_imports);
                            fn_return_types.insert(
                                format!("{}::{}", qualified_ty, item_const.ident),
                                ty.clone(),
                            );
                            // Bare-key alias for `Self::CONST_BIT`-shape
                            // references whose qualifier strips to the
                            // last segment. Mirrors `Item::Const`'s
                            // file-level registration.
                            fn_return_types.insert(item_const.ident.to_string(), ty);
                        }
                    }
                }
            }
            Item::Trait(trait_def) => {
                let trait_root = qualify_type_name_with_imports(
                    &trait_def.ident.to_string(),
                    prefix,
                    use_imports,
                );
                for sub in &trait_def.items {
                    if let syn::TraitItem::Fn(method) = sub {
                        let ret_ty = match &method.sig.output {
                            syn::ReturnType::Type(_, ty) => {
                                qualified_full_type_string_with_imports(
                                    ty,
                                    prefix,
                                    use_imports,
                                    known_struct_names,
                                    known_trait_names,
                                )
                            }
                            syn::ReturnType::Default => Some("()".to_string()),
                        };
                        if let Some(ret_ty) = ret_ty {
                            fn_return_types
                                .insert(format!("{}::{}", trait_root, method.sig.ident), ret_ty);
                        }
                    }
                }
            }
            Item::Enum(e) => {
                // RPython sum types are multiple subclasses inheriting from a
                // common ancestor; each subclass owns its own
                // `concretetype` field set keyed by the lltype object
                // identity.  Pyre carries identity as a flat string
                // table, and the only stable identity for a variant is
                // the fully-qualified `prefix::Enum::Variant` path.
                // Earlier drafts also registered the bare `Variant`
                // fallback and the bare-enum `Enum::Variant` (without
                // module prefix), but both forms collided across
                // unrelated enums with the same variant name (e.g.
                // `a::Foo::Empty` vs `b::Foo::Empty`, or
                // `Foo::Empty` vs `Bar::Empty`). Register only the
                // fully-qualified key. `field_type` accepts a shorter
                // registered key for callers with extra crate prefixes
                // only when that suffix match is unique. Tuple/unit
                // variants carry no named fields and need no entry.
                let bare_enum = e.ident.to_string();
                let qualified_enum = if prefix.is_empty() {
                    bare_enum.clone()
                } else {
                    format!("{}::{}", prefix, bare_enum)
                };
                for variant in &e.variants {
                    if let syn::Fields::Named(named) = &variant.fields {
                        let var_name = variant.ident.to_string();
                        let fields: Vec<(String, String)> = named
                            .named
                            .iter()
                            .filter_map(|f| {
                                let field_name = f.ident.as_ref()?.to_string();
                                let field_type = qualified_full_type_string_with_imports(
                                    &f.ty,
                                    prefix,
                                    use_imports,
                                    known_struct_names,
                                    known_trait_names,
                                )?;
                                Some((field_name, field_type))
                            })
                            .collect();
                        struct_fields
                            .fields
                            .insert(format!("{}::{}", qualified_enum, var_name), fields);
                    }
                }
            }
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_fields_and_returns(
                        sub_items,
                        &mod_prefix,
                        use_imports,
                        known_struct_names,
                        known_trait_names,
                        struct_fields,
                        fn_return_types,
                        immutable_fields,
                    );
                }
            }
            _ => {}
        }
    }
}

/// `promote_args` selector for `#[elidable_promote(...)]` synthesis.
///
/// Mirrors `rlib/jit.py:189-191` — RPython splits a literal string at
/// `,` when the value is not the special `"all"` marker:
///
/// ```python
/// if promote_args != 'all':
///     args = [args[int(i)] for i in promote_args.split(",")]
/// ```
#[derive(Debug, Clone)]
enum PromoteArgsSelector {
    /// `jit.py:180` default `promote_args='all'` — every non-self
    /// positional arg flows through `hint(..., promote=True)`.
    All,
    /// `jit.py:189-191` index list (`"0,2"` → `[0, 2]`).  Indices are
    /// 0-based and point into the **positional arg list including
    /// `self`** when the decorated function is a method (RPython's
    /// `_get_args(func)` reads the raw co_varnames; pyre mirrors the
    /// same convention by treating `self` as index 0 when present).
    Indices(Vec<usize>),
}

/// `rlib/jit.py:180-191` — parse the literal attribute value attached
/// to `#[elidable_promote(promote_args = "...")]`.  Bare
/// `#[elidable_promote]` defaults to `"all"` per the upstream default
/// argument at jit.py:180.
fn parse_elidable_promote_args(attr: &syn::Attribute) -> syn::Result<PromoteArgsSelector> {
    if matches!(attr.meta, syn::Meta::Path(_)) {
        return Ok(PromoteArgsSelector::All);
    }
    let mut selector = None;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("promote_args") {
            let value = meta.value()?;
            let lit: syn::LitStr = value.parse()?;
            selector = Some(if lit.value() == "all" {
                PromoteArgsSelector::All
            } else {
                let mut indices = Vec::new();
                for piece in lit.value().split(',') {
                    indices.push(piece.trim().parse::<usize>().map_err(|err| {
                        syn::Error::new(lit.span(), format!("promote_args: {err}"))
                    })?);
                }
                PromoteArgsSelector::Indices(indices)
            });
            Ok(())
        } else {
            Err(meta.error("unsupported elidable_promote argument"))
        }
    })?;
    Ok(selector.unwrap_or(PromoteArgsSelector::All))
}

/// Expand a `syn::ItemFn` into `[orig, wrapper]` when it carries
/// `#[elidable_promote]` / `#[purefunction_promote]`, else return the
/// single function unchanged.  Mirrors `rlib/jit.py:184-201`'s "module
/// import installs two callables" semantics at every entry point that
/// hands a `syn::ItemFn` to `build_function_graph*` — free functions
/// (`build_graphs_from_items`), inherent / trait-impl methods
/// (`parse::extract_inherent_impl_methods`,
/// `parse::extract_trait_impls`), and trait default methods
/// (`parse::extract_trait_impls`).  Centralising the expansion here
/// keeps the decorator behaviour uniform across all four lowering
/// surfaces.
///
/// `impl_type` carries the qualified type root (`"S"`, `"a::S"`) when
/// the source item lives inside an `impl` block, so the synthesizer can
/// emit a type-qualified tail call (`a::S::_orig_<name>_unlikely_name(
/// self, args)`) that matches the impl-method registration path built
/// by `parse::CallPath::for_impl_method` at `lib.rs:531-537`.  Free
/// functions and trait-default methods (which have no concrete `Self`
/// type at synthesis time) pass `None` and fall back to the bare-path
/// tail call.
pub fn synthesize_or_passthrough(fake_fn: ItemFn, impl_type: Option<&str>) -> Vec<ItemFn> {
    match extract_elidable_promote_selector(&fake_fn.attrs) {
        Some(selector) => {
            let (orig, wrapper) = synthesize_elidable_promote_pair(&fake_fn, &selector, impl_type);
            vec![orig, wrapper]
        }
        None => vec![fake_fn],
    }
}

/// Locate `#[elidable_promote]` (or its deprecated `#[purefunction_promote]`
/// alias from `rlib/jit.py:203-205`) and return its parsed
/// `promote_args` selector, or `None` if neither attribute is present.
///
/// `rlib/jit.py:189-191` `args[int(i)]` propagates `ValueError` /
/// `IndexError` to the caller — the decorator does not silently drop
/// malformed input.  Pyre mirrors that fail-loud behaviour here: a
/// malformed `promote_args = "..."` literal panics with the
/// underlying `syn::Error` rather than falling through.
fn extract_elidable_promote_selector(attrs: &[syn::Attribute]) -> Option<PromoteArgsSelector> {
    for attr in attrs {
        if let Some(segment) = attr.path().segments.last() {
            let name = segment.ident.to_string();
            if name == "elidable_promote" || name == "purefunction_promote" {
                return Some(parse_elidable_promote_args(attr).unwrap_or_else(|err| {
                    panic!("#[{name}(...)]: {err}");
                }));
            }
        }
    }
    None
}

/// Synthesize the wrapper / original function pair from a single
/// `#[elidable_promote] fn foo(...)` source item.
///
/// Line-by-line port of `rlib/jit.py:184-201`:
///
/// ```python
/// def decorator(func):
///     elidable(func)                                # func._elidable_function_ = True
///     args = _get_args(func)
///     code = ["def f(%s):\n" % (argstring,)]
///     if promote_args != 'all':
///         args = [args[int(i)] for i in promote_args.split(",")]
///     for arg in args:
///         code.append("    %s = hint(%s, promote=True, promote_string=True)\n" % …)
///     code.append("    return _orig_func_unlikely_name(%s)\n" % …)
///     d = {"_orig_func_unlikely_name": func, "hint": hint}
///     exec py.code.Source("\n".join(code)).compile() in d
///     result = d["f"]
///     result.__name__ = func.__name__ + "_promote"
///     return result
/// ```
///
/// Pyre's syn-tree adaptation:
///
///   * RPython's `_orig_func_unlikely_name` closure capture becomes a
///     module-scope sibling `fn _orig_<name>_unlikely_name` carrying
///     the original body and the `#[elidable]` attribute (so
///     `collect_jit_hints` registers it via `mark_elidable`).
///   * RPython's `exec compile` synthesis of `def f(...)` becomes a
///     `syn::Block` whose stmts are
///     `let <arg> = hint_promote_or_string(<arg>);` for each selected
///     arg, followed by the tail call.
///   * The user-facing name keeps its identity (matching pyre's
///     proc-macro naming convention `_orig_<name>_unlikely_name` at
///     `majit-macros::elidable_promote`); RPython's `__name__ + "_promote"`
///     rename is the symmetric choice and equivalently produces two
///     distinct callable identifiers.
fn synthesize_elidable_promote_pair(
    func: &ItemFn,
    selector: &PromoteArgsSelector,
    impl_type: Option<&str>,
) -> (ItemFn, ItemFn) {
    use quote::format_ident;
    // jit.py:186 args = _get_args(func) — positional names, self included.
    let arg_names: Vec<syn::Ident> = func
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(pt) => match &*pt.pat {
                syn::Pat::Ident(pi) => pi.ident.clone(),
                _ => panic!(
                    "#[elidable_promote] on `fn {}`: unsupported binder \
                     pattern in arg position — RPython `_get_args(func)` \
                     (jit.py:172-178) reads positional names off \
                     `co_varnames`, which never include destructured \
                     binders.  Rewrite the parameter as a plain `name: \
                     Ty` instead.",
                    func.sig.ident
                ),
            },
            // `&self` / `&mut self` map to a positional name "self" so
            // index 0 in `Indices` continues to address the receiver as
            // RPython does (`_get_args(func)` reads co_varnames raw).
            syn::FnArg::Receiver(_) => format_ident!("self"),
        })
        .collect();

    let orig_ident = format_ident!("_orig_{}_unlikely_name", func.sig.ident);

    // jit.py:184-185 — original keeps the body, gains `_elidable_function_`.
    let orig_attrs: Vec<syn::Attribute> = func
        .attrs
        .iter()
        .filter(|a| {
            let name = a
                .path()
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            name != "elidable_promote" && name != "purefunction_promote"
        })
        .cloned()
        .collect();
    let elidable_attr: syn::Attribute = syn::parse_quote!(#[elidable]);
    let mut orig_fn = func.clone();
    orig_fn.attrs = orig_attrs;
    orig_fn.attrs.push(elidable_attr);
    orig_fn.sig.ident = orig_ident.clone();

    // jit.py:189-191 — promote_args=='all' → all indices; else parsed list.
    // RPython `args[int(i)]` raises `IndexError` on out-of-range
    // indices; pyre fails loudly here for the same reason.
    let promote_indices: Vec<usize> = match selector {
        PromoteArgsSelector::All => (0..arg_names.len()).collect(),
        PromoteArgsSelector::Indices(ix) => {
            for &i in ix {
                if i >= arg_names.len() {
                    panic!(
                        "#[elidable_promote(promote_args = ...)] on `fn {}`: \
                         index {} is out of range for a {}-arg function \
                         (jit.py:191 `args[int(i)]` would IndexError)",
                        func.sig.ident,
                        i,
                        arg_names.len()
                    );
                }
            }
            ix.clone()
        }
    };
    // jit.py:191-194 — `for arg in args: hint(arg, promote=True,
    // promote_string=True)`.  RPython emits a *single* dual-flag hint
    // per arg and defers the dispatch to
    // `rpython/jit/codewriter/jtransform.py:599-606`, which only
    // matches `concretetype == lltype.Ptr(rstr.STR)`:
    //
    //   * `Ptr(rstr.STR)` → keep `promote_string`, delete `promote`
    //     → emit `str_guard_value` (jit.py:615-631).
    //   * everything else, **including `Ptr(rstr.UNICODE)`** → delete
    //     `promote_string`, keep `promote` → emit
    //     `<kind>_guard_value` with `kind = getkind(concretetype)`,
    //     which is `"ref"` for any GC pointer
    //     (`rpython/jit/metainterp/history.py:64-67`).  No
    //     `unicode_guard_value` opname exists in RPython; the
    //     unicode-specific `str_guard_value/OS_UNIEQ_NONNULL` shape
    //     is only reachable via the dedicated `promote_unicode`
    //     hint (`jit_codewriter/jtransform.py:632-648`,
    //     `rpython/rlib/jit.py:130-131`), which `elidable_promote`
    //     never emits.
    //
    // Pyre mirrors that shape line-by-line: every promoted arg gets
    // `hint_promote_or_string`, and the `PromoteOrString` arm in
    // `jit_codewriter/jtransform.rs` falls through to the plain
    // `<kind>_guard_value` family because pyre lacks a
    // `Ptr(rstr.STR)`-equivalent GC layout — `del hints['promote_string']`
    // is the upstream `else` branch at `jtransform.py:603-606`.
    //
    // `args` includes `self` since `_get_args(func)` reads
    // `co_varnames` raw.  Rust forbids re-binding the `self` keyword,
    // so the receiver is routed through a fresh `__self_promoted`
    // local; non-receiver args keep RPython's shadow pattern.
    let promote_self = promote_indices.iter().any(|&i| arg_names[i] == "self");
    let promote_stmts: Vec<syn::Stmt> = promote_indices
        .iter()
        .map(|&i| {
            let id = &arg_names[i];
            if id == "self" {
                syn::parse_quote!(let __self_promoted = hint_promote_or_string(self);)
            } else {
                syn::parse_quote!(let #id = hint_promote_or_string(#id);)
            }
        })
        .collect();

    // jit.py:195 — return _orig_func_unlikely_name(args).  When `self` was
    // promoted above, the tail call uses the promoted local; otherwise it
    // threads the unmodified receiver through.
    let call_args = arg_names.iter().map(|id| -> syn::Expr {
        if id == "self" {
            if promote_self {
                syn::parse_quote!(__self_promoted)
            } else {
                syn::parse_quote!(self)
            }
        } else {
            syn::parse_quote!(#id)
        }
    });
    // jit.py:195-197 — `_orig_func_unlikely_name` is bound in the
    // wrapper's `exec` namespace (`d = {"_orig_func_unlikely_name":
    // func, ...}`), so the wrapper invokes it as a closure-local name.
    // Pyre's lowering surface does not have an analogue closure scope;
    // for free fns the orig is registered as a bare-path function (so a
    // bare-path call resolves), but for impl methods the orig is
    // registered under `<ImplType>::_orig_<name>_unlikely_name` via
    // `parse::CallPath::for_impl_method` at `lib.rs:531-537`.  Emit a
    // type-qualified tail call in that case so the wrapper's IR `Call`
    // target matches the registration path and the elidable flag bound
    // to the orig graph is visible at the call-site.
    let tail_call: syn::Expr = match impl_type {
        Some(ty_str) => {
            let ty_path: syn::Path = syn::parse_str(ty_str).unwrap_or_else(|err| {
                panic!(
                    "synthesize_elidable_promote_pair: failed to parse impl type `{ty_str}`: {err}"
                )
            });
            syn::parse_quote!(#ty_path::#orig_ident(#(#call_args),*))
        }
        None => syn::parse_quote!(#orig_ident(#(#call_args),*)),
    };
    let tail_stmt = syn::Stmt::Expr(tail_call, None);

    // jit.py:198-201 — wrapper is the user-facing decorated name with
    // the promote+forward body; `#[elidable_promote]` is stripped so
    // `collect_jit_hints` does not register the wrapper itself as
    // elidable (the binary flag belongs on the original alone, per
    // jit.py:185 `elidable(func)`).
    let wrapper_attrs: Vec<syn::Attribute> = func
        .attrs
        .iter()
        .filter(|a| {
            let name = a
                .path()
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            name != "elidable_promote" && name != "purefunction_promote"
        })
        .cloned()
        .collect();
    let mut wrapper_block = syn::Block {
        brace_token: Default::default(),
        stmts: Vec::new(),
    };
    wrapper_block.stmts.extend(promote_stmts);
    wrapper_block.stmts.push(tail_stmt);

    let mut wrapper_fn = func.clone();
    wrapper_fn.attrs = wrapper_attrs;
    wrapper_fn.block = Box::new(wrapper_block);

    (orig_fn, wrapper_fn)
}

/// Walk the statements of a fn body and register the return types of
/// any nested `fn` items declared inside.  Used by `collect_fields_and_
/// returns`'s `Item::Fn` arm so the classifier sees nested-fn
/// signatures.  Recurses through `Stmt::Item(Item::Fn(_))` only
/// (nested-mod / nested-impl inside a fn body are vanishingly rare in
/// pyre source and would require their own qualified prefix).
fn collect_nested_fn_returns(
    stmts: &[syn::Stmt],
    prefix: &str,
    use_imports: &HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
    fn_return_types: &mut HashMap<String, String>,
) {
    for stmt in stmts {
        if let syn::Stmt::Item(Item::Fn(nested)) = stmt {
            let ret_ty = match &nested.sig.output {
                syn::ReturnType::Type(_, ty) => qualified_full_type_string_with_imports(
                    ty,
                    prefix,
                    use_imports,
                    known_struct_names,
                    known_trait_names,
                ),
                syn::ReturnType::Default => Some("()".to_string()),
            };
            if let Some(ret_ty) = ret_ty {
                let key = if prefix.is_empty() {
                    nested.sig.ident.to_string()
                } else {
                    format!("{}::{}", prefix, nested.sig.ident)
                };
                fn_return_types.entry(key).or_insert(ret_ty);
            }
            collect_nested_fn_returns(
                &nested.block.stmts,
                prefix,
                use_imports,
                known_struct_names,
                known_trait_names,
                fn_return_types,
            );
        }
    }
}


/// Public entry for building a graph from a single function AST node.
/// Lower a standalone expression into an existing graph.
/// Used to build semantic graphs from opcode match arm bodies.
///
/// RPython `flowspace/objspace.py:38` — `build_flow()` re-raises
/// `FlowingError` so callers observe the unsupported construct as an
/// error rather than receiving a partially-constructed graph.  The
/// `Unknown` marker op that `stop_unsupported` already emitted stays in
/// the graph; the caller decides whether to keep, discard, or close it.
pub fn lower_expr_into_graph(
    graph: &mut FunctionGraph,
    expr: &syn::Expr,
) -> Result<(), FlowingError> {
    lower_expr_into_graph_with_signature(graph, expr, None)
}

/// Variant of [`lower_expr_into_graph`] that pre-registers a function
/// signature's formal parameters as startblock `OpKind::Input` ops +
/// `Block.inputargs` entries + `GraphBuildContext.bind_local_id`
/// bindings.
///
/// Closes the "adapter cross-block body Input" Skip family
/// for `__opcode_dispatch__::*` synthesized arm graphs:
/// without this pre-binding, an arm body that references
/// `execute_opcode_step`'s formal parameters (`frame`, `instruction`,
/// `executor`, ...) falls through to the naked body-`Input` emit at
/// the `Expr::Path` fallback (`front/ast.rs:4559`), which the
/// flowspace adapter rejects as a producer-side gap.  Pre-binding
/// puts each formal parameter at a known startblock inputarg so
/// same-block reads dedup against the binding and cross-block reads
/// resolve via `lazy_install_local_at_current_block` — matching the
/// `RPython`/PyPy shape where every per-opcode handler method has the
/// dispatcher's parameters in its formal signature.
///
/// The parameter-registration loop mirrors [`build_function_graph`]
/// (`front/ast.rs:3056-3156`) but skips module-prefix / use-imports /
/// struct / trait registries, since opcode-dispatch arm graphs are
/// synthesized without whole-program context.
pub fn lower_expr_into_graph_with_signature(
    graph: &mut FunctionGraph,
    expr: &syn::Expr,
    sig: Option<&syn::Signature>,
) -> Result<(), FlowingError> {
    let mut block = graph.startblock;
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = HashMap::new();
    let empty_suffix_index = MethodSuffixIndex::default();
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    let mut ctx = GraphBuildContext::new(
        &empty_registry,
        &empty_fn_ret,
        &empty_suffix_index,
        "",
        HashMap::new(),
        &empty_names,
        &empty_trait_names,
    );
    if let Some(sig) = sig {
        for param in &sig.inputs {
            match param {
                syn::FnArg::Receiver(recv) => {
                    let self_ty = classify_fn_arg_ty(&recv.ty);
                    ctx.local_value_types
                        .insert("self".to_string(), self_ty.clone());
                    if let Some(var) = graph.push_op_var(
                        block,
                        OpKind::Input {
                            name: "self".to_string(),
                            ty: self_ty,
                        },
                        true,
                    ) {
                        graph.name_value_var(&var, "self".to_string());
                        graph.push_inputarg_var(block, var.clone());
                        ctx.bind_local_id_var("self".to_string(), &var, graph, block);
                    }
                }
                syn::FnArg::Typed(pat_type) => {
                    let name = canonical_pat_name(&pat_type.pat);
                    if let Some(type_root) = type_root_ident(&pat_type.ty) {
                        ctx.local_type_roots.insert(name.clone(), type_root);
                    }
                    let arg_ty = classify_fn_arg_ty(&pat_type.ty);
                    ctx.local_value_types.insert(name.clone(), arg_ty.clone());
                    if let Some(var) = graph.push_op_var(
                        block,
                        OpKind::Input {
                            name: name.clone(),
                            ty: arg_ty,
                        },
                        true,
                    ) {
                        graph.name_value_var(&var, name.clone());
                        graph.push_inputarg_var(block, var.clone());
                        ctx.bind_local_id_var(name, &var, graph, block);
                    }
                }
            }
        }
    }
    ctx.assert_stack_empty_at_stmt_boundary("lower_expr_into_graph entry");
    let lowered = lower_expr(
        graph,
        &mut block,
        expr,
        &AstGraphOptions::default(),
        &mut ctx,
    )?;
    ctx.assert_stack_empty_at_stmt_boundary("lower_expr_into_graph exit");
    if graph.block(block).is_open() {
        graph.set_return(block, lowered.value_var(graph));
    }
    Ok(())
}


/// Expose `collect_jit_hints` so `parse.rs` can hoist trait-method
/// hints from AST attributes when the strict graph build returns Err.
pub fn collect_jit_hints_pub(attrs: &[syn::Attribute]) -> Vec<String> {
    collect_jit_hints(attrs, None)
}

/// `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]`
/// — companion to `collect_jit_hints` that exposes the function's
/// positional parameter names when `#[oopspec(...)]` is present.
///
/// Used by the walker to populate `CallControl::oopspec_argnames`
/// (`lib.rs:598-600` consumes the `"oopspec_argnames:..."` hint
/// alongside the `"oopspec:..."` hint).  Pyre's `parse_oopspec`
/// (`support.py:701-715` port) needs argnames to resolve identifier
/// slots in the spec's `(...)` pattern to `Index(n)` placeholders.
pub fn collect_jit_hints_with_sig(attrs: &[syn::Attribute], sig: &syn::Signature) -> Vec<String> {
    collect_jit_hints(attrs, Some(sig))
}

/// Resolution of a `(receiver_leaf, method)` suffix against the
/// `fn_return_types` keys: either the single unique key carrying that
/// suffix or `Ambiguous` when two distinct keys share it.
#[derive(Debug, Clone)]
enum SuffixMatch {
    Unique(String),
    Ambiguous,
}

/// Compile-time resolver for `lookup_method_return_type`'s leaf-suffix
/// fallback. Maps `(receiver_leaf, method)` to the unique `fn_return_types`
/// key with that suffix (or `Ambiguous`). Built once per whole-program
/// build from the frozen `fn_return_types` map and borrowed into every
/// per-function `GraphBuildContext`, so the fallback is an O(1) lookup
/// instead of a linear scan over every registered return type. pyre-only
/// resolution aid — RPython carries the callee's `concretetype` on the
/// annotated op, so there is no name-suffix resolution upstream.
///
/// Convergence path: retired once the annotator binds a `concretetype` to
/// each call result (`call.py:98 funcobj.graph` is the codewriter-side
/// analog), so method return types are read off the op rather than
/// recovered by matching a `(receiver_leaf, method)` name suffix.
#[derive(Debug, Clone, Default)]
struct MethodSuffixIndex {
    by_suffix: HashMap<(String, String), SuffixMatch>,
}

impl MethodSuffixIndex {
    fn from_fn_return_types(map: &HashMap<String, String>) -> Self {
        let mut by_suffix: HashMap<(String, String), SuffixMatch> = HashMap::new();
        for key in map.keys() {
            let Some((owner, method)) = key.rsplit_once("::") else {
                continue;
            };
            let leaf = owner.rsplit("::").next().unwrap_or(owner);
            match by_suffix.entry((leaf.to_string(), method.to_string())) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(SuffixMatch::Unique(key.clone()));
                }
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    if let SuffixMatch::Unique(existing) = o.get()
                        && existing != key
                    {
                        o.insert(SuffixMatch::Ambiguous);
                    }
                }
            }
        }
        Self { by_suffix }
    }

    /// The unique `fn_return_types` key whose last two segments are
    /// `(receiver_leaf, method)`, or `None` when no key or more than one
    /// key carries that suffix.
    fn unique_key(&self, receiver_leaf: &str, method: &str) -> Option<&str> {
        match self
            .by_suffix
            .get(&(receiver_leaf.to_string(), method.to_string()))?
        {
            SuffixMatch::Unique(key) => Some(key.as_str()),
            SuffixMatch::Ambiguous => None,
        }
    }
}

#[derive(Debug, Clone)]
struct GraphBuildContext<'a> {
    local_type_roots: HashMap<String, String>,
    local_type_strings: HashMap<String, String>,
    local_value_types: HashMap<String, ValueType>,
    /// RPython `LOAD_FAST` parity: locals' definition sites carried as
    /// `(Variable, defining BlockId)` so a body `Expr::Path` reference
    /// can reuse the existing definition's `Variable` instead of
    /// emitting a fresh `OpKind::Input`. Same-block reuse only —
    /// cross-block reads keep the legacy fresh-`Input` behaviour
    /// because pyre does not yet thread the locals stack across
    /// `Link.args` / `inputarg` the way RPython
    /// `flowspace/flowcontext.py:835 LOAD_FAST` does. Closing the
    /// cross-block gap is a deferred Cat 3.2 follow-up; this field
    /// owns the same-block half of the parity.
    local_value_ids: HashMap<String, (crate::flowspace::model::Variable, BlockId)>,
    local_trait_bound_roots: HashMap<String, String>,
    generic_trait_roots: HashMap<String, String>,
    /// RPython: ARRAY element type identity — maps variable name to the
    /// element type of its array (e.g. "arr" → "Point" for `arr: Vec<Point>`).
    /// This is the Rust equivalent of RPython's `GcArray(T)` where T is the
    /// element type that determines the ARRAY identity for `cpu.arraydescrof()`.
    local_array_types: HashMap<String, String>,
    /// Receiver-trait lookup for locals/parameters bound to `&mut dyn T`
    /// / `&dyn T` / `Box<dyn T>` / `dyn T`.  Populated at let-statement
    /// and fn-parameter binding time; consumed by `dyn_trait_for_receiver`
    /// so method-call lowering can emit `CallTarget::Indirect`
    /// (`jtransform.py:410-412`).
    local_dyn_trait_roots: HashMap<String, String>,
    /// Closure-bound locals: `let f = |args| body` registers `f` →
    /// closure body's return type so a downstream `f(...)` call gets a
    /// known return type. Pyre's walker has no closure visibility
    /// (`fn_return_types` only registers `Item::Fn` / `Item::Const` /
    /// `Item::Impl` methods); RPython's `bookkeeper.getdesc(value)`
    /// resolves any callable in scope by host-identity, so this side
    /// table substitutes by registering closure return types under the
    /// bare local ident.  Read by `lookup_function_return_type`'s
    /// bare-key fallback chain.
    local_closure_returns: HashMap<String, String>,
    /// RPython: program-level struct field types, available for resolving
    /// field access array identity (e.g. `self.array[i]` → owner.field_type).
    struct_fields: &'a StructFieldRegistry,
    /// RPython: op.result.concretetype — function return types from the annotator.
    /// Maps function name (or "Type::method") → return type string.
    /// Used by array_type_id_from_expr to resolve Call/MethodCall expressions.
    fn_return_types: &'a HashMap<String, String>,
    /// O(1) leaf-suffix resolver over `fn_return_types`, built once per
    /// whole-program build and shared across every per-function context.
    /// Consulted by `lookup_method_return_type` when the exact
    /// `receiver::method` key misses.
    method_suffix_index: &'a MethodSuffixIndex,
    /// Module path prefix for qualifying bare type names.
    /// RPython: lltype identity is globally unique — bare "Foo" in mod "a"
    /// must resolve to "a::Foo" in struct_fields lookups.
    module_prefix: String,
    /// The file's source-module path — `parsed.module_path` for the file
    /// that owns this graph, constant through nested `mod` recursion.
    /// Distinct from `module_prefix`, which tracks the *nested-mod*
    /// segment chain (top-level functions get `module_prefix = ""` so
    /// that bare type qualification doesn't accidentally prepend the
    /// file's path).  `source_module` is the key used to scope module-
    /// static lookups so a `pub const FOO: i64 = 1;` declared in
    /// `file_a` resolves only for graphs built from `file_a` — matching
    /// PyPy's per-frame `globals` (`flowcontext.py:845`) which only sees
    /// the defining file's module globals.
    source_module: String,
    /// Per-source `use <path> as alias` table — `(alias → fully_qualified_path)`
    /// resolved from this source file's top-level `Item::Use` declarations.
    /// PyPy peer: `bookkeeper.getdesc(value)` walks the host Python lexical
    /// scope of the current frame's import resolutions; pyre carries names
    /// as strings, so this map is the per-graph slice of
    /// `CallControl.use_imports` aggregated by `analyze_pipeline_from_parsed`.
    /// Empty when the file has no use-aliases or the build entry bypassed
    /// the parsed-file plumbing (`parse::collect_function_graphs` tests).
    use_imports: HashMap<String, String>,
    /// Program-wide `pub const` / `pub static` declarations keyed by
    /// `(module_path, name)`.  Aggregated by
    /// [`build_semantic_program_from_parsed_files_with_options`] from
    /// every parsed file's
    /// [`crate::parse::ParsedInterpreter::module_statics`].
    /// Populated via [`Self::with_module_statics`] at the function-graph
    /// build site; defaults to empty when callers (tests, legacy entry
    /// points) construct a context without program-wide aggregation.
    module_statics: HashMap<(String, String), crate::parse::ModuleStaticDecl>,
    known_struct_names: &'a std::collections::HashSet<String>,
    known_trait_names: &'a std::collections::HashSet<String>,
    /// Loop targets active at the current lowering point.  Pushed on
    /// entry to `Loop` / `While` / `ForLoop` and popped after the body
    /// is walked.  `break` closes the current block with a goto to the
    /// innermost `break_target`; `continue` goes to `continue_target`.
    /// RPython: `flowspace/flowcontext.py:525` BreakLoop signal +
    /// `:1341` LoopBlock.handle_signal dispatches to end/header.
    loop_stack: Vec<LoopFrame>,
    /// First-bind positional order of local names — graph-wide
    /// append-only.  Each name is appended at the moment it is first
    /// bound anywhere in the function (`bind_local_id`); subsequent
    /// rebinds update `local_value_ids` in place but do not move the
    /// entry, and `LocalBindingSnapshot::restore` does NOT roll the
    /// order back so a name's slot index is invariant across the
    /// entire lowering even when a sibling arm's bindings get
    /// restored.  RPython parity: `co_varnames` slot order is
    /// assigned at compile time and never reshuffled
    /// (`flowspace/flowcontext.py:835 LOAD_FAST` reads slots by
    /// index); `FrameState::union`'s positional zip
    /// (`framestate.py:14 _union`) relies on this graph-wide
    /// invariant so two predecessors of the same merge point line
    /// up slot-by-slot.
    local_first_bind_order: Vec<String>,
    /// Graph-wide membership set complementing
    /// `local_first_bind_order`.  Used by `bind_local_id` to detect
    /// the *first ever* bind of a name in this function — distinct
    /// from "currently bound" (`local_value_ids.contains_key`)
    /// because a `LocalBindingSnapshot::restore` may unbind a name
    /// from `local_value_ids` while leaving its slot in
    /// `local_first_bind_order`.  Without this set a re-bind after
    /// restore would double-append the name and violate the slot
    /// invariant.
    local_first_bind_seen: std::collections::HashSet<String>,
    /// Flow-space value stack analogous to `flowspace/flowcontext.py:
    /// 314-345 FlowContext.stack`.  Empty until slice Z4.B+ converts
    /// leaf `Expr` lowering (Lit / Path) to the push/pop walker shape;
    /// scaffolded here so the stack-helper API surface (`pushvalue` /
    /// `popvalue` / `peekvalue` / `popvalues` / `dropvaluesuntil`) can
    /// be ported in advance and validated in isolation.  Cell type is
    /// `StackElem` (Hlvalue cells per `flowcontext.py:285 self.stack`,
    /// a polymorphic `Variable | Constant | FlowSignal` list).
    /// The cfg(test) `pushvid(slot)` helper mints a fresh `Variable` and
    /// registers `(variable, slot)` so the cell index round-trips through
    /// `bridge_variable`; cfg(test) `popvid() -> usize` reverses the
    /// bridge. Production code uses `pushvalue(cell: StackElem)` /
    /// `popvalue() -> StackElem` directly without slot projection.
    /// `Hlvalue::Constant` and `FlowSignal` arms become callable once
    /// the Z4 walker (slice Z4.B+) introduces a non-Variable push site.
    #[allow(dead_code)]
    value_stack: Vec<crate::flowspace::framestate::StackElem>,
    /// Pending FSException at the current flow point — analogue of
    /// `flowcontext.py:354 self.last_exception`.  None until the Z4
    /// walker (slice Z4.B+) populates a real value at SETUP_EXCEPT /
    /// RAISE_VARARGS sites; scaffolded here so `getstate` /
    /// `setstate` can construct / restore the full 5-tuple
    /// `FrameState` shape ahead of the walker rewrite.
    #[allow(dead_code)]
    last_exception: Option<crate::flowspace::model::FSException>,
    /// Pending frame-block stack at the current flow point — analogue
    /// of `flowcontext.py:285 self.blockstack`.  Empty until the Z4
    /// walker pushes `LoopBlock` / `ExceptBlock` / `FinallyBlock` /
    /// `WithBlock` / `IterBlock` entries on SETUP_* opcode equivalents
    /// (Z4.E+ for the loop variants; Z4.H+ for the exception-handling
    /// variants).  Scaffolded here so `getstate` can populate the
    /// `FrameState.blocklist` projection without a separate side
    /// table.
    #[allow(dead_code)]
    blockstack: Vec<crate::flowspace::flowcontext::FrameBlock>,
    /// `flowcontext.py:293 self.joinpoints = {}` — candidate block list
    /// keyed by `FrameState.next_offset`.  Each entry holds the
    /// SpamBlocks already created for that join point in arrival order
    /// (newest at index 0, matching upstream's `candidates.insert(0,
    /// newblock)` convention at `flowcontext.py:435/462`).  Read by
    /// `mergeblock`, which iterates candidates to find one whose
    /// `framestate.union(currentstate)` returns non-None; written when
    /// a new SpamBlock is created (`make_next_block` arm or the
    /// generalization arm) or an existing candidate is generalized
    /// out (`recloseblock` retire).
    ///
    /// Empty until callers route through `mergeblock`.  The tree-
    /// recursive lowering today builds merge blocks directly without
    /// consulting this map; downstream slices will migrate per-site.
    joinpoints: HashMap<i64, Vec<BlockId>>,
    /// `flowcontext.py:402` `self.pendingblocks`.  Generalized SpamBlocks
    /// enqueued by the `mergeblock` generalize arm (`flowcontext.py:463`)
    /// and consumed by the re-walk drain
    /// (`drain_pendingblocks_for_merge_offset`, the loop-scoped port of
    /// `build_flow`, `flowcontext.py:399-422`).  A `Vec` rather than a
    /// `collections.deque`: the drain is loop-scoped and filters by
    /// `merge_offset`, so FIFO order is irrelevant.  Unlike upstream
    /// (`:402` seeds `[graph.startblock]`) pyre starts empty — the
    /// recursive walker drives the first walk inline via `make_next_block`
    /// continuation, so only the retroactive generalize case enqueues
    /// here.  Empty on every production graph (the back-edge always
    /// direct-links on borrow-checked input — see [`PendingBlock`]).
    pendingblocks: Vec<PendingBlock>,
    /// Counter for generating unique negative `next_offset` keys for
    /// loop merge points.  Each loop gets a unique key via
    /// `next_loop_merge_offset()` so `mergeblock` can partition
    /// candidates per loop header.  Negative keys avoid collision with
    /// future real bytecode offsets (non-negative).
    loop_merge_counter: i64,
}

#[derive(Debug, Clone)]
struct LoopFrame {
    continue_target: BlockId,
    merge_offset: i64,
    header_slot_count: usize,
    break_target: BlockId,
    /// Original header FrameState from `build_loop_header_state`.
    /// Before calling `mergeblock` on the back-edge, we restore this
    /// onto `block.framestate` because pyre's body-walk may overwrite
    /// it during lowering (RPython's SpamBlock.framestate is immutable
    /// post-creation).
    header_state: FrameState,
    /// Termination budget for the re-walk drain (task #91).  pyre
    /// safety adaptation (parity rule 5) with NO RPython basis:
    /// `flowcontext.py:402 while self.pendingblocks` is unbounded,
    /// trusting only the merge lattice.  The lattice is monotone — a
    /// generalize can only `None`-kill a loop-carried slot
    /// (`framestate.py:110-111`) or freshen a `Variable`, never
    /// re-grow a slot — and `build_loop_header_state` pre-widens every
    /// slot to a `Variable`, so the only headroom is `Variable`→`None`
    /// per slot.  Each re-walk that generalizes therefore strictly
    /// raises the merged-state `None`-count, bounding the drain at
    /// `header_slot_count + 1` iterations per `merge_offset`.  Stored on
    /// the frame (not the ctx) so nested loops with distinct negative
    /// `merge_offset` keys keep independent budgets.  Decremented per
    /// re-walk by `drain_pendingblocks_for_merge_offset` (see
    /// [`PendingBlock`]).
    rewalk_budget_remaining: usize,
}

/// `flowcontext.py:402` `self.pendingblocks` element — a generalized
/// SpamBlock awaiting re-walk.  Enqueued by `mergeblock`'s generalize
/// arm (`flowcontext.py:463 self.pendingblocks.append(newblock)`).
/// `state` is the block's `framestate` (what `record_block`'s
/// `setstate` restores, `flowcontext.py:408`); `merge_offset` scopes
/// the drain to its owning loop.
///
/// The re-walk drain that consumes this queue
/// (`drain_pendingblocks_for_merge_offset`, the loop-scoped port of
/// `build_flow` / `record_block`, `flowcontext.py:399-422`) is wired
/// into the While / Loop / ForLoop arms.  On borrow-checked input the
/// queue stays empty, so the drain is a no-op there: flowcontext.py
/// needs it because Python bytecode permits `DELETE_FAST` (`del x`)
/// followed by a read on a later loop iteration — a runtime
/// `UnboundLocalError` the flow analysis models by leaving the slot
/// `None`, forcing a generalize + re-walk — but Rust's borrow checker
/// STATICALLY rejects that pattern: moving/`drop`-ing a loop-carried
/// local without rebinding it before the back-edge is a cross-iteration
/// use-after-move (`E0382`).  Since pyre only lowers functions that
/// compile, no production graph `None`-kills a loop-carried slot, so the
/// back-edge always direct-links and this queue never fills outside the
/// `drain_*` unit tests that drive the path directly (task #91).
#[derive(Clone, Debug)]
struct PendingBlock {
    block: BlockId,
    state: FrameState,
    merge_offset: i64,
}

/// The loop arm prologue replayed by
/// [`GraphBuildContext::rewalk_loop_body`] when the re-walk drain
/// (`flowcontext.py:399-422`) revisits a generalized header.  A CLOSED
/// ENUM rather than a `Box<dyn FnMut>`: the three loop kinds are a
/// finite set, and a closure would force a second `&mut graph` /
/// `&mut ctx` re-borrow inside a method already holding them (exactly
/// the borrow-checker-driven redesign the parity rules forbid).  Each
/// variant carries only the arm-specific data the prologue needs.
///
/// ⚠️ `replay` is a SECOND copy of the first-walk arm prologue
/// (`Expr::While` cond+branch / `Expr::Loop` no-op / `Expr::ForLoop`
/// abort+branch).  If a first-walk prologue changes, update `replay` in
/// lockstep — the two are intentionally duplicated to keep the
/// first-walk op stream byte-identical for the 39 benchmarks, at the
/// cost of this drift hazard.
///
/// `Copy` so the drain can replay the same prologue for every
/// generalized header at one `merge_offset` (the fields are a shared
/// AST reference + the exit block, both `Copy`).
#[derive(Clone, Copy)]
enum LoopPrologue<'a> {
    While { cond: &'a syn::Expr, exit: BlockId },
    Loop,
    For { exit: BlockId },
}

impl LoopPrologue<'_> {
    /// Re-emit the arm prologue onto the generalized header
    /// `header_block` and return the body entry block (`Some`), or
    /// `None` when the prologue itself closed the path (no body to
    /// re-walk).
    fn replay(
        self,
        graph: &mut FunctionGraph,
        ctx: &mut GraphBuildContext,
        header_block: BlockId,
        header_state: &FrameState,
        options: &AstGraphOptions,
    ) -> Result<Option<BlockId>, FlowingError> {
        match self {
            // `Expr::While` cond+branch.  `get_value_var!` is inlined
            // here because its early `return Ok(Lowered::...)` does not
            // type-check in a `Result<Option<BlockId>, _>` context.
            LoopPrologue::While { cond, exit } => {
                let mut header_tail = header_block;
                let lowered = lower_expr(graph, &mut header_tail, cond, options, ctx)?;
                if lowered.path_closed {
                    return Ok(None);
                }
                let cond_pre_var = match lowered.value_var(graph) {
                    Some(v) => v,
                    None => {
                        return Err(FlowingError::Unsupported {
                            kind: UnknownKind::UnsupportedExpr {
                                variant: UnsupportedExprKind::OtherExpr,
                            },
                        });
                    }
                };
                ctx.pushvid_var(&cond_pre_var);
                let cond_var = ctx.popvid_var(graph);
                let body_entry = graph.create_block();
                let header_branch_snapshot = ctx.getstate(graph, 0);
                graph.set_branch(header_tail, cond_var, body_entry, vec![], exit, vec![]);
                graph.block_mut(header_tail).framestate = Some(header_branch_snapshot);
                graph.block_mut(header_block).framestate = Some(header_state.clone());
                Ok(Some(body_entry))
            }
            // `Expr::Loop` — no prologue; the header IS the body entry
            // (`continue_target == body_entry` in the first walk).
            LoopPrologue::Loop => Ok(Some(header_block)),
            // `Expr::ForLoop` — emit the ForLoop Abort marker, then
            // branch/goto header → body.
            LoopPrologue::For { exit } => {
                let for_cond_var = graph.push_op_var(
                    header_block,
                    OpKind::Abort {
                        kind: UnknownKind::UnsupportedExpr {
                            variant: UnsupportedExprKind::ForLoop,
                        },
                    },
                    true,
                );
                let body_entry = graph.create_block();
                if let Some(cond_var) = for_cond_var {
                    graph.set_branch(header_block, cond_var, body_entry, vec![], exit, vec![]);
                } else {
                    graph.set_goto(header_block, body_entry, vec![]);
                }
                graph.block_mut(header_block).framestate = Some(header_state.clone());
                Ok(Some(body_entry))
            }
        }
    }
}

/// `flowcontext.py:465-472 make_next_block` — create a SpamBlock
/// whose FrameState is `pre_loop_snapshot.copy()` (framestate.py:42).
/// Every Variable-typed slot gets a fresh phi Variable;
/// `prune_dead_phis` cleans up unused ones post-build.
///
/// Returns the copied state plus phi-info for each Variable-typed
/// slot (slot index, fresh Variable, name, value type) — needed by
/// the caller to emit paired `OpKind::Input` ops and rebind
/// `ctx.local_value_ids`.
fn build_loop_header_state(
    graph: &mut FunctionGraph,
    ctx: &GraphBuildContext<'_>,
    pre_loop_snapshot: &FrameState,
) -> (
    FrameState,
    Vec<(usize, crate::flowspace::model::Variable, String, ValueType)>,
) {
    // The caller emits a paired `OpKind::Input` op for every entry in
    // the returned `phi_info` (locals only), while
    // `create_block_from_framestate` derives the header block's
    // `inputargs` from `header_state.getvariables()` =
    // `locals + flatten(stack) + [exc_type, exc_value]` Variables
    // (`framestate.py:50`).  Those two sets must coincide, else the
    // phi-block invariant (`model.rs` `is_phi_block`: every inputarg
    // paired with an `Input` op) breaks.  They coincide exactly when
    // the loop header carries no stack/exception Variables — i.e. the
    // value stack is drained and `last_exception` is None at the
    // statement boundary, which the Z4 walker maintains until it
    // activates real stack/exc cells.  Fence the precondition with a
    // release-active `assert!` (build-time, runs once per loop, off the
    // JITted hot path) so a future non-empty-stack loop header panics
    // loudly rather than silently emitting a phi-block with unpaired
    // inputargs.  Full stack/exc phi materialisation here is Z4-walker
    // work (task #91); until it lands the precondition holds by
    // construction.
    assert!(
        pre_loop_snapshot.stack.is_empty() && pre_loop_snapshot.last_exception.is_none(),
        "build_loop_header_state: loop header carries stack/exc state \
         (stack_len={}, has_exc={}); locals-only phi emission would leave \
         getvariables() stack/exc inputargs unpaired with Input ops \
         (stack/exc phi materialisation is Z4-walker work)",
        pre_loop_snapshot.stack.len(),
        pre_loop_snapshot.last_exception.is_some(),
    );

    // Collect pre-loop Variable types BEFORE copy() mints fresh vars.
    let pre_loop_var_types: Vec<(usize, ValueType)> = pre_loop_snapshot
        .locals_w_view(graph)
        .iter()
        .enumerate()
        .filter_map(|(i, slot)| match slot {
            Some(crate::flowspace::model::Hlvalue::Variable(v)) => {
                let vt = graph_value_type_var(graph, v).unwrap_or(ValueType::Unknown);
                Some((i, vt))
            }
            _ => None,
        })
        .collect();

    // `framestate.py:42 state.copy()` — every Variable replaced with
    // a fresh copy; Constants and None slots unchanged.
    let header_state = pre_loop_snapshot.copy(graph);

    // Derive phi_info from the freshened state.
    let phi_info: Vec<(usize, crate::flowspace::model::Variable, String, ValueType)> =
        pre_loop_var_types
            .into_iter()
            .filter_map(|(slot_idx, value_type)| {
                if let Some(crate::flowspace::model::Hlvalue::Variable(phi_var)) =
                    header_state.locals_w.get(slot_idx).and_then(|s| s.as_ref())
                {
                    let name = ctx.local_first_bind_order[slot_idx].clone();
                    Some((slot_idx, phi_var.clone(), name, value_type))
                } else {
                    None
                }
            })
            .collect();
    (header_state, phi_info)
}

impl<'a> GraphBuildContext<'a> {
    fn new(
        struct_fields: &'a StructFieldRegistry,
        fn_return_types: &'a HashMap<String, String>,
        method_suffix_index: &'a MethodSuffixIndex,
        module_prefix: &str,
        use_imports: HashMap<String, String>,
        known_struct_names: &'a std::collections::HashSet<String>,
        known_trait_names: &'a std::collections::HashSet<String>,
    ) -> Self {
        Self {
            local_type_roots: HashMap::new(),
            local_type_strings: HashMap::new(),
            local_value_types: HashMap::new(),
            local_value_ids: HashMap::new(),
            local_trait_bound_roots: HashMap::new(),
            generic_trait_roots: HashMap::new(),
            local_array_types: HashMap::new(),
            local_dyn_trait_roots: HashMap::new(),
            local_closure_returns: HashMap::new(),
            struct_fields,
            fn_return_types,
            method_suffix_index,
            module_prefix: module_prefix.to_string(),
            source_module: String::new(),
            use_imports,
            module_statics: HashMap::new(),
            known_struct_names,
            known_trait_names,
            loop_stack: Vec::new(),
            local_first_bind_order: Vec::new(),
            local_first_bind_seen: std::collections::HashSet::new(),
            value_stack: Vec::new(),
            last_exception: None,
            blockstack: Vec::new(),
            joinpoints: HashMap::new(),
            pendingblocks: Vec::new(),
            loop_merge_counter: 0,
        }
    }

    /// Builder that attaches the program-wide module-static table to
    /// this graph build context.  Mirrors the additive `with_*`
    /// builder pattern already used by `bind_local_id_*` helpers — the
    /// `new()` constructor stays signature-stable so existing call
    /// sites (5 production + several test fixtures) keep compiling
    /// without per-site edits, and the build sites that have access
    /// to a program-wide `module_statics` table (the multi-file
    /// pipeline entry in `build_semantic_program_from_parsed_files_
    /// with_options`) opt in via this setter.
    fn with_module_statics(
        mut self,
        module_statics: HashMap<(String, String), crate::parse::ModuleStaticDecl>,
    ) -> Self {
        self.module_statics = module_statics;
        self
    }

    /// Stamp the file-level source-module path used to scope module-
    /// static lookups.  Empty string means "no per-file scope known"
    /// (test fixtures and legacy public entry points); the production
    /// pipeline calls this with `parsed.module_path`.
    fn with_source_module(mut self, source_module: &str) -> Self {
        self.source_module = source_module.to_string();
        self
    }

    // ------------------------------------------------------------------
    // Flow-space value-stack helpers (slice Z4.A scaffolding).
    //
    // Line-by-line port of `flowspace/flowcontext.py:317-345` —
    // `stackdepth` / `pushvalue` / `popvalue` / `peekvalue` /
    // `settopvalue` / `popvalues` / `dropvaluesuntil`.  Identifier names
    // and behavioural shapes match upstream exactly.  Pyre's stack
    // holds `StackElem` cells matching upstream's polymorphic
    // `Variable | Constant | FlowSignal` shape — `StackElem::Value(
    // Hlvalue::Variable(_))` for live operands and `StackElem::Signal(_)`
    // for `Return` / `Raise` / `Break` / `Continue` flow signals.

    /// `flowcontext.py:317-319` — current value stack size.
    #[allow(dead_code)]
    fn stackdepth(&self) -> usize {
        self.value_stack.len()
    }

    /// `flowcontext.py:321-322 pushvalue(self, w_object)` — push the
    /// Hlvalue cell `w_object` onto the value stack.  PyPy signature
    /// takes the cell verbatim; pyre wraps as `StackElem::Value` /
    /// `StackElem::Signal` at the caller.
    #[allow(dead_code)]
    fn pushvalue(&mut self, cell: crate::flowspace::framestate::StackElem) {
        self.value_stack.push(cell);
    }

    /// `flowcontext.py:324-325 popvalue(self)` — pop the topmost cell.
    /// Panics when the stack is empty (upstream's `list.pop()` raises
    /// `IndexError`; pyre raises `unwrap`).
    #[allow(dead_code)]
    fn popvalue(&mut self) -> crate::flowspace::framestate::StackElem {
        self.value_stack
            .pop()
            .expect("popvalue: empty stack (flowcontext.py:325 list.pop on empty)")
    }

    /// cfg(test)-only `pushvalue` for a graph-wide slot index — fetches
    /// the backing `Variable` from `graph.value_variables[slot]` (every
    /// slot minted via `alloc_value_var` has one) and wraps it as a
    /// `StackElem::Value(Hlvalue::Variable(_))` before pushing.
    ///
    /// Production lower_expr / lower_stmt push through
    /// [`Self::pushvid_var`] directly with the upstream `Variable`
    /// handle (no `slot → graph.variable_at` bridge).  This slot form
    /// is retained as the cfg(test) counterpart of
    /// [`Self::popvid`] for the `pushvid_popvid_round_trip_through_value_stack`
    /// + `pushvid_panics_when_vid_has_no_backing_variable` fixtures
    /// that pin the slot-resolution contract.
    #[cfg(test)]
    fn pushvid(&mut self, graph: &FunctionGraph, slot: usize) {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::Hlvalue;
        let var = graph
            .variable_at(slot)
            .unwrap_or_else(|| {
                panic!(
                    "pushvid: slot {} has no backing Variable on graph {:?}; \
                     callers must mint the slot via `alloc_value_var` / \
                     `ensure_variable_registered_void` before pushing onto value_stack",
                    slot, graph.name,
                )
            })
            .clone();
        self.pushvalue(StackElem::Value(Hlvalue::Variable(var)));
    }

    /// Variable-direct sibling of [`Self::pushvid`] — pushes the upstream
    /// `Variable` straight onto `value_stack` as
    /// `StackElem::Value(Hlvalue::Variable(var.clone()))` without
    /// `graph.variable_at(slot)` projection.  Callers that already hold
    /// the `Variable` handle (e.g. from `Lowered::value_var(graph)` or
    /// `push_op_var`) skip the slot → Variable round-trip.
    fn pushvid_var(&mut self, var: &crate::flowspace::model::Variable) {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::Hlvalue;
        self.pushvalue(StackElem::Value(Hlvalue::Variable(var.clone())));
    }

    /// cfg(test) `popvalue` recovering a graph-wide slot index from the
    /// topmost cell — counterpart to [`Self::pushvid`].  Panics when
    /// the stack is empty (`popvalue`'s precondition) or when the
    /// topmost cell is not a `StackElem::Value(Hlvalue::Variable(_))`.
    ///
    /// Pyre's analogue of upstream's `w_obj = self.popvalue()` followed
    /// by `op.result = w_obj` — upstream consumes Variables directly,
    /// pyre projects them back to the pyre IR's slot carrier via
    /// `graph.slot_of(&var)`.
    ///
    /// The Variable-only restriction matches the Z4.B.1 push contract
    /// (only `pushvid`-style cells today); once Z4.G+ activates
    /// Constant-cell pushes the helper will widen with a matching
    /// `LinkArg::Const`-style return.
    #[cfg(test)]
    fn popvid(&mut self, graph: &FunctionGraph) -> usize {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::Hlvalue;
        match self.popvalue() {
            StackElem::Value(Hlvalue::Variable(v)) => graph.bridge_variable(&v),
            other => panic!(
                "popvid: expected StackElem::Value(Hlvalue::Variable), got {other:?} \
                 (graph {:?})",
                graph.name,
            ),
        }
    }

    /// Variable-returning sibling of [`Self::popvid`] — pops the same
    /// `StackElem::Value(Hlvalue::Variable(_))` cell but returns the
    /// `Variable` directly so callers can skip the
    /// `graph.slot_of(&var).expect(...)` → `graph.must_variable_at(slot)`
    /// round-trip and feed the carrier straight into ops that already
    /// take `Vec<Variable>` (e.g. `OpKind::Call.args`,
    /// `OpKind::IndirectCall.args`).
    #[allow(dead_code)]
    fn popvid_var(&mut self, graph: &FunctionGraph) -> crate::flowspace::model::Variable {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::Hlvalue;
        match self.popvalue() {
            StackElem::Value(Hlvalue::Variable(v)) => v,
            other => panic!(
                "popvid_var: expected StackElem::Value(Hlvalue::Variable), got {other:?} \
                 (graph {:?})",
                graph.name,
            ),
        }
    }

    /// `flowcontext.py:327-330 peekvalue(self, index_from_top=0)` —
    /// look at the cell `index_from_top` positions below the top.
    /// Top of stack is `peekvalue(0)`.
    #[allow(dead_code)]
    fn peekvalue(&self, index_from_top: usize) -> &crate::flowspace::framestate::StackElem {
        let len = self.value_stack.len();
        assert!(
            index_from_top < len,
            "peekvalue: depth {index_from_top} exceeds stack size {len} (flowcontext.py:329)"
        );
        &self.value_stack[len - 1 - index_from_top]
    }

    /// `flowcontext.py:332-334 settopvalue(self, w_object,
    /// index_from_top=0)` — overwrite the cell `index_from_top`
    /// positions below the top.
    #[allow(dead_code)]
    fn settopvalue(
        &mut self,
        cell: crate::flowspace::framestate::StackElem,
        index_from_top: usize,
    ) {
        let len = self.value_stack.len();
        assert!(
            index_from_top < len,
            "settopvalue: depth {index_from_top} exceeds stack size {len} (flowcontext.py:333)"
        );
        let idx = len - 1 - index_from_top;
        self.value_stack[idx] = cell;
    }

    /// `flowcontext.py:336-341 popvalues(self, n)` — pop `n` cells in
    /// stack order (oldest first), returning them as a `Vec`.  `n == 0`
    /// returns an empty vec without touching the stack.
    ///
    /// PyPy uses Python negative-slice semantics — `self.stack[-n:]`
    /// silently clamps when `n > len(self.stack)`, so the upstream
    /// helper returns the entire stack and clears it without raising.
    /// Mirror that behaviour by clamping `n` to `len` before splitting;
    /// asserting here would diverge from upstream on overflow inputs.
    #[allow(dead_code)]
    fn popvalues(&mut self, n: usize) -> Vec<crate::flowspace::framestate::StackElem> {
        if n == 0 {
            return Vec::new();
        }
        let len = self.value_stack.len();
        let take = n.min(len);
        self.value_stack.split_off(len - take)
    }

    /// `flowcontext.py:343-344 dropvaluesuntil(self, finaldepth)` —
    /// shrink the stack to exactly `finaldepth` cells.  Used by
    /// `FrameBlock.cleanupstack` (`flowcontext.py:1335-1336`) when a
    /// SETUP_* block is unwound.
    ///
    /// PyPy's `del self.stack[finaldepth:]` is a no-op when
    /// `finaldepth >= len(self.stack)` (the slice past the end is
    /// empty).  `Vec::truncate` matches: it leaves the stack
    /// unchanged when `finaldepth >= len`.  No assert — overflow
    /// inputs silently pass.
    #[allow(dead_code)]
    fn dropvaluesuntil(&mut self, finaldepth: usize) {
        self.value_stack.truncate(finaldepth);
    }

    /// `flowcontext.py:346-348 getstate(self, next_offset)` —
    /// construct a `FrameState` from the current ctx fields.
    ///
    /// ```python
    /// def getstate(self, next_offset):
    ///     return FrameState(self.locals_w, self.stack[:],
    ///             self.last_exception, self.blockstack[:], next_offset)
    /// ```
    ///
    /// Pyre's locals projection (`entries`) is sourced from
    /// `local_first_bind_order` indexed into `local_value_ids`.
    /// `last_exception` and `blocklist` thread directly because
    /// their cell types match between ctx and `FrameState`.
    ///
    /// `value_stack` (now `Vec<StackElem>` per Z4.A.5) is cloned
    /// verbatim into `FrameState.stack` — direct match of upstream's
    /// `self.stack[:]` because both carriers are Hlvalue-shaped.
    /// `last_exception` and `blocklist` thread directly because their
    /// cell types match between ctx and `FrameState`.
    ///
    /// Allocate a unique negative joinpoint key for a loop header
    /// merge point.  Negative keys partition loop headers from other
    /// merge points (which use next_offset ≥ 0).
    fn next_loop_merge_offset(&mut self) -> i64 {
        self.loop_merge_counter -= 1;
        self.loop_merge_counter
    }

    /// Production capture path: every `set_branch` / `set_goto` site
    /// in `Expr::If` / `Expr::Match` / loop variants stamps
    /// `Block.framestate = Some(ctx.getstate(graph, 0))` so the lazy
    /// installer can walk predecessor framestates back to a binding
    /// site.  `next_offset = 0` until the Z4 walker rewrite threads
    /// real bytecode-equivalent offsets through.
    fn getstate(&self, _graph: &FunctionGraph, next_offset: i64) -> FrameState {
        let entries: Vec<Option<crate::flowspace::model::Variable>> = self
            .local_first_bind_order
            .iter()
            .map(|name| {
                self.local_value_ids
                    .get(name)
                    .map(|(var, _defining_block)| var.clone())
            })
            .collect();
        // Populate the `locals_w` Hlvalue carrier in lockstep with
        // `entries` — `flowcontext.py:346 getstate` carries
        // `self.locals_w` directly into the new FrameState.  Each
        // captured Variable wraps into `Hlvalue::Variable(v)` (the
        // matching upstream cell type for the AST-frontend's locals
        // domain; pyre has no Constant production at this site).
        let locals_w: Vec<Option<crate::flowspace::model::Hlvalue>> = entries
            .iter()
            .map(|slot| {
                slot.as_ref()
                    .cloned()
                    .map(crate::flowspace::model::Hlvalue::Variable)
            })
            .collect();
        FrameState {
            entries,
            locals_w,
            stack: self.value_stack.clone(),
            last_exception: self.last_exception.clone(),
            blocklist: self.blockstack.clone(),
            next_offset,
        }
    }

    /// `flowcontext.py:350-356 setstate(self, state)` — reset the ctx
    /// to a previously-captured `FrameState`.
    ///
    /// ```python
    /// def setstate(self, state):
    ///     self.locals_w = state.locals_w[:]
    ///     self.stack = state.stack[:]
    ///     self.last_exception = state.last_exception
    ///     self.blockstack = state.blocklist[:]
    ///     self._normalize_raise_signals()
    /// ```
    ///
    /// Pyre's locals projection update goes through `local_value_ids`
    /// — a name→(vid, defining_block) HashMap rather than upstream's
    /// positional list, because pyre carries the (vid, BlockId)
    /// reuse-gate.  `local_first_bind_order` is graph-wide append-
    /// only and stays untouched (slot indices never reshuffle —
    /// upstream `co_varnames` parity).  `last_exception` and
    /// `blocklist` thread directly.  `_normalize_raise_signals`
    /// (`flowcontext.py:358-362`) is a no-op for pyre because no
    /// `RaiseImplicit` cells live on `value_stack` yet (Z4.H ports
    /// the exception-handling sites).
    ///
    /// `value_stack` (now `Vec<StackElem>` per Z4.A.5) is replaced by
    /// `state.stack.clone()` — direct match of upstream's
    /// `self.stack = state.stack[:]` because both carriers are
    /// Hlvalue-shaped.
    ///
    /// Currently unused — `LocalBindingSnapshot::restore` continues
    /// to be the production restore path; this is the structural
    /// API surface ahead of the walker rewrite (Z4.B+).
    #[allow(dead_code)]
    fn setstate(&mut self, state: &FrameState, graph: &FunctionGraph) {
        // Locals: rebind each name from `local_first_bind_order` to
        // the slot's Variable in `state.entries`, dropping the binding
        // when the slot is `None`-killed.  The defining_block is set
        // to `BlockId(0)` as a placeholder — the Z4 walker rewrite
        // will need to thread the originating block alongside the
        // Variable (or retire the (Variable, BlockId) gate entirely
        // as part of Z2.5 absorption at Z4.last).
        for (slot_idx, name) in self.local_first_bind_order.clone().iter().enumerate() {
            match state.entry_var(slot_idx, graph) {
                Some(var) => {
                    self.local_value_ids.insert(name.clone(), (var, BlockId(0)));
                }
                None => {
                    self.local_value_ids.remove(name);
                }
            }
        }
        // `flowcontext.py:352 self.stack = state.stack[:]` — direct
        // copy now that `value_stack` and `FrameState.stack` share the
        // `Vec<StackElem>` carrier (Z4.A.5).
        self.value_stack = state.stack.clone();
        self.last_exception = state.last_exception.clone();
        self.blockstack = state.blocklist.clone();
        self.normalize_raise_signals();
    }

    /// `flowcontext.py:350-356 setstate` companion threaded with an
    /// explicit `owner_block` so the post-merge `local_value_ids`
    /// records carry the merge block itself as their `defining_block`
    /// — every `Variable` in `state.locals_w` is materialised in
    /// `owner_block.inputargs` by `create_block_from_framestate`, so
    /// `(var, owner_block)` is the structurally honest pairing of the
    /// pyre `(Variable, BlockId)` reuse-gate.
    ///
    /// Also names freshly-minted phi `Variable`s via
    /// `graph.name_value(vid, name)` so the cutover's
    /// `flowspace_adapter.rs:1706-1708` cross-block aliasing recovers
    /// the merge-block inputarg under the local's name.  Upstream
    /// `framestate.py:113 Variable()` mints anonymous Variables at
    /// NeedsPhi cells — pyre's IR side-table for `value_name` is the
    /// graph-side carrier that `register_variable_valueid` leaves
    /// untouched, so the naming step lives here at the setstate
    /// boundary.
    ///
    /// Refreshes `local_value_types` from `graph_value_type(vid)` so
    /// later `read_local` / `STORE_FAST` re-entry sees the merged kind
    /// (carry-through type widening: `Unknown` cells inherit the
    /// concrete sibling kind via `FrameState::union`'s wildcard rule).
    ///
    /// Production entry — used by `lower_if_expr`'s
    /// `!want_phi && both_open` migration to rebind ctx after the
    /// merge block has been created via `create_block_from_framestate`
    /// + `set_goto_from_framestate`.  Loop-body scope cleanup (#134)
    /// drops body-local bindings on `Expr::ForLoop` / `Expr::While` /
    /// `Expr::Loop` close, and the migration's
    /// `can_thread_variable_to_block` dry-run skips orphan-rooted
    /// graphs (`>2-arm Expr::Match` fallback at `ast.rs:6045-6052`),
    /// so `merged.locals_w` no longer surfaces orphan Variables that
    /// would trip `ensure_variable_at_block`'s pred-chain reachability
    /// assert.
    fn setstate_at_block(
        &mut self,
        state: &FrameState,
        owner_block: BlockId,
        graph: &mut FunctionGraph,
    ) {
        // Snapshot the slot view + Variables + names before re-borrowing
        // graph mutably for `name_value_var` / `local_value_types` updates.
        let entries: Vec<(
            usize,
            String,
            Option<(crate::flowspace::model::Variable, ValueType)>,
        )> = {
            let view = state.locals_w_view(graph);
            self.local_first_bind_order
                .iter()
                .enumerate()
                .map(|(slot_idx, name)| {
                    let payload = view
                        .get(slot_idx)
                        .and_then(|c| c.as_ref())
                        .and_then(|cell| {
                            match cell {
                                crate::flowspace::model::Hlvalue::Variable(v) => Some((
                                    v.clone(),
                                    graph_value_type_var(graph, v).unwrap_or(ValueType::Unknown),
                                )),
                                // Constant cells in locals: pyre keys
                                // identity through Variable, and Constants
                                // are emitted on-demand by the reader
                                // (Stmt::Local / Expr::Path lower constant
                                // literals via push_op directly), so we
                                // drop the binding here and let the next
                                // read re-emit.  Upstream `framestate.py`
                                // carries the Constant cell directly; the
                                // structural divergence is accepted at
                                // `setstate`-boundary today (rare path —
                                // `framestate.py:113` mints Variables not
                                // Constants on NeedsPhi).
                                crate::flowspace::model::Hlvalue::Constant(_) => None,
                            }
                        });
                    (slot_idx, name.clone(), payload)
                })
                .collect()
        };
        for (_slot_idx, name, payload) in entries {
            match payload {
                Some((var, ty)) => {
                    if graph.value_name_for(&var).is_none() {
                        graph.name_value_var(&var, name.clone());
                    }
                    self.local_value_ids
                        .insert(name.clone(), (var, owner_block));
                    self.local_value_types.insert(name, ty);
                }
                None => {
                    self.local_value_ids.remove(&name);
                    self.local_value_types.remove(&name);
                }
            }
        }
        self.value_stack = state.stack.clone();
        self.last_exception = state.last_exception.clone();
        self.blockstack = state.blocklist.clone();
        self.normalize_raise_signals();
    }

    /// `flowcontext.py:358-362 _normalize_raise_signals` — every
    /// `RaiseImplicit` cell on the stack is downgraded to a plain
    /// `Raise(same w_exc)` after `setstate` restores a captured
    /// snapshot.  Upstream rationale: a stored `RaiseImplicit` no
    /// longer carries its "produced inside `do_op`" context once
    /// it has been replayed from a `FrameState`, so the stricter
    /// `Raise` semantics take over.  Today a no-op because the AST
    /// walker has not yet pushed `FlowSignal` cells onto
    /// `value_stack` — gates activate when Z4.H wires exception
    /// handling.  Surface kept in lockstep with upstream's setstate
    /// body so the wiring drops in without scope creep when the
    /// Z4 walker materialises signal cells.
    fn normalize_raise_signals(&mut self) {
        use crate::flowspace::flowcontext::FlowSignal;
        use crate::flowspace::framestate::StackElem;
        for cell in &mut self.value_stack {
            if let StackElem::Signal(FlowSignal::RaiseImplicit { w_exc }) = cell {
                *cell = StackElem::Signal(FlowSignal::Raise {
                    w_exc: w_exc.clone(),
                });
            }
        }
    }

    /// Z4.B.0 tripwire — assert `value_stack` is empty at a statement
    /// boundary.
    ///
    /// Upstream `flowcontext.py:413 handle_bytecode` runs once per
    /// bytecode; at the simple-stmt level the Python compiler always
    /// emits matching push/pop counts so the stack returns to zero
    /// depth between simple stmts within a function body.
    ///
    /// Pyre's AST frontend collapses each Rust `syn::Stmt` to a single
    /// dispatch through `lower_stmt`.  Once Z4.B.1+ wire leaf-`Expr`
    /// pushes alongside the existing `lower_expr` returns, every
    /// consumer (Stmt::Local STORE_FAST analogue, Stmt::Expr POP_TOP
    /// analogue, every operator/call-arg site that today reads via
    /// `get_value_var!(lower_expr(...), graph)`) must pop in lockstep.  Any
    /// imbalance trips this assert at the *next* statement boundary,
    /// flagging the specific Stmt whose push/pop pair drifted.
    ///
    /// Today (Z4.B.0) `value_stack` is never written, so the assert
    /// is a no-op safety net — its job is to be in place before
    /// Z4.B.1 starts pushing.
    fn assert_stack_empty_at_stmt_boundary(&self, where_: &str) {
        debug_assert!(
            self.value_stack.is_empty(),
            "stmt-boundary stack imbalance at {where_}: depth {} expected 0 \
             (flowcontext.py:413 simple-stmt invariant; Z4.B.0 tripwire)",
            self.value_stack.len()
        );
    }

    /// `flowspace/flowcontext.py:424-463 mergeblock` — line-by-line
    /// port of the candidate-list-driven join-point handler.
    ///
    /// ```python
    /// def mergeblock(self, currentblock, currentstate):
    ///     next_offset = currentstate.next_offset
    ///     candidates = self.joinpoints.setdefault(next_offset, [])
    ///     for block in candidates:
    ///         newstate = block.framestate.union(currentstate)
    ///         if newstate is not None:
    ///             break
    ///     else:
    ///         newblock = self.make_next_block(currentblock, currentstate)
    ///         candidates.insert(0, newblock)
    ///         return
    ///
    ///     if newstate.matches(block.framestate):
    ///         outputargs = currentstate.getoutputargs(newstate)
    ///         currentblock.closeblock(Link(outputargs, block))
    ///         return
    ///
    ///     newblock = SpamBlock(newstate)
    ///     ...
    ///     outputargs = currentstate.getoutputargs(newstate)
    ///     link = Link(outputargs, newblock)
    ///     currentblock.closeblock(link)
    ///
    ///     block.dead = True
    ///     block.operations = ()
    ///     block.exitswitch = None
    ///     outputargs = block.framestate.getoutputargs(newstate)
    ///     block.recloseblock(Link(outputargs, newblock))
    ///     candidates.remove(block)
    ///
    ///     candidates.insert(0, newblock)
    ///     self.pendingblocks.append(newblock)
    /// ```
    ///
    /// Three control-flow arms:
    ///   - **No candidate accepts union** (`flowcontext.py:433-436`):
    ///     fresh SpamBlock via `make_next_block`, `currentstate` is
    ///     also the new block's framestate, candidate registered at
    ///     head.
    ///   - **First union-non-None candidate's merge equals it under
    ///     `matches`** (`:438-441`): no generalization needed; close
    ///     `currentblock` with a direct Link to the existing
    ///     candidate.
    ///   - **First union-non-None candidate's merge generalizes**
    ///     (`:443-463`): create a new SpamBlock for the merged state,
    ///     close `currentblock` to it, retire the old candidate by
    ///     clearing its body and replacing its exits with a forward to
    ///     the new block, then swap the candidate-list entry.
    ///
    /// Pyre adaptation: only `make_next_block`'s `pendingblocks.append`
    /// (`:472`, the first-arrival case) is omitted — the tree-recursive
    /// lowering visits every reachable block synchronously, so the new
    /// block becomes the merge target the caller lowers into directly.
    /// The generalize arm's `:463` append IS ported (it feeds the
    /// re-walk drain).  Returns the BlockId of the merge target so the
    /// caller can continue.
    ///
    /// Production migration blocker: pyre's existing AST merge sites
    /// build "lean" merge blocks via `create_block_with_arg_vars(0)` /
    /// `create_block_with_arg_vars(1)` with the locals threaded
    /// per-slot through `lazy_install_local_at_current_block` only
    /// when a fresh phi is actually needed.  This helper instead
    /// follows `flowcontext.py:443 SpamBlock(newstate)` and emits a
    /// merge block whose `inputargs` are ALL Variables in
    /// `newstate.getvariables()` (locals + flattened stack + exc).
    /// Switching a callsite over therefore changes the merge
    /// block's inputarg arity in ways that downstream `set_goto`
    /// callers (other merge sites that target this block) are not
    /// prepared for, and exposes the latent gap where pyre's
    /// `ctx.local_value_ids[name] = (vid, defining_block)` records
    /// can carry a `defining_block` that isn't transitively reachable
    /// from the predecessor (e.g. a local rebound in a sibling
    /// arm).  `ensure_variable_at_block` in
    /// `set_goto_from_framestate` then fails to backfill that
    /// Variable through the predecessor chain.
    ///
    /// Closing the gap structurally requires either (a) making every
    /// pyre AST merge block a SpamBlock with the full
    /// `getvariables()` inputarg shape (so set_goto callers thread
    /// the same N args uniformly) AND making `ctx.setstate` thread
    /// the actual merge-block as the `defining_block` for the phi
    /// locals it installs; or (b) a reachability-aware filter that
    /// strips graph-unreachable Variables from the framestate before
    /// `mergeblock`.  Multi-session.
    fn mergeblock(
        &mut self,
        graph: &mut FunctionGraph,
        currentblock: BlockId,
        currentstate: FrameState,
    ) -> BlockId {
        let next_offset = currentstate.next_offset;
        // `flowcontext.py:428` — `candidates = self.joinpoints.setdefault(
        // next_offset, [])`.  Snapshot the candidate list so the framestate
        // reads below can hold immutable graph borrows without conflicting
        // with the mutable `self.joinpoints` borrow we'll need at the end.
        let candidates_snapshot: Vec<BlockId> = self
            .joinpoints
            .get(&next_offset)
            .cloned()
            .unwrap_or_default();

        // `flowcontext.py:429-432` — walk candidates looking for a union
        // hit; on first non-None break out, preserving `block` (the
        // candidate that succeeded) and `newstate` (the union result).
        // Python's `for ... else` runs the else clause only when the
        // loop completes without breaking, which we encode as
        // `hit.is_none()`.
        let mut hit: Option<(BlockId, FrameState)> = None;
        for &cand in &candidates_snapshot {
            let cand_fs = graph
                .block(cand)
                .framestate
                .clone()
                .expect("mergeblock: candidate must be a SpamBlock (framestate-bearing)");
            if let Some(merged) = cand_fs.union(&currentstate, graph) {
                hit = Some((cand, merged));
                break;
            }
        }

        match hit {
            None => {
                // `flowcontext.py:465-472 make_next_block` —
                // `newstate = state.copy(); newblock = SpamBlock(newstate);
                //  outputargs = state.getoutputargs(newstate);
                //  block.closeblock(Link(outputargs, newblock))`
                let newstate = currentstate.copy(graph);
                let newblock = graph.create_block_from_framestate(&newstate);
                let outputargs = currentstate.getoutputargs(&newstate, graph);
                let link = Link::new_mixed(outputargs, newblock, None);
                graph.closeblock(currentblock, vec![link]);
                self.joinpoints
                    .entry(next_offset)
                    .or_default()
                    .insert(0, newblock);
                newblock
            }
            Some((cand, newstate)) => {
                let cand_fs = graph
                    .block(cand)
                    .framestate
                    .clone()
                    .expect("mergeblock: candidate must be a SpamBlock (framestate-bearing)");
                if newstate.matches(&cand_fs, graph) {
                    // `flowcontext.py:438-441` — direct link.
                    // `outputargs = currentstate.getoutputargs(newstate);
                    //  currentblock.closeblock(Link(outputargs, block))`
                    graph.closeblock_link(currentblock, cand, &currentstate, &cand_fs);
                    cand
                } else {
                    // `flowcontext.py:443-463` — generalize: new
                    // SpamBlock, link current→new, retire old candidate.
                    let newblock = graph.create_block_from_framestate(&newstate);
                    // `:444-447` —
                    //   varnames = self.pycode.co_varnames
                    //   for name, w_value in zip(varnames, newstate.locals_w):
                    //       if isinstance(w_value, Variable):
                    //           w_value.rename(name)
                    // `local_first_bind_order` is pyre's `co_varnames`
                    // (dense local-slot → source name).  `rename` honours
                    // its "don't rename twice" guard.
                    for (slot_idx, slot) in newstate.locals_w.iter().enumerate() {
                        if let Some(crate::flowspace::model::Hlvalue::Variable(v)) = slot {
                            if let Some(name) = self.local_first_bind_order.get(slot_idx).cloned() {
                                // `w_value.rename(name)` — `_name` is
                                // identity-shared (Slice 1), so renaming a
                                // clone propagates to the block's inputargs
                                // and the registry entry.
                                let mut vc = v.clone();
                                vc.rename(&name);
                            }
                        }
                    }
                    // `:449-451`
                    graph.closeblock_link(currentblock, newblock, &currentstate, &newstate);
                    // `:454-459`
                    let old_outputargs = cand_fs.getoutputargs(&newstate, graph);
                    {
                        let blk = graph.block_mut(cand);
                        blk.operations.clear();
                        blk.exitswitch = None;
                        blk.dead = true;
                    }
                    // `:458-459`
                    let link = Link::new_mixed(old_outputargs, newblock, None);
                    graph.recloseblock(cand, vec![link]);
                    // `:460-462`
                    let candidates = self.joinpoints.entry(next_offset).or_default();
                    candidates.retain(|&c| c != cand);
                    candidates.insert(0, newblock);
                    // `:463` — `self.pendingblocks.append(newblock)`.
                    // Enqueue the generalized SpamBlock for the re-walk
                    // drain (`drain_pendingblocks_for_merge_offset`,
                    // `flowcontext.py:399-422`).
                    //
                    // On every (borrow-checked) production graph this arm is
                    // unreachable — proof: `mergeblock`'s only production
                    // caller is the loop back-edge / continue site
                    // (`close_loop_back_edge_inner`), where the header is
                    // the sole candidate.  The header is
                    // `pre_loop_snapshot.copy()` (`build_loop_header_state`),
                    // which freshens every Variable slot; every loop-carried
                    // slot IS a Variable, because `getstate` wraps each
                    // local as `Hlvalue::Variable` (`local_value_ids` stores
                    // `(Variable, BlockId)` — never a `Constant`) and the
                    // header carries no stack/exc cells (fenced in
                    // `build_loop_header_state`).  So for every slot
                    // `union(Variable_header, X_body)` yields a fresh
                    // Variable (`framestate.py:113`) and `matches(merged,
                    // header)` sees Variable-vs-Variable everywhere → the
                    // direct-link arm above returns the header.  Only a
                    // None-killed loop-carried slot would route here, and
                    // that requires moving/`drop`-ing a loop-carried local
                    // without rebinding it before the back-edge — a
                    // cross-iteration use-after-move the borrow checker
                    // rejects (`E0382`).  So the drain is faithfully ported
                    // but its queue stays empty in production; the `drain_*`
                    // unit tests drive it directly.
                    self.pendingblocks.push(PendingBlock {
                        block: newblock,
                        state: newstate.clone(),
                        merge_offset: next_offset,
                    });
                    newblock
                }
            }
        }
    }

    /// The set of loop-header blocks whose `inputargs` arity is fixed by
    /// `create_block_from_framestate` and must NOT grow during a
    /// `thread_loop_link_args` backfill: every enclosing loop's
    /// `continue_target` on `loop_stack`, plus `header` (the loop being
    /// closed, which may not be on the stack yet — pre-loop close — or
    /// already popped — back-edge close).  Passed as `forbidden` to
    /// `can_thread_variable_to_block`.
    fn loop_forbidden_set(&self, header: BlockId) -> std::collections::HashSet<BlockId> {
        let mut forbidden: std::collections::HashSet<BlockId> =
            self.loop_stack.iter().map(|f| f.continue_target).collect();
        forbidden.insert(header);
        forbidden
    }

    /// `flowcontext.py:438-463` back-edge close mechanics — returns the
    /// `mergeblock` result so the caller distinguishes a direct-link
    /// (`== continue_target`) from a generalize (`!= continue_target`,
    /// with the new header enqueued on `pendingblocks` for the drain).
    /// Shared by the While / Loop / ForLoop arms, the `continue` site,
    /// and the re-walk drain (`flowcontext.py:399-422`).
    ///
    /// Snapshot the body-tail state, truncate it to the header's
    /// loop-carried slot count (body-internal locals at slots >=
    /// `header_slot_count` are dropped before the back-edge `union`),
    /// restamp the body-tail and header framestates, then `mergeblock`
    /// against the header.  On borrow-checked input the pre-widening
    /// invariant (`build_loop_header_state` freshens every loop-carried
    /// slot to a `Variable`) makes the `union` always `matches()` the
    /// header, so the arm's first close direct-links and the drain finds
    /// an empty queue — the re-walk path is faithfully ported but inert.
    fn close_loop_back_edge_inner(
        &mut self,
        graph: &mut FunctionGraph,
        body_tail: BlockId,
        loop_frame: &LoopFrame,
    ) -> BlockId {
        if !graph.block(body_tail).is_open() {
            return loop_frame.continue_target;
        }
        let mut body_exit_state = self.getstate(graph, loop_frame.merge_offset);
        body_exit_state
            .entries
            .truncate(loop_frame.header_slot_count);
        body_exit_state
            .locals_w
            .truncate(loop_frame.header_slot_count);
        graph.block_mut(body_tail).framestate = Some(body_exit_state.clone());
        // Restore header framestate before mergeblock — the body walk may
        // have overwritten it (pyre stamps framestates during lowering;
        // SpamBlock.framestate is immutable post-creation upstream).
        graph.block_mut(loop_frame.continue_target).framestate =
            Some(loop_frame.header_state.clone());
        // Backfill loop-carried slots the body never read on this path
        // (e.g. `cond` in `while i < n { if cond { break; } i = i + 1; }`,
        // or an outer var across an inner loop's back-edge) so the
        // `getoutputargs` back-edge link references only `body_tail`-defined
        // slots.  See `FunctionGraph::thread_loop_link_args`.
        let forbidden = self.loop_forbidden_set(loop_frame.continue_target);
        graph.thread_loop_link_args(
            body_tail,
            &body_exit_state,
            &loop_frame.header_state,
            &forbidden,
        );
        self.mergeblock(graph, body_tail, body_exit_state)
    }

    /// `flowcontext.py:407-422 record_block` — re-walk one generalized
    /// header block from its (post-generalize, NARROWER) framestate.
    /// Restore ctx to `pending.state`, replay the loop arm prologue onto
    /// the fresh header, re-walk the body, and re-close the back-edge —
    /// returning the `mergeblock` result so the drain loop can detect a
    /// fixpoint (`== pending.block`) or a further generalize (a new
    /// enqueue on `pendingblocks`).
    ///
    /// Uses `pending.state` (the narrower header), NOT
    /// `loop_frame.header_state`: re-stamping the pristine wide header at
    /// `close_loop_back_edge_inner` would re-generalize on every pass and
    /// never converge.
    fn rewalk_loop_body(
        &mut self,
        graph: &mut FunctionGraph,
        pending: &PendingBlock,
        body: &syn::Block,
        prologue: LoopPrologue<'_>,
        loop_frame: &LoopFrame,
        options: &AstGraphOptions,
    ) -> Result<BlockId, FlowingError> {
        // `:408 setstate(block.framestate)` — rebind ctx locals to the
        // generalized header's Variables.  `setstate_at_block` threads
        // the honest defining_block (`pending.block`), vs bare
        // `setstate`'s `BlockId(0)` placeholder.
        self.setstate_at_block(&pending.state, pending.block, graph);
        // `setstate` does not restore `loop_stack` (it ends at
        // `normalize_raise_signals`), so push a synthetic LoopFrame
        // around the re-walk for body break/continue.  continue_target
        // and header_state come from the NARROWER pending block.
        self.loop_stack.push(LoopFrame {
            continue_target: pending.block,
            merge_offset: loop_frame.merge_offset,
            header_slot_count: pending.state.entries.len(),
            break_target: loop_frame.break_target,
            header_state: pending.state.clone(),
            rewalk_budget_remaining: loop_frame.rewalk_budget_remaining,
        });
        let snapshot = LocalBindingSnapshot::capture(self);
        let body_entry =
            match prologue.replay(graph, self, pending.block, &pending.state, options)? {
                Some(b) => b,
                None => {
                    // The replayed prologue terminated the header without a
                    // body (a path-closing condition) — no back-edge to
                    // merge; the header is its own fixpoint.
                    self.loop_stack.pop();
                    snapshot.restore(self);
                    return Ok(pending.block);
                }
            };
        // `:412-414` — re-run the body bytecode (here: the body stmts).
        let mut body_tail = body_entry;
        for stmt in &body.stmts {
            let closed = lower_stmt(graph, &mut body_tail, stmt, options, self)?;
            if closed {
                break;
            }
        }
        let frame = self.loop_stack.pop().unwrap();
        let merged = self.close_loop_back_edge_inner(graph, body_tail, &frame);
        snapshot.restore(self);
        Ok(merged)
    }

    /// `flowcontext.py:402-405 build_flow` drain, SCOPED to one loop's
    /// `merge_offset`.  Re-walks every generalized header that
    /// `mergeblock` enqueued for this loop (`:463`) until the queue (for
    /// this offset) is empty, returning the converged `mergeblock`
    /// results.
    ///
    /// Scoped via `position()` by `merge_offset` rather than draining the
    /// flat `pendingblocks` Vec, so nested loops — whose distinct
    /// negative `merge_offset` keys interleave on the shared queue — keep
    /// independent drains (the inner loop fully drains before the outer
    /// arm runs).  FIFO within an offset (`:403 popleft`).
    fn drain_pendingblocks_for_merge_offset(
        &mut self,
        graph: &mut FunctionGraph,
        merge_offset: i64,
        body: &syn::Block,
        prologue: LoopPrologue<'_>,
        loop_frame: &LoopFrame,
        options: &AstGraphOptions,
    ) -> Result<Vec<BlockId>, FlowingError> {
        let mut converged = Vec::new();
        // Termination guard (parity rule 5, no RPython basis — see
        // `LoopFrame::rewalk_budget_remaining`).  `flowcontext.py:402`
        // trusts the monotone None-kill lattice alone; pyre adds the
        // bound as a release-safe backstop.
        let mut budget = loop_frame.rewalk_budget_remaining;
        loop {
            // `:403 popleft` — FIFO first entry matching this merge_offset.
            let pos = match self
                .pendingblocks
                .iter()
                .position(|pb| pb.merge_offset == merge_offset)
            {
                Some(p) => p,
                None => break,
            };
            let pending = self.pendingblocks.remove(pos);
            // `:404 if not block.dead`
            if graph.block(pending.block).dead {
                continue;
            }
            assert!(
                budget > 0,
                "re-walk drain exceeded header_slot_count + 1 iterations for \
                 merge_offset {merge_offset}; the monotone None-kill lattice \
                 should bound it (task #91)"
            );
            budget -= 1;
            // `:405 record_block(block)` — re-walk the generalized header.
            let merged =
                self.rewalk_loop_body(graph, &pending, body, prologue, loop_frame, options)?;
            converged.push(merged);
        }
        Ok(converged)
    }

    /// cfg(test) sibling of [`Self::bind_local_id`] that accepts the
    /// raw dense-slot index — lets test fixtures spell the local
    /// binding as `ctx.bind_local_id_at(name, N, block, &mut graph)`
    /// instead of allocating a `Variable` first.  If slot `idx` does
    /// not yet exist on `graph`, a placeholder `Variable` is minted
    /// via [`FunctionGraph::bind_variable_at`] so the carrier always
    /// holds a real handle.
    #[cfg(test)]
    fn bind_local_id_at(
        &mut self,
        name: String,
        idx: usize,
        defining_block: BlockId,
        graph: &mut FunctionGraph,
    ) {
        let var = match graph.variable_at(idx) {
            Some(v) => v.clone(),
            None => {
                let v = crate::flowspace::model::Variable::new();
                graph.bind_variable_at(idx, v.clone());
                v
            }
        };
        self.bind_local_id_var(name, &var, graph, defining_block);
    }

    /// Bind a local name to a `(Variable, defining_block)` pair via the
    /// existing `local_value_ids` side-table.  On *first* bind (name
    /// never seen by this graph) the name is also appended to
    /// `local_first_bind_order` and recorded in
    /// `local_first_bind_seen` so its slot position is fixed for the
    /// remainder of the build, even across `LocalBindingSnapshot::
    /// restore`.  On rebind the slot position is preserved.  RPython
    /// parity: `co_varnames` slot indices are assigned at compile time
    /// and never reshuffled.  The `(Variable, BlockId)` carrier holds
    /// the upstream identity directly so no projection is needed at the
    /// write site.
    fn bind_local_id_var(
        &mut self,
        name: String,
        var: &crate::flowspace::model::Variable,
        _graph: &crate::model::FunctionGraph,
        defining_block: BlockId,
    ) {
        if !self.local_first_bind_seen.contains(&name) {
            self.local_first_bind_seen.insert(name.clone());
            self.local_first_bind_order.push(name.clone());
        }
        self.local_value_ids
            .insert(name, (var.clone(), defining_block));
    }

    /// Project `local_value_ids[name]` into a `(Variable, BlockId)`
    /// pair.  Lets readers extract the backing `Variable` handle
    /// directly without re-projecting through `graph.must_variable_at`.
    /// `_graph` is retained for caller-side symmetry with the
    /// previous signature.
    fn local_var_of(
        &self,
        name: &str,
        _graph: &FunctionGraph,
    ) -> Option<(crate::flowspace::model::Variable, BlockId)> {
        self.local_value_ids.get(name).cloned()
    }
}

#[derive(Clone)]
struct LocalBindingSnapshot {
    local_type_roots: HashMap<String, String>,
    local_type_strings: HashMap<String, String>,
    local_value_types: HashMap<String, ValueType>,
    local_value_ids: HashMap<String, (crate::flowspace::model::Variable, BlockId)>,
    local_trait_bound_roots: HashMap<String, String>,
    local_array_types: HashMap<String, String>,
    local_dyn_trait_roots: HashMap<String, String>,
    local_closure_returns: HashMap<String, String>,
    // NB: `local_first_bind_order` and `local_first_bind_seen` are
    // intentionally NOT captured here — they are graph-wide
    // append-only and survive sibling-arm restores so the slot index
    // of any name stays invariant across the entire lowering.
    // RPython parity: `co_varnames` is fixed at compile time, so
    // the slot map cannot be rolled back by control flow.
}

impl LocalBindingSnapshot {
    fn capture(ctx: &GraphBuildContext<'_>) -> Self {
        Self {
            local_type_roots: ctx.local_type_roots.clone(),
            local_type_strings: ctx.local_type_strings.clone(),
            local_value_types: ctx.local_value_types.clone(),
            local_value_ids: ctx.local_value_ids.clone(),
            local_trait_bound_roots: ctx.local_trait_bound_roots.clone(),
            local_array_types: ctx.local_array_types.clone(),
            local_dyn_trait_roots: ctx.local_dyn_trait_roots.clone(),
            local_closure_returns: ctx.local_closure_returns.clone(),
        }
    }

    fn restore(self, ctx: &mut GraphBuildContext<'_>) {
        ctx.local_type_roots = self.local_type_roots;
        ctx.local_type_strings = self.local_type_strings;
        ctx.local_value_types = self.local_value_types;
        ctx.local_value_ids = self.local_value_ids;
        ctx.local_trait_bound_roots = self.local_trait_bound_roots;
        ctx.local_array_types = self.local_array_types;
        ctx.local_dyn_trait_roots = self.local_dyn_trait_roots;
        ctx.local_closure_returns = self.local_closure_returns;
        // local_first_bind_order / local_first_bind_seen NOT
        // restored — see struct doc comment.
    }
}

/// Lazy cross-block local installer.
///
/// Triggered from `Expr::Path`'s cross-block branch when the local
/// `name` is bound in a block other than `current_block`.  Allocates a
/// fresh `OpKind::Input { name, ty }` in `current_block`, registers it
/// as `current_block.inputargs`, rewrites `ctx.local_value_ids[name]`
/// to point at the new inputarg `Variable`, and **threads back** the
/// predecessor side of the join: for every immediate predecessor edge
/// `(pred_block, exit_idx)` landing at `current_block`, the snapshot
/// recorded in `pred_block.framestate` supplies a candidate
/// predecessor-side `Variable` for `name`.  When that
/// candidate is itself defined in `pred_block` (its inputarg or an
/// op result), it is appended to `pred_block.exits[exit_idx].args`
/// directly.  When it was inherited from a dominator (not defined
/// in `pred_block` itself — e.g. an empty intermediate merge block
/// that just forwards a parameter), the installer **recurses** to
/// install `name` as an inputarg of `pred_block` first, walking the
/// predecessor chain back until the recursion lands on a block that
/// defines the local.  RPython: equivalent of
/// `flowspace/flowcontext.py:835 LOAD_FAST` pulling a fresh
/// `Variable` into the merge block while `flowspace/model.py:114
/// Link.__init__` keeps `len(args) == len(target.inputargs)`
/// invariant — RPython does this work implicitly because every basic
/// block edge in the recorder threads the `frame.locals_w` slot
/// array; pyre's frontend builds the same shape on demand only at
/// cross-block reads (lazy) instead of preemptively at every block
/// boundary (eager) so blocks with no cross-block readers stay
/// zero-arity.
///
/// **Stage B2 (final)**: the conservative fence is gone — the
/// installer fires for every cross-block local read regardless of
/// `current_block`'s op count.  The hazard the fence used to mask was
/// (a) duplicate phi inputargs for the same name (closed by the
/// idempotency check below — a second read of `name` in
/// `current_block` reuses the inputarg from an earlier install), and
/// (b) UnaryOp("neg", Float) annotating to Int by default and
/// poisoning downstream phi-merge inputargs through union(Int,
/// Float) → Unknown → GcRef backfill (closed by the
/// `rfloat.py:rtype_neg` parity arm in
/// `translator/rtyper/legacy_annotator.rs` and the matching
/// `infer_concrete_from_op` Unknown-pass-through in
/// `translator/rtyper/legacy_resolve.rs`).
///
/// Returns `Some(new_vid)` on success, `None` if any predecessor lacks
/// a recorded snapshot or whose snapshot lacks `name` (Slice P2
/// retired the type-disagreement abort path — the wildcard fold
/// widens disagreeing concrete kinds to Unknown instead).  The call
/// site falls back to the legacy naked-`Input` emit when `None` is
/// returned.
/// `pre_allocated_var`: when `Some(var)`, use the caller-supplied
/// `Variable` for the fresh phi instead of allocating a new one.
/// Used by union callers (`Expr::If`, `Expr::Match`) that pre-allocate
/// phi variables inside `FrameState::union(_into)` so the merged
/// state can be returned with variables materialised; the install is
/// then emitted with the same `Variable` the merged state already
/// carries.  `None` preserves the legacy behaviour (allocate inside).
fn lazy_install_local_at_current_block_var(
    graph: &mut crate::model::FunctionGraph,
    ctx: &mut GraphBuildContext<'_>,
    current_block: BlockId,
    name: &str,
    pre_allocated_var: Option<crate::flowspace::model::Variable>,
) -> Option<crate::flowspace::model::Variable> {
    // Reuse — `name` may already have been installed at `current_block`
    // by an earlier read in the same block (prior recursion into a
    // shared predecessor, etc.).  Treat the same-block hit as the
    // canonical answer.
    if let Some((var, def_block)) = ctx.local_var_of(name, graph)
        && def_block == current_block
    {
        return Some(var);
    }

    // Idempotency by graph state: if a prior lazy install for `name`
    // already added an inputarg-anchored `OpKind::Input { name }` to
    // `current_block`, reuse it.  RPython parity:
    // `flowcontext.py:407 setstate(block.framestate)` makes the same
    // local slot read multiple times in a block resolve to the same
    // Variable; the duplicate-install hazard arises because pyre's
    // `LocalBindingSnapshot::restore` (Stage B1) wipes the ctx-side
    // `local_value_ids` cache between then/else arms while the
    // graph-side inputarg from the then-arm's lazy install still
    // exists, so the else-arm's recursion would otherwise allocate a
    // second phi slot for the same name.  Checking the graph's
    // inputargs list directly closes the gap without depending on
    // ctx state.
    {
        let block = graph.block(current_block);
        for op in &block.operations {
            if let (Some(result_var), OpKind::Input { name: op_name, .. }) =
                (op.result.as_ref(), &op.kind)
                && op_name == name
                && block.inputargs.contains(result_var)
            {
                return Some(result_var.clone());
            }
        }
    }

    let pred_edges: Vec<(BlockId, usize)> = graph
        .blocks
        .iter()
        .flat_map(|b| {
            let bid = b.id;
            b.exits.iter().enumerate().filter_map(move |(i, exit)| {
                if exit.target == current_block {
                    Some((bid, i))
                } else {
                    None
                }
            })
        })
        .collect();
    if pred_edges.is_empty() {
        return None;
    }

    // Audit Cat 2-1 cycle-safe shape: split into two phases.  The
    // outer `OpKind::Input` is pushed and the inputarg slot is
    // reserved BEFORE any recursive predecessor walk, so a recursion
    // back into `current_block` (e.g. a back-edge predecessor whose
    // own pred-walk reaches `current_block`) finds the in-progress
    // phi via the graph-state idempotency check above and short-
    // circuits with the pre-allocated vid.  RPython
    // `flowspace/flowcontext.py:438-451 mergeblock` reaches the same
    // fixpoint by iteratively applying `union` until predecessor
    // links converge — pyre's static AST collapses that to a single
    // pass with the in-flight inputarg slot acting as the iteration
    // checkpoint.
    //
    // Phase 1 (read-only): collect each predecessor's snap_vid and
    // decide whether the snapshot's vid is directly usable in
    // `pred_block` or whether `pred_block` will need its own
    // recursive install.  Predecessor snapshots fold their observed
    // `graph_value_type` via the wildcard rule (`Unknown` carries
    // the concrete sibling through; concrete-vs-different-concrete
    // widens to Unknown — the abort-on-disagreement path was retired,
    // path, types are downstream concerns per
    // `framestate.py:union`).
    struct PredSnap {
        pred_block: BlockId,
        exit_idx: usize,
        snap_var: crate::flowspace::model::Variable,
        needs_recurse: bool,
    }
    let mut pred_snaps: Vec<PredSnap> = Vec::with_capacity(pred_edges.len());
    let mut shared_value_type: Option<ValueType> = None;
    for (pred_block, exit_idx) in &pred_edges {
        // Stage A1: predecessor framestate is now per-block, captured
        // at close time (`Block.framestate`).  All exits of one
        // `set_branch` share the same close-time state; the per-edge
        // `ctx.exit_snapshots` HashMap is gone.  RPython parity:
        // `flowspace/flowcontext.py:38 SpamBlock.framestate`.
        let snap_var = {
            // `framestate.py:locals_w` is a positional slot vector;
            // resolve `name → slot_idx` via the graph-wide first-bind
            // order, mirroring upstream's `co_varnames.index(name)`
            // lookup pattern.  Consult `snap.locals_w` per upstream's
            // source of truth — pyre's `getstate` populates the
            // `Hlvalue` carrier in lockstep with `entries`, so the
            // `locals_w`-driven lookup recovers the same Variable
            // identity the framestate captured.
            let slot_idx = ctx.local_first_bind_order.iter().position(|n| n == name)?;
            let snap = graph.block(*pred_block).framestate.as_ref()?;
            let view = snap.locals_w_view(graph);
            let cell = view.get(slot_idx)?;
            match cell {
                Some(crate::flowspace::model::Hlvalue::Variable(v)) => v.clone(),
                _ => return None,
            }
        };
        let needs_recurse = !graph.variable_defined_in_block(*pred_block, &snap_var);
        // Type folds across predecessors via the wildcard rule:
        // concrete-vs-Unknown carries the concrete kind through;
        // concrete-vs-different-concrete widens to Unknown so the
        // freshly-installed inputarg's `ty` is the most-general kind
        // observable across this merge.  Mirrors upstream's pattern
        // where flow-space `Variable` carries no type and rtyper
        // assigns `concretetype` post-flow (`framestate.py:union`
        // never inspects types).  Pyre's prior deviation here
        // failed the install on concrete disagreement; that gate
        // was retired after a dry-run audit confirmed
        // zero fixture / production conflicts (cargo lib 2557/0/3
        // + check.py 14/14×2 PASS).
        let observed_type = graph_value_type_var(graph, &snap_var).unwrap_or(ValueType::Unknown);
        match (&shared_value_type, &observed_type) {
            (None, _) => shared_value_type = Some(observed_type.clone()),
            (Some(prior), new) if prior == new => {}
            (Some(ValueType::Unknown), _) => shared_value_type = Some(observed_type.clone()),
            (Some(_), ValueType::Unknown) => {}
            (Some(ValueType::Ref(_)), ValueType::Ref(_)) => {
                shared_value_type = Some(ValueType::Ref(None))
            }
            (Some(_), _) => shared_value_type = Some(ValueType::Unknown),
        }
        pred_snaps.push(PredSnap {
            pred_block: *pred_block,
            exit_idx: *exit_idx,
            snap_var,
            needs_recurse,
        });
    }

    // If every predecessor's `graph_value_type(snap_vid)` returned
    // Unknown (e.g. the snap_vid points to an Input op that was itself
    // installed by an earlier lazy_install whose own predecessors
    // could not resolve a concrete kind), fall back to
    // `ctx.local_value_types[name]` — populated by `Stmt::Local`
    // (`let mut x: bool = ...`), `build_loop_header_state` (loop
    // phi's resolved kind), and function-parameter registration.
    //
    // RPython parity: `Variable.concretetype` is the rtyper-side type
    // tracking, which carries across cross-block reads independently
    // of the framestate Hlvalue identity.  Pyre's `local_value_types`
    // is the annotator-lattice analogue at the AST frontend.  Slice
    // 2.3 retired the per-slot `FrameStateEntry::value_type` (NEW-
    // DEVIATION); the convergence path defers type unification to a
    // future annotator port.  Until that port lands, this fallback
    // keeps the lazy installer's freshly-emitted Input op tagged with
    // the lattice's best-known kind so downstream consumers
    // (`expr_unary_not_operand_kind` and the assembler's per-kind
    // copy invariant) see a concrete type for locals that were
    // concretely-typed at let-binding time.
    let value_type = shared_value_type.unwrap_or(ValueType::Unknown).clone();
    let value_type = if matches!(value_type, ValueType::Unknown) {
        ctx.local_value_types
            .get(name)
            .cloned()
            .unwrap_or(ValueType::Unknown)
    } else {
        value_type
    };

    // Phase 2 (graph mutation): push the outer `OpKind::Input` +
    // inputarg slot first, snapshot the prior ctx binding for
    // rollback, then walk predecessors (recursively installing where
    // needed).  The recursion's same-block idempotency check on
    // `block.inputargs` finds this outer call's freshly-installed
    // inputarg and returns its vid, breaking the
    // `current_block → pred → ... → current_block` cycle that
    // arises when a back-edge predecessor's snap_vid is defined in
    // `current_block` (e.g. body_tail / continue source whose
    // framestate snap for a header-phi name points back at the
    // header).
    let prior_ctx_lvi = ctx.local_var_of(name, graph);
    let prior_ctx_lvt = ctx.local_value_types.get(name).cloned();
    let new_var = if let Some(var) = pre_allocated_var {
        graph.push_op_with_result_var(
            current_block,
            OpKind::Input {
                name: name.to_string(),
                ty: value_type.clone(),
            },
            var.clone(),
        );
        var
    } else {
        graph.push_op_var(
            current_block,
            OpKind::Input {
                name: name.to_string(),
                ty: value_type.clone(),
            },
            true,
        )?
    };
    graph.name_value_var(&new_var, name.to_string());
    graph.push_inputarg_var(current_block, new_var.clone());
    ctx.bind_local_id_var(name.to_string(), &new_var, graph, current_block);
    ctx.local_value_types
        .insert(name.to_string(), value_type.clone());
    // Predecessor source vids whose graph result type was Unknown when
    // observed.  If the wildcard rule above promoted `value_type` to
    // a concrete kind from a sibling predecessor, every Unknown
    // source must be retagged so the freshly-installed merge
    // inputarg's `i`/`r`/`f` kind matches every incoming link arg at
    // codegen.  Without retag, the merge inputarg gets concrete `ty`
    // but a predecessor's source op still produces an Unknown-banked
    // value, tripping the assembler's `int_copy` / `ref_copy` /
    // `float_copy` same-kind invariant.
    let mut unknown_predecessor_vars: Vec<crate::flowspace::model::Variable> = Vec::new();
    let mut pred_link_args: Vec<(BlockId, usize, crate::flowspace::model::Variable)> =
        Vec::with_capacity(pred_snaps.len());
    let mut rollback = false;
    for snap in pred_snaps {
        let resolved_var = if snap.needs_recurse {
            match lazy_install_local_at_current_block_var(graph, ctx, snap.pred_block, name, None) {
                Some(var) => var,
                None => {
                    rollback = true;
                    break;
                }
            }
        } else {
            snap.snap_var
        };
        // Type-validation: retired.  Phase 1's
        // wildcard fold (`(Some(_), _) => Unknown` arm) widens
        // disagreeing concrete kinds to Unknown so the freshly-
        // installed inputarg's `ty` is the upper bound observable
        // across this merge.  Per-link resolved-type validation here
        // was a deviation duplicating that fold; the audit dry-
        // run confirmed it never fired on production fixtures.
        // Types are downstream concerns (annotator + rtyper);
        // flowspace's job is the merge shape, not the kind
        // reconciliation.  The Unknown-source-pred retag below
        // (driven by `unknown_predecessor_vars`) survives as the
        // wildcard widening's link-arg side.
        let resolved_type =
            graph_value_type_var(graph, &resolved_var).unwrap_or(ValueType::Unknown);
        if matches!(resolved_type, ValueType::Unknown) {
            unknown_predecessor_vars.push(resolved_var.clone());
        }
        pred_link_args.push((snap.pred_block, snap.exit_idx, resolved_var));
    }

    if rollback {
        // Undo Phase 2's graph + ctx mutations so the caller's
        // naked-`Input` fallback emits without a dangling inputarg
        // at `current_block`.  Recursive installs that succeeded at
        // sibling pred_blocks before the failing one are NOT rolled
        // back — those are valid installs at their own blocks and
        // remain useful for any later read.  The only orphan to
        // clean is `current_block`'s own outer inputarg.
        let block = graph.block_mut(current_block);
        let popped_op = block.operations.pop();
        debug_assert!(
            matches!(popped_op.as_ref().map(|op| &op.kind), Some(OpKind::Input { name: n, .. }) if n == name),
            "rollback expected Input op for {name:?} at the operations tail",
        );
        let popped_inputarg = block.inputargs.pop();
        debug_assert_eq!(popped_inputarg.as_ref(), Some(&new_var));
        match prior_ctx_lvi {
            Some((var, def_block)) => {
                ctx.bind_local_id_var(name.to_string(), &var, graph, def_block);
            }
            None => {
                ctx.local_value_ids.remove(name);
            }
        }
        match prior_ctx_lvt {
            Some(vt) => {
                ctx.local_value_types.insert(name.to_string(), vt);
            }
            None => {
                ctx.local_value_types.remove(name);
            }
        }
        return None;
    }

    // If the wildcard rule promoted `value_type` to a concrete kind,
    // retag every Unknown-typed predecessor source so the assembler's
    // same-kind copy invariant holds across each incoming link arg.
    // Mirrors the `Expr::Try` `retag_result_value_type` widening used
    // after `?` unwraps a `Result<T, E>` — same operation, here
    // driven by cross-arm wildcard widening rather than the
    // type-extractor's `Ok`-arm narrowing.
    if !matches!(value_type, ValueType::Unknown) {
        for var in &unknown_predecessor_vars {
            retag_result_value_type(graph, var, value_type.clone());
        }
    }

    for (pred_block, exit_idx, pred_var) in pred_link_args {
        let arg = crate::model::LinkArg::Value(pred_var);
        graph.block_mut(pred_block).exits[exit_idx].args.push(arg);
    }

    // Re-establish `ctx`'s binding for `name` to *this* block's freshly
    // installed inputarg.  The predecessor-threading loop above recurses
    // into `lazy_install_local_at_current_block_var(pred_block, ..)` when
    // a predecessor inherited `name` from a dominator, and each recursive
    // frame rebinds `ctx.local_value_ids[name]` to *its* block's inputarg
    // (the bind before the loop).  Because `ctx` is a single shared frame
    // those inner rebinds leak past their own scope, so without this
    // restore a caller that reads `name` right after the install resolves
    // to the deepest predecessor's Variable instead of `current_block`'s
    // — the read then threads an out-of-`current_block`-scope slot onto a
    // later branch's `Link.args` and trips the adapter's
    // "undefined operand slot" invariant.  RPython has no such leak:
    // `flowspace/flowcontext.py:407 setstate(block.framestate)`
    // re-establishes `frame.locals_w` for whichever block is being
    // recorded, so a read of `name` while recording `current_block`
    // always yields that block's slot Variable.  Re-asserting the bind
    // here is pyre's static-AST analogue of that per-block setstate.
    ctx.bind_local_id_var(name.to_string(), &new_var, graph, current_block);
    ctx.local_value_types
        .insert(name.to_string(), value_type.clone());

    Some(new_var)
}

// Build a SemanticFunction from a Rust function AST. Mirrors RPython
// `flowspace/objspace.py:38` `build_flow()` — `FlowingError` propagates to
// the caller rather than producing a partial graph.
thread_local! {
    /// MAJIT_UNKNOWN_DUMP diagnostic context: name of the function
    /// currently being lowered. Set on `build_function_graph` entry
    /// and restored on exit so the per-`syn::Expr` Unknown emit sites
    /// can attribute their stub to the source function. Read-only
    /// elsewhere — purely cosmetic for the dump output.
    static CURRENT_LOWERING_FN_NAME: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII guard for `CURRENT_LOWERING_FN_NAME` — restores the previous
/// fn name on Drop so a `?` early-exit inside `build_function_graph`
/// still leaves the thread-local in a sane state for sibling lowerings.
struct LoweringFnNameGuard {
    previous: Option<String>,
}

impl Drop for LoweringFnNameGuard {
    fn drop(&mut self) {
        let prev = self.previous.take();
        CURRENT_LOWERING_FN_NAME.with(|c| *c.borrow_mut() = prev);
    }
}


/// RPython: extract function-level JIT hints from attributes.
/// Maps JIT hint attributes to effectinfo classification strings.
///
/// Recognizes both legacy `jit_*` and RPython-parity names.
///
/// For `#[oopspec("spec")]`, returns `"oopspec:spec_string"` so the hint
/// consumer can extract the spec via `hint.strip_prefix("oopspec:")`.
fn collect_jit_hints(attrs: &[syn::Attribute], sig: Option<&syn::Signature>) -> Vec<String> {
    let mut hints = Vec::new();
    let mut saw_oopspec = false;
    for attr in attrs {
        if let Some(segment) = attr.path().segments.last() {
            let name = segment.ident.to_string();
            match name.as_str() {
                // RPython-parity names (rlib/jit.py)
                "elidable" | "jit_elidable" => hints.push("elidable".into()),
                // `rlib/jit.py:184-201 elidable_promote` is no longer
                // collapsed onto the user-facing function's hints here.
                // `build_graphs_from_items` synthesizes the
                // (`_orig_<NAME>_unlikely_name`, wrapper) pair before
                // this collector ever runs, attaches the synthetic
                // `#[elidable]` attribute to the original, and strips
                // `#[elidable_promote]` from the wrapper's `attrs`.
                // The orig is what RPython's `elidable(func)` at
                // jit.py:185 marks with `_elidable_function_`; the
                // wrapper (`result` at jit.py:198-201) carries no
                // binary flag.  `synthesize_elidable_promote_pair`
                // always succeeds — unrecognised binder patterns (which
                // Python signatures cannot express anyway) panic with
                // a citation to `jit.py:172-178 _get_args(func)`, so
                // there is no silent single-graph fallback.
                //
                // `call.py:292-299 getcalldescr` runs `_canraise(op)` on
                // every elidable callsite to recover the `EF_ELIDABLE_*`
                // 3-way split.  Pyre's `_canraise` is conservative for
                // callees outside `function_graphs` (Vec::len,
                // pyframe_get_pycode, etc.) — `analyze_external_call`
                // defaults to `True` (`call.rs:3631`), so the analyser
                // alone cannot recover `EF_ELIDABLE_CANNOT_RAISE` /
                // `EF_ELIDABLE_OR_MEMORYERROR` even on callees the user
                // has explicitly annotated.  Preserve the assertion as a
                // distinct hint string alongside the canonical
                // `"elidable"` so `lib.rs` can register it with
                // `mark_cannot_raise_assertion` /
                // `mark_memerror_only_assertion` and
                // `getcalldescr`'s elidable branch can honour the
                // user-asserted shape before falling back to
                // `_canraise`.
                "elidable_cannot_raise" => {
                    hints.push("elidable".into());
                    hints.push("elidable_cannot_raise".into());
                }
                "elidable_or_memerror" => {
                    hints.push("elidable".into());
                    hints.push("elidable_or_memerror".into());
                }
                "dont_look_inside" => hints.push("dont_look_inside".into()),
                "unroll_safe" => hints.push("unroll_safe".into()),
                "loop_invariant" | "jit_loop_invariant" => {
                    hints.push("loopinvariant".into());
                }
                "not_in_trace" => hints.push("not_in_trace".into()),
                // rlib/jit.py:250 — `@oopspec(spec)`: extract spec string.
                "oopspec" => {
                    if let Ok(lit) = attr.parse_args::<syn::LitStr>() {
                        hints.push(format!("oopspec:{}", lit.value()));
                    } else {
                        hints.push("oopspec".into());
                    }
                    saw_oopspec = true;
                }
                // majit-specific
                "jit_close_stack" => hints.push("close_stack".into()),
                "jit_cannot_collect" => hints.push("cannot_collect".into()),
                "jit_gc_effects" => hints.push("gc_effects".into()),
                _ => {}
            }
        }
    }
    // `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]`
    // — when `#[oopspec(...)]` is present and the function signature
    // is available, emit a paired `"oopspec_argnames:arg1,arg2,..."`
    // hint so `lib.rs:598-600` can populate
    // `CallControl::mark_oopspec_argnames` alongside `mark_oopspec`.
    //
    // `support.py:713 argname2index = dict(zip(argnames, [Index(n) for n
    // in range(nb_args)]))` requires the declaration-order names.
    // Upstream's `co_varnames[:nb_args]` includes the method's
    // `self` parameter when it's a bound method (Python's
    // `co_varnames[0] = 'self'` convention), so for strict parity the
    // Rust port maps `FnArg::Receiver` to the synthetic name `"self"`.
    // Non-`Pat::Ident` patterns (tuple destructuring, wildcards) have
    // no single name to bind — upstream `co_varnames` would record
    // the destructured locals individually, but pyre's
    // `argname2index` lookup is positional and cannot multiplex one
    // slot to many names.  Emitting a `"_"` placeholder there would
    // be a deviation (the hint would shadow no real upstream
    // identifier and silently mis-bind any oopspec literal that
    // happens to spell `_`).  When such a pattern appears, refuse to
    // emit the argnames hint entirely so `decode_builtin_call` falls
    // back to the positional / bare-name path — the same behaviour
    // upstream gets when `co_varnames` is unavailable.
    if saw_oopspec {
        if let Some(sig) = sig {
            let mut argnames: Vec<String> = Vec::with_capacity(sig.inputs.len());
            let mut skip_hint = false;
            for arg in sig.inputs.iter() {
                match arg {
                    // `co_varnames[0]` for a bound method is `'self'`.
                    syn::FnArg::Receiver(_) => argnames.push("self".to_string()),
                    syn::FnArg::Typed(pat_type) => match &*pat_type.pat {
                        syn::Pat::Ident(ident) => argnames.push(ident.ident.to_string()),
                        _ => {
                            skip_hint = true;
                            break;
                        }
                    },
                }
            }
            if !skip_hint && !argnames.is_empty() {
                hints.push(format!("oopspec_argnames:{}", argnames.join(",")));
            }
        }
    }
    hints
}

// ── Statement lowering ──────────────────────────────────────────

/// Public entry point for lowering a single statement into a graph.
/// Used by the graph-based classifier in lib.rs to analyze resolved method bodies.
///
/// RPython `flowspace/objspace.py:38` — `FlowingError` propagates.  The
/// caller is responsible for handling the unsupported-construct signal
/// (typically by discarding the partially-built graph).  The boolean
/// result mirrors `lower_stmt`: `true` means the path terminated
/// (return/break/continue/raise) and the enclosing walker should stop.
pub fn lower_stmt_pub(
    graph: &mut FunctionGraph,
    block: BlockId,
    stmt: &syn::Stmt,
) -> Result<bool, FlowingError> {
    let mut block = block;
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = HashMap::new();
    let empty_suffix_index = MethodSuffixIndex::default();
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    let mut ctx = GraphBuildContext::new(
        &empty_registry,
        &empty_fn_ret,
        &empty_suffix_index,
        "",
        HashMap::new(),
        &empty_names,
        &empty_trait_names,
    );
    lower_stmt(
        graph,
        &mut block,
        stmt,
        &AstGraphOptions::default(),
        &mut ctx,
    )
}

/// Lower a sequence of statements whose final element may be a tail
/// expression (Rust block-value form: `{ stmt; stmt; expr }`).
///
/// RPython flow-space guarantee: every source expression is walked
/// exactly once (`rpython/flowspace/flowcontext.py::FlowContext.record`
/// appends each bytecode op once). Rust `syn::Block` / `ExprBlock` /
/// `ExprUnsafe` / `ExprIf.then_branch` all carry `Vec<Stmt>` with an
/// optional `Stmt::Expr(_, None)` tail whose value becomes the block's
/// value — lowering that tail via both `lower_stmt` (which delegates to
/// `lower_expr`) and a second `lower_expr` call would emit the op
/// twice and break the "walk once" invariant.
fn lower_stmt_list_with_tail_value(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    stmts: &[syn::Stmt],
    options: &AstGraphOptions,
    ctx: &mut GraphBuildContext,
) -> Result<Lowered, FlowingError> {
    let Some((last, prefix)) = stmts.split_last() else {
        return Ok(Lowered::no_value());
    };
    // Prefix stmts: walk each; if one closes the path
    // (`return x;`, `panic!();`, ...), stop — remaining stmts are
    // unreachable, mirroring RPython `flowspace/flowcontext.py`'s
    // `FlowSignal` propagation where `Return`/`Raise` halts the
    // current recorder before the next bytecode runs.
    for stmt in prefix {
        let path_closed = lower_stmt(graph, block, stmt, options, ctx)?;
        if path_closed {
            return Ok(Lowered::path_closed());
        }
    }
    match last {
        syn::Stmt::Expr(expr, None) => lower_expr(graph, block, expr, options, ctx),
        _ => {
            let path_closed = lower_stmt(graph, block, last, options, ctx)?;
            Ok(Lowered {
                value: None,
                path_closed,
            })
        }
    }
}

fn lower_stmt(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    stmt: &syn::Stmt,
    options: &AstGraphOptions,
    ctx: &mut GraphBuildContext,
) -> Result<bool, FlowingError> {
    // Z4.B.0 tripwire: simple-stmt boundary requires `value_stack`
    // depth == 0 per `flowcontext.py:413 handle_bytecode` (each
    // bytecode handler's net push/pop balances inside a simple-stmt
    // window).  Today `value_stack` is never written; the assert is a
    // no-op safety net.  Once Z4.B.1+ wires leaf push + consumer pop,
    // a miscounted producer or consumer trips this assert at the next
    // Stmt enter.  Bottom-of-stmt assert lives at the wrapper exit
    // below so successful paths re-establish the invariant for the
    // next stmt.
    ctx.assert_stack_empty_at_stmt_boundary("lower_stmt entry");
    let result = lower_stmt_inner(graph, block, stmt, options, ctx);
    if result.is_ok() {
        ctx.assert_stack_empty_at_stmt_boundary("lower_stmt exit (Ok)");
    }
    result
}

fn lower_stmt_inner(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    stmt: &syn::Stmt,
    options: &AstGraphOptions,
    ctx: &mut GraphBuildContext,
) -> Result<bool, FlowingError> {
    match stmt {
        syn::Stmt::Expr(expr, _) => {
            let lowered = lower_expr(graph, block, expr, options, ctx)?;
            return Ok(lowered.path_closed);
        }
        syn::Stmt::Local(local) => {
            // RPython: rtyper assigns concretetype to let-bound variables.
            // Extract array element type from type annotations on let bindings.
            if let syn::Pat::Type(pat_type) = &local.pat {
                let name = canonical_pat_name(&pat_type.pat);
                if let Some(type_root) = type_root_ident(&pat_type.ty) {
                    let qualified = qualify_type_name_with_imports(
                        &type_root,
                        &ctx.module_prefix,
                        &ctx.use_imports,
                    );
                    ctx.local_type_roots.insert(name.clone(), qualified);
                }
                ctx.local_value_types
                    .insert(name.clone(), classify_fn_arg_ty(&pat_type.ty));
                if let Some(full_type) = qualified_full_type_string_with_imports(
                    &pat_type.ty,
                    &ctx.module_prefix,
                    &ctx.use_imports,
                    ctx.known_struct_names,
                    ctx.known_trait_names,
                ) {
                    ctx.local_type_strings
                        .insert(name.clone(), full_type.clone());
                    ctx.local_array_types.insert(name.clone(), full_type);
                }
                if let Some(trait_root) = extract_dyn_trait_root_with_context(
                    &pat_type.ty,
                    &ctx.module_prefix,
                    ctx.known_trait_names,
                ) {
                    ctx.local_dyn_trait_roots.insert(name.clone(), trait_root);
                }
            }
            if let Some(init) = &local.init {
                let init_type_string = infer_init_type_string(&init.expr, ctx);
                let lowered = lower_expr(graph, block, &init.expr, options, ctx)?;
                if lowered.path_closed {
                    return Ok(true);
                }
                // Record variable name (RPython Variable._name)
                if let Some(var) = lowered.value_var(graph) {
                    let name = if let syn::Pat::Ident(pat_ident) = &local.pat {
                        Some(pat_ident.ident.to_string())
                    } else if let syn::Pat::Type(pat_type) = &local.pat {
                        Some(canonical_pat_name(&pat_type.pat))
                    } else {
                        None
                    };
                    if let syn::Pat::Ident(pat_ident) = &local.pat {
                        graph.name_value_var(&var, pat_ident.ident.to_string());
                    } else if let syn::Pat::Type(pat_type) = &local.pat {
                        let name = canonical_pat_name(&pat_type.pat);
                        graph.name_value_var(&var, name);
                    }
                    if let Some(name) = name {
                        // Prefer the statically-bool classification when
                        // the init expression is a `Lit::Bool` / `!x` /
                        // comparison / `&&`/`||` / registered
                        // bool-returning call, etc. (`expr_is_statically_bool`).
                        // `graph_value_type` would otherwise return
                        // `ValueType::Int` for a `Lit::Bool` lowered as
                        // `OpKind::ConstInt(0/1)`, which would make the
                        // next `!b` classifier choose the bitwise-invert
                        // path — `let b = true; !b` would emit
                        // `int_invert` instead of bool+branch. RPython
                        // annotates `Constant(True)` with `SomeBool`
                        // (`annotator/model.py:185-227`) so the let-bind
                        // here records `ValueType::Bool` to keep the
                        // lattice node distinct from Int.
                        let bool_override = expr_is_statically_bool(&init.expr, ctx);
                        if bool_override {
                            ctx.local_value_types.insert(name.clone(), ValueType::Bool);
                        } else if let Some(ty) = graph_value_type_var(graph, &var) {
                            ctx.local_value_types.insert(name.clone(), ty);
                        }
                        // RPython `LOAD_FAST` parity: record the
                        // let-binding's `(Variable, defining BlockId)`
                        // so a same-block `Expr::Path` reference
                        // reuses this Variable instead of emitting a
                        // fresh `OpKind::Input`
                        // (`flowspace/flowcontext.py:835`).
                        ctx.bind_local_id_var(name.clone(), &var, graph, *block);
                        if let Some(type_string) = init_type_string {
                            // Mirror `bind_ident_type` on let-with-annotation:
                            // record the receiver root so subsequent
                            // `receiver_type_root` lookups for field access
                            // can resolve `(*x).field` against
                            // `ctx.struct_fields`.  Without this, lets bound
                            // by inference from a Cast/Call init lose their
                            // owner root and field reads land with `ty:
                            // Unknown` → cast arms fire downstream.
                            if let Some(root) = type_root_from_type_string(&type_string) {
                                ctx.local_type_roots.insert(name.clone(), root);
                            }
                            ctx.local_type_strings.insert(name.clone(), type_string);
                        }
                        // `let f = |args| body;` / `let f = |args| ->
                        // RetTy body;` — the rhs is a closure, which
                        // pyre's walker doesn't surface as a graph
                        // function. Register the closure's return type
                        // under the local ident so a downstream Call
                        // `f(...)` resolves through `lookup_function_
                        // return_type`'s bare-key fallback and
                        // classifier sites (`expr_unary_not_operand_
                        // kind`) get a kind. RPython peer:
                        // `bookkeeper.getdesc(value)` resolves any
                        // callable in scope by host-identity; the
                        // static walker substitutes by registering the
                        // closure return type under the bare ident.
                        if let syn::Expr::Closure(closure) = &*init.expr {
                            let closure_ret = match &closure.output {
                                syn::ReturnType::Type(_, ty) => {
                                    qualified_full_type_string_with_imports(
                                        ty,
                                        &ctx.module_prefix,
                                        &ctx.use_imports,
                                        ctx.known_struct_names,
                                        ctx.known_trait_names,
                                    )
                                }
                                syn::ReturnType::Default => {
                                    expression_type_string(&closure.body, ctx)
                                }
                            };
                            if let Some(ret_ty) = closure_ret {
                                ctx.local_closure_returns.insert(name, ret_ty);
                            }
                        }
                    } else if !matches!(&local.pat, syn::Pat::Ident(_) | syn::Pat::Type(_)) {
                        // Destructure let (`let Some(x) = ...;`,
                        // `let Foo { a, b } = ...;`, `let A | B { f, .. }
                        // = ...;`) introduces names that the simple
                        // Pat::Ident / Pat::Type binding above misses.
                        // RPython parity: `flowspace/flowcontext.py` walks
                        // the BUILD_TUPLE_UNPACK / unpack_sequence paths
                        // and binds each leaf name with its rtyped
                        // concretetype.  Pyre routes the same shape
                        // through `bind_pattern_locals`, which already
                        // unwraps `Some(_)` / `Ok(_)` / `Err(_)` and
                        // recurses into struct / or patterns.
                        bind_pattern_locals(&local.pat, init_type_string.as_deref(), ctx);
                    }
                }
            }
        }
        syn::Stmt::Macro(stmt_macro) => {
            // Rust macros are syntactic, not part of the flow graph —
            // RPython has no construct counterpart.  Only forward
            // macros whose Rust semantics have an explicit RPython
            // mapping through `lower_expr`:
            //   * abort-family (`panic!`, `unreachable!`, `todo!`,
            //     `unimplemented!`) → `set_raise` (canonical
            //     exceptblock Link per `flowspace/model.py:21-25`).
            //   * assert-family (`assert!`, `assert_eq!`, `assert_ne!`,
            //     and `debug_` variants) → conditional `set_branch` +
            //     `set_raise` on the failing arm.
            // Other statement-position macros (`dbg!`, `println!`,
            // `vec!`, `format!`, `write!`, `writeln!`, ...) are
            // skipped, matching the pre-`92725722af` behaviour.
            let name = stmt_macro
                .mac
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            if matches!(
                name.as_str(),
                "panic"
                    | "unreachable"
                    | "todo"
                    | "unimplemented"
                    | "assert"
                    | "assert_eq"
                    | "assert_ne"
                    | "debug_assert"
                    | "debug_assert_eq"
                    | "debug_assert_ne"
            ) {
                let expr_macro = syn::ExprMacro {
                    attrs: stmt_macro.attrs.clone(),
                    mac: stmt_macro.mac.clone(),
                };
                let expr = syn::Expr::Macro(expr_macro);
                let lowered = lower_expr(graph, block, &expr, options, ctx)?;
                return Ok(lowered.path_closed);
            }
        }
        syn::Stmt::Item(_) => {}
    }
    Ok(false)
}

// ── Expression lowering (block-splitting for control flow) ───────

/// Lower an `if` / `if let` / `if … else if …` expression.
///
/// Extracted out of [`lower_expr`] so the recursive descent through an
/// `else if` chain runs on a small stack frame: [`lower_expr`]'s frame
/// has to reserve space for every match-arm's locals at once (it
/// dispatches over the full [`syn::Expr`] surface) and overflows the
/// default 2 MB thread stack at ~17 nested arms; this helper only
/// carries the `If`-arm locals so the frame shrinks by roughly an
/// order of magnitude.  The same shape PyPy's bytecode walker has —
/// `flowspace/flowcontext.py` keeps each opcode handler in its own
/// frame rather than one mega-frame for the dispatch loop.
fn lower_if_expr(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    if_expr: &syn::ExprIf,
    options: &AstGraphOptions,
    ctx: &mut GraphBuildContext,
) -> Result<Lowered, FlowingError> {
    // ── if-let desugaring ──
    // `if let pat = scrutinee { then } else { else }` is exact
    // syntactic sugar for `match scrutinee { pat => then, _ =>
    // else }` (Rust Reference, "If let expressions"). We build
    // the synthetic `Expr::Match` AST and recurse so the
    // existing `Expr::Match` lowering (the path immediately
    // below at `syn::Expr::Match(m)`) handles the pattern
    // dispatch — keeps a single match-emit codepath rather than
    // duplicating the merge / phi / arm-entry logic.
    //
    // Without this desugar, `if_expr.cond` would be lowered as
    // a regular expression and trip the catch-all `Expr::Let`
    // arm below, emitting `OpKind::Abort { Let }`. That stub
    // makes any function carrying an `if let` un-portal-able
    // (Phase G G.4.4 path A.1) since a BH resume could land on
    // it and crash on "unknown bhimpl_*".
    if let syn::Expr::Let(let_expr) = if_expr.cond.as_ref() {
        let then_expr = syn::Expr::Block(syn::ExprBlock {
            attrs: vec![],
            label: None,
            block: if_expr.then_branch.clone(),
        });
        let else_expr: syn::Expr = match &if_expr.else_branch {
            Some((_, else_branch)) => (**else_branch).clone(),
            None => syn::parse_quote!({}),
        };
        let then_arm = syn::Arm {
            attrs: vec![],
            pat: (*let_expr.pat).clone(),
            guard: None,
            fat_arrow_token: Default::default(),
            body: Box::new(then_expr),
            comma: Some(Default::default()),
        };
        let else_arm = syn::Arm {
            attrs: vec![],
            pat: syn::parse_quote!(_),
            guard: None,
            fat_arrow_token: Default::default(),
            body: Box::new(else_expr),
            comma: None,
        };
        let synthetic = syn::Expr::Match(syn::ExprMatch {
            attrs: vec![],
            match_token: Default::default(),
            expr: let_expr.expr.clone(),
            brace_token: Default::default(),
            arms: vec![then_arm, else_arm],
        });
        return lower_expr(graph, block, &synthetic, options, ctx);
    }

    // RPython `flowspace/flowcontext.py:91,107,364`: unsupported
    // cond raises `FlowingError`, halting the walk.  A child
    // that closed its path (`if return_early { ... } else ...`)
    // also has no truth value — propagate via `get_value!`.
    //
    // Z4.B.1.b first production round-trip — push the cond vid onto
    // `value_stack` and pop it back before the branch.  Equivalent to
    // upstream `flowcontext.py:1095 if_jump` where the cond was pushed
    // by the prior `COMPARE_OP` / `LOAD_FAST` / `POP_JUMP_IF_FALSE`
    // pops it (`flowcontext.py:1097 cond = self.popvalue()`).  Pyre's
    // `lower_expr` still returns `Lowered::from_value_var`; the pushvid/popvid
    // pair exercises the production stack helpers so a later slice can
    // flip `lower_expr` to push internally and drop the explicit
    // push side here, leaving only the `popvid` consume.
    let cond_pre_var = get_value_var!(
        lower_expr(graph, block, &if_expr.cond, options, ctx)?,
        graph
    );
    ctx.pushvid_var(&cond_pre_var);
    let cond_var = ctx.popvid_var(graph);

    let mut then_block = graph.create_block();
    let mut else_block = graph.create_block();

    // Capture the locals frame as it was when `*block` closed via
    // `set_branch` so a later cross-block read in the merge block
    // can thread back through either arm's `Link.args` even when the
    // arm itself rebinds nothing.  Stored on `Block.framestate`
    // (per-block, captured at close time) — both exits of one
    // set_branch share the same pre-branch snapshot, so the per-edge
    // duplication collapses into a single field.
    // RPython parity: `flowspace/flowcontext.py:38
    // SpamBlock.framestate`.
    let pre_branch_snapshot = ctx.getstate(graph, 0);
    graph.set_branch(*block, cond_var, then_block, vec![], else_block, vec![]);
    graph.block_mut(*block).framestate = Some(pre_branch_snapshot);

    // Stage B1: capture the pre-branch ctx state BEFORE
    // lowering the then-arm so the else-arm can re-enter
    // `*block`'s scope.  RPython parity:
    // `flowspace/flowcontext.py:407-408 record_block(block)`
    // calls `setstate(block.framestate)` at every block
    // entry; pyre snapshots the analogue `LocalBindingSnapshot`
    // here and restores it before the else-arm.  Without this
    // restore the else-arm sees the then-arm's mutations to
    // `ctx.local_value_ids` / `local_value_types` etc.
    let pre_branch_ctx = LocalBindingSnapshot::capture(ctx);

    // Lower then branch — collect result value
    let then_lowered = lower_stmt_list_with_tail_value(
        graph,
        &mut then_block,
        &if_expr.then_branch.stmts,
        options,
        ctx,
    )?;
    // Snapshot then-arm's locals state BEFORE
    // else-arm lowering mutates `ctx.local_value_ids`.  Used
    // only if then-arm is open (will `set_goto` to merge); a
    // closed arm's snapshot is unused.
    let then_exit_snapshot = ctx.getstate(graph, 0);
    // Capture the full ctx as well so we can restore the surviving
    // arm's `local_value_ids` / `local_value_types` if the other arm
    // closes (return/raise/break).  Without this, e.g.
    // `if cond { x = 1; } else { return 0; } x` would leave
    // `ctx.local_value_ids["x"]` at the pre-branch state and the
    // post-merge `x` read would lower to the wrong SSA value.
    let then_exit_ctx = LocalBindingSnapshot::capture(ctx);

    // Stage B1: restore pre-branch ctx state before lowering
    // the else-arm so its `LOAD_FAST`-style reads see the
    // pre-If bindings, not the then-arm's rebinds.
    pre_branch_ctx.restore(ctx);

    // Lower else branch.  When the else-branch is itself a chained
    // `if` (`if … else if …`), recurse through [`lower_if_expr`]
    // directly rather than going back through [`lower_expr`].  syn's
    // AST nests each `else if` as `else_branch: Some(Expr::If(_))`,
    // so a long chain would otherwise drive [`lower_expr`]'s ~70KB
    // match-frame N levels deep and exhaust the 2 MB default stack.
    let mut else_lowered = Lowered::no_value();
    if let Some((_, else_branch)) = &if_expr.else_branch {
        else_lowered = match else_branch.as_ref() {
            syn::Expr::If(else_if_expr) => {
                lower_if_expr(graph, &mut else_block, else_if_expr, options, ctx)?
            }
            _ => lower_expr(graph, &mut else_block, else_branch, options, ctx)?,
        };
    }
    let else_exit_snapshot = ctx.getstate(graph, 0);
    // Companion ctx capture for the else-arm — same rationale as
    // `then_exit_ctx`.
    let else_exit_ctx = LocalBindingSnapshot::capture(ctx);

    // RPython `flowspace/flowcontext.py` merges via Link: a
    // branch whose path is closed (`return`/`raise`/`break`)
    // does not `goto` the merge — the `is_open` check below
    // already skips it.  A phi inputarg is introduced when both
    // arms *produced a value*, mirroring the old all-or-nothing
    // shape; arity is kept consistent by skipping the closed
    // arm's goto so only the open arm sends a `vec![value]` to
    // the one-inputarg merge block.
    let then_value_var = then_lowered.value_var(graph);
    let else_value_var = else_lowered.value_var(graph);
    let then_open = graph.block(then_block).is_open();
    let else_open = graph.block(else_block).is_open();
    let want_phi = then_value_var.is_some() && else_value_var.is_some();
    let both_open = then_open && else_open;

    // Pre-compute the unioned framestate when both arms are open — it is
    // reused below for (a) the migration path
    // (`create_block_from_framestate` + `set_goto_from_framestate`) and
    // (b) the lean-merge-block ctx update (`None`-kill + lazy
    // phi-install).  Doing the union once avoids duplicating
    // `FrameState::union`'s O(slots) walk.
    let merged_when_both_open: Option<FrameState> = if both_open {
        Some(then_exit_snapshot.union(&else_exit_snapshot, graph).expect(
            "AST frontend: union is total — entries domain has no UnionError, \
                 stack / last_exception / blocklist / next_offset are vestigial \
                 (framestate.py:78 None-return reachable only post-Z4 walker)",
        ))
    } else {
        None
    };

    // When both arms are open and there is no value-phi to
    // thread (the result is `()` — a statement-shaped `if`), the merge
    // joins via `flowcontext.py:443 SpamBlock(newstate)` — a block whose
    // `inputargs` are every Variable in `merged.getvariables()` plus
    // per-pred links built from `currentstate.getoutputargs(newstate)`.
    // `create_block_from_framestate` + `set_goto_from_framestate`
    // implement that shape; `ctx.setstate_at_block` rebinds
    // `ctx.local_value_ids` to the merge's slot Variables so post-merge
    // reads see the freshly-minted phi Variables without re-driving the
    // lazy installer.
    //
    // Eligibility safety check: pyre's existing AST blocks are not
    // SpamBlocks — many call sites set `Link.args` from name lists
    // captured BEFORE `ensure_variable_at_block` may grow a block's
    // `inputargs`, so unconditionally migrating risks two failure
    // modes:
    //
    //   1. Orphan-rooted blocks.  The >2-arm `Expr::Match` fallback
    //      at `ast.rs:6045-6052` only wires arms[0..2] via
    //      `set_branch`, leaving arms[2..] orphan.  Inside their
    //      bodies, the migration's `set_goto_from_framestate` would
    //      call `ensure_variable_at_block` against an orphan and
    //      panic ("no transitive predecessor chain leads to a
    //      definition site").
    //   2. Loop-header arity contracts.  `build_loop_header_state`
    //      + `create_block_from_framestate` fixes the header's
    //      inputargs to the framestate's Variable slots.
    //      `ensure_variable_at_block` adds carry-through Variables
    //      as unnamed inputargs — the back-edge `closeblock_link`
    //      would then trip the arity assert because
    //      `getoutputargs` produces fewer args than the header's
    //      grown inputargs count.
    //
    // `can_thread_variable_to_block` mirrors `ensure_variable_at_block`'s
    // recursion without mutation, and `forbidden_growth` lists the
    // current loop headers (continue_targets) so the dry-run also
    // rejects a walk that would have to grow a header.  When the
    // migration is skipped, the legacy lean-merge-block path below
    // copes silently — its merge block carries no inputargs and the
    // lazy installer only touches blocks that are actually reachable.
    let forbidden_growth: std::collections::HashSet<BlockId> = ctx
        .loop_stack
        .iter()
        .map(|frame| frame.continue_target)
        .collect();
    let migrate: bool = if let Some(merged) = merged_when_both_open.as_ref() {
        if want_phi {
            false
        } else {
            let then_outargs = then_exit_snapshot.getoutputargs(merged, graph);
            let else_outargs = else_exit_snapshot.getoutputargs(merged, graph);
            let safe_then = then_outargs.iter().all(|a| match a {
                LinkArg::Value(v) => {
                    graph.can_thread_variable_to_block(then_block, v, &forbidden_growth)
                }
                _ => true,
            });
            let safe_else = else_outargs.iter().all(|a| match a {
                LinkArg::Value(v) => {
                    graph.can_thread_variable_to_block(else_block, v, &forbidden_growth)
                }
                _ => true,
            });
            safe_then && safe_else
        }
    } else {
        false
    };

    let (merge_block, phi_result) = if want_phi {
        let (merge, phi_args) = graph.create_block_with_arg_vars(1);
        if then_open {
            let then_var = then_value_var.clone().unwrap();
            graph.set_goto(then_block, merge, vec![then_var]);
            graph.block_mut(then_block).framestate = Some(then_exit_snapshot.clone());
        }
        if else_open {
            let else_var = else_value_var.clone().unwrap();
            graph.set_goto(else_block, merge, vec![else_var]);
            graph.block_mut(else_block).framestate = Some(else_exit_snapshot.clone());
        }
        (merge, Some(phi_args[0].clone()))
    } else if migrate {
        let merged = merged_when_both_open
            .as_ref()
            .expect("migrate => merged_when_both_open is Some");
        let merge = graph.create_block_from_framestate(merged);
        graph.set_goto_from_framestate(then_block, merge, &then_exit_snapshot, merged);
        graph.block_mut(then_block).framestate = Some(then_exit_snapshot.clone());
        graph.set_goto_from_framestate(else_block, merge, &else_exit_snapshot, merged);
        graph.block_mut(else_block).framestate = Some(else_exit_snapshot.clone());
        (merge, None)
    } else {
        let merge = graph.create_block();
        if then_open {
            graph.set_goto(then_block, merge, vec![]);
            graph.block_mut(then_block).framestate = Some(then_exit_snapshot.clone());
        }
        if else_open {
            graph.set_goto(else_block, merge, vec![]);
            graph.block_mut(else_block).framestate = Some(else_exit_snapshot.clone());
        }
        (merge, None)
    };

    // FrameState::union-driven merge when both arms reach the
    // merge block.  Routes through `FrameState::union` for
    // explicit per-slot classification per RPython
    // `flowspace/framestate.py:105-128 union`:
    //   - One-sided None → None-kill (`framestate.py:110-111`):
    //     the slot is dropped from `ctx` so post-merge reads
    //     of that name surface as undefined-local.
    //   - CarryThrough (same vid both arms): kept; the merged
    //     entry's `value_type` may have widened from `Unknown`
    //     to a concrete kind via the wildcard rule and the
    //     source `OpKind`'s `ty` is retagged below to keep
    //     `graph_value_type` in agreement with the framestate.
    //   - NeedsPhi (disagreeing vids): eager phi install at
    //     union time per `framestate.py:113-114 union`'s
    //     fresh `Variable()` semantics —
    //     `lazy_install_local_at_current_block` allocates the
    //     merge-block inputarg, threads per-arm vids onto
    //     each predecessor's goto args, and rebinds ctx so
    //     post-merge reads of the name resolve to the new
    //     phi vid without re-driving the lazy installer.
    if migrate {
        // Migration path: `create_block_from_framestate`
        // already threaded every Variable in `merged.getvariables()`
        // into `merge_block.inputargs`, and `set_goto_from_framestate`
        // pushed the per-arm `getoutputargs` projection onto each
        // predecessor's link.  Fresh-phi slot Variables (minted by
        // `FrameState::union` as `Variable::new()`) have no upstream
        // defining op, so `graph_value_type_var` would surface Unknown
        // when `setstate_at_block` derives the post-merge
        // `ctx.local_value_types` entry — and a subsequent
        // `Expr::Unary` `!` on the rebound local would trip
        // `expr_unary_not_operand_kind`'s
        // `UnaryNotUnknownOperand` arm.  Emit a paired
        // `OpKind::Input { name, ty }` op in `merge_block` for every
        // fresh phi so `graph_value_type_var` finds the op's `ty`
        // upstream and the per-name registration carries through.
        // The type fold mirrors
        // `lazy_install_local_at_current_block_var`'s wildcard rule
        // (`ast.rs:3367-3374`): concrete + same-concrete keeps the
        // concrete kind, concrete + Unknown lifts to the concrete
        // sibling, concrete + different-concrete widens to Unknown.
        // `setstate_at_block` then rebinds ctx in lockstep with
        // `merged.locals_w` — slots whose Variable carried through
        // both arms rebind to the merge-block's inputarg, None-killed
        // slots drop, fresh-phi slots rebind to the freshly-minted
        // merge-block Variable now carrying a proper Input op.
        let merged = merged_when_both_open
            .as_ref()
            .expect("migrate => merged_when_both_open is Some");
        let phi_info: Vec<(usize, crate::flowspace::model::Variable, ValueType)> = {
            let then_view = then_exit_snapshot.locals_w_view(graph);
            let else_view = else_exit_snapshot.locals_w_view(graph);
            let merged_view = merged.locals_w_view(graph);
            let mut info = Vec::new();
            for (i, slot) in merged_view.iter().enumerate() {
                let Some(crate::flowspace::model::Hlvalue::Variable(merged_var)) = slot else {
                    continue;
                };
                let then_var = then_view
                    .get(i)
                    .and_then(|s| s.as_ref())
                    .and_then(|c| match c {
                        crate::flowspace::model::Hlvalue::Variable(v) => Some(v.clone()),
                        _ => None,
                    });
                if then_var.as_ref() == Some(merged_var) {
                    continue;
                }
                let else_var = else_view
                    .get(i)
                    .and_then(|s| s.as_ref())
                    .and_then(|c| match c {
                        crate::flowspace::model::Hlvalue::Variable(v) => Some(v.clone()),
                        _ => None,
                    });
                let then_ty = then_var
                    .as_ref()
                    .map(|v| graph_value_type_var(graph, v).unwrap_or(ValueType::Unknown))
                    .unwrap_or(ValueType::Unknown);
                let else_ty = else_var
                    .as_ref()
                    .map(|v| graph_value_type_var(graph, v).unwrap_or(ValueType::Unknown))
                    .unwrap_or(ValueType::Unknown);
                let merged_ty = match (then_ty.clone(), else_ty) {
                    (a, b) if a == b => a,
                    (ValueType::Unknown, b) => b,
                    (a, ValueType::Unknown) => a,
                    (ValueType::Ref(_), ValueType::Ref(_)) => ValueType::Ref(None),
                    _ => ValueType::Unknown,
                };
                info.push((i, merged_var.clone(), merged_ty));
            }
            info
        };
        for (slot_idx, phi_var, ty) in phi_info {
            let name = ctx.local_first_bind_order[slot_idx].clone();
            graph.push_op_with_result_var(
                merge_block,
                OpKind::Input {
                    name: name.clone(),
                    ty: ty.clone(),
                },
                phi_var.clone(),
            );
            graph.name_value_var(&phi_var, name);
        }
        ctx.setstate_at_block(merged, merge_block, graph);
    } else if then_open && else_open {
        let merged =
            merged_when_both_open.expect("both arms open => merged_when_both_open is Some");
        // Locals projection walks `merged.locals_w` per upstream
        // `framestate.py:19 self.locals_w` — pyre's `union` populates
        // the `Hlvalue` carrier in lockstep with `entries`, so this
        // walk is bit-identical to a `merged.entries` traversal while
        // keeping the read side in agreement with the upstream source
        // of truth.  Materialise the view once and reuse across the
        // None-kill + phi-install passes.
        let merged_locals_w = merged.locals_w_view(graph);
        let then_locals_w = then_exit_snapshot.locals_w_view(graph);
        for (slot_idx, slot) in merged_locals_w.iter().enumerate() {
            if matches!(slot, Some(crate::flowspace::model::Hlvalue::Variable(_))) {
                continue;
            }
            if let Some(name) = ctx.local_first_bind_order.get(slot_idx).cloned() {
                ctx.local_value_ids.remove(&name);
                ctx.local_value_types.remove(&name);
            }
        }
        // Materialise (slot_idx, merged_vid, then_vid) tuples up front
        // so the immutable `graph` borrow inside the locals_w walk
        // releases before the mutable `lazy_install_local_at_current_block`
        // call below.
        let phi_candidates: Vec<(
            usize,
            crate::flowspace::model::Variable,
            Option<crate::flowspace::model::Variable>,
        )> = merged_locals_w
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Some(crate::flowspace::model::Hlvalue::Variable(v)) => {
                    let then_var = then_locals_w.get(i).and_then(|slot| match slot {
                        Some(crate::flowspace::model::Hlvalue::Variable(v)) => Some(v.clone()),
                        _ => None,
                    });
                    Some((i, v.clone(), then_var))
                }
                _ => None,
            })
            .collect();
        drop(merged_locals_w);
        drop(then_locals_w);
        for (slot_idx, slot_var, then_var) in phi_candidates {
            let is_fresh_phi = then_var.as_ref() != Some(&slot_var);
            if is_fresh_phi {
                let name = ctx.local_first_bind_order[slot_idx].clone();
                let _ = lazy_install_local_at_current_block_var(
                    graph,
                    ctx,
                    merge_block,
                    &name,
                    Some(slot_var.clone()),
                );
            }
        }
    } else if then_open {
        // The else-arm closed (return/raise/break) so the post-merge
        // ctx must reflect the then-arm's `local_value_ids`/
        // `local_value_types` rebinds.  At this point ctx still
        // holds the else-arm's terminal state (or the pre-branch
        // state if there was no else); restore the then-arm
        // snapshot we captured before the pre-branch restore.
        then_exit_ctx.restore(ctx);
    } else if else_open {
        // Symmetric case — then-arm closed, else-arm is the only
        // reaching predecessor of the merge block.  `ctx` still
        // holds the else-arm's terminal bindings via the chain of
        // `lower_*` mutations, but be explicit so any future
        // rearrangement of the lowering order does not silently
        // break this contract.
        else_exit_ctx.restore(ctx);
    }

    // Stamp the merge block's entry framestate for the lean merge
    // paths.  `want_phi` / `migrate` build the merge via
    // `create_block_with_arg_vars` / `create_block_from_framestate`
    // and already carry a usable per-slot view; the lean paths use a
    // bare `create_block()` (0 inputargs) and would otherwise leave
    // the merge with no recorded snapshot until it later closes via
    // its own branch.  A back-edge or post-merge cross-block read
    // that recurses through this merge before it closes then hits the
    // "no recorded snapshot" bail in
    // `lazy_install_local_at_current_block_var` (predecessor
    // `framestate.as_ref()?`) and falls back to a body-`Input`
    // (rejected as "adapter cross-block body Input") or threads an
    // out-of-scope slot.  At this point `ctx` already reflects the
    // surviving arm's bindings (closed-arm restore) or the merged
    // bindings (both-open None-kill + phi install), so its snapshot
    // is the merge's entry state.  RPython parity:
    // `flowspace/flowcontext.py:407-408 record_block(block)` calls
    // `setstate(block.framestate)` at every block entry.
    if !want_phi && !migrate && graph.block(merge_block).framestate.is_none() {
        let merge_entry_snapshot = ctx.getstate(graph, 0);
        graph.block_mut(merge_block).framestate = Some(merge_entry_snapshot);
    }

    *block = merge_block;
    // If NEITHER arm remains open, the merge block is
    // unreachable — mark the enclosing path as closed so the
    // caller stops lowering into it.  RPython parity:
    // `flowspace/flowcontext.py` never keeps a merge block
    // reachable when all incoming links closed with
    // `FlowSignal::Return` / `Raise`.
    if !then_open && !else_open {
        Ok(Lowered::path_closed())
    } else {
        match phi_result {
            Some(var) => Ok(Lowered::from_value_var(graph, &var)),
            None => Ok(Lowered {
                value: None,
                path_closed: false,
            }),
        }
    }
}

/// Lower an expression, potentially splitting blocks for control flow.
///
/// RPython equivalent: FlowContext.handle_bytecode() + guessbool().
/// When `if`/`match` is encountered, the current block is terminated
/// with a Branch, new blocks are created for each arm, and `block`
/// is updated to the merge/continuation block.
fn lower_expr(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    expr: &syn::Expr,
    options: &AstGraphOptions,
    ctx: &mut GraphBuildContext,
) -> Result<Lowered, FlowingError> {
    // RPython `flowspace/flowcontext.py:258,417` — when the abstract
    // interpreter hits an unsupported bytecode it raises `FlowingError`
    // and the walk stops at once.  Pyre's analogue: emit an
    // `UnsupportedExpr` marker op in *block (so downstream passes see
    // evidence of the drop) and return `Err(FlowingError::Unsupported)`
    // so every caller in the chain aborts via `?` rather than
    // synthesising a fabricated SSA value.  The helper centralises
    // that pair so every failure site emits exactly one Unknown.
    let stop_unsupported = |graph: &mut FunctionGraph,
                            block: BlockId,
                            variant: UnsupportedExprKind|
     -> Result<Lowered, FlowingError> {
        graph.push_op_var(
            block,
            OpKind::Abort {
                kind: UnknownKind::UnsupportedExpr { variant },
            },
            true,
        );
        Err(FlowingError::Unsupported {
            kind: UnknownKind::UnsupportedExpr { variant },
        })
    };
    // Non-fatal counterpart of `stop_unsupported`: emit the `Unknown`
    // marker so coverage auditing still flags the gap, but hand its
    // Variable back so the enclosing walker keeps going.  Matches
    // RPython `LOAD_CONST` (`flowspace/flowcontext.py:841`) — the
    // bytecode pushes a value of an un-modelled shape and the flow
    // walk continues without raising `FlowingError`.
    let continue_with_unknown =
        |graph: &mut FunctionGraph, block: BlockId, variant: UnsupportedExprKind| -> Lowered {
            let var = graph
                .push_op_var(
                    block,
                    OpKind::Abort {
                        kind: UnknownKind::UnsupportedExpr { variant },
                    },
                    true,
                )
                .expect("OpKind::Abort has has_result=true");
            Lowered::from_value_var(graph, &var)
        };
    let continue_with_unknown_literal =
        |graph: &mut FunctionGraph, block: BlockId, variant: UnsupportedLiteralKind| -> Lowered {
            let var = graph
                .push_op_var(
                    block,
                    OpKind::Abort {
                        kind: UnknownKind::UnsupportedLiteral { variant },
                    },
                    true,
                )
                .expect("OpKind::Abort has has_result=true");
            Lowered::from_value_var(graph, &var)
        };
    match expr {
        // ── receiver.field / arr[i].field ──
        syn::Expr::Field(field) => {
            if let syn::Expr::Index(idx) = &*field.base {
                // RPython: getinteriorfield_gc — arr[i].field as a single op.
                let base_pre_var =
                    get_value_var!(lower_expr(graph, block, &idx.expr, options, ctx)?, graph);
                ctx.pushvid_var(&base_pre_var);
                let index_pre_var =
                    get_value_var!(lower_expr(graph, block, &idx.index, options, ctx)?, graph);
                ctx.pushvid_var(&index_pre_var);
                let index_var = ctx.popvid_var(graph);
                let base_var = ctx.popvid_var(graph);
                let field_name = member_name(&field.member);
                let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
                // Element struct type is the field owner for interiorfield descriptors.
                let elem_type = array_type_id
                    .as_ref()
                    .and_then(|atid| extract_element_type_from_str(atid));
                // RPython: getkind(op.result.concretetype) — resolve field type
                // from struct field registry for the kind suffix (i/r/f).
                let item_field_type_string = elem_type
                    .as_ref()
                    .and_then(|owner| {
                        ctx.struct_fields.field_type_in_scope(
                            owner,
                            &field_name,
                            &ctx.module_prefix,
                            &ctx.use_imports,
                        )
                    })
                    .map(ToOwned::to_owned);
                let item_ty = item_field_type_string
                    .as_deref()
                    .map(type_string_to_value_type)
                    .unwrap_or(ValueType::Unknown);
                let var = graph
                    .push_op_var(
                        *block,
                        OpKind::InteriorFieldRead {
                            base: base_var,
                            index: index_var,
                            field: crate::model::FieldDescriptor::new(field_name, elem_type),
                            item_ty,
                            array_type_id,
                        },
                        true,
                    )
                    .expect("OpKind::InteriorFieldRead has has_result=true");
                Ok(Lowered::from_value_var(graph, &var))
            } else {
                let base_pre_var =
                    get_value_var!(lower_expr(graph, block, &field.base, options, ctx)?, graph);
                ctx.pushvid_var(&base_pre_var);
                let base_var = ctx.popvid_var(graph);
                let field_name = member_name(&field.member);
                let field_type_string =
                    field_type_string_from_expr(&field.base, &field.member, ctx);
                let ty = field_type_string
                    .as_deref()
                    .map(type_string_to_value_type)
                    .unwrap_or(ValueType::Unknown);
                let var = graph
                    .push_op_var(
                        *block,
                        OpKind::FieldRead {
                            base: base_var,
                            field: crate::model::FieldDescriptor::new(
                                field_name,
                                receiver_type_root(&field.base, ctx),
                            ),
                            ty,
                            pure: false,
                        },
                        true,
                    )
                    .expect("OpKind::FieldRead has has_result=true");
                Ok(Lowered::from_value_var(graph, &var))
            }
        }

        // ── base[index] ──
        syn::Expr::Index(idx) => {
            let base_pre_var =
                get_value_var!(lower_expr(graph, block, &idx.expr, options, ctx)?, graph);
            ctx.pushvid_var(&base_pre_var);
            let index_pre_var =
                get_value_var!(lower_expr(graph, block, &idx.index, options, ctx)?, graph);
            ctx.pushvid_var(&index_pre_var);
            let index_var = ctx.popvid_var(graph);
            let base_var = ctx.popvid_var(graph);
            let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
            let item_ty = array_item_value_type_from_array_type_id(array_type_id.as_deref())
                .unwrap_or(ValueType::Unknown);
            let var = graph
                .push_op_var(
                    *block,
                    OpKind::ArrayRead {
                        base: base_var,
                        index: index_var,
                        item_ty,
                        nolength: nolength_from_array_type_id(array_type_id.as_deref()),
                        array_type_id,
                    },
                    true,
                )
                .expect("OpKind::ArrayRead has has_result=true");
            Ok(Lowered::from_value_var(graph, &var))
        }

        // ── lhs = rhs ──
        syn::Expr::Assign(assign) => {
            // RPython `flowcontext.py` evaluates rhs first; if it raises
            // `FlowingError`, the whole assignment is dropped.  `get_value!`
            // propagates both `FlowingError` (`Err(..)`) and `path_closed`
            // (`Ok(Lowered { path_closed: true })`) up the walk.
            let value_pre_var = get_value_var!(
                lower_expr(graph, block, &assign.right, options, ctx)?,
                graph
            );
            ctx.pushvid_var(&value_pre_var);
            let value_var = ctx.popvid_var(graph);

            match &*assign.left {
                syn::Expr::Field(field) => {
                    if let syn::Expr::Index(idx) = &*field.base {
                        // RPython: setinteriorfield_gc — arr[i].field = value.
                        let base_pre_var = get_value_var!(
                            lower_expr(graph, block, &idx.expr, options, ctx)?,
                            graph
                        );
                        ctx.pushvid_var(&base_pre_var);
                        let index_pre_var = get_value_var!(
                            lower_expr(graph, block, &idx.index, options, ctx)?,
                            graph
                        );
                        ctx.pushvid_var(&index_pre_var);
                        let index_var = ctx.popvid_var(graph);
                        let base_var = ctx.popvid_var(graph);
                        let field_name = member_name(&field.member);
                        let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
                        let elem_type = array_type_id
                            .as_ref()
                            .and_then(|atid| extract_element_type_from_str(atid));
                        // RPython: getkind(v_value.concretetype) — resolve field type
                        // from struct field registry for the kind suffix (i/r/f).
                        let item_ty = elem_type
                            .as_ref()
                            .and_then(|owner| {
                                ctx.struct_fields.field_type_in_scope(
                                    owner,
                                    &field_name,
                                    &ctx.module_prefix,
                                    &ctx.use_imports,
                                )
                            })
                            .map(type_string_to_value_type)
                            .unwrap_or(ValueType::Unknown);
                        graph.push_op_var(
                            *block,
                            OpKind::InteriorFieldWrite {
                                base: base_var,
                                index: index_var,
                                field: crate::model::FieldDescriptor::new(field_name, elem_type),
                                value: value_var,
                                item_ty,
                                array_type_id,
                            },
                            false,
                        );
                    } else {
                        let base_pre_var = get_value_var!(
                            lower_expr(graph, block, &field.base, options, ctx)?,
                            graph
                        );
                        ctx.pushvid_var(&base_pre_var);
                        let base_var = ctx.popvid_var(graph);
                        let field_name = member_name(&field.member);
                        let ty = field_value_type_from_expr(&field.base, &field.member, ctx)
                            .unwrap_or(ValueType::Unknown);
                        graph.push_op_var(
                            *block,
                            OpKind::FieldWrite {
                                base: base_var,
                                field: crate::model::FieldDescriptor::new(
                                    field_name,
                                    receiver_type_root(&field.base, ctx),
                                ),
                                value: value_var,
                                ty,
                            },
                            false,
                        );
                    }
                }
                syn::Expr::Index(idx) => {
                    let base_pre_var =
                        get_value_var!(lower_expr(graph, block, &idx.expr, options, ctx)?, graph);
                    ctx.pushvid_var(&base_pre_var);
                    let index_pre_var =
                        get_value_var!(lower_expr(graph, block, &idx.index, options, ctx)?, graph);
                    ctx.pushvid_var(&index_pre_var);
                    let index_var = ctx.popvid_var(graph);
                    let base_var = ctx.popvid_var(graph);
                    let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
                    let item_ty =
                        array_item_value_type_from_array_type_id(array_type_id.as_deref())
                            .unwrap_or(ValueType::Unknown);
                    graph.push_op_var(
                        *block,
                        OpKind::ArrayWrite {
                            base: base_var,
                            index: index_var,
                            value: value_var,
                            item_ty,
                            nolength: nolength_from_array_type_id(array_type_id.as_deref()),
                            array_type_id,
                        },
                        false,
                    );
                }
                syn::Expr::Path(path) if path.path.segments.len() == 1 && path.qself.is_none() => {
                    // Generic local assignment `x = rhs` — RPython STORE_FAST
                    // parity (`flowspace/flowcontext.py:878-885`):
                    //
                    //     w_newvalue = self.popvalue()
                    //     ...
                    //     self.locals_w[varindex] = w_newvalue
                    //     if isinstance(w_newvalue, Variable):
                    //         w_newvalue.rename(self.getlocalvarname(varindex))
                    //
                    // Two effects: replace the locals slot for `x`
                    // with the rhs `Variable`, and rename the rhs
                    // `Variable` to the local name so diagnostics and
                    // the adapter's `name_to_value` lookup pick the
                    // rhs up under that name.  Same-block dedup
                    // machinery installed at `lower_stmt`'s let arm
                    // (`ast.rs:1389 local_value_ids.insert`) caches
                    // `(let-rhs Variable, defining block)`; without
                    // this STORE_FAST update a later `x` read returns
                    // the stale let value.
                    let name = path
                        .path
                        .segments
                        .iter()
                        .map(|seg| seg.ident.to_string())
                        .collect::<Vec<_>>()
                        .join("::");
                    ctx.bind_local_id_var(name, &value_var, graph, *block);
                }
                _ => {
                    // Generic assignment — value already lowered
                }
            }
            Ok(Lowered::no_value())
        }

        // ── function call ──
        syn::Expr::Call(call) => {
            for a in &call.args {
                let v_pre_var = get_value_var!(lower_expr(graph, block, a, options, ctx)?, graph);
                ctx.pushvid_var(&v_pre_var);
            }
            let mut args_vars: Vec<crate::flowspace::model::Variable> =
                Vec::with_capacity(call.args.len());
            for _ in 0..call.args.len() {
                args_vars.push(ctx.popvid_var(graph));
            }
            args_vars.reverse();
            // `<prim>::from(x)` is the function-call spelling of a
            // numeric widening.  RPython has no `from`; it spells the
            // same conversion as the `int(v)` / `r_uint(v)` builtin
            // calls (rbuiltin.py:178), so route the single-arg primitive
            // `from` through the same coercion chain as `x as T`
            // (`Expr::Cast`) instead of emitting an unregistered
            // `FunctionPath` call that misses `PyreCallRegistry`.
            if let syn::Expr::Path(p) = &*call.func
                && args_vars.len() == 1
                && let Some(target_ty) = numeric_from_target_type(&p.path)
            {
                let operand_var = args_vars.into_iter().next().expect("args_vars.len() == 1");
                let source_ty = graph_value_type_var(graph, &operand_var);
                let var = lower_value_cast(graph, *block, operand_var, source_ty, target_ty);
                return Ok(Lowered::from_value_var(graph, &var));
            }
            // `std::ptr::eq(a, b)` is pointer identity — the same
            // comparison pyre's `BinOp { op: "eq" }` on two Ref operands
            // produces, which jtransform rewrites to `ptr_eq`
            // (jtransform.rs:849 / jtransform.py:1243 rewrite_op_ptr_eq).
            // Emit that BinOp instead of an unregistered FunctionPath
            // call that misses `PyreCallRegistry`.
            if let syn::Expr::Path(p) = &*call.func
                && args_vars.len() == 2
                && is_ptr_eq_path(&p.path)
            {
                let mut it = args_vars.into_iter();
                let lhs = it.next().expect("args_vars.len() == 2");
                let rhs = it.next().expect("args_vars.len() == 2");
                let var = graph
                    .push_op_var(
                        *block,
                        OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs,
                            rhs,
                            result_ty: ValueType::Bool,
                        },
                        true,
                    )
                    .expect("OpKind::BinOp has has_result=true");
                return Ok(Lowered::from_value_var(graph, &var));
            }
            let target = canonical_call_target(&call.func, ctx);
            // RPython parity: same rationale as the MethodCall arm above
            // — `op.result.concretetype` is set from the registered
            // FuncDesc.  Look up the qualified function path in
            // `ctx.fn_return_types` (populated in pass 1) so calls to
            // free functions returning `usize` / `bool` / `i64` propagate
            // a `Signed` result kind through rtyper instead of defaulting
            // to GcRef.
            let call_return_type_string = if let syn::Expr::Path(p) = &*call.func {
                let segments: Vec<String> = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                lookup_function_return_type(ctx, &segments).cloned()
            } else {
                None
            };
            let result_ty = if let syn::Expr::Path(p) = &*call.func {
                let segments: Vec<String> = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                intrinsic_call_result_type(&segments)
                    .or_else(|| {
                        call_return_type_string
                            .as_deref()
                            .map(type_string_to_value_type)
                    })
                    .unwrap_or(ValueType::Unknown)
            } else {
                ValueType::Unknown
            };
            let var = graph
                .push_op_var(
                    *block,
                    OpKind::Call {
                        target,
                        args: args_vars,
                        result_ty,
                    },
                    true,
                )
                .expect("OpKind::Call has has_result=true");
            Ok(Lowered::from_value_var(graph, &var))
        }

        // ── method call ──
        syn::Expr::MethodCall(mc) => {
            let recv_pre_var =
                get_value_var!(lower_expr(graph, block, &mc.receiver, options, ctx)?, graph);
            ctx.pushvid_var(&recv_pre_var);
            for a in &mc.args {
                let v_pre_var = get_value_var!(lower_expr(graph, block, a, options, ctx)?, graph);
                ctx.pushvid_var(&v_pre_var);
            }
            let total = 1 + mc.args.len();
            let mut args_vars: Vec<crate::flowspace::model::Variable> = Vec::with_capacity(total);
            for _ in 0..total {
                args_vars.push(ctx.popvid_var(graph));
            }
            args_vars.reverse();
            // RPython `jtransform.py:410-412`: a polymorphic receiver
            // (dyn Trait) lowers to `indirect_call`, not `direct_call`.
            // Detect via the collected local_dyn_trait_roots map so
            // locals / params / Box<dyn> receivers all participate
            // (Issue 3 coverage).
            let receiver_root = receiver_type_root(&mc.receiver, ctx);
            let trait_bound_root = trait_bound_root_for_receiver(&mc.receiver, ctx);
            let target = if let Some(trait_root) = dyn_trait_root_for_receiver(&mc.receiver, ctx) {
                CallTarget::indirect(trait_root, mc.method.to_string())
            } else {
                CallTarget::method(mc.method.to_string(), receiver_root.clone())
            };
            // RPython parity: `op.result.concretetype` is set from the
            // callee graph's return signature at flowspace time
            // (`flowspace/objspace.py` consults the registered FuncDesc).
            // Pyre's pass 1 collected method return types into
            // `ctx.fn_return_types` keyed by `Type::method`; resolving
            // here lets the rtyper produce `Signed` operands for pure
            // integer ops (otherwise `value_type_to_kind` defaults to
            // `'r'` and the result reaches the assembler as a Ref-kind
            // operand, surfacing as `int_ge/ir>i` etc.).
            let method_return_type_string =
                lookup_method_return_type(ctx, receiver_root.as_deref(), &mc.method)
                    .or_else(|| {
                        lookup_method_return_type(ctx, trait_bound_root.as_deref(), &mc.method)
                    })
                    .cloned();
            let result_ty = primitive_method_result_type(graph, &args_vars, &mc.method)
                .or_else(|| transparent_option_method_result_type(graph, &args_vars, &mc.method))
                .or_else(|| {
                    method_return_type_string
                        .as_deref()
                        .map(type_string_to_value_type)
                })
                .unwrap_or(ValueType::Unknown);
            let var = graph
                .push_op_var(
                    *block,
                    OpKind::Call {
                        target,
                        args: args_vars,
                        result_ty,
                    },
                    true,
                )
                .expect("OpKind::Call has has_result=true");
            Ok(Lowered::from_value_var(graph, &var))
        }

        // ── if/else → block split (RPython FlowContext.guessbool) ──
        //
        // Creates: then_block, else_block, merge_block
        // If both branches produce a value, merge_block gets an inputarg
        // (Phi node) that receives the value from each branch via Link args.
        syn::Expr::If(if_expr) => lower_if_expr(graph, block, if_expr, options, ctx),

        // ── return ──
        syn::Expr::Return(ret) => {
            // RPython `RETURN_VALUE` (`flowspace/flowcontext.py`):
            // `popvalue()` then `raise Return(w_result)`.  Pyre
            // equivalent: evaluate the return value (propagating
            // path_closed / FlowingError), then `set_return(..)` closes
            // the block and `Lowered::path_closed()` tells the caller
            // to stop walking this path.
            let val_var = if let Some(e) = &ret.expr {
                let lowered = lower_expr(graph, block, e, options, ctx)?;
                if lowered.path_closed {
                    return Ok(Lowered::path_closed());
                }
                lowered.value_var(graph)
            } else {
                None
            };
            graph.set_return(*block, val_var);
            Ok(Lowered::path_closed())
        }

        // ── block { stmts } ──
        syn::Expr::Block(blk) => {
            lower_stmt_list_with_tail_value(graph, block, &blk.block.stmts, options, ctx)
        }

        // ── literals ──
        // RPython `rpython/annotator/model.py` + `rtyper/rclass.py` resolve
        // every literal to a concrete SSA value at annotation time.  pyre
        // handles the common RPython-usable cases here; cases that RPython
        // itself does not support (f64 literals, char/str/byte literals
        // inside annotated code) still fall through to `OpKind::Abort`
        // and are tracked as rtyper follow-ups.
        syn::Expr::Lit(lit) => {
            match &lit.lit {
                syn::Lit::Int(int_lit) => {
                    if let Ok(v) = int_lit.base10_parse::<i64>() {
                        let var = graph
                            .push_op_var(*block, OpKind::ConstInt(v), true)
                            .expect("ConstInt has has_result=true");
                        return Ok(Lowered::from_value_var(graph, &var));
                    }
                }
                // RPython lowers `True`/`False` to `Constant(True/False)`
                // of `lltype.Bool` (annotator/model.py:227 SomeBool).  At
                // the codewriter level `getkind(Bool)` returns `'int'`
                // (`rpython/jit/codewriter/flatten.py:getkind`) so the
                // value lives in an int register, but the annotator-side
                // distinction (SomeBool vs SomeInteger) is preserved by
                // emitting the dedicated `OpKind::ConstBool` variant.
                syn::Lit::Bool(b) => {
                    let var = graph
                        .push_op_var(*block, OpKind::ConstBool(b.value), true)
                        .expect("ConstBool has has_result=true");
                    return Ok(Lowered::from_value_var(graph, &var));
                }
                // RPython treats `chr(x)` / single-char byte literals as
                // `lltype.Char` which is also kind `'int'` (single unsigned
                // byte).  Rust `b'x'` (syn::Lit::Byte) and `'x'`
                // (syn::Lit::Char as u32) map to the same shape.
                syn::Lit::Byte(b) => {
                    let var = graph
                        .push_op_var(*block, OpKind::ConstInt(b.value() as i64), true)
                        .expect("ConstInt has has_result=true");
                    return Ok(Lowered::from_value_var(graph, &var));
                }
                syn::Lit::Char(c) => {
                    let var = graph
                        .push_op_var(*block, OpKind::ConstInt(c.value() as i64), true)
                        .expect("ConstInt has has_result=true");
                    return Ok(Lowered::from_value_var(graph, &var));
                }
                // RPython `flowmodel.py:Constant(rfloat)`: float literals
                // become `Constant` nodes with `lltype.Float` concretetype.
                // Pyre stores the bit pattern (`history.py:265
                // ConstFloat.getfloatstorage`) so PartialEq/Hash stay
                // derivable; the assembler materialises this through the
                // existing `constants_f` pool with a `float_copy` op.
                syn::Lit::Float(f) => {
                    if let Ok(v) = f.base10_parse::<f64>() {
                        let var = graph
                            .push_op_var(*block, OpKind::ConstFloat(v.to_bits()), true)
                            .expect("ConstFloat has has_result=true");
                        return Ok(Lowered::from_value_var(graph, &var));
                    }
                }
                _ => {}
            }
            // Unsupported literal kind — tag the specific variant so
            // the `Unknown` marker + diagnostics still identify the
            // remaining rtyper-side port gap (Str / Float / ByteStr /
            // Verbatim).  RPython `LOAD_CONST`
            // (`flowspace/flowcontext.py:841`) pushes the constant and
            // the flow walk continues; Err here would abort the whole
            // function graph and cascade through consumers like
            // `assert!("...")` / `panic!("...")` (which can legitimately
            // carry string literals next to side-effecting args).
            let variant = match &lit.lit {
                syn::Lit::Str(_) => UnsupportedLiteralKind::Str,
                syn::Lit::Float(_) => UnsupportedLiteralKind::Float,
                syn::Lit::ByteStr(_) => UnsupportedLiteralKind::ByteStr,
                syn::Lit::Verbatim(_) => UnsupportedLiteralKind::Verbatim,
                _ => UnsupportedLiteralKind::Other,
            };
            if std::env::var("MAJIT_UNKNOWN_DUMP").is_ok() {
                println!("cargo:warning=[UnsupportedLit] variant={variant:?}");
            }
            Ok(continue_with_unknown_literal(graph, *block, variant))
        }

        // ── path (variable reference) ──
        syn::Expr::Path(path) => {
            let name = path
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            // RPython `flowspace/flowcontext.py:835 LOAD_FAST`: the
            // bytecode reads the existing local-stack entry rather
            // than introducing a new `Variable`.  Pyre's analogue:
            // when a single-segment path names a local whose
            // definition lives in the *same* block as the read,
            // forward the bound `Variable` directly so downstream
            // passes see a single SSA definition with multiple uses
            // — matching upstream's frame-locals model.
            if path.path.segments.len() == 1
                && path.qself.is_none()
                && let Some((var, defining_block)) = ctx.local_var_of(&name, graph)
                && defining_block == *block
            {
                return Ok(Lowered::from_value_var(graph, &var));
            }
            // Cross-block read of a single-segment local — try lazy
            // install first (allocates an inputarg in `*block` +
            // threads `Link.args` back to every predecessor whose
            // closing site recorded a snapshot in `Block.framestate`).
            // Falls back to the legacy naked `OpKind::Input` emit when
            // any predecessor lacks a recorded snapshot.
            if path.path.segments.len() == 1
                && path.qself.is_none()
                && ctx
                    .local_value_ids
                    .get(&name)
                    .is_some_and(|(_, defining_block)| *defining_block != *block)
                && let Some(threaded_var) =
                    lazy_install_local_at_current_block_var(graph, ctx, *block, &name, None)
            {
                return Ok(Lowered::from_value_var(graph, &threaded_var));
            }
            // Path-as-value unit-variant ctor route — `StepResult::
            // Continue` / `JitAction::Return` etc. reach the Expr::Path
            // arm without going through `canonical_call_target`'s
            // Call-site routing.  Without this branch they fall through
            // to the naked `OpKind::Input` emit below and the rtyper
            // adapter rejects them as "adapter cross-block body Input"
            // because the qualified-path string is never a real local.
            // Route only the unit-variant subset of `is_synthetic_
            // ctor_path` (the Pyre-side `Class::Variant` group) so the
            // 0-arg `HostObject::new_class` lands as a
            // `SomeInstance(classdef)`.  Result/Option wrappers
            // (`Ok`/`Err`/`Some`) are deliberately excluded — they are
            // only valid as one-argument calls, and jtransform's
            // transparent elision (`is_synthetic_result_option_ctor`)
            // handles `args.len() == 1` only.
            let segments_vec: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect();
            // Path-as-value numeric constants — `flowspace/flowcontext.py:858
            // LOAD_GLOBAL` parity: when a path resolves to a statically
            // known module-level numeric attribute, emit a `Constant` node
            // (`flowmodel.py:Constant(value)`) instead of a body-`Input` that
            // would mis-classify as a cross-block local read.
            //
            // Match Rust's `f64::{INFINITY,NEG_INFINITY,NAN}` (and the
            // `std::f64::*` / `core::f64::*` long forms) directly to the
            // existing `OpKind::ConstFloat(bits)` lowering used at the
            // `syn::Lit::Float` arm above.
            if let Some(bits) = path_as_value_float_constant(&segments_vec) {
                let var = graph
                    .push_op_var(*block, OpKind::ConstFloat(bits), true)
                    .expect("ConstFloat has has_result=true");
                return Ok(Lowered::from_value_var(graph, &var));
            }
            // Parse-side `pub const NAME: <primitive> = <literal>`
            // resolution.  `ctx.module_statics` (populated at
            // `build_semantic_program_*_with_options` from each
            // `ParsedInterpreter::module_statics`) carries every
            // file-root const/static whose initialiser was a single
            // primitive `syn::Lit::{Bool, Int, Float}` literal.  When
            // the Expr::Path resolves to such a decl, emit the matching
            // typed `OpKind::Const*` directly — mirroring how the
            // `syn::Lit::*` arm above lowers an in-place literal.
            //
            // Same RPython parity argument as `path_as_value_float_constant`
            // above: `flowspace/flowcontext.py:858 LOAD_GLOBAL` produces
            // `Constant(value)` when the resolved module attribute is
            // statically known.
            if let Some(literal) = lookup_module_static_literal(&segments_vec, ctx) {
                let const_op = match literal {
                    crate::parse::ModuleStaticLiteral::Bool(b) => OpKind::ConstBool(b),
                    crate::parse::ModuleStaticLiteral::Int(v) => OpKind::ConstInt(v),
                    crate::parse::ModuleStaticLiteral::Float(bits) => OpKind::ConstFloat(bits),
                };
                let var = graph
                    .push_op_var(*block, const_op, true)
                    .expect("Const{Bool,Int,Float} has has_result=true");
                return Ok(Lowered::from_value_var(graph, &var));
            }
            if is_synthetic_unit_variant_path(&segments_vec)
                && !registered_function_path(&segments_vec, ctx)
            {
                let (last_idx, last) = segments_vec
                    .iter()
                    .enumerate()
                    .last()
                    .map(|(i, s)| (i, s.clone()))
                    .expect("synthetic ctor path is non-empty");
                let owner_path = segments_vec[..last_idx].to_vec();
                let target = CallTarget::synthetic_transparent_ctor_with_owner(owner_path, last);
                let result_var = graph.push_op_var(
                    *block,
                    OpKind::Call {
                        target,
                        args: Vec::new(),
                        result_ty: ValueType::Unknown,
                    },
                    true,
                );
                return Ok(result_var
                    .map(|var| Lowered::from_value_var(graph, &var))
                    .unwrap_or_else(Lowered::no_value));
            }
            let ty = ctx
                .local_value_types
                .get(&name)
                .cloned()
                .unwrap_or(ValueType::Unknown);
            let value_var = graph.push_op_var(
                *block,
                OpKind::Input {
                    name: name.clone(),
                    ty: ty.clone(),
                },
                true,
            );
            // RPython `LOAD_FAST` parity (`flowspace/flowcontext.py:835`):
            // once the local-stack slot is read into a Variable in the
            // current block, subsequent reads of the same name in the
            // same block must return the same Variable — RPython's
            // bytecode reads the slot, not a fresh copy.  Pyre's
            // single-segment `Expr::Path` reaches this fallback only
            // when the same-block reuse and cross-block lazy install
            // both decline (no recorded predecessor framestate, or
            // graph-recoverable kind disagreement at
            // `lazy_install_local_at_current_block:1751-1757`); we
            // register the freshly-emitted `Input` result as the
            // authoritative `(Variable, current_block)` so further
            // reads of `name` within the same block dedup against this
            // synthetic Input.
            //
            // `LocalBindingSnapshot` saves and restores
            // `ctx.local_value_ids` across `If` / `Match` / `Loop` /
            // `While` / `ForLoop` boundaries, so the cached `(vid,
            // block)` does not leak into a sibling control-flow arm.
            if let Some(ref var) = value_var
                && path.path.segments.len() == 1
                && path.qself.is_none()
            {
                ctx.bind_local_id_var(name.clone(), var, graph, *block);
            }
            Ok(value_var
                .map(|var| Lowered::from_value_var(graph, &var))
                .unwrap_or_else(Lowered::no_value))
        }

        // ── reference &expr ──
        syn::Expr::Reference(r) => lower_expr(graph, block, &r.expr, options, ctx),

        // `&raw const/mut expr` (`syn::Expr::RawAddr`) is intentionally
        // *not* pass-through here.  Unlike `&expr`, the raw-address
        // operator yields the *address* of the inner expr rather than
        // its value, so reusing the inner lowering would silently
        // misrepresent semantics (a downstream `as usize` cast would
        // see the dereferenced value instead of the pointer).  Falling
        // through to the `_ => other` unsupported handler classifies
        // it as `UnsupportedExprKind::RawAddr` (data-creation arm),
        // walks the inner expr for side effects via the `match other`
        // RawAddr branch below, and emits an `Unknown` marker so the
        // graph remains opaque rather than incorrect.

        // ── parenthesized (expr) ──
        syn::Expr::Paren(p) => lower_expr(graph, block, &p.expr, options, ctx),

        // ── unary *x, !x, -x ──
        syn::Expr::Unary(u) => {
            // `*x` (Rust deref) has no flowspace counterpart —
            // `flowspace/operation.py:465-474` registers only `pos` /
            // `neg` / `invert` / `bool` as unary ops.  The
            // RPython-parity `build_flow.rs::lower_unary`
            // (`flowspace/rust_source/build_flow.rs:3301`) treats
            // `UnOp::Deref` as `lower_expr(b, &u.expr)` pass-through:
            // the annotator tracks identity + type regardless of
            // borrow form, so emitting an aliasing op here is
            // redundant.  Pyre's codewriter independently aliases
            // `OpKind::UnaryOp { op: "deref", .. }` to its operand
            // at `jit_codewriter/jtransform.rs:711` (same arm as
            // `same_as`), confirming no semantic load is lost.
            // Pass-through here lets the rtyper-side adapter
            // (`translator/rtyper/flowspace_adapter.rs:359`) skip
            // the `deref` Skip category — the production graph
            // never carries `deref` ops past this point.  The
            // fail-loud invariant at adapter level remains: any
            // synthetic graph that injects `OpKind::UnaryOp {
            // op: "deref", .. }` directly still surfaces a
            // `TyperError` (anchor test
            // `cutover.rs:anchor_unary_deref_surfaces_failloud_no_flowspace_peer`).
            if matches!(u.op, syn::UnOp::Deref(_)) {
                return lower_expr(graph, block, &u.expr, options, ctx);
            }
            // ── Rust `!x` lowering — RPython has TWO opcodes; pyre
            //    folds them at this single site.
            //    TODO(unary-not-split): when the front-end gains
            //    receiver-typed dispatch, split this back into the
            //    UNARY_NOT (`flowcontext.py:531-538`) vs UNARY_INVERT
            //    (`flowcontext.py:188-191`) shape so each surfaces
            //    its own opname.
            //
            // Upstream RPython distinguishes two unary-not operators:
            //
            //   * `UNARY_NOT`   (`flowcontext.py:531-538`) — *logical* not
            //     on booleans / truthy values.  Lowered as `op.bool(w_value)`
            //     followed by `guessbool` + constant-tail join.
            //
            //   * `UNARY_INVERT` (`flowcontext.py:188-191`) — *bitwise* not
            //     on integers.  Lowered as `op.invert(w_value)`, registered
            //     at `operation.py:474 add_operator('invert', 1, ..)` /
            //     `lloperation.py int_invert`.
            //
            // Rust's `!` is overloaded by the `std::ops::Not` trait:
            // `!bool` → logical not, `!i64` (and other integer types) →
            // *bitwise* not.  The frontend must classify the operand
            // before lowering: bool goes through the `UNARY_NOT` shape,
            // int goes through `UNARY_INVERT`.  Unknown operands
            // fail-loud (`stop_unsupported`) since RPython's
            // `flowcontext.py:194,535-538` dispatches strictly at the
            // bytecode token; guessing would collapse two distinct
            // RPython bytecodes.
            //
            // The `UNARY_NOT` branch desugars via `bool(x)` + branch +
            // constant tail, mirroring `flowcontext.py:531-538`:
            //
            //     w_value = self.popvalue()
            //     w_bool  = op.bool(w_value).eval(self)
            //     self.pushvalue(const(not self.guessbool(w_bool)))
            //
            // Twin of `flowspace/rust_source/build_flow.rs:1337
            // lower_unary_not`.  Both arms Link straight to the join with
            // a constant tail — no separate evaluation block (the simpler
            // twin of `&&`/`||`'s `lower_short_circuit`).
            //
            if matches!(u.op, syn::UnOp::Not(_)) {
                // ── Statically-obvious int operand: emit `invert` op ──
                // RPython `flowcontext.py:188-191 UNARY_INVERT` dispatches
                // to `op.invert(w_arg)` (`operation.py:474
                // add_operator('invert', 1, ..)`).  Pyre's rtyper has the
                // matching `"invert"` arm in `RPythonTyper::translate_op`
                // routing to `Repr::rtype_invert`.  Lowering an
                // int-literal `!` directly to `OpKind::UnaryOp { op:
                // "invert", .. }` skips the bool-branch detour entirely
                // — the result is the bitwise complement, matching
                // `~lit` in Python.
                // ── Dynamic operand-type dispatch ──
                // Rust's `!` is overloaded via `std::ops::Not`:
                // `!T where T: Not` is bitwise complement for integer
                // types and logical negation for `bool`.  Mirror PyPy's
                // bytecode-level distinction (`UNARY_INVERT` vs
                // `UNARY_NOT`) by inspecting the operand's static type
                // through `local_type_strings`/`local_value_types`
                // tracking populated at let-binding / fn-parameter
                // time.
                //
                // Statically detected as int → emit `invert` op
                // (UNARY_INVERT). Statically detected as bool → fall
                // through to the UNARY_NOT bool+branch desugar.
                // Unknown fail-louds via `stop_unsupported`.
                match expr_unary_not_operand_kind(&u.expr, ctx) {
                    UnaryNotOperandKind::Int => {
                        let operand_pre_var =
                            get_value_var!(lower_expr(graph, block, &u.expr, options, ctx)?, graph);
                        ctx.pushvid_var(&operand_pre_var);
                        let operand_var = ctx.popvid_var(graph);
                        // The classifier returns `Int` for both
                        // primitive integer kinds (lowered as
                        // `ValueType::Int`) and arbitrary-precision
                        // integers like `BigInt` (lowered as
                        // `ValueType::Ref`). RPython's
                        // `IntegerRepr.rtype_invert` /
                        // `LongRepr.rtype_invert` dispatch on the
                        // operand's lattice node; pyre projects that
                        // through `graph_value_type_var(operand_var)` so
                        // the emitted `OpKind::UnaryOp.result_ty` matches
                        // the operand's actual lowered shape and the
                        // function's declared return type
                        // (`bigint_invert(a: BigInt) -> BigInt` →
                        // `Ref`).
                        let result_ty = graph_value_type_var(graph, &operand_var)
                            .filter(|ty| matches!(ty, ValueType::Int | ValueType::Ref(_)))
                            .unwrap_or(ValueType::Int);
                        let var = graph
                            .push_op_var(
                                *block,
                                OpKind::UnaryOp {
                                    op: "invert".into(),
                                    operand: operand_var,
                                    result_ty,
                                },
                                true,
                            )
                            .expect("OpKind::UnaryOp has has_result=true");
                        return Ok(Lowered::from_value_var(graph, &var));
                    }
                    UnaryNotOperandKind::Bool => {}
                    UnaryNotOperandKind::Unknown => {
                        // RPython `flowcontext.py:194,535-538`
                        // dispatches `UNARY_NOT` vs `UNARY_INVERT`
                        // strictly at the Python bytecode token; pyre
                        // mirrors that contract by fail-louding when
                        // the operand kind cannot be recovered.
                        // `build_flow.rs:4404-4416` is fail-loud on the
                        // same shape. The classifier
                        // (`expr_unary_not_operand_kind`,
                        // `front/ast.rs:5582`) handles the production
                        // patterns surfaced in
                        // `pyre-{object,interpreter,jit}/src/` plus
                        // `majit-ir/src/resoperation.rs`.
                        return stop_unsupported(
                            graph,
                            *block,
                            UnsupportedExprKind::UnaryNotUnknownOperand,
                        );
                    }
                }
                let operand_pre_var =
                    get_value_var!(lower_expr(graph, block, &u.expr, options, ctx)?, graph);
                ctx.pushvid_var(&operand_pre_var);
                let operand_var = ctx.popvid_var(graph);
                let cond_var = graph
                    .push_op_var(
                        *block,
                        OpKind::UnaryOp {
                            op: "bool".into(),
                            operand: operand_var,
                            result_ty: ValueType::Bool,
                        },
                        true,
                    )
                    .expect("UnaryOp { op: \"bool\", .. } produces a value");
                // RPython pushes `Constant(not python_bool)` per arm of
                // `lltype.Bool` — the false arm produces `True`
                // (`!false == true`); the true arm produces `False`
                // (`!true == false`).  Emit as `ConstBool` so the
                // annotator picks `SomeBool` rather than `SomeInteger`.
                let const_true_var = graph
                    .push_op_var(*block, OpKind::ConstBool(true), true)
                    .expect("ConstBool produces a value");
                let const_false_var = graph
                    .push_op_var(*block, OpKind::ConstBool(false), true)
                    .expect("ConstBool produces a value");
                // ── Locals threading ──
                // RPython `flowcontext.py:531-538 UNARY_NOT` propagates
                // frame locals across the bool-fork via `Link.args` ↔
                // `inputargs`.  Mirror of build_flow.rs:1363
                // lower_unary_not's `[tail, ...locals]` join shape.
                // Both arms' Links carry pre-fork local values plus the
                // constant tail; ctx.local_value_ids rebinds to join
                // inputargs after the merge so post-join reads resolve
                // to the merged values.
                let pre_fork_locals = ctx.local_value_ids.clone();
                let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
                merged_names.sort();
                // Each pre-fork local must be defined in `*block` before
                // it can ride the fork's `Link.args`.  Pyre threads
                // locals lazily (on cross-block read), so a local bound
                // in a dominator but never read in `*block` still points
                // at the dominator's Variable; threading that
                // out-of-scope slot onto an arm link trips the adapter's
                // "undefined operand slot" invariant.  Lazy-install the
                // slot at `*block` (walking the predecessor chain to
                // install inputargs and append each edge's `Link.args`)
                // so the carried Variable is `*block`-local.  RPython
                // needs no analogue: `flowcontext.py:835 LOAD_FAST` /
                // `getoutputargs` thread every `frame.locals_w` slot at
                // every edge eagerly.
                let pre_fork_local_vars: Vec<crate::flowspace::model::Variable> = merged_names
                    .iter()
                    .map(|name| {
                        let var = pre_fork_locals[name].0.clone();
                        if graph.variable_defined_in_block(*block, &var) {
                            var
                        } else {
                            lazy_install_local_at_current_block_var(graph, ctx, *block, name, None)
                                .unwrap_or(var)
                        }
                    })
                    .collect();

                let (join_block, join_arg_vars) =
                    graph.create_block_with_arg_vars(merged_names.len() + 1);
                let tail_var = join_arg_vars[0].clone();
                let join_local_arg_vars: &[crate::flowspace::model::Variable] = &join_arg_vars[1..];

                let mut false_arm_vars: Vec<crate::flowspace::model::Variable> =
                    Vec::with_capacity(merged_names.len() + 1);
                false_arm_vars.push(const_false_var);
                false_arm_vars.extend(pre_fork_local_vars.iter().cloned());
                let mut true_arm_vars: Vec<crate::flowspace::model::Variable> =
                    Vec::with_capacity(merged_names.len() + 1);
                true_arm_vars.push(const_true_var);
                true_arm_vars.extend(pre_fork_local_vars.iter().cloned());

                // Two Links into the same join: cond truthy → tail
                // is `0` (false); cond falsy → tail is `1` (true).
                graph.set_branch(
                    *block,
                    cond_var,
                    join_block,
                    false_arm_vars,
                    join_block,
                    true_arm_vars,
                );

                // Rebind locals to join_block's inputargs.  Same
                // pattern as the `&&`/`||` arm above.
                for (name, arg_var) in merged_names.iter().zip(join_local_arg_vars.iter()) {
                    ctx.bind_local_id_var(name.clone(), arg_var, graph, join_block);
                }

                *block = join_block;
                return Ok(Lowered::from_value_var(graph, &tail_var));
            }
            let operand_pre_var =
                get_value_var!(lower_expr(graph, block, &u.expr, options, ctx)?, graph);
            ctx.pushvid_var(&operand_pre_var);
            let operand_var = ctx.popvid_var(graph);
            let var = graph
                .push_op_var(
                    *block,
                    OpKind::UnaryOp {
                        op: unary_op_name(&u.op).into(),
                        operand: operand_var,
                        result_ty: ValueType::Unknown,
                    },
                    true,
                )
                .expect("OpKind::UnaryOp has has_result=true");
            Ok(Lowered::from_value_var(graph, &var))
        }

        // ── binary a + b ──
        syn::Expr::Binary(bin) => {
            // Short-circuit `&&` / `||` are control flow in RPython
            // (`flowspace/operation.py:475-510` does NOT register
            // short-circuit `and`/`or` as binary operators).  Mirror
            // `flowspace/rust_source/build_flow.rs:1191
            // lower_short_circuit` line-by-line: emit `bool(lhs)` as the
            // exitswitch discriminator, fork the block via
            // `set_branch`, evaluate rhs in a separate block, and merge
            // both arms into a `[tail, ...locals]`-shaped join carrying
            // either `lhs_raw` (short-circuit arm) or `rhs_raw` (full
            // eval) as `tail` plus every pre-fork frame local threaded
            // through (so an `STORE_FAST` inside rhs propagates past the
            // join — `build_flow.rs:1218-1232 / :1281-1292`).  Upstream
            // bytecode basis — `flowcontext.py:766-777
            // JUMP_IF_FALSE_OR_POP` (`&&`) / `JUMP_IF_TRUE_OR_POP`
            // (`||`):
            //
            //     w_value = self.peekvalue()
            //     if not self.guessbool(op.bool(w_value).eval(self)):
            //         return target          # short-circuit on False
            //     self.popvalue()             # else evaluate rhs
            //
            // The surviving value is the original `lhs_raw` (matching
            // `peekvalue()` then `popvalue()`), not `bool(lhs_raw)` —
            // `bool` is only the switch discriminator.  `&&`
            // short-circuits on False (lhs falsy is the result; rhs is
            // dead); `||` short-circuits on True.  Without this
            // desugar, `binary_op_name(&bin.op)` emits the literal
            // `"and"` / `"or"` opname which then trips
            // `flowspace_adapter.rs:422 normalize_binop_name`'s
            // fail-loud arm, blocking the rtyper cutover.
            if matches!(bin.op, syn::BinOp::And(_) | syn::BinOp::Or(_)) {
                let is_and = matches!(bin.op, syn::BinOp::And(_));

                let lhs_raw_pre_var =
                    get_value_var!(lower_expr(graph, block, &bin.left, options, ctx)?, graph);
                ctx.pushvid_var(&lhs_raw_pre_var);
                let lhs_raw_var = ctx.popvid_var(graph);
                let cond_var = graph
                    .push_op_var(
                        *block,
                        OpKind::UnaryOp {
                            op: "bool".into(),
                            operand: lhs_raw_var.clone(),
                            result_ty: ValueType::Bool,
                        },
                        true,
                    )
                    .expect("UnaryOp { op: \"bool\", .. } produces a value");

                // ── Locals threading through fork/join ──
                //
                // RPython's frame-locals model (`flowcontext.py:835
                // LOAD_FAST` / `:872-884 STORE_FAST`) propagates locals
                // across every fork via `Link.args` ↔ target
                // `inputargs`.  Pyre's
                // `flowspace/rust_source/build_flow.rs:1191
                // lower_short_circuit` mirrors this with a
                // `[tail, ...merged_names]` join shape — the parallel
                // port done here for `front/ast.rs::Expr::Binary`
                // `&&`/`||`.
                //
                // Pre-fork snapshot of `ctx.local_value_ids` provides
                // the names that must thread through the join; rhs
                // lowers against rhs_block's inputargs so any `STORE_
                // FAST` inside rhs writes to a local already-in-flight
                // through the join.
                //
                // A future migration brings the
                // same pattern to `Expr::If` / `Expr::Match` so all
                // fork/join shapes use the consistent
                // `[result, ...locals]` Link.args contract; the
                // single-fork forms (`!`-bool desugar) follow as a
                // sibling slice.
                let pre_fork_locals = ctx.local_value_ids.clone();
                let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
                merged_names.sort();
                // Each pre-fork local must be defined in `*block` before
                // it can ride the fork's `Link.args`.  Pyre threads
                // locals lazily (on cross-block read), so a local bound
                // in a dominator but never read in `*block` still points
                // at the dominator's Variable; threading that
                // out-of-scope slot onto an arm link trips the adapter's
                // "undefined operand slot" invariant.  Lazy-install the
                // slot at `*block` (walking the predecessor chain to
                // install inputargs and append each edge's `Link.args`)
                // so the carried Variable is `*block`-local.  RPython
                // needs no analogue: `flowcontext.py:835 LOAD_FAST` /
                // `getoutputargs` thread every `frame.locals_w` slot at
                // every edge eagerly.
                let pre_fork_local_vars: Vec<crate::flowspace::model::Variable> = merged_names
                    .iter()
                    .map(|name| {
                        let var = pre_fork_locals[name].0.clone();
                        if graph.variable_defined_in_block(*block, &var) {
                            var
                        } else {
                            lazy_install_local_at_current_block_var(graph, ctx, *block, name, None)
                                .unwrap_or(var)
                        }
                    })
                    .collect();

                let (mut rhs_block, rhs_local_arg_vars) =
                    graph.create_block_with_arg_vars(merged_names.len());
                let (join_block, join_arg_vars) =
                    graph.create_block_with_arg_vars(merged_names.len() + 1);
                let tail_var = join_arg_vars[0].clone();
                let join_local_arg_vars: &[crate::flowspace::model::Variable] = &join_arg_vars[1..];

                // Short-circuit Link.args = [lhs_raw, ...pre_fork_locals];
                // rhs Link.args         = [...pre_fork_locals].
                let mut shortcut_link_vars: Vec<crate::flowspace::model::Variable> =
                    Vec::with_capacity(merged_names.len() + 1);
                shortcut_link_vars.push(lhs_raw_var.clone());
                shortcut_link_vars.extend(pre_fork_local_vars.iter().cloned());
                let rhs_link_vars: Vec<crate::flowspace::model::Variable> =
                    pre_fork_local_vars.clone();

                if is_and {
                    // `&&`: cond truthy → eval rhs; cond falsy →
                    // short-circuit `lhs_raw` straight to the join.
                    graph.set_branch(
                        *block,
                        cond_var,
                        rhs_block,
                        rhs_link_vars,
                        join_block,
                        shortcut_link_vars,
                    );
                } else {
                    // `||`: cond truthy → short-circuit `lhs_raw`
                    // straight to the join; cond falsy → eval rhs.
                    graph.set_branch(
                        *block,
                        cond_var,
                        join_block,
                        shortcut_link_vars,
                        rhs_block,
                        rhs_link_vars,
                    );
                }

                // Rebind locals to rhs_block's inputargs so rhs
                // lowering sees them via same-block reads.
                for (name, arg_var) in merged_names.iter().zip(rhs_local_arg_vars.iter()) {
                    ctx.bind_local_id_var(name.clone(), arg_var, graph, rhs_block);
                }

                // Lower rhs in `rhs_block`; if the path remains open,
                // link to the join carrying `rhs_raw` plus current
                // local values (which rhs may have rebound through
                // assigns / nested control flow).  An rhs that closes
                // its path (`return` / `raise` / `break` inside the
                // right operand) leaves only the short-circuit arm
                // reaching the join — same open-arm-only pattern as
                // `lower_if` above.
                let rhs_lowered = lower_expr(graph, &mut rhs_block, &bin.right, options, ctx)?;
                if graph.block(rhs_block).is_open() {
                    // `rhs_lowered.value` is `None` only when rhs was a
                    // statement-form sub-expression with no value;
                    // `lhs_raw_var` keeps the join arity-correct in that
                    // unusual case (defensive, mirrors `lower_if`'s
                    // arity guard).
                    let rhs_raw_var = rhs_lowered
                        .value_var(graph)
                        .unwrap_or_else(|| lhs_raw_var.clone());
                    let rhs_exit_local_vars: Vec<crate::flowspace::model::Variable> = merged_names
                        .iter()
                        .map(|name| {
                            ctx.local_value_ids
                                .get(name)
                                .map(|(var, _)| var.clone())
                                .unwrap_or_else(|| {
                                    pre_fork_locals
                                        .get(name)
                                        .map(|(var, _)| var.clone())
                                        .expect("local must remain in scope after rhs lower")
                                })
                        })
                        .collect();
                    let mut rhs_to_join_vars: Vec<crate::flowspace::model::Variable> =
                        Vec::with_capacity(merged_names.len() + 1);
                    rhs_to_join_vars.push(rhs_raw_var);
                    rhs_to_join_vars.extend(rhs_exit_local_vars);
                    graph.set_goto(rhs_block, join_block, rhs_to_join_vars);
                }

                // Rebind locals to join_block's inputargs so post-join
                // reads of each name resolve to the merged phi value
                // — `(join_inputarg, join_block)` is the same-block
                // tuple `Expr::Path` checks at line 2114 to elide the
                // `OpKind::Input` emit.  Mirror of build_flow.rs:1294-
                // 1300's `b.open_new_block(... join_locals ...)`.
                for (name, arg_var) in merged_names.iter().zip(join_local_arg_vars.iter()) {
                    ctx.bind_local_id_var(name.clone(), arg_var, graph, join_block);
                }

                *block = join_block;
                return Ok(Lowered::from_value_var(graph, &tail_var));
            }

            let lhs_pre_var =
                get_value_var!(lower_expr(graph, block, &bin.left, options, ctx)?, graph);
            ctx.pushvid_var(&lhs_pre_var);
            let rhs_pre_var =
                get_value_var!(lower_expr(graph, block, &bin.right, options, ctx)?, graph);
            ctx.pushvid_var(&rhs_pre_var);
            let rhs_var = ctx.popvid_var(graph);
            let lhs_var = ctx.popvid_var(graph);
            let op_name = binary_op_name(&bin.op);
            let result_ty = binary_result_value_type_var(graph, &lhs_var, &rhs_var, op_name);
            let value_var = graph.push_op_var(
                *block,
                OpKind::BinOp {
                    op: op_name.into(),
                    lhs: lhs_var,
                    rhs: rhs_var,
                    result_ty,
                },
                true,
            );
            // RPython INPLACE_* + STORE_FAST parity
            // (`flowspace/flowcontext.py:878-885`): compound assignment
            // `x += y` (and -=, *=, /=, %=, &=, |=, ^=, <<=, >>=) push
            // the inplace result and immediately replace the locals
            // slot for `x`, then renames the resulting `Variable` to
            // the local name.  Without the local_value_ids update,
            // the same-block dedup cache (`ast.rs:1389` let arm,
            // `:1724` simple-assign arm) still points at the
            // pre-inplace Variable, so a later same-block read of
            // `x` returns the stale value.  Without the
            // `graph.name_value_var` rename, the adapter's
            // `name_to_value` lookup continues to resolve `x` to the
            // pre-inplace Variable.  Simple assignment `x = y` is
            // handled at the Expr::Assign arm above; this branch
            // owns the compound path that lowers as Expr::Binary.
            // The compound BinOp result is always a Variable (not a
            // `ConstInt`/`ConstFloat` define-op).
            if op_name.ends_with("_assign")
                && let (Some(var), syn::Expr::Path(path)) = (&value_var, &*bin.left)
                && path.path.segments.len() == 1
                && path.qself.is_none()
            {
                let name = path
                    .path
                    .segments
                    .iter()
                    .map(|seg| seg.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                ctx.bind_local_id_var(name, var, graph, *block);
            }
            Ok(value_var
                .map(|var| Lowered::from_value_var(graph, &var))
                .unwrap_or_else(Lowered::no_value))
        }

        // ── cast: expr as T ──
        syn::Expr::Cast(cast) => {
            let operand_pre_var =
                get_value_var!(lower_expr(graph, block, &cast.expr, options, ctx)?, graph);
            ctx.pushvid_var(&operand_pre_var);
            let operand_var = ctx.popvid_var(graph);
            let result_ty = classify_fn_arg_ty(&cast.ty);
            if result_ty == ValueType::Unknown {
                return Ok(Lowered::from_value_var(graph, &operand_var));
            }
            if result_ty == ValueType::Void {
                return Ok(Lowered::no_value());
            }
            let source_ty = graph_value_type_var(graph, &operand_var);
            // `Ref → Unsigned` has no single-call canonical in upstream
            // RPython.  `llmemory.cast_adr_to_uint(addr)` is defined as
            // `r_uint(cast_adr_to_int(addr))` (llmemory.py), and
            // `cast_ptr_to_int(p)` returns `Signed`, so the
            // Rust `ptr as usize` surface lowers to the same two-step
            // composition: `cast_ptr_to_int(p)` → intermediate `Signed`,
            // then `r_uint(intermediate)` → `Unsigned`.  Routing
            // through `same_as` would propagate the operand's Ref
            // concretetype to the result and leak a kind-mismatched
            // `int_mod/ri>i` opname when the result is later used as
            // an integer operand (lock-in at
            // `pyre-jit-trace/src/jitcode_runtime.rs:1340`).
            if matches!(source_ty.as_ref(), Some(ValueType::Ref(_)))
                && result_ty == ValueType::Unsigned
            {
                let intermediate_var = graph
                    .push_op_var(
                        *block,
                        OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: [
                                    "rpython",
                                    "rtyper",
                                    "lltypesystem",
                                    "lltype",
                                    "cast_ptr_to_int",
                                ]
                                .iter()
                                .map(|s| s.to_string())
                                .collect(),
                            },
                            args: vec![operand_var],
                            result_ty: ValueType::Int,
                        },
                        true,
                    )
                    .expect("cast_ptr_to_int call op must produce a result slot");
                let var = graph
                    .push_op_var(
                        *block,
                        OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: ["rpython", "rlib", "rarithmetic", "r_uint"]
                                    .iter()
                                    .map(|s| s.to_string())
                                    .collect(),
                            },
                            args: vec![intermediate_var],
                            result_ty: ValueType::Unsigned,
                        },
                        true,
                    )
                    .expect("OpKind::Call has has_result=true");
                return Ok(Lowered::from_value_var(graph, &var));
            }
            // The canonical `int()` / `float()` / `bool()` / `cast_*`
            // coercion chain — and the identity `same_as` fallback
            // (rtyper.py:478-481) for source-type-unknown / identity
            // casts — is shared with the `<prim>::from(x)` function-call
            // spelling; see `lower_value_cast`.
            let var = lower_value_cast(graph, *block, operand_var, source_ty, result_ty);
            Ok(Lowered::from_value_var(graph, &var))
        }

        // ── match expr { arms } → multi-block (RPython switch) ──
        syn::Expr::Match(m) => {
            // Z4.B.1.b: scrutinee eval routes through pushvid/popvid
            // (cf. Expr::If cond at line 3852).  Equivalent to upstream
            // `flowcontext.py:1180 build_class` / `:1207 setup_with`
            // patterns where the scrutinee was on the stack from its
            // producing opcode and the dispatching opimpl pops it.
            let scrutinee_pre_var =
                get_value_var!(lower_expr(graph, block, &m.expr, options, ctx)?, graph);
            ctx.pushvid_var(&scrutinee_pre_var);
            let scrutinee_var = ctx.popvid_var(graph);
            let scrutinee_type_string = expression_type_string(&m.expr, ctx);

            if m.arms.is_empty() {
                return Ok(Lowered::no_value());
            }
            let bool_exitcases = classify_match_bool_exitcases(&m.arms);
            let switch_exitcases = if bool_exitcases.is_none() {
                classify_match_switch_exitcases(&m.arms)
            } else {
                None
            };

            // Lower each arm body into its own block, collecting both
            // the ENTRY block (what the outer Branch/Goto jumps to)
            // and the TAIL block (what jumps to merge). lower_expr
            // takes `&mut arm_block` and may rewire arm_block to the
            // arm's tail (e.g., nested if/match's merge). We capture
            // the entry before calling so the outer terminator targets
            // the right landing pad.
            //
            // The merge block's inputarg list must have the same
            // length as every outgoing Goto's args (flatten.py:308
            // assumption), so we defer merge creation until we know
            // whether any arm actually produced a value.
            //
            // RPython `flowspace/flowcontext.py:417` — `FlowingError`
            // from any arm aborts the whole function graph, not just
            // the current arm.  `?` here propagates that out of the
            // whole match so the enclosing `build_function_graph` body
            // loop breaks at the first unsupported construct, matching
            // upstream's all-or-nothing flowgraph semantics.
            // Pre-create every arm's entry block so the scrutinee's exits
            // + framestate are installed BEFORE the arm bodies are
            // lowered — mirroring `lower_if_expr`, where `set_branch`
            // (creating the then/else edges) and the framestate stamp both
            // precede the then/else walk (ast.rs:4919-4920).  A cross-block
            // read of a pre-match local — whether inside an arm body or in
            // the post-merge continuation — then finds the scrutinee→entry
            // edge and the recorded snapshot already present, so the lazy
            // installer threads `Link.args` back through the scrutinee
            // block instead of falling to the naked body-`Input` emit the
            // rtyper adapter rejects.
            let arm_entries: Vec<BlockId> = m.arms.iter().map(|_| graph.create_block()).collect();

            // Capture the scrutinee block's locals frame as it stands when
            // it branches to the arm entries; stamped on `Block.framestate`
            // below (after the exits are installed).  RPython parity:
            // `flowspace/flowcontext.py:38 SpamBlock.framestate`.
            let pre_match_snapshot = ctx.getstate(graph, 0);

            // Dispatch (set_branch / exitswitch / single-goto fallback).
            // Runs BEFORE the arm bodies are lowered so (a) a cross-block
            // read inside an arm body threads back through the
            // scrutinee→entry edge, and (b) the migration's dry-run below
            // sees `*block` as a predecessor of each `arm_entries[i]`
            // rather than dead-ending at an orphan entry.  Slice 4.2-style
            // `set_goto` to merge is order-independent (links from tail are
            // unaffected by `*block`'s branch shape).
            if let Some(arm_exitcases) = bool_exitcases {
                // RPython `flatten.py:240-267` lowers Bool exitswitch
                // blocks through the two-link `goto_if_not` path, not the
                // integer `switch` path.  A Rust `match flag { true => ...,
                // _ => ... }` is therefore expanded to explicit
                // True/False exitcases instead of using the switch
                // `"default"` sentinel.
                let mut exits = Vec::new();
                for (entry, exitcases) in arm_entries.iter().zip(arm_exitcases.iter()) {
                    for exitcase in exitcases {
                        exits.push(
                            Link::new_mixed(Vec::new(), *entry, Some(exitcase.clone()))
                                .with_llexitcase_from_exitcase(),
                        );
                    }
                }
                graph.set_control_flow_metadata(
                    *block,
                    Some(ExitSwitch::Value(scrutinee_var)),
                    exits,
                );
            } else if let Some(arm_exitcases) = switch_exitcases {
                // RPython `flatten.py:278-308` switch shape:
                // `exitswitch` is the scrutinee and each primitive arm
                // contributes one Link with a concrete `exitcase`.  A
                // wildcard arm uses the same `"default"` sentinel that
                // upstream treats as the fall-through switch path.
                let mut exits = Vec::new();
                for (entry, exitcases) in arm_entries.iter().zip(arm_exitcases.iter()) {
                    for exitcase in exitcases {
                        exits.push(
                            Link::new_mixed(Vec::new(), *entry, Some(exitcase.clone()))
                                .with_llexitcase_from_exitcase(),
                        );
                    }
                }
                graph.set_control_flow_metadata(
                    *block,
                    Some(ExitSwitch::Value(scrutinee_var)),
                    exits,
                );
            } else if m.arms.len() == 1 {
                graph.set_goto(*block, arm_entries[0], vec![]);
            } else {
                // Structural adaptation for Rust composite patterns
                // (`if let Some(_)`, `Err(_)`, tuple/struct variants).
                // This front-end lacks a typed enum-discriminant op; keep
                // the existing two-arm truthy split for those cases rather
                // than inventing a fake switch key. Primitive literal
                // patterns use the switch path above.
                graph.set_branch(
                    *block,
                    scrutinee_var,
                    arm_entries[0],
                    vec![],
                    arm_entries[1],
                    vec![],
                );
            }

            // Stamp the scrutinee block's framestate now that its exits to
            // the arm entries are installed (all four dispatch shapes above
            // set exits on `*block`).  This is what lets a cross-block read
            // of a pre-match local thread `Link.args` back through the
            // scrutinee block; without it the lazy installer's predecessor
            // recursion reaches the scrutinee block, finds no recorded
            // snapshot, and falls back to the naked body-`Input` emit.
            // Mirror of `lower_if_expr`'s `set_branch` + framestate stamp
            // pair (ast.rs:4919-4920).
            graph.block_mut(*block).framestate = Some(pre_match_snapshot);

            // Each arm carries (tail, value, exit_framestate,
            // exit_local_bindings).  The trailing
            // `LocalBindingSnapshot` is the per-arm version of the
            // `then_exit_ctx` / `else_exit_ctx` captures in
            // [`lower_if_expr`]: when exactly one arm survives the
            // merge (siblings all closed via `return` / `raise` /
            // `break` / `panic!`), the post-merge `ctx` must
            // restore to that arm's bindings rather than the
            // pre-match snapshot, so post-merge reads of the
            // surviving arm's rebinds resolve to the correct SSA
            // values.  This is the if-let path's regression carrier
            // — `if let pat = scrut { x = 1; } else { return 0; } x`
            // desugars to a 2-arm match where only the open arm
            // contributes bindings, and the bound `x` must be
            // visible past the merge.
            let mut arm_tails: Vec<(
                BlockId,
                Option<crate::flowspace::model::Variable>,
                FrameState,
                LocalBindingSnapshot,
            )> = Vec::with_capacity(m.arms.len());
            for (arm_idx, arm) in m.arms.iter().enumerate() {
                let entry = arm_entries[arm_idx];
                let mut tail = entry;
                let saved_locals = LocalBindingSnapshot::capture(ctx);
                bind_pattern_locals(&arm.pat, scrutinee_type_string.as_deref(), ctx);
                let arm_lowered_result = lower_expr(graph, &mut tail, &arm.body, options, ctx);
                // Snapshot this arm's exit framestate BEFORE
                // `restore` wipes the per-arm rebinds.  The merge
                // block's lazy installer uses
                // `Block.framestate` on each predecessor to thread
                // `Link.args` back through the arm tail (RPython
                // parity: `flowspace/flowcontext.py:407-408
                // record_block(block)` calls `setstate(block.framestate)`
                // and `flowspace/flowcontext.py:449
                // currentstate.getoutputargs(newstate)` reads the
                // predecessor's snapshot to produce link args).
                // Without this, the predecessor walk in
                // `lazy_install_local_at_current_block` finds no
                // snapshot and falls back to the legacy naked-`Input`
                // emit, which loses the per-arm rebind information.
                let arm_exit_snapshot = ctx.getstate(graph, 0);
                let arm_exit_locals = LocalBindingSnapshot::capture(ctx);
                saved_locals.restore(ctx);
                let arm_lowered = arm_lowered_result?;
                // A closed arm (body is `return x` / `break` / `panic!`
                // / `raise`) does not contribute a value to the merge —
                // its path terminates inside `tail` and no outgoing
                // goto is synthesised.  Per RPython
                // `flowspace/flowcontext.py:1253` `Raise.nomoreblocks`,
                // sibling walks continue irrespective of this arm's
                // closure.
                let arm_lowered_var = arm_lowered.value_var(graph);
                arm_tails.push((tail, arm_lowered_var, arm_exit_snapshot, arm_exit_locals));
            }

            // Merge gets a Phi inputarg iff every arm that actually
            // reaches the merge carries a value.  Closed arms (early
            // `return` / `break`) don't emit a goto to merge, so they
            // contribute nothing to the phi arity.  Mixing some-value
            // and no-value open arms would require a fake phi arg for
            // the no-value arms (RPython `jit/codewriter/flatten.py:308`
            // — every outgoing goto's arg list must match the target's
            // inputarg arity), so in that case we emit no phi at all.
            let all_open_arms_have_value = arm_tails
                .iter()
                .all(|(tail, r, _, _)| !graph.block(*tail).is_open() || r.is_some());

            let open_arm_snapshots: Vec<(
                BlockId,
                Option<crate::flowspace::model::Variable>,
                FrameState,
                LocalBindingSnapshot,
            )> = arm_tails
                .iter()
                .filter(|(tail, _, _, _)| graph.block(*tail).is_open())
                .map(|(tail, result, exit_snapshot, exit_locals)| {
                    (
                        *tail,
                        result.clone(),
                        exit_snapshot.clone(),
                        exit_locals.clone(),
                    )
                })
                .collect();
            let any_open = !open_arm_snapshots.is_empty();

            // Iterative left-fold over 2-way `FrameState::union` —
            // direct port of `flowspace/flowcontext.py:430-436
            // mergeblock`'s repeated 2-way union over arriving
            // candidates.  Pyre's static AST shape knows every open
            // arm at lowering time, so the fold runs them in order
            // (first arm = initial running state; each subsequent
            // arm = `acc.union(arm)`).  Computed up front so both
            // the migration path (Slice 4.2 N-way generalisation)
            // and the legacy iterative-fold None-kill + lazy
            // phi-install path below share the same `merged` view.
            let merged_when_multi_open: Option<FrameState> = if open_arm_snapshots.len() >= 2 {
                let mut acc = open_arm_snapshots[0].2.clone();
                for (_, _, arm, _) in &open_arm_snapshots[1..] {
                    acc = acc.union(arm, graph).expect(
                        "AST frontend: union is total — entries domain has no UnionError, \
                         stack / last_exception / blocklist / next_offset are vestigial \
                         (framestate.py:78 None-return reachable only post-Z4 walker)",
                    );
                }
                Some(acc)
            } else {
                None
            };

            // Slice 4.2 N-way migration eligibility: switch to
            // `create_block_from_framestate` + per-arm
            // `set_goto_from_framestate` + fresh-phi
            // `OpKind::Input` + `ctx.setstate_at_block` when 2+ open
            // arms reach a merge that carries no value phi (the
            // `()`-shaped match) AND each arm's
            // `getoutputargs(merged)` can backfill through its tail
            // without entering an orphan-rooted chain or growing a
            // `ctx.loop_stack` continue_target's `inputargs`.  Same
            // gate as `lower_if_expr` (see ast.rs:~4426 for the
            // 2-arm specialisation).  When dry-run fails, the legacy
            // lean-merge-block + iterative-fold lazy-installer path
            // below copes silently — its merge carries no
            // framestate-derived inputargs and the installer only
            // touches blocks actually reachable.
            let forbidden_growth: std::collections::HashSet<BlockId> = ctx
                .loop_stack
                .iter()
                .map(|frame| frame.continue_target)
                .collect();
            let migrate: bool = if all_open_arms_have_value {
                false
            } else if let Some(merged) = merged_when_multi_open.as_ref() {
                open_arm_snapshots.iter().all(|(tail, _, state, _)| {
                    let outargs = state.getoutputargs(merged, graph);
                    outargs.iter().all(|a| match a {
                        LinkArg::Value(v) => {
                            graph.can_thread_variable_to_block(*tail, v, &forbidden_growth)
                        }
                        _ => true,
                    })
                })
            } else {
                false
            };

            let (merge, merge_phi) = if migrate {
                let merged = merged_when_multi_open
                    .as_ref()
                    .expect("migrate => merged_when_multi_open is Some");
                let m = graph.create_block_from_framestate(merged);
                (m, None)
            } else if all_open_arms_have_value {
                let (m_block, phi_args) = graph.create_block_with_arg_vars(1);
                (m_block, Some(phi_args[0].clone()))
            } else {
                (graph.create_block(), None)
            };

            // Per-arm close.  Migration threads
            // `getoutputargs(merged)` via `set_goto_from_framestate`
            // so each predecessor link carries the full mergeable
            // (locals + flattened stack + exception args); legacy
            // sends either the value-phi arg (when every open arm
            // contributes) or empty args (no-value merge).  The
            // arm-tail framestate stamp mirrors `Expr::If`'s
            // then/else stamp at the equivalent close site —
            // `flowspace/flowcontext.py:407-408 record_block`'s
            // per-block framestate carries through to the merge
            // block's lazy installer when the migration path is not
            // taken.
            for (tail, result, exit_snapshot, _) in &open_arm_snapshots {
                if migrate {
                    let merged = merged_when_multi_open
                        .as_ref()
                        .expect("migrate => merged_when_multi_open is Some");
                    graph.set_goto_from_framestate(*tail, merge, exit_snapshot, merged);
                } else {
                    let goto_args: Vec<crate::flowspace::model::Variable> =
                        if all_open_arms_have_value {
                            vec![result.clone().expect(
                                "filter above guarantees every open arm's `result` is Some",
                            )]
                        } else {
                            Vec::new()
                        };
                    graph.set_goto(*tail, merge, goto_args);
                }
                graph.block_mut(*tail).framestate = Some(exit_snapshot.clone());
            }

            // Iterative-fold-driven merge when 2+ arms reach the
            // merge block.  Per-slot semantics mirror Expr::If's
            // both-open-arm merge, generalised to N arms — direct
            // port of `flowspace/flowcontext.py: 430-436 mergeblock`'s
            // repeated 2-way union over arriving candidates.
            //
            //   - Carry-through Unknown→concrete retag keeps
            //     `graph_value_type(vid)` in agreement with the
            //     merged framestate when an inferred-only arm is
            //     unioned with an annotated arm.
            //   - None-kill drops slots that any arm left unbound
            //     (`framestate.py:110-111`); post-merge reads of
            //     those names surface as undefined-local.
            //   - Fresh-phi install (per-slot vid disagreement)
            //     allocates the merge-block inputarg, threads
            //     predecessor link args via
            //     `lazy_install_local_at_current_block`, and
            //     rebinds ctx so post-merge reads resolve to the
            //     new phi.
            //
            // Single-open-arm case is intentionally NOT handled
            // here: that's the audit's pre-existing one-open-arm
            // fragility shared with Expr::If, requiring a separate
            // ctx-restore strategy that lives outside this slice.
            if migrate {
                // Slice 4.2 N-way migration ctx update: emit
                // `OpKind::Input { name, ty }` ops in `merge` for
                // every fresh-phi slot Variable (minted by
                // `FrameState::union` as `Variable::new()`).  Without
                // the paired Input op, `graph_value_type_var` returns
                // None for the fresh phi and `setstate_at_block`
                // would surface `ValueType::Unknown` into
                // `ctx.local_value_types`, tripping downstream
                // `Expr::Unary` `!` on the rebound local.  Type fold
                // mirrors `lazy_install_local_at_current_block_var`'s
                // wildcard rule (`ast.rs:3367-3374`) generalised to
                // N arms by left-folding the per-slot per-arm types.
                let merged = merged_when_multi_open
                    .as_ref()
                    .expect("migrate => merged_when_multi_open is Some");
                let phi_info: Vec<(usize, crate::flowspace::model::Variable, ValueType)> = {
                    let merged_view = merged.locals_w_view(graph);
                    let arm_views: Vec<Vec<Option<crate::flowspace::model::Variable>>> =
                        open_arm_snapshots
                            .iter()
                            .map(|(_, _, state, _)| {
                                state
                                    .locals_w_view(graph)
                                    .iter()
                                    .map(|cell| match cell {
                                        Some(crate::flowspace::model::Hlvalue::Variable(v)) => {
                                            Some(v.clone())
                                        }
                                        _ => None,
                                    })
                                    .collect()
                            })
                            .collect();
                    let mut info = Vec::new();
                    for (i, slot) in merged_view.iter().enumerate() {
                        let Some(crate::flowspace::model::Hlvalue::Variable(merged_var)) = slot
                        else {
                            continue;
                        };
                        let first_arm_var =
                            arm_views.first().and_then(|v| v.get(i).cloned()).flatten();
                        if first_arm_var.as_ref() == Some(merged_var) {
                            continue;
                        }
                        let mut folded_ty: Option<ValueType> = None;
                        for arm_view in &arm_views {
                            let arm_var = arm_view.get(i).cloned().flatten();
                            let arm_ty = arm_var
                                .as_ref()
                                .map(|v| {
                                    graph_value_type_var(graph, v).unwrap_or(ValueType::Unknown)
                                })
                                .unwrap_or(ValueType::Unknown);
                            folded_ty = Some(match folded_ty {
                                None => arm_ty,
                                Some(prior) => match (prior, arm_ty) {
                                    (a, b) if a == b => a,
                                    (ValueType::Unknown, b) => b,
                                    (a, ValueType::Unknown) => a,
                                    _ => ValueType::Unknown,
                                },
                            });
                        }
                        info.push((
                            i,
                            merged_var.clone(),
                            folded_ty.unwrap_or(ValueType::Unknown),
                        ));
                    }
                    info
                };
                for (slot_idx, phi_var, ty) in phi_info {
                    let name = ctx.local_first_bind_order[slot_idx].clone();
                    graph.push_op_with_result_var(
                        merge,
                        OpKind::Input {
                            name: name.clone(),
                            ty: ty.clone(),
                            class_root: match &ty {
                                ValueType::Ref(Some(root)) => Some(root.clone()),
                                _ => None,
                            },
                        },
                        phi_var.clone(),
                    );
                    graph.name_value_var(&phi_var, name);
                }
                ctx.setstate_at_block(merged, merge, graph);
            } else if open_arm_snapshots.len() >= 2 {
                // `FrameState::union` returns `Option<FrameState>`
                // per `framestate.py:78-89`'s `try/except UnionError:
                // return None` envelope.  The
                // `framestate.py:117/126 UnionError` paths (SpecTag
                // mismatch, FlowSignal-type mismatch) can only fire
                // on the stack / exception projections, which are
                // vestigially empty / `None` on the AST frontend
                // until the Z4 walker populates them; the
                // `.expect(...)` documents the AST-frontend total
                // invariant.  Per-slot type unification across match
                // arms is annotator-side per upstream
                // `framestate.py:union` (Hlvalue identity only).
                // Iterative left-fold over 2-way `FrameState::union`
                // — direct port of upstream's `flowspace/
                // flowcontext.py:430-436 mergeblock` loop:
                //
                //     for block in candidates:
                //         newstate = block.framestate.union(currentstate)
                //         if newstate is not None:
                //             break
                //
                // Pyre's static AST shape knows every open arm at
                // lowering time, so the fold runs them in order
                // (first arm = initial running state; each
                // subsequent arm = `acc.union(arm)`).  rustc has
                // already rejected source whose arms bind the same
                // local to two different concrete kinds, so the
                // `.expect(...)` documents the contract.
                //
                // TODO: no SpamBlock / recloseblock chain — fused into
                // direct construction.
                // Upstream's `mergeblock` (`flowcontext.py:425-463`)
                // generalises by creating a fresh `SpamBlock(newstate)`
                // (`:443`), marking the prior block dead via
                // `block.dead = True` + `block.operations = ()`
                // (`:455-456`), and patching the dead block's exits
                // to forward to the new block via
                // `block.recloseblock(Link(outputargs, newblock))`
                // (`:458-459`).  `simplify.eliminate_empty_blocks`
                // (`simplify.py:52-69`) then collapses the dead-block
                // forwarding chain into a single multi-incoming merge
                // block — which is exactly the shape pyre's static
                // AST produces directly.
                //
                // Pyre's tree-recursive lowering has no per-bytecode
                // dispatch loop, so the chain ceremony is fused into
                // direct construction; the resulting CFG is the
                // post-collapse upstream-orthodox shape.  The Block
                // `dead` field, `model::eliminate_empty_blocks` pass,
                // and the `simplify_graph` invocation in
                // `build_function_graph` are all in place; once Z4's
                // flowcontext-walker rewrite materialises intermediate
                // SpamBlocks per fold step, the chain becomes load-
                // bearing without further infrastructure changes.
                let merged = merged_when_multi_open
                    .expect("open_arm_snapshots.len() >= 2 => merged_when_multi_open is Some");
                let first_arm = &open_arm_snapshots[0].2;
                // Type unification across arms is annotator-side per
                // upstream `framestate.py:union` (Hlvalue identity
                // only).  The prior carry-through retag block was a
                // deviation dependent on the retired
                // `FrameStateEntry::value_type` field; it has been
                // removed (mirrors the If/else
                // counterpart).  Convergence: same as If/else —
                // annotator/rtyper port handles type unification at
                // its proper layer.
                // Locals projection walks `merged.locals_w` per
                // upstream `framestate.py:19 self.locals_w` — pyre's
                // `union` populates the `Hlvalue` carrier in lockstep
                // with `entries`, so the walk is bit-identical while
                // keeping the read side in agreement with the upstream
                // source of truth.  Materialise the view once and
                // reuse across the None-kill + phi-install passes.
                let merged_locals_w = merged.locals_w_view(graph);
                let first_arm_locals_w = first_arm.locals_w_view(graph);
                for (slot_idx, slot) in merged_locals_w.iter().enumerate() {
                    if matches!(slot, Some(crate::flowspace::model::Hlvalue::Variable(_))) {
                        continue;
                    }
                    // None-kill: resolve `slot_idx → name` via
                    // `local_first_bind_order` per the framestate
                    // positional-zip invariant (mirrors the If-merge
                    // counterpart).
                    if let Some(name) = ctx.local_first_bind_order.get(slot_idx).cloned() {
                        ctx.local_value_ids.remove(&name);
                        ctx.local_value_types.remove(&name);
                    }
                }
                // Materialise (slot_idx, merged_vid, first_arm_vid)
                // tuples up front so the immutable `graph` borrow inside
                // the locals_w walk releases before
                // `lazy_install_local_at_current_block`'s mutable call.
                let phi_candidates: Vec<(
                    usize,
                    crate::flowspace::model::Variable,
                    Option<crate::flowspace::model::Variable>,
                )> = merged_locals_w
                    .iter()
                    .enumerate()
                    .filter_map(|(i, slot)| match slot {
                        Some(crate::flowspace::model::Hlvalue::Variable(v)) => {
                            let first_var = first_arm_locals_w.get(i).and_then(|slot| match slot {
                                Some(crate::flowspace::model::Hlvalue::Variable(v)) => {
                                    Some(v.clone())
                                }
                                _ => None,
                            });
                            Some((i, v.clone(), first_var))
                        }
                        _ => None,
                    })
                    .collect();
                drop(merged_locals_w);
                drop(first_arm_locals_w);
                for (slot_idx, slot_var, first_var) in phi_candidates {
                    let is_fresh_phi = first_var.as_ref() != Some(&slot_var);
                    if is_fresh_phi {
                        let name = ctx.local_first_bind_order[slot_idx].clone();
                        let _ = lazy_install_local_at_current_block_var(
                            graph,
                            ctx,
                            merge,
                            &name,
                            Some(slot_var.clone()),
                        );
                    }
                }
            } else if open_arm_snapshots.len() == 1 {
                // Companion of `lower_if_expr`'s `then_exit_ctx` /
                // `else_exit_ctx` restore for the case where exactly
                // one arm reaches the merge.  Without this restore,
                // ctx still holds the pre-match snapshot (because
                // every arm called `saved_locals.restore(ctx)` after
                // its body walk), and post-merge reads of the
                // surviving arm's rebinds would resolve to the wrong
                // (stale) SSA values.  The if-let desugar at the top
                // of [`lower_if_expr`] funnels patterns like
                // `if let pat = scrut { x = 1; } else { return 0; } x`
                // through this match path, so the regression carrier
                // is `Expr::Match` — restore here keeps the post-`x`
                // read on the surviving open arm's binding.
                open_arm_snapshots[0].3.clone().restore(ctx);
            }

            *block = merge;
            if !any_open {
                // All arms terminated — the enclosing walk has no open
                // path to continue.
                Ok(Lowered::path_closed())
            } else {
                match merge_phi {
                    Some(var) => Ok(Lowered::from_value_var(graph, &var)),
                    None => Ok(Lowered {
                        value: None,
                        path_closed: false,
                    }),
                }
            }
        }

        // ── while → header block + body block + exit block ──
        syn::Expr::While(w) => {
            let exit = graph.create_block();
            let merge_offset = ctx.next_loop_merge_offset();

            // `flowcontext.py:465-473 make_next_block` — state.copy()
            // + create SpamBlock + closeblock entry link.
            let pre_loop_block = *block;
            let pre_loop_snapshot = ctx.getstate(graph, 0);
            let (mut header_state, phi_info) =
                build_loop_header_state(graph, ctx, &pre_loop_snapshot);
            header_state.next_offset = merge_offset;
            let header_entry = graph.create_block_from_framestate(&header_state);
            for (_slot_idx, phi_var, name, ty) in &phi_info {
                graph.push_op_with_result_var(
                    header_entry,
                    OpKind::Input {
                        name: name.clone(),
                        ty: ty.clone(),
                        class_root: match &ty {
                            ValueType::Ref(Some(root)) => Some(root.clone()),
                            _ => None,
                        },
                    },
                    phi_var.clone(),
                );
                graph.name_value_var(phi_var, name.clone());
            }
            // Backfill loop-carried slots the pre-loop block does not
            // define locally (e.g. an outer-loop var an inner loop merges)
            // so the entry link references only `pre_loop_block`-defined
            // slots.  See `FunctionGraph::thread_loop_link_args`.
            let forbidden = ctx.loop_forbidden_set(header_entry);
            graph.thread_loop_link_args(
                pre_loop_block,
                &pre_loop_snapshot,
                &header_state,
                &forbidden,
            );
            graph.closeblock_link(
                pre_loop_block,
                header_entry,
                &pre_loop_snapshot,
                &header_state,
            );
            graph.block_mut(pre_loop_block).framestate = Some(pre_loop_snapshot);
            // Register header in joinpoints so back-edge mergeblock
            // finds it as a candidate.
            ctx.joinpoints
                .entry(merge_offset)
                .or_default()
                .insert(0, header_entry);
            ctx.setstate_at_block(&header_state, header_entry, graph);

            // Push LoopFrame BEFORE cond eval so the header is in
            // `forbidden_growth` during condition lowering — prevents
            // `ensure_variable_at_block` from growing header inputargs.
            ctx.loop_stack.push(LoopFrame {
                continue_target: header_entry,
                merge_offset,
                header_slot_count: header_state.entries.len(),
                header_state: header_state.clone(),
                break_target: exit,
                rewalk_budget_remaining: header_state.entries.len() + 1,
            });

            let post_eager_phi_locals = LocalBindingSnapshot::capture(ctx);

            // Header: evaluate condition, branch to body or exit.
            let mut header_tail = header_entry;
            let cond_pre_var = get_value_var!(
                lower_expr(graph, &mut header_tail, &w.cond, options, ctx)?,
                graph
            );
            ctx.pushvid_var(&cond_pre_var);
            let cond_var = ctx.popvid_var(graph);
            let body_entry = graph.create_block();
            let header_branch_snapshot = ctx.getstate(graph, 0);
            graph.set_branch(header_tail, cond_var, body_entry, vec![], exit, vec![]);
            graph.block_mut(header_tail).framestate = Some(header_branch_snapshot);
            graph.block_mut(header_entry).framestate = Some(header_state.clone());

            let mut body_tail = body_entry;
            for stmt in &w.body.stmts {
                let closed = lower_stmt(graph, &mut body_tail, stmt, options, ctx)?;
                if closed {
                    break;
                }
            }
            let loop_frame = ctx.loop_stack.pop().unwrap();
            // `flowcontext.py:399-422` — close the back-edge, then drain
            // any generalized header it (or a body `continue`) enqueued.
            // On borrow-checked input the back-edge always direct-links,
            // so the queue is empty and the drain is a no-op.
            ctx.close_loop_back_edge_inner(graph, body_tail, &loop_frame);
            ctx.drain_pendingblocks_for_merge_offset(
                graph,
                loop_frame.merge_offset,
                &w.body,
                LoopPrologue::While {
                    cond: &w.cond,
                    exit,
                },
                &loop_frame,
                options,
            )?;

            *block = exit;
            post_eager_phi_locals.restore(ctx);
            Ok(Lowered::no_value())
        }
        syn::Expr::Loop(l) => {
            let exit = graph.create_block();
            let merge_offset = ctx.next_loop_merge_offset();

            let pre_loop_block = *block;
            let pre_loop_snapshot = ctx.getstate(graph, 0);
            let (mut header_state, phi_info) =
                build_loop_header_state(graph, ctx, &pre_loop_snapshot);
            header_state.next_offset = merge_offset;
            let body_entry = graph.create_block_from_framestate(&header_state);
            for (_slot_idx, phi_var, name, ty) in &phi_info {
                graph.push_op_with_result_var(
                    body_entry,
                    OpKind::Input {
                        name: name.clone(),
                        ty: ty.clone(),
                        class_root: match &ty {
                            ValueType::Ref(Some(root)) => Some(root.clone()),
                            _ => None,
                        },
                    },
                    phi_var.clone(),
                );
                graph.name_value_var(phi_var, name.clone());
            }
            // Backfill loop-carried slots not defined locally at the
            // pre-loop block.  See `FunctionGraph::thread_loop_link_args`.
            let forbidden = ctx.loop_forbidden_set(body_entry);
            graph.thread_loop_link_args(
                pre_loop_block,
                &pre_loop_snapshot,
                &header_state,
                &forbidden,
            );
            graph.closeblock_link(
                pre_loop_block,
                body_entry,
                &pre_loop_snapshot,
                &header_state,
            );
            graph.block_mut(pre_loop_block).framestate = Some(pre_loop_snapshot);
            ctx.joinpoints
                .entry(merge_offset)
                .or_default()
                .insert(0, body_entry);
            ctx.setstate_at_block(&header_state, body_entry, graph);

            let post_eager_phi_locals = LocalBindingSnapshot::capture(ctx);

            ctx.loop_stack.push(LoopFrame {
                continue_target: body_entry,
                merge_offset,
                header_slot_count: header_state.entries.len(),
                header_state: header_state.clone(),
                break_target: exit,
                rewalk_budget_remaining: header_state.entries.len() + 1,
            });
            let mut body_tail = body_entry;
            for stmt in &l.body.stmts {
                let closed = lower_stmt(graph, &mut body_tail, stmt, options, ctx)?;
                if closed {
                    break;
                }
            }
            let loop_frame = ctx.loop_stack.pop().unwrap();
            // `flowcontext.py:399-422` — close the back-edge, then drain.
            // Empty queue → no-op on borrow-checked input.
            ctx.close_loop_back_edge_inner(graph, body_tail, &loop_frame);
            ctx.drain_pendingblocks_for_merge_offset(
                graph,
                loop_frame.merge_offset,
                &l.body,
                LoopPrologue::Loop,
                &loop_frame,
                options,
            )?;

            *block = exit;
            post_eager_phi_locals.restore(ctx);
            Ok(Lowered::no_value())
        }
        syn::Expr::ForLoop(f) => {
            let iterable_pre_var =
                get_value_var!(lower_expr(graph, block, &f.expr, options, ctx)?, graph);
            ctx.pushvid_var(&iterable_pre_var);
            let iterable = ctx.popvid_var(graph);
            let _ = iterable;

            let body_entry = graph.create_block();
            let exit = graph.create_block();
            let merge_offset = ctx.next_loop_merge_offset();

            let pre_loop_block = *block;
            let pre_loop_snapshot = ctx.getstate(graph, 0);
            let (mut header_state, phi_info) =
                build_loop_header_state(graph, ctx, &pre_loop_snapshot);
            header_state.next_offset = merge_offset;
            let header_entry = graph.create_block_from_framestate(&header_state);
            for (_slot_idx, phi_var, name, ty) in &phi_info {
                graph.push_op_with_result_var(
                    header_entry,
                    OpKind::Input {
                        name: name.clone(),
                        ty: ty.clone(),
                        class_root: match &ty {
                            ValueType::Ref(Some(root)) => Some(root.clone()),
                            _ => None,
                        },
                    },
                    phi_var.clone(),
                );
                graph.name_value_var(phi_var, name.clone());
            }
            // Backfill loop-carried slots not defined locally at the
            // pre-loop block.  See `FunctionGraph::thread_loop_link_args`.
            let forbidden = ctx.loop_forbidden_set(header_entry);
            graph.thread_loop_link_args(
                pre_loop_block,
                &pre_loop_snapshot,
                &header_state,
                &forbidden,
            );
            graph.closeblock_link(
                pre_loop_block,
                header_entry,
                &pre_loop_snapshot,
                &header_state,
            );
            graph.block_mut(pre_loop_block).framestate = Some(pre_loop_snapshot);
            ctx.joinpoints
                .entry(merge_offset)
                .or_default()
                .insert(0, header_entry);
            ctx.setstate_at_block(&header_state, header_entry, graph);

            ctx.loop_stack.push(LoopFrame {
                continue_target: header_entry,
                merge_offset,
                header_slot_count: header_state.entries.len(),
                header_state: header_state.clone(),
                break_target: exit,
                rewalk_budget_remaining: header_state.entries.len() + 1,
            });

            let post_eager_phi_locals = LocalBindingSnapshot::capture(ctx);

            let for_cond_var = graph.push_op_var(
                header_entry,
                OpKind::Abort {
                    kind: UnknownKind::UnsupportedExpr {
                        variant: UnsupportedExprKind::ForLoop,
                    },
                },
                true,
            );
            if let Some(cond_var) = for_cond_var {
                graph.set_branch(header_entry, cond_var, body_entry, vec![], exit, vec![]);
            } else {
                graph.set_goto(header_entry, body_entry, vec![]);
            }
            graph.block_mut(header_entry).framestate = Some(header_state.clone());

            let mut body_tail = body_entry;
            for stmt in &f.body.stmts {
                let closed = lower_stmt(graph, &mut body_tail, stmt, options, ctx)?;
                if closed {
                    break;
                }
            }
            let loop_frame = ctx.loop_stack.pop().unwrap();
            // `flowcontext.py:399-422` — close the back-edge, then drain.
            // Empty queue → no-op on borrow-checked input.
            ctx.close_loop_back_edge_inner(graph, body_tail, &loop_frame);
            ctx.drain_pendingblocks_for_merge_offset(
                graph,
                loop_frame.merge_offset,
                &f.body,
                LoopPrologue::For { exit },
                &loop_frame,
                options,
            )?;

            *block = exit;
            post_eager_phi_locals.restore(ctx);
            Ok(Lowered::no_value())
        }

        // ── break/continue ──
        //
        // RPython `flowspace/flowcontext.py:525` models these as
        // `Break` / `Continue` `FlowSignal`s; `LoopBlock.handle_signal`
        // (`:1341`) rewrites the current block with a Link to the
        // loop's end / header.  Pyre's port: look up the enclosing
        // `LoopFrame` on `ctx.loop_stack` and close the current block
        // with `set_goto(*block, target)`, then report path_closed so
        // the surrounding walker stops emitting ops into a
        // now-terminated block.  A break/continue outside any loop is
        // orphaned — `path_closed` alone gives the surrounding walker
        // the stop signal without corrupting the graph.
        syn::Expr::Break(b) => {
            if let Some(e) = &b.expr {
                let lowered = lower_expr(graph, block, e, options, ctx)?;
                if lowered.path_closed {
                    return Ok(Lowered::path_closed());
                }
            }
            if let Some(frame) = ctx.loop_stack.last().cloned() {
                if graph.block(*block).is_open() {
                    // `flowcontext.py:438 closeblock(Link(outputargs, target))`
                    // Close the break predecessor via `closeblock_link`.
                    // Read the break target's framestate when present;
                    // fall back to empty FrameState otherwise.
                    let pre_break_snapshot = ctx.getstate(graph, 0);
                    let target_state = graph
                        .block(frame.break_target)
                        .framestate
                        .clone()
                        .unwrap_or_default();
                    graph.closeblock_link(
                        *block,
                        frame.break_target,
                        &pre_break_snapshot,
                        &target_state,
                    );
                    // Stamp the break source's framestate so the
                    // post-loop lazy installer can read the locals
                    // visible on this predecessor edge — same role as
                    // the then/else arm-tail and loop body-tail
                    // stamps.  RPython parity:
                    // `flowspace/flowcontext.py:399-465 mergeblock`
                    // requires every closing predecessor to carry a
                    // FrameState so `getoutputargs` can resolve the
                    // target's inputargs slot-by-slot.
                    graph.block_mut(*block).framestate = Some(pre_break_snapshot);
                }
            }
            Ok(Lowered::path_closed())
        }
        syn::Expr::Continue(_) => {
            // A `continue` edge is a loop back-edge: link the current
            // block to the still-active loop header (peek, not pop) via
            // the shared back-edge close.
            if let Some(frame) = ctx.loop_stack.last().cloned() {
                // Close the continue back-edge WITHOUT draining — a
                // generalize here enqueues on `pendingblocks`, and the
                // ENCLOSING loop arm drains it (it holds the body AST; a
                // full body re-walk re-runs this continue).
                ctx.close_loop_back_edge_inner(graph, *block, &frame);
            }
            Ok(Lowered::path_closed())
        }

        // ── closure ──
        //
        // TODO(closure-body-compilation): the closure body is NOT
        // walked here.  `MAKE_FUNCTION`
        // (`pypy/interpreter/pyopcode.py:1144`,
        // `flowspace/flowcontext.py:1177`) materialises a *separate*
        // graph for the `def`/`lambda` body and pushes a fresh
        // function value onto the stack — the body never inlines into
        // the enclosing flow.  Pyre currently lowers the whole
        // expression to a single `Unknown` placeholder for the
        // closure *value* and leaves the body uncompiled.  An earlier
        // attempt to walk the body in-place was a deviation (it
        // treated the closure as a synchronous block, which broke
        // callers that pass the closure itself as a function-typed
        // argument — e.g. `|_| {}` produced no value for
        // `get_value!`).
        //
        // The full port needs three pieces:
        //   1. Synthesise a fresh `FunctionGraph` per closure body
        //      (parameters from `closure.inputs`, captures plumbed
        //      through inputargs as upvars).
        //   2. Register it with the surrounding `PyreCallRegistry`
        //      under a synthetic
        //      `CallPath::Closure(<host-fn>::<call-site-index>)` so
        //      `simple_call` resolution and `target_to_path` can find
        //      it.
        //   3. Replace this `Unknown` emit with an `OpKind` that
        //      pushes the synthetic graph's host identity onto the
        //      value stack — mirroring `MAKE_FUNCTION`
        //      (`flowspace/flowcontext.py:1177`).
        //
        // Downstream call sites
        // (`expression_type_string` closure-passthrough at line 7953)
        // already project the closure body's return type for the
        // common method-call patterns (`map` / `unwrap_or_else` /
        // `and_then` / `filter`), so the *call-site* analysis is
        // type-coherent even without the synthetic-graph port; the
        // *body* itself stays uncompiled.  Multi-session epic
        // (captures, multi-shot calls, indirect-call dispatch).
        syn::Expr::Closure(_) => Ok(continue_with_unknown(
            graph,
            *block,
            UnsupportedExprKind::Closure,
        )),

        // ── tuple (a, b, c) ──
        syn::Expr::Tuple(t) => {
            // RPython `BUILD_TUPLE` (`pypy/interpreter/pyopcode.py:955`,
            // `flowspace/flowcontext.py:1163`) always pushes a fresh
            // tuple object — the result is a NEW value distinct from
            // any individual element.  Pyre has no `NewTuple` op yet
            // (deferred), so the construct lowers to a single
            // `Unknown` marker tagged `Tuple` that stands in for the
            // whole tuple-builder; callers that read the result get a
            // well-formed Variable but coverage audits still flag the
            // port gap.  Elements lower for their side effects and
            // path-closed propagation but do NOT feed the result.
            for e in &t.elems {
                let lowered = lower_expr(graph, block, e, options, ctx)?;
                if lowered.path_closed {
                    return Ok(Lowered::path_closed());
                }
            }
            Ok(continue_with_unknown(
                graph,
                *block,
                UnsupportedExprKind::Tuple,
            ))
        }

        // ── try expr? ──
        //
        // RPython `flowspace/flowcontext.py:127-148 guessexception` port.
        // A can-raise op closes its containing block with
        // `block.exitswitch = c_last_exception` and two Links: the
        // normal fall-through Link and the exception Link whose
        // `args`/`extravars` both reference fresh prevblock-side
        // `Variable('last_exception')` / `Variable('last_exc_value')`
        // (`flowcontext.py:130-134`).  These fresh variables flow into
        // the exceptblock's own inputargs via `insert_renamings` — the
        // target side has its own distinct Variables
        // (`flowcontext.py:135 vars2`), matching upstream's "Link.args
        // are prevblock-side values" invariant at
        // `flowspace/model.py:114`.
        syn::Expr::Try(t) => {
            let ok_ty = expression_type_string(&t.expr, ctx)
                .as_deref()
                .and_then(transparent_result_ok_type)
                .map(type_string_to_value_type);
            // RPython `flowspace/flowcontext.py:379-393 do_op`:
            // `guessexception` (which installs
            // `block.exitswitch = c_last_exception`) runs only AFTER an
            // operation is recorded, and only when `op.canraise` is
            // non-empty.  A bare-local read (`kwargs?`) records no
            // operation, so the block must NOT be closed as canraise.
            // Snapshot the operations cursor before lowering the operand,
            // then classify the tail op the operand actually appended by
            // its `op.canraise` (`flowspace_adapter::op_canraise`): a
            // raising tail op (a call — including a non-unit transparent
            // ctor, which lowers to a non-builtin `simple_call` —, getitem /
            // setitem, or a div / mod / shift binop) is what `?` closes the
            // block against; a non-raising tail op (an elided unit-variant
            // ctor, getattr / setattr, a pure binop / cast / const, or a
            // cross-block `OpKind::Input`) records nothing that can raise,
            // so the block must NOT be closed as canraise.
            let block_before = *block;
            let len_before = graph.block(*block).operations.len();
            let inner_pre_var =
                get_value_var!(lower_expr(graph, block, &t.expr, options, ctx)?, graph);
            let recorded_raising_op = {
                let ops = &graph.block(*block).operations;
                let last_can_raise = ops
                    .last()
                    .map(|op| crate::translator::rtyper::flowspace_adapter::op_canraise(&op.kind))
                    .unwrap_or(false);
                if *block == block_before {
                    // Operand stayed in this block: it raises iff it
                    // appended a raising op as the new tail.
                    ops.len() > len_before && last_can_raise
                } else {
                    // Operand moved the cursor (nested `?`, if-expr
                    // operand): its final op landed in the new `*block`;
                    // that op is what `?` raises against.
                    last_can_raise
                }
            };
            ctx.pushvid_var(&inner_pre_var);
            let inner_var = ctx.popvid_var(graph);
            if let Some(ok_ty) = ok_ty {
                retag_result_value_type(graph, &inner_var, ok_ty);
            }
            if recorded_raising_op {
                // ── `?` on a raising operand (a call) ──
                // The operand's recorded op is the block's last (raising)
                // op; close the block with `exitswitch = c_last_exception`
                // and a normal + exception link (`flowcontext.py:147`).
                let continuation = graph.create_block();
                let continuation_arg = graph.alloc_value_var();
                graph.push_inputarg_var(continuation, continuation_arg.clone());
                // RPython `flowcontext.py:130-133` — fresh prevblock-side
                // `Variable('last_exception')` + `Variable('last_exc_value')`.
                let last_exception_var = graph.alloc_value_var();
                let last_exc_value_var = graph.alloc_value_var();
                let exc_block = graph.exceptblock;
                graph.set_goto(*block, continuation, vec![inner_var.clone()]);
                let normal_link = Link::from_variables(graph, vec![inner_var], continuation, None);
                let exc_link = Link::from_variables(
                    graph,
                    vec![last_exception_var.clone(), last_exc_value_var.clone()],
                    exc_block,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::Value(last_exception_var)),
                    Some(LinkArg::Value(last_exc_value_var)),
                );
                graph.set_control_flow_metadata(
                    *block,
                    Some(ExitSwitch::LastException),
                    vec![normal_link, exc_link],
                );
                // Stamp the canraise `?`-source block's framestate before
                // switching to the continuation, mirroring the branch
                // source stamp (`lower_if_expr` `pre_branch_snapshot`).
                // The canraise continuation receives only `inner_var` on
                // its normal link, so a cross-block read of any other local
                // in the continuation (or a successor) walks back to this
                // `?`-source via `lazy_install_local_at_current_block_var`,
                // which bails at `pred_block.framestate.as_ref()?` when the
                // source carries no framestate — falling through to a naked
                // body-`Input` op the adapter rejects as "cross-block body
                // Input" (the cat14 skip).  Branch and loop sources already
                // stamp here; the canraise close was the lone gap.  Guarded
                // on `framestate.is_none()` so an enclosing construct that
                // already stamped this block is not overwritten.
                if graph.block(*block).framestate.is_none() {
                    let pre_try_snapshot = ctx.getstate(graph, 0);
                    graph.block_mut(*block).framestate = Some(pre_try_snapshot);
                }
                *block = continuation;
                Ok(Lowered::from_value_var(graph, &continuation_arg))
            } else {
                // ── `?` on a bare value (no recorded raising op) ──
                // RPython models a value-test early-return NOT as canraise
                // but as a `guessbool` branch on the value
                // (`flowcontext.py:107-122` `block.exitswitch =
                // w_condition`).  pyre lacks a typed enum-discriminant op,
                // so — mirroring the composite-pattern `match` arm
                // (`front/ast.rs` `match { Some(_) => .., None => .. }`,
                // "two-arm truthy split" on the scrutinee) — the
                // transparent Option/Result value itself is the branch
                // condition: truthy (`Some`/`Ok`) continues with the
                // unwrapped value; falsy (`None`/`Err`) early-returns the
                // same (transparent) value through the returnblock.  Unwrap
                // is identity at this layer (`transparent_result_ok_type` /
                // `transparent_option_inner_type` strip the wrapper string
                // only); a falsy Option is the null `None` ref and a falsy
                // Result is the `Err`-carrying value — both are valid
                // returns for the enclosing fn's transparent return repr.
                //
                // Locals threading mirrors the UNARY_NOT bool-fork: every
                // pre-fork local rides the success `Link.args` ↔
                // continuation `inputargs` so post-`?` reads resolve to the
                // threaded values (`flowcontext.py:835 LOAD_FAST` /
                // `getoutputargs` thread every `frame.locals_w` slot at
                // every edge).
                let pre_fork_locals = ctx.local_value_ids.clone();
                let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
                merged_names.sort();
                let pre_fork_local_vars: Vec<crate::flowspace::model::Variable> = merged_names
                    .iter()
                    .map(|name| {
                        let var = pre_fork_locals[name].0.clone();
                        if graph.variable_defined_in_block(*block, &var) {
                            var
                        } else {
                            lazy_install_local_at_current_block_var(graph, ctx, *block, name, None)
                                .unwrap_or(var)
                        }
                    })
                    .collect();

                let (continuation, continuation_args) =
                    graph.create_block_with_arg_vars(merged_names.len() + 1);
                let continuation_value_arg = continuation_args[0].clone();
                let continuation_local_args: Vec<crate::flowspace::model::Variable> =
                    continuation_args[1..].to_vec();

                // Failure arm receives the (transparent) failing value and
                // returns it unchanged through the returnblock.
                let (failure_arm, failure_args) = graph.create_block_with_arg_vars(1);
                let failure_value_arg = failure_args[0].clone();

                let mut success_link_args: Vec<crate::flowspace::model::Variable> =
                    Vec::with_capacity(merged_names.len() + 1);
                success_link_args.push(inner_var.clone());
                success_link_args.extend(pre_fork_local_vars.iter().cloned());

                // `set_branch` builds `ExitCase::Bool(true)` for the
                // `if_true` arm: truthy (`Some`/`Ok`) → continuation,
                // falsy (`None`/`Err`) → failure arm.
                graph.set_branch(
                    *block,
                    inner_var.clone(),
                    continuation,
                    success_link_args,
                    failure_arm,
                    vec![inner_var],
                );
                graph.set_return(failure_arm, Some(failure_value_arg));

                for (name, arg_var) in merged_names.iter().zip(continuation_local_args.iter()) {
                    ctx.bind_local_id_var(name.clone(), arg_var, graph, continuation);
                }

                *block = continuation;
                Ok(Lowered::from_value_var(graph, &continuation_value_arg))
            }
        }

        // ── unsafe { stmts } ──
        //
        // RPython flow-space has no concept of `unsafe` — in Python every
        // load/store already has the same aliasing model.  In the Rust
        // port `unsafe { stmts }` wraps raw-pointer / transmute helpers
        // whose **body** is still a regular Rust block; the `unsafe`
        // keyword is a type-system marker, not runtime semantics.  Lower
        // it by reusing the same `Block` path so the contained
        // statements + tail expression flow through normally.
        syn::Expr::Unsafe(u) => {
            lower_stmt_list_with_tail_value(graph, block, &u.block.stmts, options, ctx)
        }

        // ── fallback ──
        //
        // RPython `flowspace/flowcontext.py` evaluates sub-expressions
        // eagerly as bytecode streams in; `FlowingError` halts the
        // walk AT the unsupported op, not BEFORE the sub-expression
        // push operations.  For Rust variants whose AST carries named
        // sub-expressions (Range endpoints, Struct field values, Array
        // / Repeat elements, `if let` scrutinee) we walk those first
        // so their Call / FieldRead / etc. ops land in the graph
        // before the Unknown marker + abort.
        other => {
            // Conditional-raise macro family (assert!, debug_assert!,
            // assert_eq!, assert_ne!, debug_assert_eq!,
            // debug_assert_ne!) expand to `if !cond { panic }` — a
            // runtime check that either continues or unconditionally
            // raises.  Port to the RPython-canonical shape of a
            // `set_branch` whose false side routes through the
            // exceptblock via `set_raise`
            // (`rpython/flowspace/model.py:21-25`).  Unlike panic!, the
            // macro expression itself has type `()` — on the pass side
            // the enclosing walk continues normally.
            if let syn::Expr::Macro(m) = other {
                let macro_name = m
                    .mac
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                let is_assert = matches!(macro_name.as_str(), "assert" | "debug_assert");
                let is_assert_cmp = matches!(
                    macro_name.as_str(),
                    "assert_eq" | "assert_ne" | "debug_assert_eq" | "debug_assert_ne"
                );

                // ── matches! desugaring ──
                // `matches!(scrutinee, pat)` and `matches!(scrutinee,
                // pat if guard)` desugar (per std::matches docs) to
                // `match scrutinee { pat => true, _ => false }` (with
                // guard inlined onto the arm if present). We build the
                // synthetic `Expr::Match` AST and recurse so the
                // existing match lowering handles the dispatch — same
                // shape as the `if let` desugar above.
                //
                // Without this desugar, `matches!` flows through the
                // catch-all `Expr::Macro` arm below and emits
                // `OpKind::Abort { Macro }`. Phase G G.4.4 Path A.B.
                if macro_name == "matches" {
                    let tokens = m.mac.tokens.clone();
                    if let Some((scrutinee_tokens, rest_tokens)) =
                        split_macro_args_at_first_top_comma(tokens)
                    {
                        if let (Ok(scrutinee_expr), Ok((pat, guard))) = (
                            syn::parse2::<syn::Expr>(scrutinee_tokens),
                            syn::parse::Parser::parse2(parse_matches_pat_and_guard, rest_tokens),
                        ) {
                            let arm_then_body: syn::Expr = syn::parse_quote!(true);
                            let arm_else_body: syn::Expr = syn::parse_quote!(false);
                            let then_arm = syn::Arm {
                                attrs: vec![],
                                pat,
                                guard: guard.map(|g| (Default::default(), Box::new(g))),
                                fat_arrow_token: Default::default(),
                                body: Box::new(arm_then_body),
                                comma: Some(Default::default()),
                            };
                            let else_arm = syn::Arm {
                                attrs: vec![],
                                pat: syn::parse_quote!(_),
                                guard: None,
                                fat_arrow_token: Default::default(),
                                body: Box::new(arm_else_body),
                                comma: None,
                            };
                            let synthetic = syn::Expr::Match(syn::ExprMatch {
                                attrs: vec![],
                                match_token: Default::default(),
                                expr: Box::new(scrutinee_expr),
                                brace_token: Default::default(),
                                arms: vec![then_arm, else_arm],
                            });
                            return lower_expr(graph, block, &synthetic, options, ctx);
                        }
                    }
                    // Parse failure falls through to the catch-all
                    // Macro arm — preserves the `OpKind::Abort`
                    // diagnostic for un-portable `matches!` shapes
                    // rather than silently mis-lowering.
                }

                if is_assert || is_assert_cmp {
                    if let Ok(args) = m.mac.parse_body_with(
                        syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
                    ) {
                        let mut it = args.iter();
                        let cond_opt: Option<crate::flowspace::model::Variable> = if is_assert {
                            if let Some(cond_expr) = it.next() {
                                let lowered = lower_expr(graph, block, cond_expr, options, ctx)?;
                                if lowered.path_closed {
                                    return Ok(Lowered::path_closed());
                                }
                                lowered.value_var(graph)
                            } else {
                                None
                            }
                        } else {
                            let lhs_expr = it.next();
                            let rhs_expr = it.next();
                            match (lhs_expr, rhs_expr) {
                                (Some(le), Some(re)) => {
                                    let lhs_pre_var = get_value_var!(
                                        lower_expr(graph, block, le, options, ctx)?,
                                        graph
                                    );
                                    ctx.pushvid_var(&lhs_pre_var);
                                    let rhs_pre_var = get_value_var!(
                                        lower_expr(graph, block, re, options, ctx)?,
                                        graph
                                    );
                                    ctx.pushvid_var(&rhs_pre_var);
                                    let rhs_var = ctx.popvid_var(graph);
                                    let lhs_var = ctx.popvid_var(graph);
                                    let op_name = if macro_name.contains("_ne") {
                                        "ne"
                                    } else {
                                        "eq"
                                    };
                                    graph.push_op_var(
                                        *block,
                                        OpKind::BinOp {
                                            op: op_name.into(),
                                            lhs: lhs_var,
                                            rhs: rhs_var,
                                            result_ty: ValueType::Unknown,
                                        },
                                        true,
                                    )
                                }
                                _ => None,
                            }
                        };
                        if let Some(cond_var) = cond_opt {
                            // Split into pass/fail arms BEFORE walking
                            // the message expressions.  Per RPython
                            // `flowspace/flowcontext.py:107`
                            // (`BlockRecorder.guessbool`), the two
                            // arms of a conditional are independent
                            // walks — the message-format arguments
                            // are only reachable on the failing path
                            // and must not land ops on the pass path.
                            //
                            // Message format arguments walk on the
                            // fail branch: RPython `LOAD_CONST`
                            // (`flowspace/flowcontext.py:841`) pushes a
                            // constant and the walk continues, so Str /
                            // Float / ByteStr literals are no longer
                            // fatal — the non-fatal Lit handler above
                            // emits an `Unknown` marker and returns a
                            // value.  We therefore walk every rest arg
                            // unconditionally (side-effect-preserving
                            // order).
                            let pass_block = graph.create_block();
                            let mut fail_block = graph.create_block();
                            graph.set_branch(
                                *block,
                                cond_var,
                                pass_block,
                                vec![],
                                fail_block,
                                vec![],
                            );
                            // Walk every message-expr on the fail
                            // branch to preserve its side effects
                            // (Call / FieldRead / …) on the graph,
                            // then hand the evaluated ValueIds to the
                            // shared `exc_from_raise` lowering as the
                            // positional args of `simple_call(AssertionError,
                            // *args)`.  Upstream parity: RPython
                            // `RAISE_VARARGS` (`flowcontext.py:638-656`)
                            // popvalue's all args before reaching
                            // `exc_from_raise`; the adapter here picks
                            // `AssertionError` as the `w_arg1` for
                            // every assert-family macro so
                            // `front::raise::lower_exc_from_raise`
                            // walks the same op sequence as the
                            // flowspace port at
                            // `flowspace/flowcontext.rs:1189`.
                            let mut message_args: Vec<crate::flowspace::model::Variable> =
                                Vec::new();
                            for rest in it {
                                // The fail-branch walk is independent
                                // of the pass-branch walk; a
                                // path-closing construct inside the
                                // message format (`panic!` nested
                                // inside the format arg) still leaves
                                // the pass branch open, so we don't
                                // propagate path_closed out here.
                                // FlowingError still propagates via
                                // `?`.
                                let lowered =
                                    lower_expr(graph, &mut fail_block, rest, options, ctx)?;
                                if let Some(var) = lowered.value_var(graph) {
                                    message_args.push(var);
                                }
                            }
                            let _ = &macro_name; // name is only used for diagnostics; class is fixed.
                            crate::front::raise::lower_exc_from_raise(
                                graph,
                                fail_block,
                                "AssertionError",
                                message_args,
                            );
                            *block = pass_block;
                            // Pass block is still open — the assert
                            // expression itself has type `()`, no value.
                            return Ok(Lowered::no_value());
                        }
                    }
                }
            }
            // Abort-family macros (`panic!`, `unreachable!`, `todo!`,
            // `unimplemented!`) have type `!` and terminate the current
            // control-flow path with an unconditional raise.  Matches
            // RPython `flowspace/flowcontext.py:1253` `Raise.nomoreblocks`
            // where the enclosing block is closed with a Link to
            // `exceptblock` regardless of the exception argument shape.
            //
            // Per RPython `RAISE_VARARGS`
            // (`flowspace/flowcontext.py:638-656`), the raise target /
            // arguments are `popvalue()`'d off the stack — they have
            // already been evaluated before the Raise.  The same
            // happens in Rust: `panic!("{}", side_effect())` evaluates
            // `side_effect()` before panicking.  Walk every macro arg
            // before `set_raise` so its side effects land in the graph.
            // Literal args are no longer fatal (Lit handler above emits
            // `Unknown` + returns a value), so no skip is needed.
            if let syn::Expr::Macro(m) = other {
                let name = m
                    .mac
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                if matches!(
                    name.as_str(),
                    "panic" | "unreachable" | "todo" | "unimplemented"
                ) {
                    // Walk every message-arg for its side effects
                    // (popvalue-before-raise semantic of RPython
                    // `RAISE_VARARGS`, `flowcontext.py:638-656`), then
                    // forward the evaluated ValueIds as the positional
                    // args of `simple_call(PanicError, *args)` inside
                    // the shared `exc_from_raise` lowering
                    // (`front::raise::lower_exc_from_raise` →
                    // `flowcontext.rs:1189` parity).
                    let mut message_args: Vec<crate::flowspace::model::Variable> = Vec::new();
                    if let Ok(args) = m.mac.parse_body_with(
                        syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
                    ) {
                        for arg in args.iter() {
                            let lowered = lower_expr(graph, block, arg, options, ctx)?;
                            if lowered.path_closed {
                                // A path-closing sub-expression already
                                // terminated `*block`; the outer panic!
                                // has nothing more to do — propagate
                                // path_closed so the enclosing walker
                                // stops.
                                return Ok(Lowered::path_closed());
                            }
                            if let Some(var) = lowered.value_var(graph) {
                                message_args.push(var);
                            }
                        }
                    }
                    // RPython `flowspace/flowcontext.py:1253`
                    // `Raise.nomoreblocks`: close the current block
                    // with a Link to `exceptblock`, then signal the
                    // path terminated.  RPython raises `StopFlowing`,
                    // which is the same kind of FlowSignal as Return —
                    // sibling walks continue normally.  Pyre equivalent:
                    // `Lowered::path_closed()` on the `Ok` arm, NOT
                    // `Err(FlowingError)` (which would abort the whole
                    // function graph).  The Rust panic-family macros
                    // (panic!, unreachable!, todo!, unimplemented!)
                    // all share the `PanicError` adapter class — their
                    // runtime-distinct PanicInfo shape is not modelled
                    // at the flow-graph layer, mirroring reviewer
                    // guidance (`flowcontext.py:2861 Raise` bytecode
                    // adapter — version-specific variants converge on
                    // `exc_from_raise(w_arg1, w_arg2)`).
                    let _ = &name; // macro name carried for diagnostics only.
                    crate::front::raise::lower_exc_from_raise(
                        graph,
                        *block,
                        "PanicError",
                        message_args,
                    );
                    return Ok(Lowered::path_closed());
                }
            }
            let variant = match other {
                syn::Expr::Array(_) => UnsupportedExprKind::Array,
                syn::Expr::Async(_) => UnsupportedExprKind::Async,
                syn::Expr::Await(_) => UnsupportedExprKind::Await,
                syn::Expr::Const(_) => UnsupportedExprKind::Const,
                syn::Expr::Group(_) => UnsupportedExprKind::Group,
                syn::Expr::Infer(_) => UnsupportedExprKind::Infer,
                syn::Expr::Let(_) => UnsupportedExprKind::Let,
                syn::Expr::Macro(_) => UnsupportedExprKind::Macro,
                syn::Expr::Range(_) => UnsupportedExprKind::Range,
                syn::Expr::RawAddr(_) => UnsupportedExprKind::RawAddr,
                syn::Expr::Repeat(_) => UnsupportedExprKind::Repeat,
                syn::Expr::Struct(_) => UnsupportedExprKind::Struct,
                syn::Expr::TryBlock(_) => UnsupportedExprKind::TryBlock,
                syn::Expr::Verbatim(_) => UnsupportedExprKind::Verbatim,
                syn::Expr::Yield(_) => UnsupportedExprKind::Yield,
                _ => UnsupportedExprKind::OtherExpr,
            };
            // The diagnostic emit decision is made later in the
            // `is_data_creation` / `stop_unsupported` chain below;
            // the dump probe runs there so `[UnsupportedExpr]` covers
            // the data-creation default-arm path and
            // `[UnsupportedExpr/stop]` covers the abort path.
            // Helper: walk a sub-expression purely for its side effects
            // (the parent composite is about to be marked unsupported,
            // so the returned value is unused).  Propagate FlowingError
            // via `?`; on path_closed, bail out of the parent walk too
            // — the enclosing block is already terminated and a later
            // `stop_unsupported` would push into a closed block.
            macro_rules! walk_for_side_effects {
                ($e:expr) => {{
                    let lowered = lower_expr(graph, block, $e, options, ctx)?;
                    if lowered.path_closed {
                        return Ok(Lowered::path_closed());
                    }
                }};
            }
            // Non-fatal families mirror RPython bytecodes that push a
            // value and continue the flow walk:
            //   • Data constructors — `BUILD_LIST` / `BUILD_TUPLE` /
            //     `newslice` (`flowspace/flowcontext.py:1168`,
            //     `pypy/interpreter/pyopcode.py:960`).  Pyre does not
            //     yet emit `NewList` / `NewStruct` / `NewRange` IR
            //     ops, so element walks land in the graph and a
            //     single `Unknown` marker stands in for the
            //     allocation.  The local Rust-parity adapter
            //     `flowspace/rust_source/build_flow.rs:1889`
            //     (`lower_array -> newlist`) uses the same shape.
            //   • Generic (non-abort) macros — `format!`, `write!`,
            //     `vec!`, `matches!`, …  treat these as opaque ops
            //     whose result is an opaque value; sub-expr walks
            //     still capture side effects before the marker.
            //     Abort-family macros are handled separately by the
            //     `set_raise` branch earlier in the Macro arm above.
            let is_data_creation = matches!(
                other,
                syn::Expr::Array(_)
                    | syn::Expr::Repeat(_)
                    | syn::Expr::Struct(_)
                    | syn::Expr::Range(_)
                    | syn::Expr::Let(_)
                    | syn::Expr::Macro(_)
                    | syn::Expr::RawAddr(_)
            );
            match other {
                // `a..b` / `a..=b` / `..b` / `a..` / `..` — evaluate
                // the endpoint expressions so side effects in them are
                // captured.  Per RPython `newslice` (implicit in
                // `BUILD_SLICE` at `pypy/interpreter/pyopcode.py`), the
                // endpoints land as separate pushes before the slice
                // is constructed.
                syn::Expr::Range(r) => {
                    if let Some(from) = &r.start {
                        walk_for_side_effects!(from);
                    }
                    if let Some(to) = &r.end {
                        walk_for_side_effects!(to);
                    }
                }
                // `[a, b, c]` — evaluate each element.  RPython
                // `BUILD_LIST` (`flowspace/flowcontext.py:1168`) pops
                // N items and pushes `space.newlist(items)`; we emit
                // an `Unknown` marker for the `newlist` step, which
                // matches the local Rust-parity adapter in
                // `flowspace/rust_source/build_flow.rs:1889`
                // (`lower_array -> newlist`) until a proper
                // `OpKind::NewList` lands.
                syn::Expr::Array(a) => {
                    for e in &a.elems {
                        walk_for_side_effects!(e);
                    }
                }
                // `[v; N]` — evaluate the element expression and the
                // repeat count expression.  N is commonly a literal
                // integer; walking it emits a `ConstInt` op that the
                // annotator can still see.
                syn::Expr::Repeat(r) => {
                    walk_for_side_effects!(&r.expr);
                    walk_for_side_effects!(&r.len);
                }
                // `S { f: v, g: w, ..rest }` — evaluate each field
                // value, then any `..rest` base.  Parallels RPython
                // `newstruct` / `BUILD_MAP`-style constructors.
                syn::Expr::Struct(s) => {
                    for field in &s.fields {
                        walk_for_side_effects!(&field.expr);
                    }
                    if let Some(rest) = &s.rest {
                        walk_for_side_effects!(rest);
                    }
                }
                // `let PAT = EXPR` (only reachable as the cond of an
                // `if let` / `while let`).  Evaluate the scrutinee so
                // side effects are captured; the pattern match itself
                // remains opaque until enum-variant dispatch lands.
                syn::Expr::Let(l) => {
                    walk_for_side_effects!(&l.expr);
                }
                // `&raw const/mut EXPR` — the address operator
                // produces a pointer rather than the inner value, so
                // we emit an `Unknown` marker for the address itself
                // (handled by the data-creation arm below) but still
                // walk the inner expr so any side effects are
                // captured before the pointer is taken.
                syn::Expr::RawAddr(r) => {
                    walk_for_side_effects!(&r.expr);
                }
                // `foo!(a, b, c)` / `foo![a, b, c]` / `foo!{a, b, c}`
                // — most Rust macros whose bodies reach this point
                // (vec!, format!, matches!, write!, writeln!, ...)
                // accept comma-separated expressions as arguments.
                // Parse the token stream as `Punctuated<Expr, ,>` and
                // walk each; on parse failure (e.g. macros with
                // non-expression syntax), fall through to bare abort.
                // Matches the RPython FlowingError convention at
                // `rpython/flowspace/flowcontext.py:258` where
                // sub-expression push ops land BEFORE the abort point.
                syn::Expr::Macro(m) => {
                    if let Ok(args) = m.mac.parse_body_with(
                        syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
                    ) {
                        for arg in args.iter() {
                            walk_for_side_effects!(arg);
                        }
                    }
                }
                _ => {}
            }
            let dump_enabled = std::env::var("MAJIT_UNKNOWN_DUMP").is_ok();
            if is_data_creation {
                if dump_enabled {
                    let fn_name = CURRENT_LOWERING_FN_NAME
                        .with(|c| c.borrow().clone())
                        .unwrap_or_else(|| "<unknown>".to_string());
                    println!("cargo:warning=[UnsupportedExpr] fn={fn_name} variant={variant:?}");
                }
                Ok(continue_with_unknown(graph, *block, variant))
            } else {
                if dump_enabled {
                    let fn_name = CURRENT_LOWERING_FN_NAME
                        .with(|c| c.borrow().clone())
                        .unwrap_or_else(|| "<unknown>".to_string());
                    println!(
                        "cargo:warning=[UnsupportedExpr/stop] fn={fn_name} variant={variant:?}"
                    );
                }
                stop_unsupported(graph, *block, variant)
            }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Split a macro token stream at the first top-level (depth-0) comma,
/// returning `(prefix, suffix)` token streams without the comma. Used
/// by the `matches!` desugar to peel `(scrutinee, pat [if guard])` —
/// the first comma separates `scrutinee` from the rest, but commas
/// inside `(...)` / `[...]` / `{...}` (e.g. tuple struct patterns,
/// inner expression lists) must not split. Returns `None` when no
/// top-level comma is present.
fn split_macro_args_at_first_top_comma(
    tokens: proc_macro2::TokenStream,
) -> Option<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    let mut prefix: Vec<proc_macro2::TokenTree> = Vec::new();
    let mut iter = tokens.into_iter();
    for tt in iter.by_ref() {
        if let proc_macro2::TokenTree::Punct(ref p) = tt {
            if p.as_char() == ',' && p.spacing() == proc_macro2::Spacing::Alone {
                let suffix: proc_macro2::TokenStream = iter.collect();
                return Some((prefix.into_iter().collect(), suffix));
            }
        }
        prefix.push(tt);
    }
    None
}

/// `syn::parse::Parser` adapter for the `pat [if guard]` tail of a
/// `matches!` invocation. Mirrors the std `matches!` macro grammar
/// (`std::macros::matches`): a single `Pat`, optionally followed by
/// `if Expr`. The `Pat` parse uses `Pat::parse_multi_with_leading_vert`
/// so top-level `|` alternations (`Some(_) | None`) are accepted.
fn parse_matches_pat_and_guard(
    input: syn::parse::ParseStream,
) -> syn::Result<(syn::Pat, Option<syn::Expr>)> {
    let pat = syn::Pat::parse_multi_with_leading_vert(input)?;
    let guard = if input.peek(syn::Token![if]) {
        let _: syn::Token![if] = input.parse()?;
        Some(input.parse::<syn::Expr>()?)
    } else {
        None
    };
    Ok((pat, guard))
}

fn classify_match_bool_exitcases(arms: &[syn::Arm]) -> Option<Vec<Vec<ExitCase>>> {
    if arms.len() != 2 {
        return None;
    }
    let mut all = Vec::with_capacity(arms.len());
    let mut seen_bools = Vec::<bool>::new();
    for (arm_idx, arm) in arms.iter().enumerate() {
        if arm.guard.is_some() {
            return None;
        }
        let is_last_arm = arm_idx + 1 == arms.len();
        let mut sub_pats = Vec::new();
        flatten_or_pattern(&arm.pat, &mut sub_pats);
        if sub_pats.len() > 1 && sub_pats.iter().any(|pat| matches!(pat, syn::Pat::Wild(_))) {
            return None;
        }
        let mut arm_cases = Vec::with_capacity(sub_pats.len());
        for (sub_idx, sub_pat) in sub_pats.iter().enumerate() {
            let is_last = is_last_arm && sub_idx + 1 == sub_pats.len();
            match classify_bool_pattern(sub_pat, is_last, &seen_bools)? {
                BoolPatternCase::Value(value) => {
                    if seen_bools.contains(&value) {
                        return None;
                    }
                    seen_bools.push(value);
                    arm_cases.push(ExitCase::Bool(value));
                }
                BoolPatternCase::Default(values) => {
                    if values.is_empty() {
                        return None;
                    }
                    for value in values {
                        seen_bools.push(value);
                        arm_cases.push(ExitCase::Bool(value));
                    }
                }
            }
        }
        all.push(arm_cases);
    }
    let mut sorted = seen_bools;
    sorted.sort();
    sorted.dedup();
    if sorted.as_slice() == &[false, true] {
        Some(all)
    } else {
        None
    }
}

enum BoolPatternCase {
    Value(bool),
    Default(Vec<bool>),
}

fn classify_bool_pattern(
    pat: &syn::Pat,
    is_last: bool,
    seen_bools: &[bool],
) -> Option<BoolPatternCase> {
    match pat {
        syn::Pat::Wild(_) if is_last => {
            let mut values = Vec::new();
            if !seen_bools.contains(&false) {
                values.push(false);
            }
            if !seen_bools.contains(&true) {
                values.push(true);
            }
            Some(BoolPatternCase::Default(values))
        }
        syn::Pat::Wild(_) => None,
        syn::Pat::Lit(lit) => match &lit.lit {
            syn::Lit::Bool(value) => Some(BoolPatternCase::Value(value.value)),
            _ => None,
        },
        _ => None,
    }
}

fn classify_match_switch_exitcases(arms: &[syn::Arm]) -> Option<Vec<Vec<ExitCase>>> {
    if arms.len() < 2 {
        return None;
    }
    let mut all = Vec::with_capacity(arms.len());
    let mut seen = Vec::<ExitCase>::new();
    for (arm_idx, arm) in arms.iter().enumerate() {
        if arm.guard.is_some() {
            return None;
        }
        let is_last_arm = arm_idx + 1 == arms.len();
        let mut sub_pats = Vec::new();
        flatten_or_pattern(&arm.pat, &mut sub_pats);
        if sub_pats.len() > 1 && sub_pats.iter().any(|pat| matches!(pat, syn::Pat::Wild(_))) {
            return None;
        }
        let mut arm_cases = Vec::with_capacity(sub_pats.len());
        for (sub_idx, sub_pat) in sub_pats.iter().enumerate() {
            let is_last = is_last_arm && sub_idx + 1 == sub_pats.len();
            let exitcase = classify_switch_pattern(sub_pat, is_last)?;
            if !is_default_exitcase(&exitcase) {
                if seen.iter().any(|existing| existing == &exitcase) {
                    return None;
                }
                seen.push(exitcase.clone());
            }
            arm_cases.push(exitcase);
        }
        all.push(arm_cases);
    }
    Some(all)
}

fn flatten_or_pattern<'a>(pat: &'a syn::Pat, out: &mut Vec<&'a syn::Pat>) {
    match pat {
        syn::Pat::Or(or_pat) => {
            for case in &or_pat.cases {
                flatten_or_pattern(case, out);
            }
        }
        syn::Pat::Paren(paren) => flatten_or_pattern(&paren.pat, out),
        other => out.push(other),
    }
}

fn classify_switch_pattern(pat: &syn::Pat, is_last: bool) -> Option<ExitCase> {
    match pat {
        syn::Pat::Wild(_) if is_last => Some(default_exitcase()),
        syn::Pat::Wild(_) => None,
        syn::Pat::Lit(lit) => match &lit.lit {
            syn::Lit::Int(int_lit) => int_lit
                .base10_parse::<i64>()
                .ok()
                .map(|value| ExitCase::Const(ConstValue::Int(value))),
            syn::Lit::Byte(value) => Some(ExitCase::Const(ConstValue::Int(value.value() as i64))),
            syn::Lit::Char(value) => Some(ExitCase::Const(ConstValue::Int(value.value() as i64))),
            _ => None,
        },
        _ => None,
    }
}

fn default_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::byte_str("default"))
}

fn is_default_exitcase(exitcase: &ExitCase) -> bool {
    matches!(
        exitcase,
        ExitCase::Const(ConstValue::ByteStr(bytes)) if bytes.as_slice() == b"default"
    )
}

fn unary_op_name(op: &syn::UnOp) -> &'static str {
    match op {
        syn::UnOp::Deref(_) => "deref",
        syn::UnOp::Not(_) => "not",
        syn::UnOp::Neg(_) => "neg",
        _ => "unknown_unary",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnaryNotOperandKind {
    Bool,
    Int,
    Unknown,
}

/// Classify Rust's overloaded `!` operand into the RPython opcode it
/// can be lowered to. RPython/PyPy has distinct bytecode shapes:
/// `UNARY_NOT` lowers through `op.bool` plus a branch, while
/// `UNARY_INVERT` lowers directly to `op.invert`.  `Unknown` triggers
/// `stop_unsupported(UnaryNotUnknownOperand)` at the call site
/// (`lower_expr`'s `Expr::Unary UnOp::Not` arm, ~line 3737), mirroring
/// RPython `flowcontext.py:194,535-538` and `build_flow.rs:4404-4416`
/// which fail-loud on the same shape; the classifier extensions below
/// ensure every production-source operand resolves to `Bool` or `Int`.
///
/// TODO(receiver-typed-dispatch): RPython
/// `bookkeeper.py:353-409 getdesc(value)` keys on Python object
/// identity — same-named methods on different types cannot alias
/// because each host callable is a distinct PyObject.  Pyre's
/// classifier operates on the AST surface (no callable-identity
/// resolution available pre-rtyper), so it falls back to
/// `local_type_strings` / `fn_return_types` lookup, last-segment
/// fallback for multi-segment `Path`s, and a receiver-independent
/// method shortlist for known-shape predicates (`is_empty`,
/// `is_constant`, ...).  Two methods of the same name returning
/// different types would collide here even if they belong to
/// different types — the classifier is structurally not 1:1 with
/// `getdesc`.  Retire when pyre's surface DSL learns to emit a
/// callable-resolved Path (via the `resolved_path` producer
/// `codewriter.rs::stamp_classdef_hints_on_graph`,
/// `description.py:407-519` parity); the classifier can then route
/// through the resolved callable and the shortlist
/// retires.  Until then, name-shortlist hits that diverge
/// from upstream behaviour surface as dual-gate `Skip` (typed-Ref
/// classdef-less SomeInstance) or production divergence panic.
fn expr_unary_not_operand_kind(expr: &syn::Expr, ctx: &GraphBuildContext) -> UnaryNotOperandKind {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Bool(_),
            ..
        }) => UnaryNotOperandKind::Bool,
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(_) | syn::Lit::Byte(_) | syn::Lit::Char(_),
            ..
        }) => UnaryNotOperandKind::Int,
        // Multi-segment Path expressions — `Self::CONST_BIT`,
        // `Type::CONST`, `Module::CONST`. Pyre's walker registers
        // `Item::Impl::Const` under both `Type::CONST` and bare `CONST`
        // (front/ast.rs Item::Impl arm), so the last-segment fallback
        // resolves `Self::CONST_BIT` references via the bare alias.
        // TODO(receiver-typed-dispatch): replace `fn_return_types`'s
        // string map with a host-identity-keyed `Bookkeeper.descs`
        // populated by the walker so the last-segment lookup at line
        // 5602 collapses into a single qualified lookup
        // (`bookkeeper.py:353` parity).  Until then the bare-name
        // fallback can alias when same-named symbols live on
        // different scopes/types.
        syn::Expr::Path(path) if path.qself.is_none() && path.path.segments.len() >= 2 => {
            let n = path.path.segments.len();
            let last = path.path.segments[n - 1].ident.to_string();
            if let Some(ty) = ctx.fn_return_types.get(&last) {
                let kind = type_string_to_unary_not_kind(ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            let bare_impl = format!(
                "{}::{}",
                path.path.segments[n - 2].ident,
                path.path.segments[n - 1].ident
            );
            if let Some(ty) = ctx.fn_return_types.get(&bare_impl) {
                let kind = type_string_to_unary_not_kind(ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            UnaryNotOperandKind::Unknown
        }
        syn::Expr::Path(path)
            if path.qself.is_none()
                && path.path.leading_colon.is_none()
                && path.path.segments.len() == 1 =>
        {
            let segment = &path.path.segments[0];
            if !matches!(segment.arguments, syn::PathArguments::None) {
                return UnaryNotOperandKind::Unknown;
            }
            let name = segment.ident.to_string();
            if let Some(ty) = ctx.local_type_strings.get(&name) {
                let kind = type_string_to_unary_not_kind(ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            if let Some(kind) = ctx
                .local_value_types
                .get(&name)
                .map(value_type_to_unary_not_kind)
                && kind != UnaryNotOperandKind::Unknown
            {
                return kind;
            }
            // Module-level `pub const FOO: bool = ...` — the walker
            // registers `Item::Const` typed names into
            // `fn_return_types` (front/ast.rs:535+ `Item::Const`
            // arm; key namespace is shared but Rust convention
            // (SCREAMING_SNAKE_CASE consts vs snake_case fns) keeps
            // the lookups disjoint in pyre source).  RPython resolves
            // these via the bookkeeper's PBC table at LOAD_GLOBAL
            // time (`bookkeeper.py:329-340 immutablevalue`).
            if let Some(ty) = ctx.fn_return_types.get(&name) {
                let kind = type_string_to_unary_not_kind(ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            UnaryNotOperandKind::Unknown
        }
        syn::Expr::Binary(bin) => expr_binary_unary_not_operand_kind(ctx, bin),
        syn::Expr::Unary(unary) => match unary.op {
            syn::UnOp::Not(_) => expr_unary_not_operand_kind(&unary.expr, ctx),
            syn::UnOp::Neg(_) => {
                if expr_unary_not_operand_kind(&unary.expr, ctx) == UnaryNotOperandKind::Int {
                    UnaryNotOperandKind::Int
                } else {
                    UnaryNotOperandKind::Unknown
                }
            }
            syn::UnOp::Deref(_) => UnaryNotOperandKind::Unknown,
            _ => UnaryNotOperandKind::Unknown,
        },
        syn::Expr::Paren(paren) => expr_unary_not_operand_kind(&paren.expr, ctx),
        syn::Expr::Group(group) => expr_unary_not_operand_kind(&group.expr, ctx),
        // `unsafe { expr }` — Rust's unsafe block is a transparent
        // wrapper for analyser purposes; RPython has no syntactic
        // analogue so the inner expression's classification is what
        // matters.  Mirror of the Paren/Group arms.
        syn::Expr::Unsafe(u) => {
            if let Some(syn::Stmt::Expr(tail, None)) = u.block.stmts.last() {
                expr_unary_not_operand_kind(tail, ctx)
            } else {
                UnaryNotOperandKind::Unknown
            }
        }
        syn::Expr::Block(b) => {
            if let Some(syn::Stmt::Expr(tail, None)) = b.block.stmts.last() {
                expr_unary_not_operand_kind(tail, ctx)
            } else {
                UnaryNotOperandKind::Unknown
            }
        }
        // `if cond { a } else { b }` — RPython `flowcontext.py:413`
        // joins the two arms at the merge block; if both arms project
        // to the same `SomeBool` / `SomeInteger`, the join carries
        // that lattice node forward.  Mirror by classifying both arm
        // tails and accepting only when they agree.  Missing else (or
        // non-Block arms) collapses to Unknown.
        syn::Expr::If(if_expr) => {
            let then_kind =
                if let Some(syn::Stmt::Expr(tail, None)) = if_expr.then_branch.stmts.last() {
                    expr_unary_not_operand_kind(tail, ctx)
                } else {
                    UnaryNotOperandKind::Unknown
                };
            if then_kind == UnaryNotOperandKind::Unknown {
                return UnaryNotOperandKind::Unknown;
            }
            match &if_expr.else_branch {
                Some((_, else_expr)) => {
                    let else_kind = expr_unary_not_operand_kind(else_expr, ctx);
                    if else_kind == then_kind {
                        then_kind
                    } else {
                        UnaryNotOperandKind::Unknown
                    }
                }
                None => UnaryNotOperandKind::Unknown,
            }
        }
        // `match e { arm => ..., ... }` — `flowcontext.py:413` join-
        // shape parity, but with N arms.  Accept only when every arm
        // projects to the same kind.
        syn::Expr::Match(match_expr) => {
            let mut acc: Option<UnaryNotOperandKind> = None;
            for arm in &match_expr.arms {
                let kind = expr_unary_not_operand_kind(&arm.body, ctx);
                if kind == UnaryNotOperandKind::Unknown {
                    return UnaryNotOperandKind::Unknown;
                }
                match acc {
                    None => acc = Some(kind),
                    Some(prev) if prev == kind => {}
                    _ => return UnaryNotOperandKind::Unknown,
                }
            }
            acc.unwrap_or(UnaryNotOperandKind::Unknown)
        }
        syn::Expr::Macro(mac)
            if mac.mac.path.segments.len() == 1
                && mac.mac.path.segments[0].ident.to_string() == "matches" =>
        {
            UnaryNotOperandKind::Bool
        }
        // `expr?` — Rust try operator desugars to `match expr { Ok(v)
        // => v, Err(e) => return Err(e.into()) }` (or the Option
        // counterpart). Mirror the `expression_type_string` Try arm
        // by classifying the inner expression's kind through the
        // Result/Option unwrap. RPython peer: `flowcontext.py`'s
        // POP_BLOCK / END_FINALLY exception-channel join, projected to
        // the success-arm tail.
        syn::Expr::Try(t) => {
            if let Some(inner_ty) = expression_type_string(&t.expr, ctx)
                && let Some(unwrapped) = unwrap_result_or_option(&inner_ty)
            {
                let kind = type_string_to_unary_not_kind(unwrapped);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            UnaryNotOperandKind::Unknown
        }
        // `(*func).can_change_code` / `frame.is_root` — struct field
        // access whose declared type is recorded in
        // `ctx.struct_fields`.  RPython resolves this through
        // `SomeInstance.find_attribute` (`annotator/model.py:430+`);
        // pyre's `expression_type_string` already routes
        // `Expr::Field` to `field_type_string_from_expr`, so the
        // classifier just needs to project the resulting type string.
        syn::Expr::Field(_) => {
            if let Some(ty) = expression_type_string(expr, ctx) {
                let kind = type_string_to_unary_not_kind(&ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            UnaryNotOperandKind::Unknown
        }
        syn::Expr::Call(_) | syn::Expr::MethodCall(_) => {
            if let Some(ty) = expression_type_string(expr, ctx) {
                let kind = type_string_to_unary_not_kind(&ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            // `.unwrap()` / `.expect(_)` on a `Result<T, _>` / `Option<T>`
            // receiver projects to `T`.  RPython parity: the annotator
            // walks the method-resolution and sees `T` directly
            // (`rtyper/rmodel.py:rtype_method_unwrap`); pyre lacks
            // generic-method visibility, so synthesise the projection
            // by inspecting the receiver's return type and unwrapping
            // the `Result<...>` / `Option<...>` wrapper.
            if let syn::Expr::MethodCall(mc) = expr
                && (mc.method == "unwrap" || mc.method == "expect")
                && let Some(receiver_ty) = expression_type_string(&mc.receiver, ctx)
                && let Some(inner) = unwrap_result_or_option(&receiver_ty)
            {
                let kind = type_string_to_unary_not_kind(inner);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            // TODO(receiver-typed-dispatch): the receiver-independent
            // bool shortlist below is a heuristic substitute for
            // `bookkeeper.getdesc(receiver).find_method` (RPython
            // `unaryop.py:206-213`).  RPython matches host-stdlib
            // trait methods by *receiver class* identity; pyre's
            // annotator does not yet track stdlib/cross-crate trait
            // impls, so receiver-typed lookup
            // (`primitive_method_result_type` /
            // `lookup_method_return_type`) misses these names and
            // they reach the catch-all here.  Retire when
            // `fn_return_types` is populated from a metadata-only
            // walk over `pyre-{object,interpreter,jit}` + stdlib
            // trait impls — this entire match arm goes away.
            // Aliasing risk: a user-source method with the same name
            // on an unrelated type that does NOT return bool (e.g.
            // `eq`/`ne`/`lt`/`gt`/`ge`/`le` from `PartialOrd`,
            // `contains`/`any`/`all` from various collection traits)
            // would mis-classify here.  Trim attempts on those names
            // 2026-05-11 broke pyre source build (some user-source
            // call sites depend on the bool fallback) — full removal
            // requires receiver-typed dispatch landing first.
            if let syn::Expr::MethodCall(mc) = expr {
                let method_name = mc.method.to_string();
                if matches!(
                    method_name.as_str(),
                    "is_null"
                        | "is_some"
                        | "is_none"
                        | "is_ok"
                        | "is_err"
                        | "is_empty"
                        | "contains"
                        | "starts_with"
                        | "ends_with"
                        | "eq"
                        | "ne"
                        | "lt"
                        | "le"
                        | "gt"
                        | "ge"
                        // Float-bool predicates (`f64::is_nan`,
                        // `f64::is_infinite`, etc.) — receiver-typed
                        // arm `primitive_method_result_type:5576`
                        // already maps these to `ValueType::Bool` for
                        // typed-call resolution, but
                        // `lookup_method_return_type` only consults
                        // user-source `fn_return_types`.  Mirror the
                        // shortlist here so the overloaded `!` over
                        // a stdlib float predicate classifies as
                        // Bool without depending on receiver-type
                        // tracking propagating to the caller.
                        | "is_nan"
                        | "is_infinite"
                        | "is_finite"
                        | "is_sign_negative"
                        | "is_sign_positive"
                        // Integer sign predicates (`i64::is_positive`,
                        // `i64::is_negative`).  Same rationale as
                        // the float predicates above — RPython's
                        // `bookkeeper.getdesc(receiver).find_method`
                        // would resolve via the host stdlib.
                        | "is_positive"
                        | "is_negative"
                        // `char::is_alphabetic` / `char::is_digit` /
                        // `char::is_alphanumeric` / `char::is_whitespace`
                        // / `char::is_ascii*` — `core::char` predicates
                        // returning `bool`.  Same rationale as the
                        // numeric predicates.
                        | "is_alphabetic"
                        | "is_alphanumeric"
                        | "is_digit"
                        | "is_whitespace"
                        | "is_ascii"
                        | "is_ascii_alphabetic"
                        | "is_ascii_alphanumeric"
                        | "is_ascii_digit"
                        | "is_ascii_whitespace"
                        | "is_ascii_uppercase"
                        | "is_ascii_lowercase"
                        | "is_uppercase"
                        | "is_lowercase"
                        // `std::process::ExitStatus::success` and the
                        // related `Path::exists` family — used by
                        // pyre's stdlib-detection helpers
                        // (`pyre-interpreter/src/importing.rs`).
                        | "success"
                        | "exists"
                        // Cross-crate JIT-driver / descriptor methods
                        // declared on types defined outside
                        // `PYRE_JIT_GRAPH_SOURCES` (e.g. `WarmState`
                        // in `majit-metainterp/src/warmstate.rs:143`,
                        // `Descr::is_array_of_pointers` in
                        // `majit-ir/src/descr.rs:1551`,
                        // `Signature::has_kwarg` in
                        // `pyre-interpreter/src/gateway.rs:235`).  The
                        // walker registers these methods only when
                        // their owner type is in the analyser source
                        // set — for cross-crate types it isn't, so
                        // `lookup_method_return_type` returns None and
                        // the surface `!driver.is_tracing()` /
                        // `!bh_descr.is_array_of_pointers()` /
                        // `!sig.has_kwarg()` falls through.  Same
                        // rationale as the stdlib-method shortlist
                        // above (`is_null` / `is_some` / ...): RPython
                        // `bookkeeper.getdesc(receiver).find_method`
                        // resolves these by host-identity; pyre's
                        // static shortlist substitutes for the missing
                        // whole-program annotator visibility.
                        // Convergence with the cross-crate
                        // `pyre_object::*` Call shortlist: a
                        // metadata-only walk over the host crates
                        // retires this list.
                        | "is_tracing"
                        | "has_compiled_loop"
                        | "is_array_of_pointers"
                        | "has_kwarg"
                        | "has_vararg"
                        // `Iterator::any` / `Iterator::all` — stdlib
                        // iterator predicates that always return
                        // `bool`.  See TODO(receiver-typed-dispatch)
                        // block above.
                        | "any"
                        | "all"
                        // `majit-ir`-side predicate methods on
                        // `OpCode` / `AbstractValue` / `OpRef` /
                        // `ResOperation` — all return `bool` and are
                        // receiver-independent shape introspectors.
                        // Walker registers `Type::method` entries when
                        // `majit-ir/src/resoperation.rs` is in scope,
                        // but multi-segment / variant-call receivers
                        // (`OpCode::IntAdd.is_jit_debug()`,
                        // `AbstractValue::ConstInt(7).is_input_arg()`)
                        // bypass `receiver_type_root`. Same RPython
                        // parity rationale as the stdlib `is_*`
                        // shortlist above.
                        | "is_input_arg"
                        | "is_res_op"
                        | "is_constant"
                        | "is_always_pure"
                        | "is_foldable_guard"
                        | "is_malloc"
                        | "is_memory_access"
                        | "is_comparison"
                        | "is_setarrayitem"
                        | "is_getfield"
                        | "is_setfield"
                        | "is_getarrayitem"
                        | "is_setinteriorfield"
                        | "is_getinteriorfield"
                        | "can_malloc"
                        | "can_raise"
                        | "is_guard_exception"
                        | "is_guard_overflow"
                        | "is_call"
                        | "is_jit_debug"
                        | "is_guard"
                        | "is_ovf"
                        | "is_same_as"
                        | "is_vector_arithmetic"
                        | "is_label"
                        | "is_final"
                        | "should_trace_function_entry"
                ) {
                    return UnaryNotOperandKind::Bool;
                }
            }
            // Stdlib free-function path calls that unambiguously return
            // `bool` — the path-call analogue of the `is_null`/`is_some`
            // / etc. method shortlist above.  Pyre's walker only
            // registers user-source signatures into `fn_return_types`
            // (`front/ast.rs:462+`), so stdlib paths like `std::ptr::eq`
            // never resolve through `lookup_function_return_type`.
            // RPython resolves these via `bookkeeper.getdesc(value)`
            // keyed on the host-stdlib function object identity
            // (`annrpython.py` callee resolution); pyre has no parallel
            // descriptor, so the shortlist stays until the annotator
            // gains stdlib visibility.
            if let syn::Expr::Call(call) = expr
                && let syn::Expr::Path(path) = &*call.func
            {
                let segments: Vec<String> = path
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                let joined = segments.join("::");
                if matches!(
                    joined.as_str(),
                    "std::ptr::eq"
                        | "core::ptr::eq"
                        | "ptr::eq"
                        | "crate::is_function"
                        | "crate::is_function_with_fixed_code"
                ) {
                    return UnaryNotOperandKind::Bool;
                }
                // `Self::raw_is_constant` / `Self::raw_is_*` —
                // `majit-ir/src/resoperation.rs:185 raw_is_constant(raw: u32) -> bool`
                // and siblings declare bool predicates as `Self::method`
                // calls inside an impl block (function-call shape, not
                // method-call).  TODO(receiver-typed-dispatch): when
                // `Self`-relative paths resolve through the bookkeeper's
                // class identity, this falls out automatically.
                if segments.len() == 2 && segments[0] == "Self" {
                    let last = segments[1].as_str();
                    if matches!(
                        last,
                        "raw_is_constant"
                            | "raw_is_input_arg"
                            | "raw_is_res_op"
                            | "raw_is_const_int"
                            | "raw_is_const_float"
                    ) {
                        return UnaryNotOperandKind::Bool;
                    }
                }
                // Cross-crate `pyre_object::*` predicate family —
                // `pyre/pyre-object/src/{pyobject,typeobject,strobject,
                // excobject}.rs` declares a stable set of `pub unsafe
                // fn is_<...>(obj) -> bool` visibility helpers
                // (`pyobject.rs:308-368`: `is_int`, `is_bool`,
                // `is_float`, `is_long`, `is_int_or_long`, `is_list`,
                // `is_tuple`, `is_dict`, `is_none`, `is_not_implemented`,
                // …; plus `is_exception` / `is_str` / `is_module` /
                // `is_instance` / `is_set_or_frozenset` / `is_type` /
                // `is_bool` / `is_builtin_code` exported from sibling
                // modules).  These are the cross-crate analogue of
                // `std::ptr::eq` — pyre's analyser source set
                // (`generated::PYRE_JIT_GRAPH_SOURCES`) does not include
                // `pyre/pyre-object/src/*.rs`, so `lookup_function_return_
                // type` has no entry to project.  RPython parity:
                // `bookkeeper.getdesc(value)` resolves these helpers by
                // host-stdlib identity (`annrpython.py` callee
                // resolution); pyre's static shortlist substitutes for
                // the missing whole-program annotator visibility.
                // Convergence path: emit a metadata-only walk over
                // `pyre/pyre-object/src/*.rs` that registers
                // `fn_return_types` without subjecting raw-pointer
                // `unsafe fn` bodies to graph analysis (deferred;
                // the shortlist retires once the metadata-only walk
                // lands).
                if let Some(joined_str) = joined.strip_prefix("pyre_object::") {
                    let last = joined_str.rsplit("::").next().unwrap_or(joined_str);
                    if matches!(
                        last,
                        "is_int"
                            | "is_bool"
                            | "is_float"
                            | "is_long"
                            | "is_int_or_long"
                            | "is_list"
                            | "is_tuple"
                            | "is_dict"
                            | "is_none"
                            | "is_not_implemented"
                            | "is_str"
                            | "is_exception"
                            | "is_module"
                            | "is_instance"
                            | "is_type"
                            | "is_set_or_frozenset"
                            | "is_builtin_code"
                            | "ll_issubclass"
                            | "w_set_contains"
                            | "w_type_get_hasdict"
                            | "w_type_get_weakrefable"
                            | "w_type_get_acceptable_as_base_class"
                    ) {
                        return UnaryNotOperandKind::Bool;
                    }
                }
                // Multi-segment cross-crate / `crate::`-rooted user paths
                // (`pyre_object::is_exception`, `crate::typeobject::is_type`,
                // ...) — the walker registers free functions under the
                // file-local prefix
                // (`build_semantic_program_from_parsed_files_with_options`
                // currently passes `prefix=""` per file,
                // `front/ast.rs:751-780`), so a multi-segment lookup
                // through `lookup_function_return_type` misses the
                // single-name registration.  Cross-crate / cross-module
                // `pyre_object::`-rooted paths are functionally
                // equivalent to bare references in pyre's whole-program
                // mode (RPython `annrpython.py` resolves callees via
                // `bookkeeper.getdesc(value)` keyed on object identity,
                // not on the `module.name`).  Re-attempt the lookup
                // with the trailing single segment so the bare-key
                // fallback (`lookup_function_return_type`'s
                // `segments.len() == 1` branch) finds the registered
                // entry.  Scoped to `!` classification — wider use of
                // last-segment fallback risks cross-module name
                // collisions.
                if segments.len() > 1
                    && let Some(last) = segments.last()
                    && let Some(ret) = ctx.fn_return_types.get(last)
                {
                    let kind = type_string_to_unary_not_kind(ret);
                    if kind != UnaryNotOperandKind::Unknown {
                        return kind;
                    }
                }
                // `pyre_object::typeobject::Layout::expands_equal(...)`
                // — multi-segment Impl-method paths.  The walker
                // registers Impl methods under the bare `Type::method`
                // shape (`Item::Impl` arm at front/ast.rs:591), so a
                // crate-relative call site referencing the method
                // through its module path falls back to the last two
                // segments.  Same RPython parity rationale as the
                // last-single-segment fallback above (whole-program
                // visibility via `bookkeeper.getdesc(value)` keyed on
                // host identity, not on source path).
                if segments.len() >= 2 {
                    let n = segments.len();
                    let bare_impl = format!("{}::{}", segments[n - 2], segments[n - 1]);
                    if let Some(ret) = ctx.fn_return_types.get(&bare_impl) {
                        let kind = type_string_to_unary_not_kind(ret);
                        if kind != UnaryNotOperandKind::Unknown {
                            return kind;
                        }
                    }
                }
                // External numeric type-conversion constructors —
                // `BigInt::from(...)`, `BigUint::from(...)`,
                // `i64::from_str_radix(...)`, etc.  Pyre's walker has no
                // visibility into `num_bigint`, so the constructor's
                // return type isn't in `fn_return_types`.  RPython peer:
                // `bookkeeper.getdesc(value)` resolves host-stdlib
                // arithmetic types (`int`, `float`, `long`) to integer
                // / float annotations; pyre's static shortlist
                // mirrors that for the BigInt family used by the long
                // bytestring carrier (`pyre-interpreter/src/baseobjspace.rs`
                // BigInt arithmetic).
                if segments.len() >= 2 {
                    let n = segments.len();
                    if matches!(segments[n - 2].as_str(), "BigInt" | "BigUint") {
                        return UnaryNotOperandKind::Int;
                    }
                }
            }
            UnaryNotOperandKind::Unknown
        }
        _ => UnaryNotOperandKind::Unknown,
    }
}

fn expr_binary_unary_not_operand_kind(
    ctx: &GraphBuildContext,
    bin: &syn::ExprBinary,
) -> UnaryNotOperandKind {
    match bin.op {
        syn::BinOp::Eq(_)
        | syn::BinOp::Ne(_)
        | syn::BinOp::Lt(_)
        | syn::BinOp::Le(_)
        | syn::BinOp::Gt(_)
        | syn::BinOp::Ge(_)
        | syn::BinOp::And(_)
        | syn::BinOp::Or(_) => UnaryNotOperandKind::Bool,
        syn::BinOp::Add(_)
        | syn::BinOp::Sub(_)
        | syn::BinOp::Mul(_)
        | syn::BinOp::Div(_)
        | syn::BinOp::Rem(_)
        | syn::BinOp::Shl(_)
        | syn::BinOp::Shr(_) => {
            if expr_unary_not_operand_kind(&bin.left, ctx) == UnaryNotOperandKind::Int
                && expr_unary_not_operand_kind(&bin.right, ctx) == UnaryNotOperandKind::Int
            {
                UnaryNotOperandKind::Int
            } else {
                UnaryNotOperandKind::Unknown
            }
        }
        syn::BinOp::BitAnd(_) | syn::BinOp::BitOr(_) | syn::BinOp::BitXor(_) => {
            let lhs = expr_unary_not_operand_kind(&bin.left, ctx);
            let rhs = expr_unary_not_operand_kind(&bin.right, ctx);
            if lhs == rhs {
                lhs
            } else {
                UnaryNotOperandKind::Unknown
            }
        }
        _ => UnaryNotOperandKind::Unknown,
    }
}

fn value_type_to_unary_not_kind(ty: &ValueType) -> UnaryNotOperandKind {
    match ty {
        ValueType::Bool => UnaryNotOperandKind::Bool,
        // `lltype.Unsigned`'s `UNARY_INVERT` dispatch routes through
        // the same `int_invert` opname as `lltype.Signed` — see
        // `flowspace/operation.py:521 invert.dispatch_to_register
        // _class('int')` and `rint.py:rtype_int__invert` which both
        // Signed and Unsigned IntegerRepr inherit.  `UnaryNotOperand
        // Kind::Int` here drives the bytecode dispatch decision; the
        // result-type carrier is computed from the operand's actual
        // lowered type at the emit site, so Unsigned operands stay
        // Unsigned through the rtyper.
        ValueType::Int | ValueType::Unsigned => UnaryNotOperandKind::Int,
        _ => UnaryNotOperandKind::Unknown,
    }
}

fn type_string_to_unary_not_kind(type_str: &str) -> UnaryNotOperandKind {
    let trimmed = type_str.trim();
    // Arbitrary-precision integer types — `BigInt` / `BigUint`. Routed
    // through `UNARY_INVERT` (bitwise NOT) like primitive integers,
    // even though their lattice lowering is `ValueType::Ref`. RPython
    // peer: `LongRepr.rtype_invert` (`rtyper/rlong.py:..`) dispatches
    // bigint invert at the rtyper layer; pyre's `OpKind::UnaryOp.
    // result_ty` is computed from the operand's actual lowered type
    // at the emit site (`front/ast.rs:3667 UnaryNotOperandKind::Int`
    // arm), so this kind only drives the bytecode dispatch decision,
    // not the result-type carrier.
    if matches!(trimmed, "BigInt" | "BigUint") {
        return UnaryNotOperandKind::Int;
    }
    value_type_to_unary_not_kind(&type_string_to_value_type(type_str))
}

/// Strip the outer `Result<_, _>` or `Option<_>` wrapper from a type
/// string and return the inner ok / some payload type.  Used by the
/// `.unwrap()` / `.expect()` projection in `expr_unary_not_operand_kind`
/// so a `let x = foo().unwrap()` whose `foo() -> Result<bool, _>`
/// classifies `x` as `Bool`, not `Unknown`.
fn unwrap_result_or_option(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    let inner = trimmed
        .strip_prefix("Result<")
        .or_else(|| trimmed.strip_prefix("Option<"))?
        .strip_suffix('>')?;
    // For `Result<T, E>` keep `T` (split on the top-level comma).  Track
    // angle-bracket depth so nested generics like `Result<Vec<T>, E>` /
    // `Option<Result<bool, _>>` survive the split.
    let mut depth = 0_i32;
    for (i, ch) in inner.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => return Some(inner[..i].trim()),
            _ => {}
        }
    }
    Some(inner.trim())
}

fn expr_is_statically_bool(expr: &syn::Expr, ctx: &GraphBuildContext) -> bool {
    expr_unary_not_operand_kind(expr, ctx) == UnaryNotOperandKind::Bool
}

fn binary_op_name(op: &syn::BinOp) -> &'static str {
    match op {
        syn::BinOp::Add(_) => "add",
        syn::BinOp::Sub(_) => "sub",
        syn::BinOp::Mul(_) => "mul",
        syn::BinOp::Div(_) => "div",
        syn::BinOp::Rem(_) => "mod",
        syn::BinOp::And(_) => "and",
        syn::BinOp::Or(_) => "or",
        syn::BinOp::BitXor(_) => "bitxor",
        syn::BinOp::BitAnd(_) => "bitand",
        syn::BinOp::BitOr(_) => "bitor",
        syn::BinOp::Shl(_) => "lshift",
        syn::BinOp::Shr(_) => "rshift",
        syn::BinOp::Eq(_) => "eq",
        syn::BinOp::Lt(_) => "lt",
        syn::BinOp::Le(_) => "le",
        syn::BinOp::Ne(_) => "ne",
        syn::BinOp::Ge(_) => "ge",
        syn::BinOp::Gt(_) => "gt",
        syn::BinOp::AddAssign(_) => "add_assign",
        syn::BinOp::SubAssign(_) => "sub_assign",
        syn::BinOp::MulAssign(_) => "mul_assign",
        syn::BinOp::DivAssign(_) => "div_assign",
        syn::BinOp::RemAssign(_) => "mod_assign",
        syn::BinOp::BitXorAssign(_) => "bitxor_assign",
        syn::BinOp::BitAndAssign(_) => "bitand_assign",
        syn::BinOp::BitOrAssign(_) => "bitor_assign",
        syn::BinOp::ShlAssign(_) => "lshift_assign",
        syn::BinOp::ShrAssign(_) => "rshift_assign",
        _ => "unknown_binop",
    }
}

/// Variable-keyed BinOp result type oracle.
///
/// RPython `flowspace/operation.py:505-510` registers `lt`, `le`,
/// `eq`, `ne`, `ge`, `gt` as 2-arg operators returning lltype.Bool;
/// the annotator stamps the result `SomeBool(SomeInteger)`
/// (`annotator/model.py:185-198` — distinct lattice node from
/// SomeInteger).  Pyre mirrors that with `ValueType::Bool`, which
/// `valuetype_to_someshell` projects to `SomeValue::Bool` and the
/// rtyper picks `BoolRepr` for (`rmodel.rs:2204`).  Downstream
/// jit_codewriter sites that key off `ValueType::Int` already alias
/// Bool to Int (commit 4318ebb51b2 added the 9 wildcard / explicit
/// arms — assembler getkind, call array-descr / ir_type, jtransform
/// stamp/kind/ir).
fn binary_result_value_type_var(
    graph: &FunctionGraph,
    lhs: &crate::flowspace::model::Variable,
    rhs: &crate::flowspace::model::Variable,
    op: &str,
) -> ValueType {
    if matches!(op, "eq" | "ne" | "lt" | "le" | "gt" | "ge") {
        return ValueType::Bool;
    }
    let lhs_ty = graph_value_type_var(graph, lhs);
    let rhs_ty = graph_value_type_var(graph, rhs);
    binary_result_value_type_inner(lhs_ty, rhs_ty, op)
}

fn binary_result_value_type_inner(
    lhs_ty: Option<ValueType>,
    rhs_ty: Option<ValueType>,
    op: &str,
) -> ValueType {
    match (lhs_ty, rhs_ty) {
        (Some(ValueType::Float), Some(ValueType::Float))
        | (Some(ValueType::Float), Some(ValueType::Int))
        | (Some(ValueType::Int), Some(ValueType::Float))
            if matches!(
                op,
                "add"
                    | "sub"
                    | "mul"
                    | "div"
                    | "mod"
                    | "add_assign"
                    | "sub_assign"
                    | "mul_assign"
                    | "div_assign"
                    | "mod_assign"
            ) =>
        {
            ValueType::Float
        }
        (Some(ValueType::Int), Some(ValueType::Int))
            if matches!(
                op,
                "add"
                    | "sub"
                    | "mul"
                    | "div"
                    | "mod"
                    | "bitand"
                    | "bitor"
                    | "bitxor"
                    | "lshift"
                    | "rshift"
                    | "add_assign"
                    | "sub_assign"
                    | "mul_assign"
                    | "div_assign"
                    | "mod_assign"
                    | "bitand_assign"
                    | "bitor_assign"
                    | "bitxor_assign"
                    | "lshift_assign"
                    | "rshift_assign"
            ) =>
        {
            ValueType::Int
        }
        _ => ValueType::Unknown,
    }
}

fn member_name(member: &syn::Member) -> String {
    match member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(idx) => idx.index.to_string(),
    }
}

/// RPython: direct_call carries the exact callee graph identity.
/// Qualify single-segment bare function names with module prefix so that
/// `helper()` inside `mod a` produces `["a", "helper"]`, matching the
/// registered graph path.
fn canonical_call_target(expr: &syn::Expr, ctx: &GraphBuildContext) -> CallTarget {
    match expr {
        syn::Expr::Path(path) => {
            let mut segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect();
            if is_synthetic_ctor_path(&segments) && !registered_function_path(&segments, ctx) {
                let last = segments
                    .last()
                    .expect("transparent ctor path is non-empty")
                    .clone();
                let owner_path = segments[..segments.len() - 1].to_vec();
                return CallTarget::synthetic_transparent_ctor_with_owner(owner_path, last);
            }
            if segments.len() == 1 {
                if let Some(full) = ctx.use_imports.get(&segments[0]) {
                    // LOAD_GLOBAL parity (`flowcontext.py:845-866`): a bare
                    // call name binds in the caller's lexical scope, where
                    // an imported name resolves to its import target (the
                    // callee's home path).  `walk_use_tree` (`parse.rs:738`)
                    // records each `use` item keyed by the bare alias, so
                    // resolving the import here lets a bare
                    // `items_block_items_base()` call reach the free
                    // function registered under its callee-home path
                    // instead of being mis-qualified with the caller's own
                    // module (or left as a bare name the free-function
                    // conflict guard never registered).  Mirrors the
                    // import-first ladder in `qualify_type_name_with_imports`.
                    segments = full.split("::").map(str::to_string).collect();
                } else if !ctx.module_prefix.is_empty() {
                    let mut qualified = ctx
                        .module_prefix
                        .split("::")
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>();
                    qualified.extend(segments);
                    segments = qualified;
                }
            } else if segments.len() == 2 {
                // `Type::method` — LOAD_GLOBAL <Type> resolves the class
                // OBJECT in the caller's lexical scope and LOAD_ATTR
                // <method> reaches the method's `FunctionDesc` by that
                // class identity (`flowcontext.py:845-866`,
                // `bookkeeper.py:353-396`), never by a global name scan.
                // Resolve the Type head the same way registration does
                // (`qualify_type_name_with_imports`, `parse.rs:1163`): a
                // `use a::S as T` alias resolves through `use_imports`
                // first, then the module prefix, so an aliased spelling
                // `T::m(...)` lands on the same key the impl registered.
                //
                // Do NOT feed every `A::b` head through
                // `qualify_type_name_with_imports`: that helper is type-name
                // specific and falls back to `module_prefix::A`, which would
                // corrupt module/root paths such as `crate::helper()` inside
                // `mod a` into `a::crate::helper`.  This path resolver only
                // rewrites when lexical/global evidence identifies the head
                // as an import alias, a known/canonical struct, or a
                // same-scope nested item.
                if let Some(qualified) =
                    qualify_two_segment_call_target(&segments[0], &segments[1], ctx)
                {
                    segments = qualified;
                }
            }
            CallTarget::function_path(segments)
        }
        _ => CallTarget::UnsupportedExpr,
    }
}

fn qualify_two_segment_call_target(
    head: &str,
    leaf: &str,
    ctx: &GraphBuildContext,
) -> Option<Vec<String>> {
    if is_absolute_or_internal_path_root(head) {
        return None;
    }
    if let Some(full) = ctx.use_imports.get(head) {
        let mut resolved: Vec<String> = full.split("::").map(str::to_string).collect();
        resolved.push(leaf.to_string());
        return Some(resolved);
    }
    let canonical = majit_ir::descr::canonical_struct_name(head);
    if canonical != head {
        let mut resolved: Vec<String> = canonical.split("::").map(str::to_string).collect();
        resolved.push(leaf.to_string());
        return Some(resolved);
    }
    if ctx.module_prefix.is_empty() {
        return None;
    }
    let prefixed = format!("{}::{}", ctx.module_prefix, head);
    let prefixed_key = format!("{}::{}", prefixed, leaf);
    if ctx.known_struct_names.contains(&prefixed) || ctx.fn_return_types.contains_key(&prefixed_key)
    {
        let mut resolved: Vec<String> = prefixed.split("::").map(str::to_string).collect();
        resolved.push(leaf.to_string());
        return Some(resolved);
    }
    None
}

fn is_absolute_or_internal_path_root(head: &str) -> bool {
    matches!(head, "crate" | "self" | "super") || crate::parse::PYRE_INTERNAL_CRATES.contains(&head)
}

/// Decide whether `segments` should be lowered as
/// `CallTarget::SyntheticTransparentCtor` instead of `FunctionPath`.
///
/// Two disjoint groups share the same routing decision but receive
/// different downstream treatment:
///
/// * **Result/Option elision whitelist** (`Ok`/`Err`/`Some`):
///   - flowspace_adapter routes via `HostObject::new_class(name, [])`
///     → `ClassDesc` → `SomeInstance(classdef)`.
///   - jtransform `is_synthetic_result_option_ctor` then re-elides the
///     single-arg call back to its inner value, matching upstream
///     `lloperation:cast_to_ptr` semantics for `Ok(x)` / `Err(e)` /
///     `Some(v)`.
///
/// * **Pyre-side `Class::Variant` ctors** (`LoopResult::Done` etc.):
///   - flowspace_adapter routes identically via
///     `HostObject::new_class(name, [])` → `SomeInstance(classdef)`.
///   - jtransform does **not** elide them (they are real values, not
///     transparent wrappers); the call lowers to a normal instance
///     construction.
///   - Upstream RPython gets the same shape by lifting each enum
///     variant as a Class with a registered `__init__` FunctionDesc;
///     pyre's surface DSL lacks that lifting pass today so the route
///     is gated explicitly until a `sym_enum_variant` analyser pass
///     lands.
///
/// Both groups must skip the `PyreCallRegistry` function-path lookup
/// because neither is registered there; the discriminator is the
/// caller's check at `canonical_call_target` that
/// `registered_function_path` returns false.
fn is_synthetic_ctor_path(segments: &[String]) -> bool {
    is_synthetic_result_option_wrapper_path(segments) || is_synthetic_unit_variant_path(segments)
}

/// `Ok`/`Err`/`Some` and their qualified spellings.  These are
/// always one-argument transparent wrappers that
/// `jtransform::is_synthetic_result_option_ctor` elides at the
/// `args.len() == 1` site.  Valid only as call targets — using the
/// bare ctor name as a value would lower to a 0-arg synthetic call
/// the elision pass does not handle.
fn is_synthetic_result_option_wrapper_path(segments: &[String]) -> bool {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    matches!(
        path.as_slice(),
        ["Ok"]
            | ["Err"]
            | ["Some"]
            | ["Result", "Ok"]
            | ["Result", "Err"]
            | ["Option", "Some"]
            | ["result", "Result", "Ok"]
            | ["result", "Result", "Err"]
            | ["option", "Option", "Some"]
            | ["std", "result", "Result", "Ok"]
            | ["std", "result", "Result", "Err"]
            | ["std", "option", "Option", "Some"]
            | ["core", "result", "Result", "Ok"]
            | ["core", "result", "Result", "Err"]
            | ["core", "option", "Option", "Some"]
    )
}

/// Path-as-value numeric constants — Rust counterpart of PyPy
/// `flowspace.LOAD_GLOBAL` resolving a statically-known module
/// attribute to a `Constant(value)` node.  Returns the `f64` bit
/// pattern matching the `syn::Lit::Float` arm's
/// `OpKind::ConstFloat(v.to_bits())` form.
///
/// Only `f64::{INFINITY, NEG_INFINITY, NAN}` and their long forms
/// `{std,core}::f64::*` are recognised here; other numeric path
/// constants would each need their own typed `Const*` op (the
/// `OpKind::ConstFloat` variant is float-specific).
fn path_as_value_float_constant(segments: &[String]) -> Option<u64> {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    let leaf = match path.as_slice() {
        ["f64", leaf] => leaf,
        ["std", "f64", leaf] => leaf,
        ["core", "f64", leaf] => leaf,
        _ => return None,
    };
    match *leaf {
        "INFINITY" => Some(f64::INFINITY.to_bits()),
        "NEG_INFINITY" => Some(f64::NEG_INFINITY.to_bits()),
        "NAN" => Some(f64::NAN.to_bits()),
        _ => None,
    }
}

/// Resolve a path expression against the program-wide
/// `pub const` / `pub static` table on `ctx.module_statics` and
/// return its compile-time literal value when known.  Mirrors how
/// `qualify_type_name_with_imports` resolves a bare type name —
/// the same three-step ladder: explicit `::`-qualified path,
/// `use`-alias lookup, same-file `(source_module, name)` lookup.
///
/// PyPy parity: `flowcontext.py:845-866` LOAD_GLOBAL only consults
/// the defining function's `frame.globals` (the file's module
/// globals) before falling through to builtins.  The bare-name
/// fallback is therefore scoped to the file's `source_module`; no
/// program-wide "unique leaf name" rule (which would let one file's
/// `pub const FOO` resolve in an unrelated file by accident).
///
/// Returns `None` when the path does not resolve to a known
/// file-root decl, or when it resolves to one whose initialiser
/// is not a primitive literal (e.g. `INT_TYPE = new_pytype("int")`).
fn lookup_module_static_literal(
    segments: &[String],
    ctx: &GraphBuildContext,
) -> Option<crate::parse::ModuleStaticLiteral> {
    if segments.is_empty() {
        return None;
    }
    let leaf = segments.last().unwrap().clone();
    if segments.len() >= 2 {
        // RPython LOAD_GLOBAL + LOAD_ATTR: resolve the root segment
        // through use-imports first, then try direct / stripped / relative.
        let resolved_segments: Vec<String> = if let Some(full) = ctx.use_imports.get(&segments[0]) {
            let mut resolved: Vec<String> = full.split("::").map(String::from).collect();
            resolved.extend_from_slice(&segments[1..]);
            resolved
        } else {
            segments.to_vec()
        };
        let module = resolved_segments[..resolved_segments.len() - 1].join("::");
        let resolved_leaf = resolved_segments.last().unwrap().clone();
        if let Some(decl) = ctx
            .module_statics
            .get(&(module.clone(), resolved_leaf.clone()))
        {
            return decl.literal;
        }
        if resolved_segments[0] == "crate"
            || crate::parse::PYRE_INTERNAL_CRATES.contains(&resolved_segments[0].as_str())
        {
            let stripped_module = resolved_segments[1..resolved_segments.len() - 1].join("::");
            if let Some(decl) = ctx
                .module_statics
                .get(&(stripped_module, resolved_leaf.clone()))
            {
                return decl.literal;
            }
        }
        let current_module = qualify_module_path(&ctx.source_module, &ctx.module_prefix);
        if !current_module.is_empty() {
            let qualified = qualify_module_path(&current_module, &module);
            if let Some(decl) = ctx.module_statics.get(&(qualified, resolved_leaf.clone())) {
                return decl.literal;
            }
        }
        return None;
    }
    // Single-segment: try `use`-alias first (RPython parity with
    // `qualify_type_name_with_imports`).
    if let Some(full) = ctx.use_imports.get(&leaf) {
        if let Some(idx) = full.rfind("::") {
            let module = full[..idx].to_string();
            let name = full[idx + 2..].to_string();
            if let Some(decl) = ctx.module_statics.get(&(module, name)) {
                return decl.literal;
            }
        }
    }
    if !ctx.module_prefix.is_empty() {
        let qualified = qualify_module_path(&ctx.source_module, &ctx.module_prefix);
        if let Some(decl) = ctx.module_statics.get(&(qualified, leaf.clone())) {
            return decl.literal;
        }
    }
    // File-level same-module fallback: bare name resolves against
    // the *file's* `source_module` (PyPy `frame.globals` parity).
    // The production pipeline always populates this via
    // `with_source_module`; test helpers pass `""` and therefore
    // skip this fallback.
    if !ctx.source_module.is_empty() {
        if let Some(decl) = ctx
            .module_statics
            .get(&(ctx.source_module.clone(), leaf.clone()))
        {
            return decl.literal;
        }
    }
    None
}

/// Pyre-side `Class::Variant` unit-variant ctors.  These are valid
/// as bare path-expression values; `flowspace_adapter` pre-folds them
/// to `Hlvalue::Constant(ConstValue::HostObject(prebuilt_instance))`
/// before the rtyper sees a call (mirrors PyPy `rtyper` resolving
/// `SomePBC([InstanceDesc(<unit-variant>)])` to a singleton constant
/// before `jtransform`).  Exposed `pub(crate)` so
/// `translator::rtyper::flowspace_adapter::is_synthetic_unit_variant_call`
/// reads the same allowlist.
pub(crate) fn is_synthetic_unit_variant_path(segments: &[String]) -> bool {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    matches!(
        path.as_slice(),
        ["LoopResult", "Done"]
            | ["LoopResult", "ContinueRunningNormally"]
            | ["JitAction", "Return"]
            | ["JitAction", "Continue"]
            | ["StepResult", "Continue"]
            | ["CompareOp", "Lt"]
            | ["CompareOp", "Le"]
            | ["CompareOp", "Gt"]
            | ["CompareOp", "Ge"]
            | ["CompareOp", "Eq"]
            | ["CompareOp", "Ne"]
    )
}

fn registered_function_path(segments: &[String], ctx: &GraphBuildContext) -> bool {
    let unqualified = segments.join("::");
    if ctx.fn_return_types.contains_key(&unqualified) {
        return true;
    }
    if segments.len() == 1 && !ctx.module_prefix.is_empty() {
        let qualified = format!("{}::{}", ctx.module_prefix, segments[0]);
        return ctx.fn_return_types.contains_key(&qualified);
    }
    false
}

fn receiver_type_root(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_type_roots.get(&ident.to_string()).cloned()),
        syn::Expr::Cast(cast) => type_root_ident(&cast.ty).map(|root| {
            qualify_type_name_with_imports(&root, &ctx.module_prefix, &ctx.use_imports)
        }),
        syn::Expr::Reference(reference) => receiver_type_root(&reference.expr, ctx),
        syn::Expr::Paren(paren) => receiver_type_root(&paren.expr, ctx),
        syn::Expr::Unary(unary) => match &unary.op {
            syn::UnOp::Deref(_) => receiver_type_root(&unary.expr, ctx),
            _ => None,
        },
        syn::Expr::Field(field) => receiver_type_root(&field.base, ctx),
        syn::Expr::Index(index) => receiver_type_root(&index.expr, ctx),
        // Chained `foo().bar()` — derive the receiver root from the
        // call's registered return type so `lookup_method_return_type`
        // can resolve `.bar`'s declared signature.  Trait-object
        // returns (`-> &dyn T` / `-> Box<dyn T>`) surface as the trait
        // name; plain `-> Bar` / `-> &mut Bar` surface as `Bar`.
        syn::Expr::Call(call) => {
            if let syn::Expr::Path(p) = &*call.func {
                let key = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                let ret = ctx.fn_return_types.get(&key)?;
                return dyn_trait_root_from_type_str(ret)
                    .or_else(|| bare_type_root_from_type_str(ret));
            }
            None
        }
        // Chained `x.foo().bar()` — same as `Expr::Call` but the
        // callee key carries the receiver root prefix so the
        // `fn_return_types` lookup matches the impl-block / trait
        // registration.
        syn::Expr::MethodCall(mc) => {
            let owner = receiver_type_root(&mc.receiver, ctx)?;
            let key = format!("{}::{}", owner, mc.method);
            let ret = ctx.fn_return_types.get(&key)?;
            dyn_trait_root_from_type_str(ret).or_else(|| bare_type_root_from_type_str(ret))
        }
        _ => None,
    }
}

/// Strip leading `&` / `&mut` / lifetime / `Box<>` / `Rc<>` / `Arc<>`
/// from a type-string and return the bare type root identifier (the
/// first path segment of the innermost type).  Returns `None` for
/// `dyn Trait`-shaped strings — callers handle those via
/// [`dyn_trait_root_from_type_str`] first.
fn bare_type_root_from_type_str(s: &str) -> Option<String> {
    let mut trimmed = s.trim();
    // Drop leading `&` / `&mut` / lifetime annotations.
    while let Some(rest) = trimmed.strip_prefix('&') {
        let rest = rest.trim_start();
        let rest = rest.strip_prefix("mut ").unwrap_or(rest).trim_start();
        let rest = if let Some(after_quote) = rest.strip_prefix('\'') {
            let rest = after_quote
                .split_once(char::is_whitespace)
                .map(|(_lt, tail)| tail.trim_start())
                .unwrap_or("");
            rest
        } else {
            rest
        };
        trimmed = rest;
    }
    if trimmed.starts_with("dyn ") {
        return None;
    }
    for wrapper in ["Box", "Rc", "Arc"] {
        let prefix = format!("{wrapper}<");
        if let Some(rest) = trimmed.strip_prefix(prefix.as_str())
            && let Some(inner) = rest.strip_suffix('>')
        {
            return bare_type_root_from_type_str(inner);
        }
    }
    // `Vec<T>` / `Option<T>` / generic containers do not map to a
    // method-call receiver root.
    if trimmed.contains('<') {
        return None;
    }
    let leaf = trimmed.split_whitespace().next()?;
    if leaf.is_empty() {
        None
    } else {
        Some(leaf.to_string())
    }
}

/// Variable-direct lookup that walks the op-result chain first then the
/// link-arg unification fold (`graph_result_value_type_var` →
/// `graph_link_input_value_type_var`).  No slot projection; the
/// upstream-orthodox carrier is `op.result: Variable` and
/// `inputarg == Variable` identity-compare per `flowspace/model.py:140`.
fn graph_value_type_var(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> Option<ValueType> {
    graph_result_value_type_var(graph, var).or_else(|| graph_link_input_value_type_var(graph, var))
}

fn retag_result_value_type(
    graph: &mut FunctionGraph,
    target_var: &crate::flowspace::model::Variable,
    ty: ValueType,
) {
    for block in &mut graph.blocks {
        for op in &mut block.operations {
            if op.result.as_ref() != Some(target_var) {
                continue;
            }
            match &mut op.kind {
                OpKind::Input { ty: result_ty, .. }
                | OpKind::FieldRead { ty: result_ty, .. }
                | OpKind::VableFieldRead { ty: result_ty, .. }
                | OpKind::BinOp { result_ty, .. }
                | OpKind::UnaryOp { result_ty, .. }
                | OpKind::Call { result_ty, .. }
                | OpKind::IndirectCall { result_ty, .. } => *result_ty = ty,
                OpKind::ArrayRead { item_ty, .. }
                | OpKind::InteriorFieldRead { item_ty, .. }
                | OpKind::VableArrayRead { item_ty, .. } => *item_ty = ty,
                _ => {}
            }
            return;
        }
    }
}

/// Link-arg unification loop driven by Variable identity (the
/// upstream-orthodox carrier per `flowspace/model.py:140`).  Walks
/// every block's `inputargs` to find one matching `target_var`, then
/// folds every predecessor link's matching arg position via
/// `graph_result_value_type_var`/`const_value_value_type` and returns
/// the unified `ValueType` if every contributor agrees.
fn graph_link_input_value_type_var(
    graph: &FunctionGraph,
    target_var: &crate::flowspace::model::Variable,
) -> Option<ValueType> {
    for target_block in &graph.blocks {
        let Some(arg_index) = target_block
            .inputargs
            .iter()
            .position(|inputarg| inputarg == target_var)
        else {
            continue;
        };
        let mut inferred: Option<ValueType> = None;
        for predecessor in &graph.blocks {
            for link in &predecessor.exits {
                if link.target != target_block.id {
                    continue;
                }
                let source_ty = match link.args.get(arg_index)? {
                    arg @ LinkArg::Value(_) => {
                        let Some(source_var) = arg.as_variable() else {
                            continue;
                        };
                        match graph_result_value_type_var(graph, source_var) {
                            Some(ty) => ty,
                            None => continue,
                        }
                    }
                    // RPython `flowspace/model.py:Constant.concretetype`
                    // — `Link.args` may carry constants whose lltype is
                    // determined by the constant's Python class; the
                    // inputarg's concretetype is unified across all
                    // predecessor links the same way variable sources
                    // are.  Skipping constants leaves the inputarg
                    // Unknown, which the rtyper backfills with GcRef
                    // and forces synthetic casts at int/float
                    // operations downstream.
                    LinkArg::Const(c) => match const_value_value_type(&c.value) {
                        Some(ty) => ty,
                        None => continue,
                    },
                };
                match &inferred {
                    None => inferred = Some(source_ty),
                    Some(existing) if *existing == source_ty => {}
                    Some(_) => return None,
                }
            }
        }
        if inferred.is_some() {
            return inferred;
        }
    }
    None
}

/// RPython `flowspace/model.py:Constant.concretetype` — map a Python
/// constant value to its lltype kind.  Used by
/// `graph_link_input_value_type` to infer phi-input concretetype from
/// constant link args.  `Placeholder` is unmaterialised by definition
/// and never appears in production link args.
fn const_value_value_type(c: &ConstValue) -> Option<ValueType> {
    match c {
        ConstValue::Int(_) | ConstValue::AddressOffset(_) => Some(ValueType::Int),
        // RPython annotates `Constant(True)`/`Constant(False)` with
        // `SomeBool(SomeInteger)` (`annotator/model.py:185-227`); the
        // rtyper picks `BoolRepr` and `getkind(lltype.Bool) == 'int'`
        // so the register class merges with Int downstream.  The
        // annotation-stage type is Bool — not Int — so propagate Bool
        // here to keep the lattice node distinct.
        ConstValue::Bool(_) => Some(ValueType::Bool),
        ConstValue::Float(_) => Some(ValueType::Float),
        // GC-managed Python objects → `lltype.Ptr(GcStruct)` (Ref bank).
        ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::None
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Dict(_)
        | ConstValue::Code(_)
        | ConstValue::Function(_)
        | ConstValue::Graphs(_)
        | ConstValue::Atom(_)
        | ConstValue::LLPtr(_)
        | ConstValue::HostObject(_) => Some(ValueType::Ref(None)),
        // `LowLevelType` constants are `lltype.Void` carriers (the
        // value IS a TYPE object); RPython flow `Constant(TYPE,
        // lltype.Void)` — Void register class.
        ConstValue::LowLevelType(_) => Some(ValueType::Void),
        // `_address` is RPython's `Address` lowleveltype — distinct
        // from GcRef and Signed.  Pyre has no Address bank, but the
        // value is a raw pointer-sized integer in practice; conservative
        // None lets the rtyper Unknown→GcRef fallback handle it as it
        // does today.
        ConstValue::LLAddress(_) => None,
        // SpecTag identity carrier — never feeds an int/float/ref op.
        ConstValue::SpecTag(_) => None,
        ConstValue::Placeholder => None,
    }
}

/// Op-result scan driven by Variable identity (`op.result == Some(var)`
/// across every block's operations).  Returns the producing op's
/// declared `ValueType` via [`op_result_value_type`].
fn graph_result_value_type_var(
    graph: &FunctionGraph,
    target_var: &crate::flowspace::model::Variable,
) -> Option<ValueType> {
    graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .find_map(|op| {
            if op.result.as_ref() == Some(target_var) {
                op_result_value_type(&op.kind)
            } else {
                None
            }
        })
}

fn op_result_value_type(kind: &OpKind) -> Option<ValueType> {
    match kind {
        OpKind::Input { ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::VableFieldRead { ty, .. }
        | OpKind::BinOp { result_ty: ty, .. }
        | OpKind::UnaryOp { result_ty: ty, .. }
        | OpKind::Call { result_ty: ty, .. }
        | OpKind::IndirectCall { result_ty: ty, .. } => {
            if *ty == ValueType::Unknown {
                None
            } else {
                Some(ty.clone())
            }
        }
        OpKind::ConstInt(_) | OpKind::VtableMethodPtr { .. } | OpKind::CurrentTraceLength => {
            Some(ValueType::Int)
        }
        OpKind::ConstFloat(_) => Some(ValueType::Float),
        OpKind::ConstBool(_) => Some(ValueType::Bool),
        OpKind::ConstRef(_) | OpKind::ConstRefNull | OpKind::ConstRefAddr(_) => {
            Some(ValueType::Ref(None))
        }
        OpKind::ArrayRead { item_ty, .. }
        | OpKind::InteriorFieldRead { item_ty, .. }
        | OpKind::VableArrayRead { item_ty, .. } => {
            if *item_ty == ValueType::Unknown {
                None
            } else {
                Some(item_ty.clone())
            }
        }
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => kind_char_to_value_type(*result_kind),
        OpKind::IsConstant { .. } | OpKind::IsVirtual { .. } => Some(ValueType::Int),
        _ => None,
    }
}

fn transparent_option_method_result_type(
    graph: &FunctionGraph,
    args: &[crate::flowspace::model::Variable],
    method: &syn::Ident,
) -> Option<ValueType> {
    match method.to_string().as_str() {
        // Rust `usize`/`*const T::len` etc — RPython `lltype.Signed`.
        "as_usize" | "len" | "wrapping_mul" => Some(ValueType::Int),
        // Bool-returning predicates: RPython `SomeBool` (`annotator/
        // model.py:185-198`). Was `Int` until the Bool lattice landed
        // (`model.rs:18-42`); split out so the call result reaches
        // downstream `valuetype_to_someshell` as `SomeBool` instead of
        // `SomeInteger`.
        "is_empty" | "is_null" => Some(ValueType::Bool),
        "unwrap_or" => args
            .get(1)
            .and_then(|default| graph_value_type_var(graph, default)),
        _ => None,
    }
}

/// Inference for Rust `f64`/i64 method calls used as low-level
/// floating-point/integer helpers in pyre's port of
/// `pypy/objspace/std/floatobject.py`.
///
/// **Structural adaptation (parity rule §1):** RPython spells the
/// same low-level operations as `r_float`/`r_uint`/`r_longlong`
/// helpers (`rpython/rlib/rfloat.py`, `rpython/rlib/rarithmetic.py`),
/// not as method calls.  The Rust port relays them through `f64::*`
/// and `i64::*` because that is the only way to express the same
/// arithmetic in stable Rust.  The receiver-typed match below stamps
/// the result class so jtransform sees the same concretetype the
/// rtyper would have written onto a Variable for an `r_float` /
/// `r_uint` operand.
///
/// This mapping is for the LOW-LEVEL helpers only — it does NOT
/// translate Python-level `float.__floor__` / `int.__abs__`.  The
/// PyPy descriptors (`pypy/objspace/std/floatobject.py:descr_floor`)
/// box-return `int`, whereas Rust `f64::floor` returns `f64`.
/// Callers must already have lowered to the helper level (e.g.
/// `floatobject.py` body, not the unboxed Python descriptor) before
/// this inference applies.
fn primitive_method_result_type(
    graph: &FunctionGraph,
    args: &[crate::flowspace::model::Variable],
    method: &syn::Ident,
) -> Option<ValueType> {
    let receiver = args
        .first()
        .and_then(|recv| graph_value_type_var(graph, recv))?;
    match (receiver, method.to_string().as_str()) {
        (ValueType::Float, "abs" | "floor" | "ceil" | "trunc" | "round" | "sqrt" | "powf") => {
            Some(ValueType::Float)
        }
        // Float bool predicates: RPython `SomeBool` (`annotator/
        // model.py:185-198`) — `math.isnan` / `math.isinf` etc lower
        // through `rfloat.py` and surface as `Bool` annotations.
        (ValueType::Float, "is_nan" | "is_infinite" | "is_finite" | "is_sign_negative") => {
            Some(ValueType::Bool)
        }
        (
            ValueType::Int,
            "abs" | "wrapping_abs" | "wrapping_mul" | "wrapping_add" | "wrapping_sub",
        ) => Some(ValueType::Int),
        _ => None,
    }
}

fn intrinsic_call_result_type(segments: &[String]) -> Option<ValueType> {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    match path.as_slice() {
        // Rust's `size_of::<T>()` is a compile-time `usize`.  RPython
        // carries such layout sizes as `lltype.Signed` constants before
        // codewriter, so the equivalent register class is Int.
        ["std", "mem", "size_of"] | ["core", "mem", "size_of"] | ["mem", "size_of"] => {
            Some(ValueType::Int)
        }
        // Associated f64 helpers used by the floatobject.py port.
        ["f64", "copysign"] | ["std", "f64", "copysign"] | ["core", "f64", "copysign"] => {
            Some(ValueType::Float)
        }
        _ => None,
    }
}

fn kind_char_to_value_type(kind: char) -> Option<ValueType> {
    match kind {
        'i' => Some(ValueType::Int),
        'r' => Some(ValueType::Ref(None)),
        'f' => Some(ValueType::Float),
        'v' => Some(ValueType::Void),
        _ => None,
    }
}

/// RPython equivalent of Rust's `expr as T` for numeric / Bool /
/// pointer casts.
///
/// Upstream has no `as T` syntax — numeric conversions are spelled as
/// builtin calls `int(v)` / `float(v)` / `bool(v)` (`rbuiltin.py:178-189`),
/// each of which the rtyper routes through
/// `BuiltinFunctionRepr.rtype_simple_call → rtype_builtin_int/float/bool
/// → arg_repr.rtype_int/float/bool` per `Repr` trait method.  Those
/// per-Repr methods (rint.py:137-147, rfloat.py:48-58, rbool.py:55-70)
/// emit the low-level `cast_*_to_*` / `*_is_true` op as a side effect
/// of `inputargs(target_lltype)` coercion through the appropriate
/// `pair_X_Y_convert_from_to` (rint.py:645-675, rbool.py:49-84).
///
/// Pointer↔int conversions go through `lltype.cast_ptr_to_int(p)` /
/// `lltype.cast_int_to_ptr(T, v_int)` (`rbuiltin.py:543-557`), routed
/// through the same `BUILTIN_TYPER` registry via the module-qualified
/// HOST_ENV resolver in `flowspace_adapter`.
///
/// Maps each `(source, target)` pair to the matching callable path —
/// a single `__builtin__` name (one segment) for numeric/Bool casts,
/// or a fully qualified `["rpython", "rtyper", "lltypesystem",
/// "lltype", <attr>]` / `["rpython", "rlib", "rarithmetic", <attr>]`
/// path for pointer / Unsigned-bridging casts.  Pairs not covered
/// (identity casts, source-type-unknown casts) return `None`, letting
/// the caller emit `OpKind::UnaryOp { op: "same_as", result_ty, .. }`
/// to preserve target-type propagation through the graph
/// (rtyper.py:478-481 internal renaming op).  All 13 typed numeric /
/// ptr / Unsigned cast names retired across Slices A.3 / B.1 / A.4a /
/// A.4b / A.4c.
fn cast_builtin_name(
    source_ty: Option<&ValueType>,
    target_ty: &ValueType,
) -> Option<&'static [&'static str]> {
    match (source_ty, target_ty) {
        // upstream `float(v)`:
        //   `IntegerRepr.rtype_float` (rint.py:144-147) — `inputargs(Float)`
        //   coerces via `pair(IntegerRepr, FloatRepr).convert_from_to`
        //   (rint.py:645-655) → `genop('cast_int_to_float', ...)`.
        //   `BoolRepr.rtype_float` (rbool.py:65-70) — `inputargs(Float)`
        //   coerces via `pair(BoolRepr, FloatRepr).convert_from_to`
        //   (rbool.py:49-56) → `genop('cast_bool_to_float', ...)`.
        (Some(ValueType::Int), ValueType::Float) => Some(&["float"]),
        (Some(ValueType::Bool), ValueType::Float) => Some(&["float"]),
        // `IntegerRepr.rtype_float` (rint.py:144-147) with `self`
        // having lltype `Unsigned` — `inputargs(Float)` coerces via
        // `pair(IntegerRepr, FloatRepr).convert_from_to`
        // (rint.py:645-655) and the `r_from.lowleveltype == Unsigned`
        // arm emits `genop('cast_uint_to_float', ...)`.
        (Some(ValueType::Unsigned), ValueType::Float) => Some(&["float"]),
        // upstream `int(v)`:
        //   `FloatRepr.rtype_int` (rfloat.py:48-53) —
        //   `genop('cast_float_to_int', ...)`.
        //   `BoolRepr.rtype_int` (rbool.py:55-60) — `inputargs(Signed)`
        //   coerces via `pair(BoolRepr, IntegerRepr).convert_from_to`
        //   (rbool.py:73-78) → `genop('cast_bool_to_int', ...)`.
        (Some(ValueType::Float), ValueType::Int) => Some(&["int"]),
        (Some(ValueType::Bool), ValueType::Int) => Some(&["int"]),
        // upstream `intmask(r_uint(...))` (rbuiltin.py:220-225) —
        // `IntegerRepr.rtype_int` (rint.py:137-142) forbids `int(v)`
        // for `r_uint`, so the canonical Unsigned→Signed entry is
        // `rarithmetic.intmask(v_uint)`.  `rtype_intmask` calls
        // `hop.inputargs(Signed)` which coerces from Unsigned via
        // `pair(IntegerRepr, IntegerRepr).convert_from_to`
        // (rint.py:202-213) emitting `cast_uint_to_int`.
        (Some(ValueType::Unsigned), ValueType::Int) => {
            Some(&["rpython", "rlib", "rarithmetic", "intmask"])
        }
        // upstream `bool(v)`:
        //   `IntegerRepr.rtype_bool` (rint.py:85-88) —
        //   `genop(self.opprefix + 'is_true', ...)`.
        //   `FloatRepr.rtype_bool` (rfloat.py:32-34) —
        //   `genop('float_is_true', ...)`.
        (Some(ValueType::Int), ValueType::Bool) => Some(&["bool"]),
        (Some(ValueType::Float), ValueType::Bool) => Some(&["bool"]),
        // `IntegerRepr.rtype_bool` (rint.py:85-88) — `genop(self.\
        // opprefix + 'is_true', ...)` — emits `uint_is_true` for
        // Unsigned-prefixed Repr.
        (Some(ValueType::Unsigned), ValueType::Bool) => Some(&["bool"]),
        // upstream `lltype.cast_ptr_to_int(p)` (rbuiltin.py:543-548) —
        //   `genop('cast_ptr_to_int', vlist, resulttype=Signed)`.
        // upstream `lltype.cast_int_to_ptr(T, v_int)`
        // (rbuiltin.py:551-557) — `genop('cast_int_to_ptr', [v_input],
        //   resulttype=hop.r_result.lowleveltype)`.
        //
        // TODO: the upstream surface is
        // a 2-arg call `simple_call(lltype.cast_int_to_ptr, PTRTYPE,
        // oddint)` where `PTRTYPE` is a constant Ptr type marker
        // (`ann_cast_int_to_ptr` asserts `PtrT.is_constant()` and
        // returns `SomePtr(ll_ptrtype=PtrT.const)`,
        // lltype.py:2379-2382).  Pyre emits a 1-arg call here.
        //
        // The constant carrier exists — `ConstValue::LowLevelType(Box<\
        // LowLevelType>)` (flowspace/model.rs:1958) wraps any lltype
        // including `LowLevelType::Ptr(Box<Ptr>)` — but the **frontend**
        // does not yet have the concrete Ptr lltype at cast-lowering
        // time.  `Expr::Cast` carries `ValueType::Ref` (the opaque
        // high-level surface type); the concrete Ptr is computed only
        // later by the rtyper from the result variable's annotation.
        // The result lltype is recovered from `hop.r_result.\
        // lowleveltype` at rtype-time — the same Ptr upstream would
        // have read from `PtrT.const`.  Graduating to the 2-arg form
        // is blocked on threading the concrete result Ptr from the
        // rtyper-side annotation back to the frontend (or a deferred
        // 2-arg rewrite in jtransform / annotator after the Ptr is
        // known).
        (Some(ValueType::Ref(_)), ValueType::Int) => Some(&[
            "rpython",
            "rtyper",
            "lltypesystem",
            "lltype",
            "cast_ptr_to_int",
        ]),
        (Some(ValueType::Int), ValueType::Ref(_)) => Some(&[
            "rpython",
            "rtyper",
            "lltypesystem",
            "lltype",
            "cast_int_to_ptr",
        ]),
        // upstream `r_uint(v)` (rarithmetic.py:600
        // `build_int('r_uint', False, LONG_BIT)`) → the
        // `ForTypeEntry.specialize_call` body emits
        // `inputargs(Unsigned)` which coerces the source via
        // `pair(SrcRepr, IntegerRepr<Unsigned>).convert_from_to`
        // (rint.py:202-213 / rint.py:657-675 / rbool.py:62-66)
        // emitting `cast_int_to_uint` / `cast_float_to_uint` /
        // `cast_bool_to_uint`.
        (Some(ValueType::Int), ValueType::Unsigned)
        | (Some(ValueType::Float), ValueType::Unsigned)
        | (Some(ValueType::Bool), ValueType::Unsigned) => {
            Some(&["rpython", "rlib", "rarithmetic", "r_uint"])
        }
        _ => None,
    }
}

/// Maps a primitive numeric type name to its `ValueType` for the
/// `<prim>::from` coercion path.  Mirrors the primitive arms of
/// `classify_fn_arg_ty` (u* → `Unsigned` so `cast_builtin_name` selects
/// the `r_uint` / `intmask` bridges, not the int identity).  Returns
/// `None` for non-primitive receivers so `Foo::from` stays an ordinary
/// call.
fn numeric_type_name_to_value_type(name: &str) -> Option<ValueType> {
    match name {
        "i8" | "i16" | "i32" | "i64" | "isize" => Some(ValueType::Int),
        "u8" | "u16" | "u32" | "u64" | "usize" => Some(ValueType::Unsigned),
        "f32" | "f64" => Some(ValueType::Float),
        _ => None,
    }
}

/// Recognizes `<prim>::from` — the infallible numeric widening
/// conversion.  Returns the target `ValueType` when `path` is exactly
/// `<primitive numeric type>::from`, else `None`.  `try_from` is
/// excluded: its `Result<T, E>` result needs the Ok-elision / unwrap
/// composition handled elsewhere.
fn numeric_from_target_type(path: &syn::Path) -> Option<ValueType> {
    let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    match segments.as_slice() {
        [ty, method] if method.as_str() == "from" => numeric_type_name_to_value_type(ty),
        _ => None,
    }
}

/// Recognizes `std::ptr::eq` / `core::ptr::eq` / `ptr::eq` — pointer
/// identity comparison.  RPython spells this as `a is b`, lowered to
/// `ptr_eq` (rmodel.py:300 rtype_is_); pyre produces the same op from a
/// `BinOp { op: "eq" }` on two Ref operands, which jtransform rewrites
/// Ref==Ref → `ptr_eq` (jtransform.rs:849 / jtransform.py:1243
/// rewrite_op_ptr_eq).
fn is_ptr_eq_path(path: &syn::Path) -> bool {
    let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    match segments.as_slice() {
        [a, b, c] => {
            matches!(a.as_str(), "std" | "core") && b.as_str() == "ptr" && c.as_str() == "eq"
        }
        [a, b] => a.as_str() == "ptr" && b.as_str() == "eq",
        _ => false,
    }
}

/// Lowers a numeric / Bool / pointer value cast of `operand_var` (whose
/// source annotation is `source_ty`) to `result_ty`.  Routes through the
/// canonical `int()` / `float()` / `bool()` / `cast_*` builtin-call
/// chain (`cast_builtin_name`, rbuiltin.py:178-189) when a typed
/// coercion is required, else emits the identity-preserving `same_as`
/// (rtyper.py:478-481).  Shared by `Expr::Cast` (`x as T`) and the
/// `<prim>::from(x)` function-call spelling, which RPython expresses
/// with the same `int(v)` / `r_uint(v)` builtin calls.
fn lower_value_cast(
    graph: &mut FunctionGraph,
    block: BlockId,
    operand_var: crate::flowspace::model::Variable,
    source_ty: Option<ValueType>,
    result_ty: ValueType,
) -> crate::flowspace::model::Variable {
    if let Some(segments) = cast_builtin_name(source_ty.as_ref(), &result_ty) {
        return graph
            .push_op_var(
                block,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: segments.iter().map(|s| s.to_string()).collect(),
                    },
                    args: vec![operand_var],
                    result_ty,
                },
                true,
            )
            .expect("OpKind::Call has has_result=true");
    }
    graph
        .push_op_var(
            block,
            OpKind::UnaryOp {
                op: "same_as".to_string(),
                operand: operand_var,
                result_ty,
            },
            true,
        )
        .expect("OpKind::UnaryOp has has_result=true")
}

// `cast_op_name` retired in Slice C.1 — after the numeric / Bool /
// pointer / Unsigned cast families all routed through `cast_builtin_\
// name → simple_call(<host_callable>, v)` (Slices A.3 / B.1 /
// A.4a-c), the only remaining arm was the `_ => "same_as"` catch-all
// for identity / source-type-unknown casts.  Slice F3 restored the
// `same_as` fallback inside `Expr::Cast` directly: when
// `cast_builtin_name` returns None, the lowering emits
// `OpKind::UnaryOp { op: "same_as", result_ty, .. }` so the target
// `ValueType` propagates through the graph (`graph_value_type` reads
// the producer op's result_ty).  Downstream
// `rbuiltin::rtype_same_as` (verbatim port of `rtyper.py:478-481`)
// preserves the operand's lltype; `backendopt::removenoops::remove_\
// same_as` (removenoops.py:47-48) collapses the identity copy when
// safe.

fn lookup_method_return_type<'a>(
    ctx: &'a GraphBuildContext,
    receiver_root: Option<&str>,
    method: &syn::Ident,
) -> Option<&'a String> {
    let receiver_root = receiver_root?;
    let exact = format!("{}::{}", receiver_root, method);
    if let Some(ret) = ctx.fn_return_types.get(&exact) {
        return Some(ret);
    }

    // Rust imports can make the call-site owner path shorter or longer
    // than the impl key. Use the leaf owner only when it is unambiguous.
    let method_name = method.to_string();
    let receiver_leaf = receiver_root.rsplit("::").next().unwrap_or(receiver_root);
    let key = ctx
        .method_suffix_index
        .unique_key(receiver_leaf, &method_name)?;
    ctx.fn_return_types.get(key)
}

/// Extract the trait root from a type-string when the type is a trait
/// object — direct (`"dyn Foo"`) or wrapped (`"Box<dyn Foo>"`,
/// `"Rc<dyn Foo>"`, `"Arc<dyn Foo>"`).  The trailing `+ 'a` lifetime
/// bound is stripped.  Returns `None` for non-dyn types.
fn dyn_trait_root_from_type_str(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("dyn ") {
        // `dyn Trait + 'a` — drop everything after the first `+`.
        let head = rest.split('+').next()?.trim();
        if head.is_empty() {
            return None;
        }
        return Some(head.to_string());
    }
    for wrapper in ["Box", "Rc", "Arc"] {
        let prefix = format!("{wrapper}<");
        if let Some(rest) = trimmed.strip_prefix(prefix.as_str())
            && let Some(inner) = rest.strip_suffix('>')
        {
            return dyn_trait_root_from_type_str(inner);
        }
    }
    None
}

/// Return the trait root when the receiver's static type is a
/// `dyn Trait` (including `&dyn T` / `&mut dyn T` / `Box<dyn T>`),
/// otherwise `None`.  Looks up local/parameter bindings via
/// `ctx.local_dyn_trait_roots`, struct field types via
/// `ctx.struct_fields`, array element types via
/// `ctx.local_array_types`, and chained method-call / free-call
/// return types via `ctx.fn_return_types`.
fn dyn_trait_root_for_receiver(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        // Local/parameter bound to `dyn Trait` — directly mapped in
        // `local_dyn_trait_roots`.
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_dyn_trait_roots.get(&ident.to_string()).cloned()),
        // Strip wrappers that don't change the static type's trait-ness.
        syn::Expr::Reference(reference) => dyn_trait_root_for_receiver(&reference.expr, ctx),
        syn::Expr::Paren(paren) => dyn_trait_root_for_receiver(&paren.expr, ctx),
        syn::Expr::Group(group) => dyn_trait_root_for_receiver(&group.expr, ctx),
        // `self.handler.run()` — resolve `self.handler`'s declared field
        // type via `struct_fields[owner_type][handler]`, then check for
        // `dyn` / `Box<dyn>` / wrapper.
        syn::Expr::Field(field) => {
            let owner = receiver_type_root(&field.base, ctx)?;
            let field_name = match &field.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(_) => return None,
            };
            let field_type = ctx.struct_fields.field_type_in_scope(
                &owner,
                &field_name,
                &ctx.module_prefix,
                &ctx.use_imports,
            )?;
            dyn_trait_root_from_type_str(field_type)
        }
        // `handlers[i].run()` — `handlers`'s declared full type is
        // tracked in `local_array_types` (e.g. `"Vec<Box<dyn T>>"`);
        // strip the container wrapper to get the element type, then
        // check whether that element is a trait object.
        syn::Expr::Index(index) => {
            let container = match &*index.expr {
                syn::Expr::Path(path) => path
                    .path
                    .get_ident()
                    .and_then(|ident| ctx.local_array_types.get(&ident.to_string()).cloned()),
                _ => None,
            }?;
            let elem = extract_element_type_from_str(&container)?;
            dyn_trait_root_from_type_str(&elem)
        }
        // Chained `x.foo().bar()` — look up `x.foo`'s declared return
        // type, accepting plain `dyn T` AND wrapped (`Box<dyn T>`).
        syn::Expr::MethodCall(mc) => {
            let owner = receiver_type_root(&mc.receiver, ctx)?;
            let key = format!("{}::{}", owner, mc.method);
            let ret = ctx.fn_return_types.get(&key)?;
            dyn_trait_root_from_type_str(ret)
        }
        // Chained `foo().bar()` — free function return type, same wrapper
        // recognition as the method-call branch.
        syn::Expr::Call(call) => {
            if let syn::Expr::Path(p) = &*call.func {
                let key = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                let ret = ctx.fn_return_types.get(&key)?;
                return dyn_trait_root_from_type_str(ret);
            }
            None
        }
        _ => None,
    }
}

fn canonical_pat_name(pat: &syn::Pat) -> String {
    match pat {
        syn::Pat::Ident(ident) => ident.ident.to_string(),
        syn::Pat::Reference(reference) => canonical_pat_name(&reference.pat),
        syn::Pat::Type(typed) => canonical_pat_name(&typed.pat),
        syn::Pat::TupleStruct(tuple_struct) => tuple_struct
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        syn::Pat::Struct(strukt) => strukt
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        syn::Pat::Tuple(_) => "tuple_pat".into(),
        syn::Pat::Slice(_) => "slice_pat".into(),
        syn::Pat::Lit(_) => "lit_pat".into(),
        syn::Pat::Path(_) => "path_pat".into(),
        syn::Pat::Wild(_) => "_".into(),
        syn::Pat::Or(_) => "or_pat".into(),
        syn::Pat::Range(_) => "range_pat".into(),
        syn::Pat::Macro(_) => "macro_pat".into(),
        syn::Pat::Paren(paren) => canonical_pat_name(&paren.pat),
        _ => "unsupported_pat".into(),
    }
}

fn bind_pattern_locals(
    pat: &syn::Pat,
    matched_type: Option<&str>,
    ctx: &mut GraphBuildContext<'_>,
) {
    match pat {
        syn::Pat::Ident(ident) => {
            if let Some(type_str) = matched_type {
                bind_ident_type(&ident.ident, type_str, ctx);
            }
        }
        syn::Pat::Reference(reference) => bind_pattern_locals(&reference.pat, matched_type, ctx),
        syn::Pat::Paren(paren) => bind_pattern_locals(&paren.pat, matched_type, ctx),
        syn::Pat::Type(typed) => {
            let explicit_type = qualified_full_type_string_with_imports(
                &typed.ty,
                &ctx.module_prefix,
                &ctx.use_imports,
                ctx.known_struct_names,
                ctx.known_trait_names,
            );
            bind_pattern_locals(&typed.pat, explicit_type.as_deref().or(matched_type), ctx);
        }
        syn::Pat::TupleStruct(tuple_struct) => {
            let path: Vec<String> = tuple_struct
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect();
            if matches_constructor_path(&path, "Some") {
                if let (Some(inner), Some(inner_pat)) = (
                    matched_type.and_then(transparent_option_inner_type),
                    tuple_struct.elems.first(),
                ) {
                    bind_pattern_locals(inner_pat, Some(inner), ctx);
                }
            } else if matches_constructor_path(&path, "Ok") {
                if let (Some(inner), Some(inner_pat)) = (
                    matched_type.and_then(transparent_result_ok_type),
                    tuple_struct.elems.first(),
                ) {
                    bind_pattern_locals(inner_pat, Some(inner), ctx);
                }
            } else if matches_constructor_path(&path, "Err") {
                if let (Some(inner), Some(inner_pat)) = (
                    matched_type.and_then(transparent_result_err_type),
                    tuple_struct.elems.first(),
                ) {
                    bind_pattern_locals(inner_pat, Some(inner), ctx);
                }
            }
        }
        syn::Pat::Struct(pat_struct) => {
            // RPython rtyper field-access shape: pattern destructure on
            // `Enum::Variant { f, .. }` resolves each field's concretetype
            // through the per-class field table (see Item::Enum branch of
            // `collect_fields_and_returns`).  Pyre's
            // `StructFieldRegistry::field_type` resolves exact owner keys and
            // accepts crate-prefix suffix recovery only when it is unique, so
            // fully-qualified destructures such as
            // `majit_ir::RdVirtualInfo::VirtualInfo` can still resolve to the
            // registered `RdVirtualInfo::VirtualInfo` identity without
            // first-wins collisions across unrelated variants.
            let owner: String = pat_struct
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            let matched_owner = matched_type.and_then(type_root_from_type_string);
            for field_pat in &pat_struct.fields {
                let field_name = match &field_pat.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(_) => continue,
                };
                let field_type = ctx
                    .struct_fields
                    .field_type_in_scope(&owner, &field_name, &ctx.module_prefix, &ctx.use_imports)
                    .or_else(|| {
                        matched_owner.as_deref().and_then(|owner| {
                            ctx.struct_fields.field_type_in_scope(
                                owner,
                                &field_name,
                                &ctx.module_prefix,
                                &ctx.use_imports,
                            )
                        })
                    })
                    .map(|s| s.to_string());
                bind_pattern_locals(&field_pat.pat, field_type.as_deref(), ctx);
            }
        }
        syn::Pat::Or(pat_or) => {
            // RPython `flowspace/flowcontext.py` does not have or-patterns;
            // `match A | B { kind, .. } => ...` desugars to two parallel
            // arms in upstream's translator.  Recurse into each case so
            // shared destructure names get bound under each variant's
            // concretetype — both variants are required to expose the
            // same field set, so the resulting concretetype is consistent
            // across the cases.
            for case in &pat_or.cases {
                bind_pattern_locals(case, matched_type, ctx);
            }
        }
        syn::Pat::Tuple(pat_tuple) => {
            // RPython `BUILD_TUPLE_UNPACK` (`flowspace/flowcontext.py`)
            // unpacks each element with the per-position concretetype
            // recorded on the source `SomeTuple`.  The matched type
            // string carries the parenthesised list of element types
            // (`(VirtualKind, &[FieldDescrInfo], &[i16], usize)`);
            // `split_tuple_type_elements` walks balanced angle / paren /
            // bracket depth so nested generics (`Option<Result<T, E>>`)
            // do not split prematurely.  When the element count
            // disagrees with the type-string arity, fall back to
            // unbound recursion — better than asserting on a
            // tuple-typed value the inference rules do not yet handle.
            let elem_types = matched_type.and_then(split_tuple_type_elements);
            for (idx, elem_pat) in pat_tuple.elems.iter().enumerate() {
                let elem_type = elem_types
                    .as_ref()
                    .and_then(|v| v.get(idx))
                    .map(|s| s.as_str());
                bind_pattern_locals(elem_pat, elem_type, ctx);
            }
        }
        _ => {}
    }
}

/// Split a parenthesised tuple type string into its element types.
/// Mirrors `extract_element_type_from_str`'s prefix walk for `&` /
/// `*const` / `*mut` so a `&(A, B)` reference is treated as `(A, B)`,
/// then walks balanced `<>` / `()` / `[]` depth so nested generics
/// (`Option<Result<T, E>>`) and inner tuples (`Vec<(A, B)>`) survive
/// the split intact.
fn split_tuple_type_elements(type_str: &str) -> Option<Vec<String>> {
    let mut s = type_str.trim();
    loop {
        let stripped = s
            .strip_prefix("*const ")
            .or_else(|| s.strip_prefix("*mut "))
            .or_else(|| s.strip_prefix("&mut "))
            .or_else(|| s.strip_prefix("&"));
        match stripped {
            Some(rest) => s = rest.trim_start(),
            None => break,
        }
    }
    let inner = s.strip_prefix('(')?.strip_suffix(')')?;
    let mut elements: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut depth: i32 = 0;
    for ch in inner.chars() {
        match ch {
            '(' | '<' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | '>' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    elements.push(trimmed.to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let last = current.trim();
    if !last.is_empty() {
        elements.push(last.to_string());
    }
    Some(elements)
}

/// `descr.py:359 ARRAY_INSIDE._hints.get('nolength', False)` source-level
/// reader. PyPy's flowgraph carries the lltype object on every array op
/// and the JIT consults `_hints` directly; pyre's source-level analysis
/// has only the Rust type spelling, so we approximate the bit by
/// inspecting the pointee after stripping pointer-like prefixes.
///
/// Default is `False` to match PyPy's `_hints.get('nolength', False)`.
/// We only return `true` when the pointee is unambiguously a contiguous
/// item run with no length header:
///
/// - `[T]` / `[T; N]` — Rust slice / fixed-size array syntax. Pointer
///   addresses items[0]; no length word stored in the block.
/// - `*const T` / `*mut T` / `&T` / `&mut T` where the pointee `T` has
///   no generic parameters (`<…>`) and no parenthesised wrapper
///   (`Ptr(…)`). A bare identifier (`i64`, `Point`, `usize`, …) is
///   read as the *element* type and the pointer addresses items[0].
///
/// All other shapes — `Vec<T>`, `GcArray<T>`, `Ptr(GcArray(T))`,
/// `*const GcArray<T>`, … — keep PyPy's default `False`. The wrapper
/// retains a length header at offset 0; a `*const GcArray<T>` is a
/// pointer to that header block, not to items[0], so its descr must
/// carry `lendescr` per `descr.py:359-362`.
pub(crate) use crate::front::syn_metadata::nolength_from_array_type_id;

fn bind_ident_type(ident: &syn::Ident, type_str: &str, ctx: &mut GraphBuildContext<'_>) {
    let name = ident.to_string();
    ctx.local_value_types
        .insert(name.clone(), type_string_to_value_type(type_str));
    let trimmed = type_str.trim().to_string();
    ctx.local_type_strings.insert(name.clone(), trimmed.clone());
    // Mirror Stmt::Local Pat::Type binding (line 1085-1087 / 1318-1320):
    // every named local with a known full-type string seeds
    // `local_array_types` so that downstream
    // `array_type_id_from_expr` / `extract_element_type_from_str`
    // resolve the element type via the same channel an explicit
    // `let x: Vec<T> = ...` would.
    ctx.local_array_types.insert(name.clone(), trimmed);
    if let Some(root) = type_root_from_type_string(type_str) {
        ctx.local_type_roots.insert(name, root);
    }
}

fn matches_constructor_path(path: &[String], leaf: &str) -> bool {
    path.last().is_some_and(|last| last == leaf)
}

/// Extract the head identifier from a Rust type string for
/// receiver / method / field-owner lookups.
///
/// **Structural adaptation (parity rule §1):** RPython carries
/// `concretetype` as an `lltype.Ptr(GcStruct)` object whose identity
/// is structural; field/method lookups compare type objects directly.
/// Rust's `syn` AST surfaces types as strings, and the analyser
/// keeps them as strings throughout, so type identity resolves by
/// head-identifier match.  Wrapper info (`Box<T>`, `Vec<T>`,
/// generic args, lifetime params) is discarded here — the caller is
/// expected to have already applied the appropriate transparent-
/// container unwrapping (`extract_element_type_from_str` for arrays,
/// `transparent_option_inner_type` etc.) when the wrapper carries
/// payload type information.  Two distinct types that happen to share
/// a head identifier (e.g. via `use crate::foo::Bar` and
/// `use crate::baz::Bar`) will collide; pyre does not currently
/// disambiguate, mirroring the analyser's flat name table.  The
/// raw-pointer / reference prefix strip preserves single-identity
/// behaviour for `let x = obj as *mut Foo;` bindings.
fn type_root_from_type_string(type_str: &str) -> Option<String> {
    let mut trimmed = type_str.trim();
    loop {
        let stripped = trimmed
            .strip_prefix("*const ")
            .or_else(|| trimmed.strip_prefix("*mut "))
            .or_else(|| trimmed.strip_prefix("&mut "))
            .or_else(|| trimmed.strip_prefix("&"));
        match stripped {
            Some(rest) => trimmed = rest.trim_start(),
            None => break,
        }
    }
    if trimmed.is_empty() || is_primitive_type_string(trimmed) {
        return None;
    }
    let head = trimmed
        .split(['<', ' ', '('])
        .next()
        .unwrap_or(trimmed)
        .trim();
    if head.is_empty() {
        None
    } else {
        Some(head.to_string())
    }
}

fn is_primitive_type_string(type_str: &str) -> bool {
    matches!(
        type_str,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "bool"
            | "char"
            | "f32"
            | "f64"
            | "()"
    )
}

/// RPython: lltype graph identity — returns the full type path.
/// For `Foo` → "Foo", for `a::Foo` → "a::Foo".
/// Classify a Rust parameter/return `syn::Type` into one of the three
/// RPython `lltype` register classes (`Int`/`Ref`/`Float`).  This is the
/// pyre-side bridge for what RPython does implicitly: each `Variable`
/// carries `concretetype`, and `getkind(concretetype)` picks the class
/// (`rpython/jit/codewriter/support.py:getkind`).  pyre's front-end
/// records only a `syn::Type` so we reproduce the mapping here.
///
/// Returned value is assigned to `OpKind::Input { ty }` so the annotator
/// + rtyper reach every function parameter with a concrete class; the
/// assembler's `lookup_reg_with_kind` then finds a coloring for every
/// operand it encounters, matching upstream's invariant that every
/// Variable reaching `assembler.py:write_insn` has a `concretetype`.
pub(crate) fn classify_fn_arg_ty(ty: &syn::Type) -> crate::model::ValueType {
    use crate::model::ValueType;
    match ty {
        syn::Type::Path(path) => {
            let last = match path.path.segments.last() {
                Some(s) => s,
                None => return ValueType::Ref(None),
            };
            if path.path.segments.len() == 2
                && path.path.segments[0].ident == "Self"
                && path.path.segments[1].ident == "Truth"
            {
                return ValueType::Int;
            }
            let name = last.ident.to_string();
            // `Box<T>` / `Rc<T>` / `Arc<T>` — classify on the inner type
            // so `Box<i64>` stays Int (RPython `lltype.Ptr(Signed)`
            // collapses to the primitive), matching the downstream
            // `ValueType::Ref` vs `Int` distinction the assembler keys
            // off.
            if matches!(name.as_str(), "Box" | "Rc" | "Arc") {
                if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            return classify_fn_arg_ty(inner);
                        }
                    }
                }
                return ValueType::Ref(type_root_ident(ty));
            }
            match name.as_str() {
                // `lltype.Signed` family.
                "i8" | "i16" | "i32" | "i64" | "isize" | "char" => ValueType::Int,
                // `lltype.Unsigned` family — `getkind(Unsigned) == 'int'`
                // collapses storage to the int register class
                // (`rpython/jit/codewriter/flatten.py:getkind`).  The
                // producer-side type tag stays Unsigned so the annotator
                // selects `SomeInteger(unsigned=True)` and the
                // rtyper-side `signed_repr_of` / `intmask` cast paths
                // distinguish signed vs unsigned at the LL boundary.
                "u8" | "u16" | "u32" | "u64" | "usize" => ValueType::Unsigned,
                // `lltype.Bool` annotates as `SomeBool(SomeInteger)`
                // (`annotator/model.py:185-198`, distinct lattice node).
                // `getkind(Bool) == 'int'` so register-class code paths
                // alias Bool to Int (jit_codewriter sites added in
                // commit 4318ebb51b2); the producer-side type tag
                // remains Bool so the rtyper picks `BoolRepr`.
                "bool" => ValueType::Bool,
                // `lltype.Float` — `f32` widens up to f64 at the SSA
                // level but stays in the Float class either way.
                "f32" | "f64" => ValueType::Float,
                // Anything else is a user type / GC ref / opaque struct.
                // Carry the joined path segments as diagnostic metadata
                // on the legacy tag. Precise typed pointers must be
                // attached by producers that can resolve the actual
                // HostObject/lltype identity; `valuetype_to_someshell`
                // deliberately keeps `Ref(_)` on the classdef-less
                // fallback.
                _ => ValueType::Ref(type_root_ident(ty)),
            }
        }
        // `&T` / `&mut T` — pointer → Ref (lltype.Ptr in RPython).
        // `type_root_ident` recursively unwraps the reference and
        // returns the inner Path's joined segments when present.
        syn::Type::Reference(_) => ValueType::Ref(type_root_ident(ty)),
        // `*const T` / `*mut T` — raw pointer, same class as Ref.  pyre
        // often stores GC objects as `*mut PyObject`; classify as Ref
        // so field/array bases reach the canonical `/rd>X` encoding
        // rather than the pyre-only `*_intbase` aliases.
        syn::Type::Ptr(_) => ValueType::Ref(type_root_ident(ty)),
        syn::Type::Paren(paren) => classify_fn_arg_ty(&paren.elem),
        syn::Type::Group(group) => classify_fn_arg_ty(&group.elem),
        // `dyn Trait` — GC pointer to a trait object.
        // `type_root_ident` returns `dyn <Trait>` so consumers can
        // distinguish concrete structs from trait objects.
        syn::Type::TraitObject(_) => ValueType::Ref(type_root_ident(ty)),
        // Tuple/array/slice: treat as Ref (bulk data, not a register
        // primitive).  RPython `lltype.Array` + `lltype.Struct` both
        // flatten to `lltype.Ptr` at the call-site boundary.  No single
        // ident makes sense as the type-root, so leave `None`.
        syn::Type::Tuple(_) | syn::Type::Array(_) | syn::Type::Slice(_) => ValueType::Ref(None),
        // `fn(T) -> T`, `impl Trait`, never — no runtime
        // representation reaches the SSA level; default to Ref for
        // safe-by-default classification.
        _ => ValueType::Ref(None),
    }
}

/// RPython's lltype.Struct objects have globally unique identities;
/// returning all path segments ensures `a::Foo` and `b::Foo` don't alias.
fn type_root_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            // `Box<dyn Trait>` / `Rc<dyn Trait>` / `Arc<dyn Trait>` —
            // unwrap the first generic arg and try again; the resulting
            // root identifies the trait, not the container.
            if let Some(last) = path.path.segments.last() {
                let wrapper = last.ident.to_string();
                if matches!(wrapper.as_str(), "Box" | "Rc" | "Arc") {
                    if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                        for arg in &args.args {
                            if let syn::GenericArgument::Type(inner) = arg {
                                if let Some(root) = type_root_ident(inner) {
                                    return Some(root);
                                }
                            }
                        }
                    }
                }
            }
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
        syn::Type::Ptr(ptr) => type_root_ident(&ptr.elem),
        syn::Type::Paren(paren) => type_root_ident(&paren.elem),
        syn::Type::Group(group) => type_root_ident(&group.elem),
        // `dyn Trait + 'a` / `&mut dyn Trait` (after deref) — return the
        // first trait bound's canonical path, rendered as `dyn <Trait>` so
        // callers can tell this is a trait object.
        syn::Type::TraitObject(obj) => {
            trait_object_root_name(&obj.bounds).map(|r| format!("dyn {r}"))
        }
        // `impl Trait` is a static opaque type (compiler monomorphizes
        // each call site to a single concrete impl), not runtime
        // family-dispatch.  RPython `indirect_call` is reserved for
        // truly polymorphic callees (`rpython/jit/codewriter/call.py:103
        // graphs_from`); treat impl Trait the same way concrete-type
        // method calls are treated and bail out so downstream emits
        // CallTarget::Method, not CallTarget::Indirect.
        syn::Type::ImplTrait(_) => None,
        _ => None,
    }
}

fn trait_bound_root_for_receiver(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_trait_bound_roots.get(&ident.to_string()).cloned()),
        syn::Expr::Reference(reference) => trait_bound_root_for_receiver(&reference.expr, ctx),
        syn::Expr::Paren(paren) => trait_bound_root_for_receiver(&paren.expr, ctx),
        syn::Expr::Group(group) => trait_bound_root_for_receiver(&group.expr, ctx),
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
            trait_bound_root_for_receiver(&unary.expr, ctx)
        }
        _ => None,
    }
}

/// Extract the declaring trait name from a `dyn T + 'a` bound list:
/// returns the first `T::Trait`-style bound's canonical path.
/// Used by `type_root_ident` / `full_type_string` / `extract_dyn_trait_root`
/// to identify the indirect-call family key.
fn trait_object_root_name(
    bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
) -> Option<String> {
    bounds.iter().find_map(|b| match b {
        syn::TypeParamBound::Trait(t) => Some(
            t.path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>()
                .join("::"),
        ),
        _ => None,
    })
}

fn collect_generic_trait_roots(
    generics: &syn::Generics,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> HashMap<String, String> {
    let mut roots = HashMap::new();
    for param in &generics.params {
        if let syn::GenericParam::Type(type_param) = param {
            if let Some(root) =
                trait_object_root_name_qualified(&type_param.bounds, prefix, known_trait_names)
            {
                roots.insert(type_param.ident.to_string(), root);
            }
        }
    }
    if let Some(where_clause) = &generics.where_clause {
        for predicate in &where_clause.predicates {
            if let syn::WherePredicate::Type(pred_ty) = predicate {
                let Some(type_name) = type_root_ident(&pred_ty.bounded_ty) else {
                    continue;
                };
                if let Some(root) =
                    trait_object_root_name_qualified(&pred_ty.bounds, prefix, known_trait_names)
                {
                    roots.insert(type_name, root);
                }
            }
        }
    }
    roots
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

fn trait_object_root_name_qualified(
    bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    trait_object_root_name(bounds)
        .map(|name| qualify_known_trait_name(&name, prefix, known_trait_names))
}

/// Returns the bare trait root (no `dyn ` prefix) when `ty` denotes a
/// `dyn Trait` / `&dyn Trait` / `Box<dyn Trait>` receiver; `None`
/// otherwise.  Used by method-call lowering to decide whether the call
/// should be modeled as an RPython `indirect_call`
/// (`rewrite_op_indirect_call` entrypoint).
pub fn extract_dyn_trait_root(ty: &syn::Type) -> Option<String> {
    extract_dyn_trait_root_with_context(ty, "", &std::collections::HashSet::new())
}

fn extract_dyn_trait_root_with_context(
    ty: &syn::Type,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    match ty {
        syn::Type::TraitObject(obj) => {
            trait_object_root_name_qualified(&obj.bounds, prefix, known_trait_names)
        }
        // `impl Trait` is a static opaque type — no runtime family-dispatch.
        // See `type_root_ident`'s ImplTrait arm for the rationale + RPython cite.
        syn::Type::ImplTrait(_) => None,
        syn::Type::Reference(r) => {
            extract_dyn_trait_root_with_context(&r.elem, prefix, known_trait_names)
        }
        syn::Type::Paren(p) => {
            extract_dyn_trait_root_with_context(&p.elem, prefix, known_trait_names)
        }
        syn::Type::Group(g) => {
            extract_dyn_trait_root_with_context(&g.elem, prefix, known_trait_names)
        }
        syn::Type::Path(path) => {
            // `Box<dyn Trait>` / `Rc<dyn Trait>` / `Arc<dyn Trait>`.
            let last = path.path.segments.last()?;
            if !matches!(last.ident.to_string().as_str(), "Box" | "Rc" | "Arc") {
                return None;
            }
            if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                for arg in &args.args {
                    if let syn::GenericArgument::Type(inner) = arg {
                        if let Some(r) =
                            extract_dyn_trait_root_with_context(inner, prefix, known_trait_names)
                        {
                            return Some(r);
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Canonical type string for a syn::Type.
///
/// Produces a string that includes generic arguments,
/// e.g. `Vec<Point>` → `"Vec<Point>"`, `Point` → `"Point"`.
pub fn full_type_string(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            let segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| {
                    let name = seg.ident.to_string();
                    match &seg.arguments {
                        syn::PathArguments::None => name,
                        syn::PathArguments::AngleBracketed(args) => {
                            let inner: Vec<String> = args
                                .args
                                .iter()
                                .filter_map(|arg| match arg {
                                    syn::GenericArgument::Type(t) => full_type_string(t),
                                    _ => None,
                                })
                                .collect();
                            if inner.is_empty() {
                                name
                            } else {
                                format!("{}<{}>", name, inner.join(","))
                            }
                        }
                        syn::PathArguments::Parenthesized(_) => name,
                    }
                })
                .collect();
            Some(segments.join("::"))
        }
        syn::Type::Reference(r) => full_type_string(&r.elem),
        syn::Type::Ptr(p) => {
            let inner = full_type_string(&p.elem)?;
            let mutability = if p.mutability.is_some() {
                "*mut"
            } else {
                "*const"
            };
            Some(format!("{mutability} {inner}"))
        }
        syn::Type::Paren(p) => full_type_string(&p.elem),
        syn::Type::Group(g) => full_type_string(&g.elem),
        syn::Type::Slice(s) => full_type_string(&s.elem).map(|t| format!("[{}]", t)),
        syn::Type::TraitObject(obj) => {
            trait_object_root_name(&obj.bounds).map(|r| format!("dyn {r}"))
        }
        // `impl Trait` is a static opaque type — render as the underlying
        // bound name without the `dyn ` prefix so downstream consumers
        // do not mistake it for a trait object (see `type_root_ident`).
        syn::Type::ImplTrait(obj) => trait_object_root_name(&obj.bounds),
        // RPython: ARRAY identity preserves full type including length.
        // [Point; 4] and [Point; 8] are different ARRAY types.
        syn::Type::Array(a) => {
            let elem = full_type_string(&a.elem)?;
            // Extract length from Expr::Lit if possible.
            let len_str = match &a.len {
                syn::Expr::Lit(lit) => match &lit.lit {
                    syn::Lit::Int(int_lit) => int_lit.base10_digits().to_string(),
                    _ => "N".to_string(),
                },
                _ => "N".to_string(),
            };
            Some(format!("[{};{}]", elem, len_str))
        }
        syn::Type::Tuple(t) if t.elems.is_empty() => Some("()".to_string()),
        syn::Type::Tuple(t) => {
            let elems: Option<Vec<String>> = t.elems.iter().map(full_type_string).collect();
            elems.map(|elems| format!("({})", elems.join(",")))
        }
        _ => None,
    }
}

/// RPython: lltype identity — `full_type_string` with module-prefix qualification.
///
/// RPython's `T.TO` always returns the actual lltype object.
/// This function qualifies single-segment leaf types that are KNOWN structs
/// (in `known_struct_names`) with the module prefix, so `Bar` in `mod a`
/// becomes `a::Bar`. Uses the actual struct name set, not a heuristic.
pub(crate) fn qualified_full_type_string(
    ty: &syn::Type,
    prefix: &str,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    qualified_full_type_string_with_imports(
        ty,
        prefix,
        &HashMap::new(),
        known_struct_names,
        known_trait_names,
    )
}

/// `qualified_full_type_string` variant that consults a per-source
/// `use <path> as alias` table when qualifying single-segment leaf
/// types — keeps struct field / fn return metadata in the same name
/// namespace as `qualify_type_name_with_imports`-driven
/// parameter/local lowering, mirroring PyPy `bookkeeper.getdesc`'s
/// single-frame `f_globals` resolution
/// (`rpython/annotator/bookkeeper.py:353-409`).
///
/// `use_imports` is the per-source map collected by
/// `parse::collect_use_imports`; an empty map reduces this back to
/// `qualified_full_type_string`'s plain `prefix::Bar` /
/// `canonical_struct_name` behaviour.
pub(crate) fn qualified_full_type_string_with_imports(
    ty: &syn::Type,
    prefix: &str,
    use_imports: &HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    // Top-level files (`prefix=""`) still need to walk the match when a
    // per-source `use_imports` table is available; PyPy `bookkeeper.getdesc`
    // resolves bare names through the importing frame's `f_globals` even at
    // module root (`rpython/annotator/bookkeeper.py:353`).  Only fall
    // through to `full_type_string` when both qualification sources are
    // empty.
    if prefix.is_empty() && use_imports.is_empty() {
        return full_type_string(ty);
    }
    match ty {
        syn::Type::Path(path) => {
            let segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| {
                    let name = seg.ident.to_string();
                    match &seg.arguments {
                        syn::PathArguments::None => {
                            // Leaf type (no generics).  Qualify when the
                            // single-segment name is a known user struct
                            // (direct match) OR aliases to one via
                            // `use foo::Bar as B` — for the rename case
                            // `B` does not itself appear in
                            // `known_struct_names`, but the resolved
                            // target's leaf name does.  Non-struct
                            // imports (`use foo::helper` for a fn,
                            // `use external_crate::Item` for an external
                            // type) leave the bare name unqualified so
                            // their identity stays distinct from the
                            // file's own struct namespace.  PyPy
                            // `bookkeeper.getdesc(value)` binds the alias
                            // to the original Python object identity.
                            let alias_targets_struct = path.path.segments.len() == 1
                                && use_imports.get(&name).is_some_and(|full| {
                                    let leaf = full
                                        .rsplit_once("::")
                                        .map(|(_, l)| l)
                                        .unwrap_or(full.as_str());
                                    known_struct_names.contains(leaf)
                                });
                            if path.path.segments.len() == 1
                                && (known_struct_names.contains(&name) || alias_targets_struct)
                            {
                                qualify_type_name_with_imports(&name, prefix, use_imports)
                            } else {
                                name
                            }
                        }
                        syn::PathArguments::AngleBracketed(args) => {
                            // Container<T,...> — qualify inner types, not the container.
                            let inner: Vec<String> = args
                                .args
                                .iter()
                                .filter_map(|arg| match arg {
                                    syn::GenericArgument::Type(t) => {
                                        qualified_full_type_string_with_imports(
                                            t,
                                            prefix,
                                            use_imports,
                                            known_struct_names,
                                            known_trait_names,
                                        )
                                    }
                                    _ => None,
                                })
                                .collect();
                            if inner.is_empty() {
                                name
                            } else {
                                format!("{}<{}>", name, inner.join(","))
                            }
                        }
                        syn::PathArguments::Parenthesized(_) => name,
                    }
                })
                .collect();
            Some(segments.join("::"))
        }
        syn::Type::Reference(r) => qualified_full_type_string_with_imports(
            &r.elem,
            prefix,
            use_imports,
            known_struct_names,
            known_trait_names,
        ),
        syn::Type::Ptr(p) => {
            let inner = qualified_full_type_string_with_imports(
                &p.elem,
                prefix,
                use_imports,
                known_struct_names,
                known_trait_names,
            )?;
            let mutability = if p.mutability.is_some() {
                "*mut"
            } else {
                "*const"
            };
            Some(format!("{mutability} {inner}"))
        }
        syn::Type::Paren(p) => qualified_full_type_string_with_imports(
            &p.elem,
            prefix,
            use_imports,
            known_struct_names,
            known_trait_names,
        ),
        syn::Type::Group(g) => qualified_full_type_string_with_imports(
            &g.elem,
            prefix,
            use_imports,
            known_struct_names,
            known_trait_names,
        ),
        syn::Type::Slice(s) => qualified_full_type_string_with_imports(
            &s.elem,
            prefix,
            use_imports,
            known_struct_names,
            known_trait_names,
        )
        .map(|t| format!("[{}]", t)),
        syn::Type::Array(a) => {
            let elem = qualified_full_type_string_with_imports(
                &a.elem,
                prefix,
                use_imports,
                known_struct_names,
                known_trait_names,
            )?;
            let len_str = match &a.len {
                syn::Expr::Lit(lit) => match &lit.lit {
                    syn::Lit::Int(int_lit) => int_lit.base10_digits().to_string(),
                    _ => "N".to_string(),
                },
                _ => "N".to_string(),
            };
            Some(format!("[{};{}]", elem, len_str))
        }
        syn::Type::Tuple(t) if t.elems.is_empty() => Some("()".to_string()),
        syn::Type::Tuple(t) => {
            let elems: Option<Vec<String>> = t
                .elems
                .iter()
                .map(|elem| {
                    qualified_full_type_string_with_imports(
                        elem,
                        prefix,
                        use_imports,
                        known_struct_names,
                        known_trait_names,
                    )
                })
                .collect();
            elems.map(|elems| format!("({})", elems.join(",")))
        }
        syn::Type::TraitObject(obj) => {
            trait_object_root_name_qualified(&obj.bounds, prefix, known_trait_names)
                .map(|r| format!("dyn {r}"))
        }
        // `impl Trait` is a static opaque — render the bound name without
        // the `dyn ` marker.  See `type_root_ident` for the full rationale.
        syn::Type::ImplTrait(obj) => {
            trait_object_root_name_qualified(&obj.bounds, prefix, known_trait_names)
        }
        _ => None,
    }
}

/// RPython: resolve ARRAY identity from an expression.
///
/// RPython: `getkind(TYPE)[0]` — map type string to ValueType for kind suffix.
/// Used by InteriorFieldRead/Write to determine the i/r/f suffix.
fn type_string_to_value_type(type_str: &str) -> ValueType {
    let type_str = type_str.trim();
    match type_str {
        // u* folded into Int alongside i* — see
        // `classify_fn_arg_ty`'s TODO(unsigned-producer-flip) for the
        // cascade list.  Must stay in sync with the producer there so
        // the InteriorFieldRead/Write kind suffix agrees with the
        // inputarg type.
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
        | "char" | "Self::Truth" => ValueType::Int,
        // `lltype.Bool` — see `classify_fn_arg_ty`'s `"bool"` arm for
        // the SomeBool/BoolRepr rationale.
        "bool" => ValueType::Bool,
        "f32" | "f64" => ValueType::Float,
        "()" => ValueType::Void,
        _ => ValueType::Ref(None),
    }
}

pub(crate) use crate::front::syn_metadata::transparent_result_ok_type;

fn transparent_result_err_type(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    for prefix in ["Result<", "std::result::Result<", "core::result::Result<"] {
        let Some(inner) = trimmed
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };
        let err_type = second_top_level_generic_arg(inner).map(str::trim)?;
        if err_type == "()" {
            return None;
        }
        return Some(err_type);
    }
    None
}

fn transparent_option_inner_type(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    for prefix in ["Option<", "std::option::Option<", "core::option::Option<"] {
        let Some(inner) = trimmed
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };
        return first_top_level_generic_arg(inner).map(str::trim);
    }
    None
}

fn first_top_level_generic_arg(args: &str) -> Option<&str> {
    let mut depth = 0usize;
    for (idx, ch) in args.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => return Some(&args[..idx]),
            _ => {}
        }
    }
    if args.is_empty() { None } else { Some(args) }
}

fn second_top_level_generic_arg(args: &str) -> Option<&str> {
    let mut depth = 0usize;
    for (idx, ch) in args.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                let rest = args[idx + 1..].trim();
                return if rest.is_empty() { None } else { Some(rest) };
            }
            _ => {}
        }
    }
    None
}

fn field_value_type_from_expr(
    base: &syn::Expr,
    member: &syn::Member,
    ctx: &GraphBuildContext,
) -> Option<ValueType> {
    field_type_string_from_expr(base, member, ctx)
        .map(|field_type| type_string_to_value_type(&field_type))
}

fn field_type_string_from_expr(
    base: &syn::Expr,
    member: &syn::Member,
    ctx: &GraphBuildContext,
) -> Option<String> {
    let field_name = member_name(member);
    let owner = receiver_type_root(base, ctx)?;
    ctx.struct_fields
        .field_type_in_scope(&owner, &field_name, &ctx.module_prefix, &ctx.use_imports)
        .map(ToOwned::to_owned)
}

/// Type-string inference for `let pat = init;` initialisers, with the
/// extra power of binding match-arm patterns into a borrowed snapshot of
/// `ctx` so an arm body's expression can be typed against the
/// destructured names the body actually uses
/// (`fielddescrs.as_slice()` after `VirtualInfo { fielddescrs, .. }`).
/// Non-match initialisers fall through to the immutable
/// `expression_type_string`.
///
/// RPython's annotator handles the same shape via `SomeBuiltin.unionof`
/// over per-branch annotations; pyre takes the first arm whose body
/// types successfully and trusts the type-checker to keep the
/// remaining arms in sync.
fn infer_init_type_string(expr: &syn::Expr, ctx: &mut GraphBuildContext<'_>) -> Option<String> {
    if let syn::Expr::Match(m) = expr {
        let scrutinee_type = expression_type_string(&m.expr, ctx);
        for arm in &m.arms {
            let saved = LocalBindingSnapshot::capture(ctx);
            bind_pattern_locals(&arm.pat, scrutinee_type.as_deref(), ctx);
            let body_type = expression_type_string(&arm.body, ctx);
            saved.restore(ctx);
            if body_type.is_some() {
                return body_type;
            }
        }
        return None;
    }
    expression_type_string(expr, ctx)
}

fn expression_type_string(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_type_strings.get(&ident.to_string()).cloned()),
        syn::Expr::Reference(reference) => expression_type_string(&reference.expr, ctx),
        syn::Expr::Paren(paren) => expression_type_string(&paren.expr, ctx),
        syn::Expr::Unary(unary) => match &unary.op {
            syn::UnOp::Deref(_) => expression_type_string(&unary.expr, ctx),
            _ => None,
        },
        syn::Expr::Cast(cast) => qualified_full_type_string_with_imports(
            &cast.ty,
            &ctx.module_prefix,
            &ctx.use_imports,
            ctx.known_struct_names,
            ctx.known_trait_names,
        ),
        syn::Expr::Field(field) => field_type_string_from_expr(&field.base, &field.member, ctx),
        syn::Expr::Call(call) => {
            let syn::Expr::Path(path) = &*call.func else {
                return None;
            };
            let segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if let Some(ret) = lookup_function_return_type(ctx, &segments).cloned() {
                return Some(ret);
            }
            // TODO(name-based-callee-resolution): receiver-/host-
            // identity-blind name shortlist below; same convergence
            // path as `lookup_function_return_type`'s TODO marker.
            // Cross-module pyre-interpreter bool / Result<bool>
            // predicate shortlist.  Pyre's `PYRE_JIT_GRAPH_SOURCES`
            // (`generated.rs:149`) is intentionally narrow so the
            // canonical-pipeline analysis time stays bounded; the
            // closure's source set excludes `baseobjspace.rs` /
            // `runtime_ops.rs` / `boolobject.rs` even though the
            // in-scope sources reference their predicates.  Aliasing
            // risk: any user-source function whose last segment
            // matches one of these names but does NOT return `bool`
            // is mis-classified here — convergence retires this
            // arm.
            let last = segments.last()?;
            // Direct `-> bool` returns.
            if matches!(
                last.as_str(),
                "is_true"
                    | "is_iterable"
                    | "w_bool_get_value"
                    | "exception_is_valid_class_w"
                    | "exception_is_valid_obj_as_class_w"
                    | "dict_storage_delete"
            ) {
                return Some("bool".to_string());
            }
            // `-> Result<bool, PyError>` returns — `?` peels to bool at
            // the let-binding's `Expr::Try` arm.
            if matches!(last.as_str(), "contains") {
                return Some("Result<bool, PyError>".to_string());
            }
            None
        }
        syn::Expr::MethodCall(mc) => {
            let receiver_root = receiver_type_root(&mc.receiver, ctx);
            let trait_bound_root = trait_bound_root_for_receiver(&mc.receiver, ctx);
            if let Some(ret) = lookup_method_return_type(ctx, receiver_root.as_deref(), &mc.method)
                .or_else(|| lookup_method_return_type(ctx, trait_bound_root.as_deref(), &mc.method))
                .cloned()
            {
                return Some(ret);
            }
            // Stdlib method shortlist that always returns `bool` —
            // mirrored from `expr_unary_not_operand_kind`'s MethodCall
            // arm so closure-body / chained-receiver type inference
            // resolves through the same identity-by-name shortcut.
            // RPython peer: `bookkeeper.getdesc(receiver).find_method`
            // resolves these by host-stdlib identity; pyre's static
            // shortlist substitutes for the missing whole-program
            // annotator visibility.  Receiver-type independence is the
            // distinguishing property — predicate methods do not
            // propagate receiver identity, so they don't need
            // `local_array_types` / `local_dyn_trait_roots` lookups.
            let method = mc.method.to_string();
            if matches!(
                method.as_str(),
                "is_null"
                    | "is_some"
                    | "is_none"
                    | "is_ok"
                    | "is_err"
                    | "is_empty"
                    | "contains"
                    | "starts_with"
                    | "ends_with"
                    | "eq"
                    | "ne"
                    | "lt"
                    | "le"
                    | "gt"
                    | "ge"
                    | "is_nan"
                    | "is_infinite"
                    | "is_finite"
                    | "is_sign_negative"
                    | "is_sign_positive"
                    | "is_positive"
                    | "is_negative"
                    | "is_alphabetic"
                    | "is_alphanumeric"
                    | "is_digit"
                    | "is_whitespace"
                    | "is_ascii"
                    | "is_ascii_alphabetic"
                    | "is_ascii_alphanumeric"
                    | "is_ascii_digit"
                    | "is_ascii_whitespace"
                    | "is_ascii_uppercase"
                    | "is_ascii_lowercase"
                    | "is_uppercase"
                    | "is_lowercase"
                    | "success"
                    | "exists"
                    | "is_tracing"
                    | "has_compiled_loop"
                    | "is_array_of_pointers"
                    | "has_kwarg"
                    | "has_vararg"
                    | "any"
                    | "all"
            ) {
                return Some("bool".to_string());
            }
            // Closure-passthrough methods — `LocalKey::with(closure)`,
            // `Option::unwrap_or_else(closure)`, `Result::unwrap_or_else
            // (closure)`, `Option::map(closure)`, `Result::map_err(
            // closure)`, `Option::and_then(closure)`,
            // `Option::or_else(closure)`, `Result::ok_or_else(closure)`,
            // `Iterator::filter_map(closure)` etc.  The receiver
            // method's return type IS the closure's return type
            // (modulo `Option<_>` / `Result<_,_>` wrapping), so project
            // the last argument's body type.
            //
            // RPython peer: `bookkeeper.getdesc(method).consider_call_
            // site` reads the callable's return annotation
            // (`bookkeeper.py:355-409`); pyre's static walker
            // substitutes by inspecting the closure body directly.
            //
            // TODO(closure-makefunction-port): RPython
            // `flowspace/flowcontext.py:1177 MAKE_FUNCTION` materialises
            // the closure body as a separate function-graph keyed by
            // the lambda's host identity, and `CALL_FUNCTION` invokes
            // it.  Pyre's closure body still lowers to a single
            // `OpKind::Abort { Closure }` placeholder at
            // `front/ast.rs:4785`, so closure side effects are not
            // analysed.  The return-type projection here is the
            // minimum-viable substitute that keeps method-call sites
            // type-coherent; full MAKE_FUNCTION parity requires
            // routing each closure through a synthetic FunctionGraph
            // + `PyreCallRegistry` entry (deferred).
            if matches!(
                method.as_str(),
                "with" | "with_borrow" | "with_borrow_mut" | "unwrap_or_else"
            ) && let Some(last) = mc.args.last()
                && let syn::Expr::Closure(closure) = last
            {
                let ret = match &closure.output {
                    syn::ReturnType::Type(_, ty) => qualified_full_type_string_with_imports(
                        ty,
                        &ctx.module_prefix,
                        &ctx.use_imports,
                        ctx.known_struct_names,
                        ctx.known_trait_names,
                    ),
                    syn::ReturnType::Default => expression_type_string(&closure.body, ctx),
                };
                if ret.is_some() {
                    return ret;
                }
            }
            // `Option<T>::map(F: FnOnce(T) -> U) -> Option<U>` and
            // `Result<T,E>::map(F: FnOnce(T) -> U) -> Result<U,E>` —
            // the receiver-method return is the *wrapper* of the
            // closure body's return.  Pyre's downstream
            // `expression_type_string` only consults the receiver
            // type-string for shape-matching (`Option<_>` / `Result<_,
            // _>`), so projecting `Option<closure_body>` /
            // `Result<closure_body, _>` keeps the wrapper visible at
            // the call site.
            if matches!(method.as_str(), "map" | "and_then" | "filter")
                && let Some(last) = mc.args.last()
                && let syn::Expr::Closure(closure) = last
                && let Some(receiver_ty) = expression_type_string(&mc.receiver, ctx)
            {
                let body_ty = match &closure.output {
                    syn::ReturnType::Type(_, ty) => qualified_full_type_string_with_imports(
                        ty,
                        &ctx.module_prefix,
                        &ctx.use_imports,
                        ctx.known_struct_names,
                        ctx.known_trait_names,
                    ),
                    syn::ReturnType::Default => expression_type_string(&closure.body, ctx),
                };
                if let Some(body) = body_ty {
                    // `Option<X>.map(|t| body) → Option<body>`,
                    // `Option<X>.and_then(|t| body) → body` (assuming
                    // body is already `Option<_>`),
                    // `Option<X>.filter(|t| <bool>) → Option<X>`
                    // (preserves receiver).  RPython parity:
                    // upstream's `SomeOption` is just a tag —
                    // pyre keeps shape via type-string.
                    if method == "filter" {
                        return Some(receiver_ty);
                    }
                    if method == "and_then" {
                        return Some(body);
                    }
                    // `map` — wrap the body in the receiver's
                    // Option/Result shape.  Strip the existing
                    // `<...>` and re-emit `Wrapper<body>`.
                    if let Some(wrapper) = receiver_ty.split('<').next()
                        && matches!(wrapper, "Option" | "Result")
                    {
                        return Some(format!("{wrapper}<{body}>"));
                    }
                }
            }
            // RPython annotator `SomeList.method_get / .method_first /
            // .method_last`-style result inference, narrowed to the
            // stdlib `Vec<T>` / `[T]` accessors that `let Some(x) =
            // lst.get(i) else { ... }` desugars from.  Pyre's
            // `local_array_types` carries the full container type
            // (`Vec<FieldDescrInfo>`); the `Option<&T>` shape is the
            // Rust-language adaptation of RPython's `lst[i]` access.
            if matches!(method.as_str(), "get" | "first" | "last") {
                if let Some(arr_ty) = array_type_id_from_expr(&mc.receiver, ctx)
                    && let Some(elem) = extract_element_type_from_str(&arr_ty)
                {
                    return Some(format!("Option<&{}>", elem));
                }
            }
            if method == "as_ref" {
                if let Some(receiver_ty) = expression_type_string(&mc.receiver, ctx)
                    && let Some(ret) = method_as_ref_return_type(&receiver_ty)
                {
                    return Some(ret);
                }
            }
            // `Vec::as_slice` / `slice::as_ref` view the receiver as
            // `&[T]` while preserving element identity.  Same parity
            // rationale as the `get` arm — RPython's `lst.tolist()` /
            // `lst[:]` aliases keep the underlying `GcArray(T)` type
            // identity for downstream `getarrayitem` lookups.
            if matches!(method.as_str(), "as_slice" | "as_ref") {
                if let Some(arr_ty) = array_type_id_from_expr(&mc.receiver, ctx)
                    && let Some(elem) = extract_element_type_from_str(&arr_ty)
                {
                    return Some(format!("&[{}]", elem));
                }
            }
            None
        }
        // RPython `BUILD_TUPLE` produces a `SomeTuple(elems)`; the type
        // surfaces as a parenthesised list of element annotations.  Pyre
        // mirrors this with a flat string so `Pat::Tuple` /
        // `split_tuple_type_elements` can route each element back to its
        // originating concretetype.
        syn::Expr::Tuple(t) => {
            let mut element_types: Vec<String> = Vec::with_capacity(t.elems.len());
            for elem in &t.elems {
                let ty = expression_type_string(elem, ctx)?;
                element_types.push(ty);
            }
            Some(format!("({})", element_types.join(", ")))
        }
        // `VirtualKind::Instance { ... }` — Rust struct-init for an enum
        // variant.  RPython's annotator returns `SomeInstance(cls)` for
        // class-instantiation; here the parent enum name is the
        // closest analog (the variant's per-class subclass identity is
        // not carried as a separate type root).  Falls back to the
        // joined path for plain struct constructors.
        syn::Expr::Struct(es) => {
            let segments: Vec<String> = es
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect();
            if segments.len() >= 2 {
                Some(segments[..segments.len() - 1].join("::"))
            } else {
                Some(segments.join("::"))
            }
        }
        // `unsafe { tail }` — RPython has no peer; transparent on the
        // analyser side.  Mirrors the `Expr::Unsafe` arm in
        // `expr_unary_not_operand_kind`.
        syn::Expr::Unsafe(u) => block_tail_type_string(&u.block, ctx),
        // `expr?` — Rust try operator; lowers to:
        //   match expr { Ok(v) => v, Err(e) => return Err(e.into()) }
        // (or the Option counterpart).  Result/Option type carriers
        // are not part of RPython's annotator surface; the closest
        // peer is `flowcontext.py:194-198` POP_BLOCK / END_FINALLY's
        // exception-channel join, which the front-end represents as
        // an `OpKind::Call` with the question-mark desugar happening
        // at the rtyper level.  For the type-string carrier we just
        // need the success-arm projection: `Result<T, E>` → `T`,
        // `Option<T>` → `T`.  Mirrors the unwrapping `outer_generic_
        // inner_type` does for `Rc`/`Arc`/`Box` in
        // `method_as_ref_return_type`.
        syn::Expr::Try(t) => {
            let inner_ty = expression_type_string(&t.expr, ctx)?;
            outer_generic_inner_type(&inner_ty, &["Result", "Option"])
        }
        // `{ ...; tail }` — same transparent handling.
        syn::Expr::Block(b) => block_tail_type_string(&b.block, ctx),
        // `if cond { a } else { b }` — RPython's annotator unifies
        // arm types via `unionof(s_then, s_else)`
        // (`annotator/model.py:UnionedSomeObject`); pyre's frontend
        // lacks a SomeObject lattice, so handle only the case where
        // the two arms agree on a single primitive type string.  The
        // else branch is required (Rust `if` without else is `()` and
        // wouldn't be the rhs of a `let` bound to a useful type).
        syn::Expr::If(if_expr) => {
            let then_ty = block_tail_type_string(&if_expr.then_branch, ctx)?;
            let (_, else_expr) = if_expr.else_branch.as_ref()?;
            let else_ty = expression_type_string(else_expr, ctx)?;
            if then_ty == else_ty {
                Some(then_ty)
            } else {
                None
            }
        }
        // `match scrut { arm => body, ... }` — same unionof rationale
        // as `Expr::If`.  All arm bodies must agree on one type
        // string for the bind site to record a known type.
        syn::Expr::Match(m) => {
            let mut arms = m.arms.iter();
            let first_ty = expression_type_string(&arms.next()?.body, ctx)?;
            for arm in arms {
                let arm_ty = expression_type_string(&arm.body, ctx)?;
                if arm_ty != first_ty {
                    return None;
                }
            }
            Some(first_ty)
        }
        _ => None,
    }
}

/// Tail-expression type string of a Rust block — `{ ...; tail }` ⇒
/// `expression_type_string(tail)`.  RPython's flowspace has no
/// block-tail concept (each opcode pushes onto a flat stack), but
/// pyre's surface DSL wraps tail-yielding blocks under `Expr::If` /
/// `Expr::Match` arms and `Expr::Block` / `Expr::Unsafe` operand
/// expressions.  Returns `None` when the block ends in a statement
/// (no tail) — e.g. `{ x = 1; }`.
fn block_tail_type_string(block: &syn::Block, ctx: &GraphBuildContext) -> Option<String> {
    if let Some(syn::Stmt::Expr(tail, None)) = block.stmts.last() {
        expression_type_string(tail, ctx)
    } else {
        None
    }
}

// TODO(name-based-callee-resolution): RPython resolves callees via
// `bookkeeper.getdesc(value)` keyed by Python object identity
// (`annrpython.py` callee resolution), so the name-based lookup
// below is a textual substitute.  Pyre's walker has no Rust `use`
// chain visibility, so a single bare ident at a call site is
// resolved through three text-keyed fallbacks (bare key →
// last-segment → last-two-segment) any of which can mis-route to
// a similarly-named unrelated function.  Convergence path: feed
// the `bookkeeper`-bound `FunctionDesc` directly through
// `GraphBuildContext` so the same host-identity dispatch
// `simple_call` uses at rtyper time also drives return-type
// lookup at AST lowering time, then retire `fn_return_types`
// entirely.  Multi-session — entry conditions are
// `PyreCallRegistry::ensure_session` reaching every call site
// before AST lowering.
fn lookup_function_return_type<'a>(
    ctx: &'a GraphBuildContext,
    segments: &[String],
) -> Option<&'a String> {
    let key = if segments.len() == 1 && !ctx.module_prefix.is_empty() {
        format!("{}::{}", ctx.module_prefix, segments[0])
    } else {
        segments.join("::")
    };
    if let Some(ret) = ctx.fn_return_types.get(&key) {
        return Some(ret);
    }
    // Fallback for cross-module calls reached via `use crate::other::*`:
    // pyre's walker inserts `fn_return_types` entries with the defining
    // module's prefix (e.g. `pyobject::is_int` for
    // `pyre-object/src/pyobject.rs::is_int`), but Rust call sites that
    // imported the function via `use crate::pyobject::*;` invoke it
    // unqualified.  The frontend does not yet track `use` chains, so
    // resolve a bare single-segment lookup against the bare key as a
    // final fallback. Matches RPython's annotator-level whole-program
    // visibility: a globally-visible `def is_int(...) -> bool` is
    // reachable from any module that imports it without re-qualifying
    // the call site (`annrpython.py` resolves callees via `bookkeeper.
    // getdesc(value)` keyed by Python object identity, not source path).
    if segments.len() == 1
        && let Some(ret) = ctx.fn_return_types.get(&segments[0])
    {
        return Some(ret);
    }
    // Closure-bound locals — `let f = |args| body` registers `f` in
    // `local_closure_returns`. Same RPython parity rationale as the
    // bare-key fallback above; the Rust adaptation is needed because
    // pyre's walker has no closure visibility, so the closure return
    // type is recorded at let-binding time.
    if segments.len() == 1
        && let Some(ret) = ctx.local_closure_returns.get(&segments[0])
    {
        return Some(ret);
    }
    // Cross-module call qualified by `crate::module::fn` /
    // `pyre_object::fn` shapes — pyre's walker registers free
    // functions under bare ident (file walker passes `prefix=""`),
    // so a multi-segment crate-relative call falls back to the last
    // segment. Same RPython parity rationale as the bare-single-
    // segment fallback above; whole-program visibility is keyed on
    // host identity, not on source path.
    if segments.len() >= 2
        && let Some(last) = segments.last()
        && let Some(ret) = ctx.fn_return_types.get(last)
    {
        return Some(ret);
    }
    // Multi-segment Impl-method paths — `pyre_object::typeobject::
    // Layout::expands_equal(...)` is registered under the bare
    // `Type::method` shape (`Item::Impl` arm at front/ast.rs:591).
    // Mirror of the same fallback in `expr_unary_not_operand_kind`'s
    // `Expr::Call` arm.
    if segments.len() >= 2 {
        let n = segments.len();
        let bare_impl = format!("{}::{}", segments[n - 2], segments[n - 1]);
        if let Some(ret) = ctx.fn_return_types.get(&bare_impl) {
            return Some(ret);
        }
        // Per-scope canonical-receiver fallback (PyPy
        // `bookkeeper.py:353-409 getdesc` lexical-resolution layering):
        // route the bare receiver leaf through the call site's own
        // `use_imports` + `module_prefix` first (PyPy `frame.f_globals`
        // role), then `STRUCT_ORIGIN_REGISTRY` + bare verbatim fallback
        // — all three encapsulated in `qualify_type_name_with_imports`.
        let canonical_recv =
            qualify_type_name_with_imports(&segments[n - 2], &ctx.module_prefix, &ctx.use_imports);
        if canonical_recv != segments[n - 2] {
            let canonical_key = format!("{}::{}", canonical_recv, segments[n - 1]);
            if let Some(ret) = ctx.fn_return_types.get(&canonical_key) {
                return Some(ret);
            }
        }
    }
    None
}

fn array_item_value_type_from_array_type_id(array_type_id: Option<&str>) -> Option<ValueType> {
    let elem_type = extract_element_type_from_str(array_type_id?)?;
    Some(type_string_to_value_type(&elem_type))
}

/// For `arr[idx]`, returns the ELEMENT TYPE of `arr` from context.
/// This is the Rust equivalent of RPython's `op.args[0].concretetype.TO`
/// which gives `GcArray(T)` — the `T` is what distinguishes array types.
/// RPython: `ARRAY.OF` — extract element type from full ARRAY type string.
///
/// Handles all Rust array/container notations:
/// - `Vec<Point>` → `"Point"` (angle brackets)
/// - `[i64]` → `"i64"` (slice)
/// - `[Point; 10]` → `"Point"` (fixed-size array)
/// - `&[Point]` / `&mut [Point]` — reference / mut-reference prefixes are
///   stripped before the slice form is matched, mirroring
///   `type_root_from_type_string`'s prefix walk so chained
///   `Vec::as_slice` results retain their element type.
fn extract_element_type_from_str(type_str: &str) -> Option<String> {
    let s = strip_pointer_like_prefixes(type_str);
    // Square brackets: [T] or [T; N]
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        // [T; N] → T (strip "; N" suffix)
        let elem = if let Some(semi) = inner.find(';') {
            inner[..semi].trim()
        } else {
            inner.trim()
        };
        if !elem.is_empty() {
            return Some(elem.to_string());
        }
    }
    // Angle brackets: Vec<T>, Box<T>, etc.  This is checked after the
    // outer slice form so `[Rc<T>]` yields `Rc<T>`, not `T`.
    if let (Some(start), Some(end)) = (s.find('<'), s.rfind('>')) {
        if start < end {
            return first_top_level_generic_arg(&s[start + 1..end])
                .map(str::trim)
                .filter(|elem| !elem.is_empty())
                .map(ToOwned::to_owned);
        }
    }
    None
}

fn strip_pointer_like_prefixes(type_str: &str) -> &str {
    let mut s = type_str.trim();
    loop {
        let stripped = s
            .strip_prefix("*const ")
            .or_else(|| s.strip_prefix("*mut "))
            .or_else(|| s.strip_prefix("&mut "))
            .or_else(|| s.strip_prefix("&"));
        match stripped {
            Some(rest) => s = rest.trim_start(),
            None => break,
        }
    }
    s
}

fn outer_generic_inner_type(type_str: &str, wrappers: &[&str]) -> Option<String> {
    let s = strip_pointer_like_prefixes(type_str);
    let start = s.find('<')?;
    let inner = s[start + 1..].strip_suffix('>')?;
    let head = s[..start].trim().rsplit("::").next().unwrap_or("").trim();
    if !wrappers.contains(&head) {
        return None;
    }
    first_top_level_generic_arg(inner)
        .map(str::trim)
        .filter(|inner| !inner.is_empty())
        .map(ToOwned::to_owned)
}

fn method_as_ref_return_type(receiver_ty: &str) -> Option<String> {
    if let Some(inner) = outer_generic_inner_type(receiver_ty, &["Rc", "Arc", "Box", "NonNull"]) {
        return Some(format!("&{}", strip_pointer_like_prefixes(&inner)));
    }
    if let Some(inner) = outer_generic_inner_type(receiver_ty, &["Option"]) {
        return Some(format!("Option<&{}>", strip_pointer_like_prefixes(&inner)));
    }
    extract_element_type_from_str(receiver_ty).map(|elem| format!("&[{}]", elem))
}

fn array_type_id_from_expr(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_array_types.get(&ident.to_string()).cloned()),
        syn::Expr::Reference(r) => array_type_id_from_expr(&r.expr, ctx),
        syn::Expr::Paren(p) => array_type_id_from_expr(&p.expr, ctx),
        // RPython: op.args[0].concretetype — for field access like `self.array`,
        // resolve the field's type from struct_fields to get element type.
        syn::Expr::Field(field) => {
            let owner_type = receiver_type_root(&field.base, ctx)?;
            let field_name = member_name(&field.member);
            // RPython: op.args[0].concretetype — returns full ARRAY type.
            let field_type_str = ctx.struct_fields.field_type_in_scope(
                &owner_type,
                &field_name,
                &ctx.module_prefix,
                &ctx.use_imports,
            )?;
            Some(field_type_str.to_string())
        }
        // RPython: op.result.concretetype — for call expressions like `make_points()[i]`,
        // resolve the return type from the exact callee graph (fn_return_types in pass 1).
        syn::Expr::Call(call) => {
            if let syn::Expr::Path(path) = &*call.func {
                // RPython: exact graph identity — join path segments to match
                // the key format produced by collect_types_from_items.
                // RPython: exact graph identity — qualify bare single-segment
                // calls with module prefix to match registered keys.
                let segments: Vec<String> = path
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                lookup_function_return_type(ctx, &segments).cloned()
            } else {
                None
            }
        }
        // RPython: op.result.concretetype — for method calls like `self.make_points()[i]`.
        // RPython resolves via the exact callee graph — no bare name fallback.
        syn::Expr::MethodCall(mc) => {
            let method_name = mc.method.to_string();
            if matches!(method_name.as_str(), "as_slice" | "as_ref")
                && let Some(ret) = expression_type_string(expr, ctx)
                && extract_element_type_from_str(&ret).is_some()
            {
                return Some(ret);
            }
            let receiver_ty = receiver_type_root(&mc.receiver, ctx)?;
            let key = format!("{}::{}", receiver_ty, method_name);
            ctx.fn_return_types.get(&key).cloned()
        }
        // RPython: op.result.concretetype — for nested index like `matrix[i][j]`,
        // resolve the outer array's element type.
        syn::Expr::Index(idx) => {
            let outer_type = array_type_id_from_expr(&idx.expr, ctx)?;
            let elem = extract_element_type_from_str(&outer_type)?;
            // If the element type is itself an array type, return it
            if elem.starts_with("Vec<") || elem.starts_with('[') {
                Some(elem)
            } else {
                None
            }
        }
        _ => None,
    }
}


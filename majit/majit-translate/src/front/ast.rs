//! AST front-end: build semantic graphs from Rust source.
//!
//! RPython equivalent: flowspace/ — converts source to Block/Link/Variable/SpaceOperation.
//! This module lowers syn AST nodes into FunctionGraph ops with proper data flow (ValueId linking).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use syn::{Item, ItemFn};

use crate::ParsedInterpreter;
use crate::model::{
    BlockId, CallTarget, ExitSwitch, FunctionGraph, ImmutableRank, Link, LinkArg, OpKind,
    UnknownKind, ValueId, ValueType, exception_exitcase,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstGraphOptions;

#[derive(Debug, Clone)]
pub struct SemanticFunction {
    pub name: String,
    pub graph: FunctionGraph,
    /// RPython: `op.result.concretetype` — full return type string.
    /// Used for array identity resolution on Call result values.
    pub return_type: Option<String>,
    /// Owner type for impl methods (e.g. "MyStruct" for `impl MyStruct { fn foo() }`).
    /// Used to construct the full CallPath for return_type registration.
    pub self_ty_root: Option<String>,
    /// RPython: function-level hints set by GC transformer / decorators.
    /// "close_stack" → _gctransformer_hint_close_stack_
    /// "cannot_collect" → _gctransformer_hint_cannot_collect_
    /// "gc_effects" → random_effects_on_gcobjs
    /// "elidable" → _elidable_function_
    /// "loopinvariant" → _jit_loop_invariant_
    pub hints: Vec<String>,
}

/// RPython: struct field type info for `heaptracker.all_interiorfielddescrs`.
/// Maps struct_name → vec of (field_name, field_element_type).
/// `field_element_type` is the array element type when the field is an
/// array container (e.g. `Vec<Point>` → `"Point"`), or the full type
/// string for non-array fields.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StructFieldRegistry {
    /// struct_name → [(field_name, full_field_type_string)]
    pub fields: HashMap<String, Vec<(String, String)>>,
}

impl StructFieldRegistry {
    /// RPython: look up a field's type.  For array-typed fields like
    /// `Vec<Point>`, this returns the full type string `"Vec<Point>"`.
    /// Callers use `array_element_type_from_str` to extract `"Point"`.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.fields.get(owner).and_then(|fields| {
            fields
                .iter()
                .find(|(name, _)| name == field_name)
                .map(|(_, ty)| ty.as_str())
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct SemanticProgram {
    pub functions: Vec<SemanticFunction>,
    /// RPython: known struct types for `get_type_flag(ARRAY.OF)` → FLAG_STRUCT.
    pub known_struct_names: std::collections::HashSet<String>,
    /// Known trait names used to canonicalize local `dyn Trait` family keys.
    pub known_trait_names: std::collections::HashSet<String>,
    /// RPython: struct field types for resolving `op.args[0].concretetype`
    /// on FieldRead-produced array bases.
    pub struct_fields: StructFieldRegistry,
    /// RPython: op.result.concretetype — whole-program function return types.
    /// Maps exact callee path (e.g. "a::helper", "Type::method") → return type.
    /// Stored here so that downstream consumers (parse.rs method graph building)
    /// can use them for array type identity resolution.
    pub fn_return_types: HashMap<String, String>,
    /// RPython: `_immutable_fields_ = [...]` declared on a class body.
    /// Maps struct name → `(field_name, rank)` pairs whose value never
    /// mutates after construction (or is quasi-immutable).  Both bare and
    /// qualified struct keys are inserted (mirroring `struct_fields`) so
    /// the same lookup logic works across module-prefix variants.  Rank
    /// encoding follows `rpython/rtyper/rclass.py:644-678 _parse_field_list`.
    pub immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>>,
}

pub fn build_semantic_program(parsed: &ParsedInterpreter) -> SemanticProgram {
    build_semantic_program_with_options(parsed, &AstGraphOptions::default())
}

pub fn build_semantic_program_from_parsed_files(
    parsed_files: &[ParsedInterpreter],
) -> SemanticProgram {
    build_semantic_program_from_parsed_files_with_options(parsed_files, &AstGraphOptions::default())
}

/// Qualify a bare type name with module prefix.
/// "Foo" with prefix "a" → "a::Foo". Already-qualified "a::Foo" unchanged.
/// Empty prefix → return bare name as-is.
pub(crate) fn qualify_type_name(bare: &str, prefix: &str) -> String {
    if prefix.is_empty() || bare.contains("::") {
        bare.to_string()
    } else {
        format!("{}::{}", prefix, bare)
    }
}

/// RPython: annotator whole-program type collection.
/// Recursively collects struct definitions, function return types, and impl
/// method return types from items, handling `mod` blocks with qualified paths.
/// The `prefix` carries the module path (e.g. "a::b") to produce exact callee
/// identities matching what `canonical_call_target` generates at call sites.
fn collect_types_from_items(
    items: &[Item],
    prefix: &str,
    known_struct_names: &mut std::collections::HashSet<String>,
    known_trait_names: &mut std::collections::HashSet<String>,
    struct_fields: &mut StructFieldRegistry,
    fn_return_types: &mut HashMap<String, String>,
    immutable_fields: &mut HashMap<String, Vec<(String, ImmutableRank)>>,
) {
    // RPython: annotator/rtyper resolves all types in a whole-program pass.
    // Two-pass: first collect ALL struct names, then field types + return types.
    // This ensures qualified_full_type_string can identify known structs
    // regardless of source order (RPython's lltype T.TO identity).
    collect_struct_names(items, prefix, known_struct_names);
    collect_trait_names(items, prefix, known_trait_names);
    collect_fields_and_returns(
        items,
        prefix,
        known_struct_names,
        known_trait_names,
        struct_fields,
        fn_return_types,
        immutable_fields,
    );
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
fn collect_fields_and_returns(
    items: &[Item],
    prefix: &str,
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
                        let field_type = qualified_full_type_string(
                            &f.ty,
                            prefix,
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
                    struct_fields.fields.insert(bare_name, fields);
                } else {
                    let qualified = format!("{}::{}", prefix, bare_name);
                    struct_fields.fields.insert(qualified, fields.clone());
                    struct_fields.fields.entry(bare_name).or_insert(fields);
                }
            }
            Item::Fn(func) => {
                // RPython: op.result.concretetype — module-qualified return type.
                let ret_ty = match &func.sig.output {
                    syn::ReturnType::Type(_, ty) => qualified_full_type_string(
                        ty,
                        prefix,
                        known_struct_names,
                        known_trait_names,
                    ),
                    syn::ReturnType::Default => None,
                };
                if let Some(ret_ty) = ret_ty {
                    let key = if prefix.is_empty() {
                        func.sig.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, func.sig.ident)
                    };
                    fn_return_types.insert(key, ret_ty);
                }
            }
            Item::Impl(impl_block) => {
                let self_ty_root = type_root_ident(&impl_block.self_ty);
                for sub in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = sub {
                        let ret_ty = match &method.sig.output {
                            syn::ReturnType::Type(_, ty) => qualified_full_type_string(
                                ty,
                                prefix,
                                known_struct_names,
                                known_trait_names,
                            ),
                            syn::ReturnType::Default => None,
                        };
                        if let Some(ret_ty) = ret_ty {
                            if let Some(ref ty_root) = self_ty_root {
                                let qualified_ty = qualify_type_name(ty_root, prefix);
                                fn_return_types.insert(
                                    format!("{}::{}", qualified_ty, method.sig.ident),
                                    ret_ty,
                                );
                            }
                        }
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

/// RPython: pass 2 graph building with Item::Mod recursion.
/// Mirrors collect_types_from_items traversal so that module-internal
/// functions get proper SemanticFunction entries with qualified names.
fn build_graphs_from_items(
    items: &[Item],
    prefix: &str,
    options: &AstGraphOptions,
    struct_fields: &StructFieldRegistry,
    fn_return_types: &HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
    functions: &mut Vec<SemanticFunction>,
) {
    for item in items {
        match item {
            Item::Fn(func) => {
                let mut sf = build_function_graph(
                    func,
                    options,
                    None,
                    struct_fields,
                    fn_return_types,
                    prefix,
                    known_struct_names,
                    known_trait_names,
                );
                // RPython: exact graph identity — module-qualified name.
                if !prefix.is_empty() {
                    sf.name = format!("{}::{}", prefix, sf.name);
                }
                functions.push(sf);
            }
            Item::Impl(impl_block) => {
                // Qualify bare self type with module prefix (RPython: unique type identity).
                let self_ty_root =
                    type_root_ident(&impl_block.self_ty).map(|t| qualify_type_name(&t, prefix));
                for sub in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = sub {
                        let fake_fn = ItemFn {
                            attrs: method.attrs.clone(),
                            vis: syn::Visibility::Inherited,
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        functions.push(build_function_graph(
                            &fake_fn,
                            options,
                            self_ty_root.clone(),
                            struct_fields,
                            fn_return_types,
                            prefix,
                            known_struct_names,
                            known_trait_names,
                        ));
                    }
                }
            }
            Item::Mod(m) => {
                if let Some((_, ref items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    build_graphs_from_items(
                        items,
                        &mod_prefix,
                        options,
                        struct_fields,
                        fn_return_types,
                        known_struct_names,
                        known_trait_names,
                        functions,
                    );
                }
            }
            _ => {}
        }
    }
}

pub fn build_semantic_program_with_options(
    parsed: &ParsedInterpreter,
    options: &AstGraphOptions,
) -> SemanticProgram {
    let mut functions = Vec::new();
    let mut known_struct_names = std::collections::HashSet::new();
    let mut known_trait_names = std::collections::HashSet::new();
    let mut struct_fields = StructFieldRegistry::default();

    // Pass 1: collect all struct definitions and function return types.
    // RPython: annotator/rtyper resolves all types in a whole-program pass.
    // We recursively traverse Item::Mod to register module-qualified paths
    // matching the exact callee identity that canonical_call_target produces.
    let mut fn_return_types: HashMap<String, String> = HashMap::new();
    let mut immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>> = HashMap::new();
    collect_types_from_items(
        &parsed.file.items,
        "",
        &mut known_struct_names,
        &mut known_trait_names,
        &mut struct_fields,
        &mut fn_return_types,
        &mut immutable_fields,
    );

    // Pass 2: build function graphs with struct_fields + fn_return_types.
    // Field types are already module-qualified at the source (via
    // qualified_full_type_string), matching RPython's lltype identity.
    build_graphs_from_items(
        &parsed.file.items,
        "",
        options,
        &struct_fields,
        &fn_return_types,
        &known_struct_names,
        &known_trait_names,
        &mut functions,
    );

    SemanticProgram {
        functions,
        known_struct_names,
        known_trait_names,
        struct_fields,
        fn_return_types,
        immutable_fields,
    }
}

pub fn build_semantic_program_from_parsed_files_with_options(
    parsed_files: &[ParsedInterpreter],
    options: &AstGraphOptions,
) -> SemanticProgram {
    // RPython: annotator/rtyper provides whole-program type info before
    // the codewriter runs. We emulate this with a 2-pass approach:
    // Pass 1: collect ALL struct definitions and function return types across ALL files.
    // Uses collect_types_from_items to handle Item::Mod recursively with
    // qualified paths matching canonical_call_target identity.
    let mut known_struct_names = std::collections::HashSet::new();
    let mut known_trait_names = std::collections::HashSet::new();
    let mut struct_fields = StructFieldRegistry::default();
    let mut fn_return_types: HashMap<String, String> = HashMap::new();
    let mut immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>> = HashMap::new();
    // RPython: whole-program — ALL types visible everywhere.
    // Collect struct names from ALL files first, then fields+returns.
    for parsed in parsed_files {
        collect_struct_names(&parsed.file.items, "", &mut known_struct_names);
        collect_trait_names(&parsed.file.items, "", &mut known_trait_names);
    }
    for parsed in parsed_files {
        collect_fields_and_returns(
            &parsed.file.items,
            "",
            &known_struct_names,
            &known_trait_names,
            &mut struct_fields,
            &mut fn_return_types,
            &mut immutable_fields,
        );
    }
    // Pass 2: build function graphs with merged struct_fields + fn_return_types visible.
    // Field types already module-qualified at source (qualified_full_type_string).
    let mut functions = Vec::new();
    for parsed in parsed_files {
        build_graphs_from_items(
            &parsed.file.items,
            "",
            options,
            &struct_fields,
            &fn_return_types,
            &known_struct_names,
            &known_trait_names,
            &mut functions,
        );
    }
    SemanticProgram {
        functions,
        known_struct_names,
        known_trait_names,
        struct_fields,
        fn_return_types,
        immutable_fields,
    }
}

/// Public entry for building a graph from a single function AST node.
/// Lower a standalone expression into an existing graph.
/// Used to build semantic graphs from opcode match arm bodies.
pub fn lower_expr_into_graph(graph: &mut FunctionGraph, expr: &syn::Expr) {
    let mut block = graph.startblock;
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = HashMap::new();
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    let mut ctx = GraphBuildContext::new(
        &empty_registry,
        &empty_fn_ret,
        "",
        &empty_names,
        &empty_trait_names,
    );
    let result = lower_expr(
        graph,
        &mut block,
        expr,
        &AstGraphOptions::default(),
        &mut ctx,
    );
    graph.set_return(block, result);
}

pub fn build_function_graph_pub(func: &ItemFn) -> SemanticFunction {
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = HashMap::new();
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    build_function_graph(
        func,
        &AstGraphOptions::default(),
        None,
        &empty_registry,
        &empty_fn_ret,
        "",
        &empty_names,
        &empty_trait_names,
    )
}

pub fn build_function_graph_with_self_ty_pub(
    func: &ItemFn,
    self_ty_root: Option<String>,
    struct_fields: &StructFieldRegistry,
    fn_return_types: &HashMap<String, String>,
    module_prefix: &str,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> SemanticFunction {
    build_function_graph(
        func,
        &AstGraphOptions::default(),
        self_ty_root,
        struct_fields,
        fn_return_types,
        module_prefix,
        known_struct_names,
        known_trait_names,
    )
}

#[derive(Debug, Clone)]
struct GraphBuildContext<'a> {
    local_type_roots: HashMap<String, String>,
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
    /// RPython: program-level struct field types, available for resolving
    /// field access array identity (e.g. `self.array[i]` → owner.field_type).
    struct_fields: &'a StructFieldRegistry,
    /// RPython: op.result.concretetype — function return types from the annotator.
    /// Maps function name (or "Type::method") → return type string.
    /// Used by array_type_id_from_expr to resolve Call/MethodCall expressions.
    fn_return_types: &'a HashMap<String, String>,
    /// Module path prefix for qualifying bare type names.
    /// RPython: lltype identity is globally unique — bare "Foo" in mod "a"
    /// must resolve to "a::Foo" in struct_fields lookups.
    module_prefix: String,
    known_struct_names: &'a std::collections::HashSet<String>,
    known_trait_names: &'a std::collections::HashSet<String>,
}

impl<'a> GraphBuildContext<'a> {
    fn new(
        struct_fields: &'a StructFieldRegistry,
        fn_return_types: &'a HashMap<String, String>,
        module_prefix: &str,
        known_struct_names: &'a std::collections::HashSet<String>,
        known_trait_names: &'a std::collections::HashSet<String>,
    ) -> Self {
        Self {
            local_type_roots: HashMap::new(),
            local_array_types: HashMap::new(),
            local_dyn_trait_roots: HashMap::new(),
            struct_fields,
            fn_return_types,
            module_prefix: module_prefix.to_string(),
            known_struct_names,
            known_trait_names,
        }
    }
}

fn build_function_graph(
    func: &ItemFn,
    options: &AstGraphOptions,
    self_ty_root: Option<String>,
    struct_fields: &StructFieldRegistry,
    fn_return_types: &HashMap<String, String>,
    module_prefix: &str,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> SemanticFunction {
    let mut graph = FunctionGraph::new(func.sig.ident.to_string());
    let mut entry = graph.startblock;
    let mut ctx = GraphBuildContext::new(
        struct_fields,
        fn_return_types,
        module_prefix,
        known_struct_names,
        known_trait_names,
    );

    // Register function parameters as Input ops AND on `Block.inputargs`.
    //
    // RPython parity: `Block.inputargs` is the function's formal parameter
    // list for the startblock (`flowspace/model.py` Block class).  Pyre
    // originally only emitted `OpKind::Input` ops here — but because body
    // `Expr::Path` lowering also emits `OpKind::Input` for plain variable
    // references (`front/ast.rs:1271-1287`), counting startblock `Input`
    // ops after lowering can no longer tell "parameter" from "body
    // reference" apart.  Populating `inputargs` during parameter
    // registration preserves the RPython `startblock.inputargs == params`
    // invariant and is what `getcalldescr`'s `FUNC.ARGS` check reads
    // (RPython `call.py:220-221`).
    for param in &func.sig.inputs {
        match param {
            syn::FnArg::Receiver(_) => {
                if let Some(self_ty_root) = &self_ty_root {
                    ctx.local_type_roots
                        .insert("self".to_string(), self_ty_root.clone());
                }
                if let Some(vid) = graph.push_op(
                    entry,
                    OpKind::Input {
                        name: "self".to_string(),
                        ty: ValueType::Unknown,
                    },
                    true,
                ) {
                    graph.name_value(vid, "self".to_string());
                    graph.block_mut(entry).inputargs.push(vid);
                }
            }
            syn::FnArg::Typed(pat_type) => {
                let name = canonical_pat_name(&pat_type.pat);
                if let Some(type_root) = type_root_ident(&pat_type.ty) {
                    // Qualify bare type with module prefix for exact identity.
                    let qualified = qualify_type_name(&type_root, &ctx.module_prefix);
                    ctx.local_type_roots.insert(name.clone(), qualified);
                }
                if let Some(full_type) = qualified_full_type_string(
                    &pat_type.ty,
                    &ctx.module_prefix,
                    ctx.known_struct_names,
                    ctx.known_trait_names,
                ) {
                    ctx.local_array_types.insert(name.clone(), full_type);
                }
                if let Some(trait_root) = extract_dyn_trait_root_with_context(
                    &pat_type.ty,
                    &ctx.module_prefix,
                    ctx.known_trait_names,
                ) {
                    ctx.local_dyn_trait_roots.insert(name.clone(), trait_root);
                }
                if let Some(vid) = graph.push_op(
                    entry,
                    OpKind::Input {
                        name: name.clone(),
                        ty: ValueType::Unknown,
                    },
                    true,
                ) {
                    graph.name_value(vid, name);
                    graph.block_mut(entry).inputargs.push(vid);
                }
            }
        }
    }

    // Lower function body
    for stmt in &func.block.stmts {
        lower_stmt(&mut graph, &mut entry, stmt, options, &mut ctx);
    }

    // Default terminator if none was set
    if graph.block(entry).is_open() {
        graph.set_return(entry, None);
    }

    // RPython: op.result.concretetype — module-qualified for exact type identity.
    let return_type = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => {
            qualified_full_type_string(ty, module_prefix, known_struct_names, known_trait_names)
        }
        syn::ReturnType::Default => None,
    };

    // RPython: function-level hints from decorators / GC transformer.
    // Scan #[jit_*] attributes to detect elidable, loopinvariant,
    // close_stack, cannot_collect, gc_effects.
    let hints = collect_jit_hints(&func.attrs);

    SemanticFunction {
        name: func.sig.ident.to_string(),
        graph,
        return_type,
        self_ty_root,
        hints,
    }
}

/// RPython: extract function-level JIT hints from attributes.
/// Maps JIT hint attributes to effectinfo classification strings.
///
/// Recognizes both legacy `jit_*` and RPython-parity names.
///
/// For `#[oopspec("spec")]`, returns `"oopspec:spec_string"` so the hint
/// consumer can extract the spec via `hint.strip_prefix("oopspec:")`.
fn collect_jit_hints(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut hints = Vec::new();
    for attr in attrs {
        if let Some(ident) = attr.path().get_ident() {
            let name = ident.to_string();
            match name.as_str() {
                // RPython-parity names (rlib/jit.py)
                "elidable" | "jit_elidable" => hints.push("elidable".into()),
                "elidable_promote" => hints.push("elidable".into()),
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
                }
                // majit-specific
                "jit_close_stack" => hints.push("close_stack".into()),
                "jit_cannot_collect" => hints.push("cannot_collect".into()),
                "jit_gc_effects" => hints.push("gc_effects".into()),
                _ => {}
            }
        }
    }
    hints
}

// ── Statement lowering ──────────────────────────────────────────

/// Public entry point for lowering a single statement into a graph.
/// Used by the graph-based classifier in lib.rs to analyze resolved method bodies.
pub fn lower_stmt_pub(graph: &mut FunctionGraph, block: BlockId, stmt: &syn::Stmt) {
    let mut block = block;
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = HashMap::new();
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    let mut ctx = GraphBuildContext::new(
        &empty_registry,
        &empty_fn_ret,
        "",
        &empty_names,
        &empty_trait_names,
    );
    lower_stmt(
        graph,
        &mut block,
        stmt,
        &AstGraphOptions::default(),
        &mut ctx,
    );
}

fn lower_stmt(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    stmt: &syn::Stmt,
    options: &AstGraphOptions,
    ctx: &mut GraphBuildContext,
) {
    match stmt {
        syn::Stmt::Expr(expr, _) => {
            lower_expr(graph, block, expr, options, ctx);
        }
        syn::Stmt::Local(local) => {
            // RPython: rtyper assigns concretetype to let-bound variables.
            // Extract array element type from type annotations on let bindings.
            if let syn::Pat::Type(pat_type) = &local.pat {
                let name = canonical_pat_name(&pat_type.pat);
                if let Some(type_root) = type_root_ident(&pat_type.ty) {
                    let qualified = qualify_type_name(&type_root, &ctx.module_prefix);
                    ctx.local_type_roots.insert(name.clone(), qualified);
                }
                if let Some(full_type) = qualified_full_type_string(
                    &pat_type.ty,
                    &ctx.module_prefix,
                    ctx.known_struct_names,
                    ctx.known_trait_names,
                ) {
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
                let result = lower_expr(graph, block, &init.expr, options, ctx);
                // Record variable name (RPython Variable._name)
                if let Some(vid) = result {
                    if let syn::Pat::Ident(pat_ident) = &local.pat {
                        graph.name_value(vid, pat_ident.ident.to_string());
                    } else if let syn::Pat::Type(pat_type) = &local.pat {
                        let name = canonical_pat_name(&pat_type.pat);
                        graph.name_value(vid, name);
                    }
                }
            }
        }
        syn::Stmt::Macro(_) => {
            graph.push_op(
                *block,
                OpKind::Unknown {
                    kind: UnknownKind::MacroStmt,
                },
                false,
            );
        }
        syn::Stmt::Item(_) => {}
    }
}

// ── Expression lowering (block-splitting for control flow) ───────

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
) -> Option<ValueId> {
    match expr {
        // ── receiver.field / arr[i].field ──
        syn::Expr::Field(field) => {
            if let syn::Expr::Index(idx) = &*field.base {
                // RPython: getinteriorfield_gc — arr[i].field as a single op.
                let base = lower_expr(graph, block, &idx.expr, options, ctx)
                    .unwrap_or_else(|| graph.alloc_value());
                let index = lower_expr(graph, block, &idx.index, options, ctx)
                    .unwrap_or_else(|| graph.alloc_value());
                let field_name = member_name(&field.member);
                let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
                // Element struct type is the field owner for interiorfield descriptors.
                let elem_type = array_type_id
                    .as_ref()
                    .and_then(|atid| extract_element_type_from_str(atid));
                // RPython: getkind(op.result.concretetype) — resolve field type
                // from struct field registry for the kind suffix (i/r/f).
                let item_ty = elem_type
                    .as_ref()
                    .and_then(|owner| ctx.struct_fields.field_type(owner, &field_name))
                    .map(type_string_to_value_type)
                    .unwrap_or(ValueType::Unknown);
                graph.push_op(
                    *block,
                    OpKind::InteriorFieldRead {
                        base,
                        index,
                        field: crate::model::FieldDescriptor::new(field_name, elem_type),
                        item_ty,
                        array_type_id,
                    },
                    true,
                )
            } else {
                let base = lower_expr(graph, block, &field.base, options, ctx)
                    .unwrap_or_else(|| graph.alloc_value());
                let field_name = member_name(&field.member);
                let ty = field_value_type_from_expr(&field.base, &field.member, ctx)
                    .unwrap_or(ValueType::Unknown);
                graph.push_op(
                    *block,
                    OpKind::FieldRead {
                        base,
                        field: crate::model::FieldDescriptor::new(
                            field_name,
                            receiver_type_root(&field.base, ctx),
                        ),
                        ty,
                        pure: false,
                    },
                    true,
                )
            }
        }

        // ── base[index] ──
        syn::Expr::Index(idx) => {
            let base = lower_expr(graph, block, &idx.expr, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            let index = lower_expr(graph, block, &idx.index, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
            let item_ty = array_item_value_type_from_array_type_id(array_type_id.as_deref())
                .unwrap_or(ValueType::Unknown);
            graph.push_op(
                *block,
                OpKind::ArrayRead {
                    base,
                    index,
                    item_ty,
                    array_type_id,
                },
                true,
            )
        }

        // ── lhs = rhs ──
        syn::Expr::Assign(assign) => {
            let value = lower_expr(graph, block, &assign.right, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());

            match &*assign.left {
                syn::Expr::Field(field) => {
                    if let syn::Expr::Index(idx) = &*field.base {
                        // RPython: setinteriorfield_gc — arr[i].field = value.
                        let base = lower_expr(graph, block, &idx.expr, options, ctx)
                            .unwrap_or_else(|| graph.alloc_value());
                        let index = lower_expr(graph, block, &idx.index, options, ctx)
                            .unwrap_or_else(|| graph.alloc_value());
                        let field_name = member_name(&field.member);
                        let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
                        let elem_type = array_type_id
                            .as_ref()
                            .and_then(|atid| extract_element_type_from_str(atid));
                        // RPython: getkind(v_value.concretetype) — resolve field type
                        // from struct field registry for the kind suffix (i/r/f).
                        let item_ty = elem_type
                            .as_ref()
                            .and_then(|owner| ctx.struct_fields.field_type(owner, &field_name))
                            .map(type_string_to_value_type)
                            .unwrap_or(ValueType::Unknown);
                        graph.push_op(
                            *block,
                            OpKind::InteriorFieldWrite {
                                base,
                                index,
                                field: crate::model::FieldDescriptor::new(field_name, elem_type),
                                value,
                                item_ty,
                                array_type_id,
                            },
                            false,
                        );
                    } else {
                        let base = lower_expr(graph, block, &field.base, options, ctx)
                            .unwrap_or_else(|| graph.alloc_value());
                        let field_name = member_name(&field.member);
                        let ty = field_value_type_from_expr(&field.base, &field.member, ctx)
                            .unwrap_or(ValueType::Unknown);
                        graph.push_op(
                            *block,
                            OpKind::FieldWrite {
                                base,
                                field: crate::model::FieldDescriptor::new(
                                    field_name,
                                    receiver_type_root(&field.base, ctx),
                                ),
                                value,
                                ty,
                            },
                            false,
                        );
                    }
                }
                syn::Expr::Index(idx) => {
                    let base = lower_expr(graph, block, &idx.expr, options, ctx)
                        .unwrap_or_else(|| graph.alloc_value());
                    let index = lower_expr(graph, block, &idx.index, options, ctx)
                        .unwrap_or_else(|| graph.alloc_value());
                    let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
                    let item_ty =
                        array_item_value_type_from_array_type_id(array_type_id.as_deref())
                            .unwrap_or(ValueType::Unknown);
                    graph.push_op(
                        *block,
                        OpKind::ArrayWrite {
                            base,
                            index,
                            value,
                            item_ty,
                            array_type_id,
                        },
                        false,
                    );
                }
                _ => {
                    // Generic assignment — value already lowered
                }
            }
            None
        }

        // ── function call ──
        syn::Expr::Call(call) => {
            let args: Vec<ValueId> = call
                .args
                .iter()
                .filter_map(|a| lower_expr(graph, block, a, options, ctx))
                .collect();
            let target = canonical_call_target(&call.func, &ctx.module_prefix);
            graph.push_op(
                *block,
                OpKind::Call {
                    target,
                    args,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── method call ──
        syn::Expr::MethodCall(mc) => {
            let mut args = Vec::new();
            if let Some(recv) = lower_expr(graph, block, &mc.receiver, options, ctx) {
                args.push(recv);
            }
            for a in &mc.args {
                if let Some(v) = lower_expr(graph, block, a, options, ctx) {
                    args.push(v);
                }
            }
            // RPython `jtransform.py:410-412`: a polymorphic receiver
            // (dyn Trait) lowers to `indirect_call`, not `direct_call`.
            // Detect via the collected local_dyn_trait_roots map so
            // locals / params / Box<dyn> receivers all participate
            // (Issue 3 coverage).
            let target = if let Some(trait_root) = dyn_trait_root_for_receiver(&mc.receiver, ctx) {
                CallTarget::indirect(trait_root, mc.method.to_string())
            } else {
                CallTarget::method(mc.method.to_string(), receiver_type_root(&mc.receiver, ctx))
            };
            graph.push_op(
                *block,
                OpKind::Call {
                    target,
                    args,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── if/else → block split (RPython FlowContext.guessbool) ──
        //
        // Creates: then_block, else_block, merge_block
        // If both branches produce a value, merge_block gets an inputarg
        // (Phi node) that receives the value from each branch via Link args.
        syn::Expr::If(if_expr) => {
            let cond = lower_expr(graph, block, &if_expr.cond, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());

            let mut then_block = graph.create_block();
            let mut else_block = graph.create_block();

            graph.set_branch(*block, cond, then_block, vec![], else_block, vec![]);

            // Lower then branch — collect result value
            let mut then_result = None;
            for stmt in &if_expr.then_branch.stmts {
                lower_stmt(graph, &mut then_block, stmt, options, ctx);
            }
            // Last expression in then_branch is the result (if no explicit return)
            if let Some(last) = if_expr.then_branch.stmts.last() {
                if let syn::Stmt::Expr(e, None) = last {
                    then_result = lower_expr(graph, &mut then_block, e, options, ctx);
                }
            }

            // Lower else branch
            let mut else_result = None;
            if let Some((_, else_branch)) = &if_expr.else_branch {
                else_result = lower_expr(graph, &mut else_block, else_branch, options, ctx);
            }

            // Create merge block with Phi if both branches have values
            let (merge_block, phi_result) = if then_result.is_some() && else_result.is_some() {
                let (merge, phi_args) = graph.create_block_with_args(1);
                // Link args: then → merge(then_result), else → merge(else_result)
                if graph.block(then_block).is_open() {
                    graph.set_goto(then_block, merge, vec![then_result.unwrap()]);
                }
                if graph.block(else_block).is_open() {
                    graph.set_goto(else_block, merge, vec![else_result.unwrap()]);
                }
                (merge, Some(phi_args[0]))
            } else {
                let merge = graph.create_block();
                if graph.block(then_block).is_open() {
                    graph.set_goto(then_block, merge, vec![]);
                }
                if graph.block(else_block).is_open() {
                    graph.set_goto(else_block, merge, vec![]);
                }
                (merge, None)
            };

            *block = merge_block;
            phi_result
        }

        // ── return ──
        syn::Expr::Return(ret) => {
            let val = ret
                .expr
                .as_ref()
                .and_then(|e| lower_expr(graph, block, e, options, ctx));
            graph.set_return(*block, val);
            None
        }

        // ── block { stmts } ──
        syn::Expr::Block(blk) => {
            let mut last = None;
            for stmt in &blk.block.stmts {
                lower_stmt(graph, block, stmt, options, ctx);
                if let syn::Stmt::Expr(e, None) = stmt {
                    last = lower_expr(graph, block, e, options, ctx);
                }
            }
            last
        }

        // ── literals ──
        syn::Expr::Lit(lit) => {
            if let syn::Lit::Int(int_lit) = &lit.lit {
                if let Ok(v) = int_lit.base10_parse::<i64>() {
                    return graph.push_op(*block, OpKind::ConstInt(v), true);
                }
            }
            graph.push_op(
                *block,
                OpKind::Unknown {
                    kind: UnknownKind::UnsupportedLiteral,
                },
                true,
            )
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
            graph.push_op(
                *block,
                OpKind::Input {
                    name,
                    ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── reference &expr ──
        syn::Expr::Reference(r) => lower_expr(graph, block, &r.expr, options, ctx),

        // ── parenthesized (expr) ──
        syn::Expr::Paren(p) => lower_expr(graph, block, &p.expr, options, ctx),

        // ── unary !x, -x ──
        syn::Expr::Unary(u) => {
            let operand = lower_expr(graph, block, &u.expr, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            graph.push_op(
                *block,
                OpKind::UnaryOp {
                    op: unary_op_name(&u.op).into(),
                    operand,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── binary a + b ──
        syn::Expr::Binary(bin) => {
            let lhs = lower_expr(graph, block, &bin.left, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            let rhs = lower_expr(graph, block, &bin.right, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            graph.push_op(
                *block,
                OpKind::BinOp {
                    op: binary_op_name(&bin.op).into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── cast: expr as T ──
        syn::Expr::Cast(cast) => lower_expr(graph, block, &cast.expr, options, ctx),

        // ── match expr { arms } → multi-block (RPython switch) ──
        syn::Expr::Match(m) => {
            let scrutinee = lower_expr(graph, block, &m.expr, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());

            if m.arms.is_empty() {
                return None;
            }

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
            let mut arm_entries: Vec<BlockId> = Vec::with_capacity(m.arms.len());
            let mut arm_tails: Vec<(BlockId, Option<ValueId>)> = Vec::with_capacity(m.arms.len());
            for arm in &m.arms {
                let entry = graph.create_block();
                let mut tail = entry;
                let result = lower_expr(graph, &mut tail, &arm.body, options, ctx);
                arm_entries.push(entry);
                arm_tails.push((tail, result));
            }

            // Any-arm-has-value → merge gets a Phi inputarg; otherwise
            // merge is a plain block and all arm gotos carry no args.
            let any_has_value = arm_tails.iter().any(|(_, r)| r.is_some());
            let (merge, merge_phi) = if any_has_value {
                let (m_block, phi_args) = graph.create_block_with_args(1);
                (m_block, Some(phi_args[0]))
            } else {
                (graph.create_block(), None)
            };

            for (tail, result) in &arm_tails {
                if !graph.block(*tail).is_open() {
                    continue;
                }
                let goto_args = if any_has_value {
                    vec![result.unwrap_or_else(|| graph.alloc_value())]
                } else {
                    Vec::new()
                };
                graph.set_goto(*tail, merge, goto_args);
            }

            // First arm as default branch (simplified)
            if m.arms.len() == 1 {
                graph.set_goto(*block, arm_entries[0], vec![]);
            } else {
                // Binary branch on scrutinee for first arm, else second.
                // If the else-fallthrough goes to `merge` directly
                // (arm count < 2), carry the matching phi argument so
                // merge's inputarg arity is respected.
                let first_block = arm_entries[0];
                let second_block = arm_entries.get(1).copied().unwrap_or(merge);
                let false_args = if second_block == merge && any_has_value {
                    vec![graph.alloc_value()]
                } else {
                    Vec::new()
                };
                graph.set_branch(
                    *block,
                    scrutinee,
                    first_block,
                    vec![],
                    second_block,
                    false_args,
                );
            }

            *block = merge;
            merge_phi
        }

        // ── while → header block + body block + exit block ──
        syn::Expr::While(w) => {
            let header_entry = graph.create_block();
            let body_entry = graph.create_block();
            let exit = graph.create_block();

            // Current block → header_entry (loop-head, 0 inputargs).
            graph.set_goto(*block, header_entry, vec![]);

            // Header: evaluate condition, branch to body or exit.
            // `lower_expr(&mut header_tail, ...)` may rewire to a
            // sub-merge; the cond-branch attaches to header_tail so
            // the branch lives at the header's actual end.
            let mut header_tail = header_entry;
            let cond = lower_expr(graph, &mut header_tail, &w.cond, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            graph.set_branch(header_tail, cond, body_entry, vec![], exit, vec![]);

            // Body → back to header_entry (entry, not tail —
            // header_entry is the 0-inputarg loop-head).
            let mut body_tail = body_entry;
            for stmt in &w.body.stmts {
                lower_stmt(graph, &mut body_tail, stmt, options, ctx);
            }
            if graph.block(body_tail).is_open() {
                graph.set_goto(body_tail, header_entry, vec![]);
            }

            *block = exit;
            None
        }
        syn::Expr::Loop(l) => {
            let body_entry = graph.create_block();
            let exit = graph.create_block();

            graph.set_goto(*block, body_entry, vec![]);

            let mut body_tail = body_entry;
            for stmt in &l.body.stmts {
                lower_stmt(graph, &mut body_tail, stmt, options, ctx);
            }
            if graph.block(body_tail).is_open() {
                graph.set_goto(body_tail, body_entry, vec![]);
            }

            *block = exit;
            None
        }
        syn::Expr::ForLoop(f) => {
            let header_entry = graph.create_block();
            let body_entry = graph.create_block();
            let exit = graph.create_block();

            graph.set_goto(*block, header_entry, vec![]);

            let mut header_tail = header_entry;
            lower_expr(graph, &mut header_tail, &f.expr, options, ctx);
            let iter_cond = graph.alloc_value();
            graph.set_branch(header_tail, iter_cond, body_entry, vec![], exit, vec![]);

            let mut body_tail = body_entry;
            for stmt in &f.body.stmts {
                lower_stmt(graph, &mut body_tail, stmt, options, ctx);
            }
            if graph.block(body_tail).is_open() {
                graph.set_goto(body_tail, header_entry, vec![]);
            }

            *block = exit;
            None
        }

        // ── break/continue ──
        syn::Expr::Break(b) => {
            if let Some(e) = &b.expr {
                lower_expr(graph, block, e, options, ctx);
            }
            None
        }
        syn::Expr::Continue(_) => None,

        // ── closure ──
        syn::Expr::Closure(c) => lower_expr(graph, block, &c.body, options, ctx),

        // ── tuple (a, b, c) ──
        syn::Expr::Tuple(t) => {
            let mut last = None;
            for e in &t.elems {
                last = lower_expr(graph, block, e, options, ctx);
            }
            last
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
            let inner = lower_expr(graph, block, &t.expr, options, ctx)?;
            let continuation = graph.create_block();
            let continuation_arg = graph.alloc_value();
            graph
                .block_mut(continuation)
                .inputargs
                .push(continuation_arg);
            // RPython `flowcontext.py:130-133` — fresh prevblock-side
            // `Variable('last_exception')` + `Variable('last_exc_value')`.
            let last_exception = graph.alloc_value();
            let last_exc_value = graph.alloc_value();
            let exc_block = graph.exceptblock;
            graph.set_goto(*block, continuation, vec![inner]);
            graph.set_control_flow_metadata(
                *block,
                Some(ExitSwitch::LastException),
                vec![
                    // RPython `flowcontext.py:141` `Link(vars=[], egg, case=None)`
                    // for the normal fall-through.
                    Link::new(vec![inner], continuation, None),
                    // RPython `flowcontext.py:141-143` `link = Link(vars, egg, case)`
                    // + `link.extravars(last_exception=..., last_exc_value=...)`
                    // with `case is Exception`.
                    Link::new(
                        vec![last_exception, last_exc_value],
                        exc_block,
                        Some(exception_exitcase()),
                    )
                    .extravars(
                        Some(LinkArg::from(last_exception)),
                        Some(LinkArg::from(last_exc_value)),
                    ),
                ],
            );
            *block = continuation;
            Some(continuation_arg)
        }

        // ── fallback ──
        _ => graph.push_op(
            *block,
            OpKind::Unknown {
                kind: UnknownKind::UnsupportedExpr,
            },
            true,
        ),
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn unary_op_name(op: &syn::UnOp) -> &'static str {
    match op {
        syn::UnOp::Deref(_) => "deref",
        syn::UnOp::Not(_) => "not",
        syn::UnOp::Neg(_) => "neg",
        _ => "unknown_unary",
    }
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
fn canonical_call_target(expr: &syn::Expr, module_prefix: &str) -> CallTarget {
    match expr {
        syn::Expr::Path(path) => {
            let mut segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect();
            if segments.len() == 1 && !module_prefix.is_empty() {
                let mut qualified = module_prefix
                    .split("::")
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();
                qualified.extend(segments);
                segments = qualified;
            }
            CallTarget::function_path(segments)
        }
        _ => CallTarget::UnsupportedExpr,
    }
}

fn receiver_type_root(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_type_roots.get(&ident.to_string()).cloned()),
        syn::Expr::Reference(reference) => receiver_type_root(&reference.expr, ctx),
        syn::Expr::Paren(paren) => receiver_type_root(&paren.expr, ctx),
        syn::Expr::Field(field) => receiver_type_root(&field.base, ctx),
        syn::Expr::Index(index) => receiver_type_root(&index.expr, ctx),
        _ => None,
    }
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
            let field_type = ctx.struct_fields.field_type(&owner, &field_name)?;
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

/// RPython: lltype graph identity — returns the full type path.
/// For `Foo` → "Foo", for `a::Foo` → "a::Foo".
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
    if prefix.is_empty() {
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
                            // Leaf type (no generics). Qualify if it looks like
                            // a user struct: starts with uppercase, single segment.
                            if path.path.segments.len() == 1 && known_struct_names.contains(&name) {
                                qualify_type_name(&name, prefix)
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
                                    syn::GenericArgument::Type(t) => qualified_full_type_string(
                                        t,
                                        prefix,
                                        known_struct_names,
                                        known_trait_names,
                                    ),
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
        syn::Type::Reference(r) => {
            qualified_full_type_string(&r.elem, prefix, known_struct_names, known_trait_names)
        }
        syn::Type::Paren(p) => {
            qualified_full_type_string(&p.elem, prefix, known_struct_names, known_trait_names)
        }
        syn::Type::Group(g) => {
            qualified_full_type_string(&g.elem, prefix, known_struct_names, known_trait_names)
        }
        syn::Type::Slice(s) => {
            qualified_full_type_string(&s.elem, prefix, known_struct_names, known_trait_names)
                .map(|t| format!("[{}]", t))
        }
        syn::Type::Array(a) => {
            let elem =
                qualified_full_type_string(&a.elem, prefix, known_struct_names, known_trait_names)?;
            let len_str = match &a.len {
                syn::Expr::Lit(lit) => match &lit.lit {
                    syn::Lit::Int(int_lit) => int_lit.base10_digits().to_string(),
                    _ => "N".to_string(),
                },
                _ => "N".to_string(),
            };
            Some(format!("[{};{}]", elem, len_str))
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
    match type_str {
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
        | "bool" => ValueType::Int,
        "f32" | "f64" => ValueType::Float,
        "()" => ValueType::Void,
        _ => ValueType::Ref,
    }
}

fn field_value_type_from_expr(
    base: &syn::Expr,
    member: &syn::Member,
    ctx: &GraphBuildContext,
) -> Option<ValueType> {
    let owner = receiver_type_root(base, ctx)?;
    let field_name = member_name(member);
    let field_type = ctx.struct_fields.field_type(&owner, &field_name)?;
    Some(type_string_to_value_type(field_type))
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
fn extract_element_type_from_str(type_str: &str) -> Option<String> {
    let s = type_str.trim();
    // Angle brackets: Vec<T>, Box<T>, etc.
    if let (Some(start), Some(end)) = (s.find('<'), s.rfind('>')) {
        if start < end {
            return Some(s[start + 1..end].trim().to_string());
        }
    }
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
    None
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
            let field_type_str = ctx.struct_fields.field_type(&owner_type, &field_name)?;
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
                let key = if segments.len() == 1 && !ctx.module_prefix.is_empty() {
                    format!("{}::{}", ctx.module_prefix, segments[0])
                } else {
                    segments.join("::")
                };
                ctx.fn_return_types.get(&key).cloned()
            } else {
                None
            }
        }
        // RPython: op.result.concretetype — for method calls like `self.make_points()[i]`.
        // RPython resolves via the exact callee graph — no bare name fallback.
        syn::Expr::MethodCall(mc) => {
            let method_name = mc.method.to_string();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_function_with_data_flow() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64, y: i64) -> i64 {
                let z = x + y;
                z
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        assert_eq!(program.functions.len(), 1);
        let graph = &program.functions[0].graph;
        // Should have Input ops for params + ops for body
        assert!(graph.block(graph.startblock).operations.len() >= 2);
    }

    #[test]
    fn lowers_field_access_with_data_flow() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            fn read_field(s: S) -> i64 {
                s.x
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        // Should contain a FieldRead op
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "x"
            )),
            "expected FieldRead for 'x', got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_field_access_with_typed_fieldread_and_fieldwrite() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64, y: f64 }
            fn mutate(s: S) -> i64 {
                s.x = 1;
                s.x
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldWrite { field, ty, .. }
                    if field.name == "x" && *ty == ValueType::Int
            )),
            "expected typed FieldWrite for 'x', got {:?}",
            ops
        );
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, ty, .. }
                    if field.name == "x" && *ty == ValueType::Int
            )),
            "expected typed FieldRead for 'x', got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_array_access_with_typed_arrayread_and_arraywrite() {
        let parsed = crate::parse::parse_source(
            r#"
            fn mutate(xs: Vec<i64>, i: usize) -> i64 {
                xs[i] = 1;
                xs[i]
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::ArrayWrite { item_ty, .. } if *item_ty == ValueType::Int
            )),
            "expected typed ArrayWrite, got {:?}",
            ops
        );
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::ArrayRead { item_ty, .. } if *item_ty == ValueType::Int
            )),
            "expected typed ArrayRead, got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_method_call_with_args() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_example(v: Vec<i64>) {
                v.push(42);
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. } if target == &CallTarget::method("push", Some("Vec".into()))
            )),
            "expected Call to 'push', got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_impl_self_method_call_with_concrete_self_type() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Foo;
            impl Foo {
                fn helper(&self) {}
                fn run(&self) {
                    self.helper();
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let run = program
            .functions
            .iter()
            .find(|func| func.name == "run")
            .expect("run graph");
        let ops = &run.graph.block(run.graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target == &CallTarget::method("helper", Some("Foo".into()))
            )),
            "expected helper call with concrete self type, got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_path_call_to_canonical_symbol() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_example(x: i64) -> i64 {
                crate::math::w_int_add(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target == &CallTarget::function_path(["crate", "math", "w_int_add"])
            )),
            "expected canonical Call target path, got {:?}",
            ops
        );
    }

    #[test]
    fn builds_impl_methods() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Foo;
            impl Foo {
                fn bar(&self) { }
                fn baz(&self, x: i64) -> i64 { x }
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        assert_eq!(program.functions.len(), 2);
        assert_eq!(program.functions[0].name, "bar");
        assert_eq!(program.functions[1].name, "baz");
    }

    #[test]
    fn if_creates_multiple_blocks() {
        let parsed = crate::parse::parse_source(
            r#"
            fn branch(x: bool) -> i64 {
                if x { 1 } else { 2 }
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        // entry + then + else + merge = at least 4 blocks
        assert!(
            graph.blocks.len() >= 4,
            "if/else should create >=4 blocks, got {}",
            graph.blocks.len()
        );
        // Upstream `flowspace/model.py:175-180` tags a bool branch by
        // `block.exitswitch == Variable` with two exits whose
        // `exitcase` values are True / False respectively.
        let entry = graph.block(graph.startblock);
        assert!(
            matches!(entry.exitswitch, Some(crate::model::ExitSwitch::Value(_))),
            "entry exitswitch should name the branch condition, got {:?}",
            entry.exitswitch,
        );
        assert_eq!(entry.exits.len(), 2, "bool branch has two exits");
    }

    #[test]
    fn while_creates_header_body_exit() {
        let parsed = crate::parse::parse_source(
            r#"
            fn loop_fn(mut x: i64) -> i64 {
                while x > 0 { x = x - 1; }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        // entry + header + body + exit = at least 4 blocks
        assert!(
            graph.blocks.len() >= 4,
            "while should create >=4 blocks, got {}",
            graph.blocks.len()
        );
    }

    #[test]
    fn lowers_binary_ops_to_exact_names_without_token_strings() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64, y: i64) -> i64 {
                x + y
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let op = graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::BinOp { op, .. } => Some(op.clone()),
                _ => None,
            })
            .expect("binop");
        assert_eq!(op, "add");
    }

    #[test]
    fn lowers_array_field_to_interiorfield_read() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Point { x: i64, y: i64 }
            fn read_point(points: Vec<Point>, i: usize) -> i64 {
                points[i].x
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::InteriorFieldRead { field, item_ty, .. }
                    if field.name == "x" && *item_ty == ValueType::Int
            )),
            "expected InteriorFieldRead for 'x' with item_ty=Int, got {:?}",
            ops
        );
        // Should NOT generate a separate ArrayRead + FieldRead pair.
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "x"
            )),
            "should not have FieldRead for 'x' when InteriorFieldRead is present"
        );
    }

    #[test]
    fn lowers_array_field_to_interiorfield_write() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Point { x: i64, y: i64 }
            fn write_point(points: Vec<Point>, i: usize) {
                points[i].x = 42;
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::InteriorFieldWrite { field, .. } if field.name == "x"
            )),
            "expected InteriorFieldWrite for 'x', got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_unary_ops_to_exact_names_without_token_strings() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64) -> i64 {
                -x
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let op = graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::UnaryOp { op, .. } => Some(op.clone()),
                _ => None,
            })
            .expect("unary op");
        assert_eq!(op, "neg");
    }

    /// RPython `jtransform.py:410-412`: `dyn Trait` receivers from
    /// parameter bindings and `Box<dyn Trait>` locals must lower to
    /// `CallTarget::Indirect`, not to `CallTarget::Method`.
    /// Covers Issue 3 (detection too narrow).
    #[test]
    fn dyn_trait_receiver_detection_local_binding() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_via_param(h: &mut dyn Handler) {
                h.run();
            }
            fn call_via_box_local() {
                let h: Box<dyn Handler> = make_handler();
                h.run();
            }
        "#,
        );
        let program = build_semantic_program(&parsed);

        for func in &program.functions {
            let graph = &func.graph;
            let saw_indirect = graph
                .block(graph.startblock)
                .operations
                .iter()
                .any(|op| match &op.kind {
                    OpKind::Call {
                        target:
                            CallTarget::Indirect {
                                trait_root,
                                method_name,
                            },
                        ..
                    } => trait_root == "Handler" && method_name == "run",
                    _ => false,
                });
            assert!(
                saw_indirect,
                "expected CallTarget::Indirect in {}, got {:?}",
                func.graph.name,
                graph.block(graph.startblock).operations
            );
        }
    }

    #[test]
    fn dyn_trait_receiver_uses_module_qualified_trait_family_key() {
        let parsed = crate::parse::parse_source(
            r#"
            mod a {
                pub trait Handler { fn run(&mut self); }
                pub fn call_a(h: &mut dyn Handler) { h.run(); }
            }
            mod b {
                pub trait Handler { fn run(&mut self); }
                pub fn call_b(h: &mut dyn Handler) { h.run(); }
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let mut seen = std::collections::HashMap::<String, String>::new();
        for func in &program.functions {
            let Some(trait_root) = func
                .graph
                .block(func.graph.startblock)
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::Indirect { trait_root, .. },
                        ..
                    } => Some(trait_root.clone()),
                    _ => None,
                })
            else {
                continue;
            };
            seen.insert(func.name.clone(), trait_root);
        }
        assert_eq!(seen.get("a::call_a"), Some(&"a::Handler".to_string()));
        assert_eq!(seen.get("b::call_b"), Some(&"b::Handler".to_string()));
    }

    /// RPython `rpython/rtyper/rclass.py:644-678 _parse_field_list` — the
    /// `?`, `[*]`, and `?[*]` suffixes must resolve to `IR_QUASIIMMUTABLE`,
    /// `IR_IMMUTABLE_ARRAY`, and `IR_QUASIIMMUTABLE_ARRAY` respectively.
    /// Covers Issue 5 (partial port).
    #[test]
    fn parse_immutable_fields_accepts_string_literal_suffixes() {
        let parsed = crate::parse::parse_source(
            r#"
            #[jit_immutable_fields("plain", "quasi?", "arr[*]", "qarr?[*]")]
            struct S { plain: i64, quasi: i64, arr: i64, qarr: i64 }
            fn noop() {}
        "#,
        );
        let program = build_semantic_program(&parsed);
        let entries = program
            .immutable_fields
            .get("S")
            .expect("S should have immutable_fields entries");
        let by_name: std::collections::HashMap<&str, ImmutableRank> =
            entries.iter().map(|(n, r)| (n.as_str(), *r)).collect();
        assert_eq!(by_name.get("plain"), Some(&ImmutableRank::Immutable));
        assert_eq!(by_name.get("quasi"), Some(&ImmutableRank::QuasiImmutable));
        assert_eq!(by_name.get("arr"), Some(&ImmutableRank::ImmutableArray));
        assert_eq!(
            by_name.get("qarr"),
            Some(&ImmutableRank::QuasiImmutableArray)
        );
    }

    /// Bare ident entries in `#[jit_immutable_fields(foo, bar)]` continue
    /// to resolve to `IR_IMMUTABLE` — backward compatibility with pre-rank
    /// usage sites.
    #[test]
    fn parse_immutable_fields_preserves_bare_ident_backward_compat() {
        let parsed = crate::parse::parse_source(
            r#"
            #[jit_immutable_fields(foo, bar)]
            struct S { foo: i64, bar: i64 }
            fn noop() {}
        "#,
        );
        let program = build_semantic_program(&parsed);
        let entries = program
            .immutable_fields
            .get("S")
            .expect("S should have immutable_fields entries");
        for (name, rank) in entries {
            assert!(
                matches!(rank, ImmutableRank::Immutable),
                "bare ident `{}` expected Immutable rank, got {:?}",
                name,
                rank,
            );
        }
        let names: std::collections::HashSet<&str> =
            entries.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains("foo"));
        assert!(names.contains("bar"));
    }

    /// Multiple `#[jit_immutable_fields(...)]` attributes on the same
    /// struct should accumulate — `rpython/rtyper/rclass.py:638-641` rbase
    /// walk iterates ancestor `_immutable_fields_` unions similarly.
    #[test]
    fn parse_immutable_fields_merge_across_multiple_attributes() {
        let parsed = crate::parse::parse_source(
            r#"
            #[jit_immutable_fields("a?")]
            #[jit_immutable_fields("b[*]")]
            struct S { a: i64, b: i64 }
            fn noop() {}
        "#,
        );
        let program = build_semantic_program(&parsed);
        let entries = program
            .immutable_fields
            .get("S")
            .expect("S should have immutable_fields entries");
        let by_name: std::collections::HashMap<&str, ImmutableRank> =
            entries.iter().map(|(n, r)| (n.as_str(), *r)).collect();
        assert_eq!(by_name.get("a"), Some(&ImmutableRank::QuasiImmutable));
        assert_eq!(by_name.get("b"), Some(&ImmutableRank::ImmutableArray));
    }

    /// Rust `impl Trait` is a static opaque type — the compiler
    /// monomorphizes each call site to a single concrete impl.  RPython
    /// `indirect_call` is reserved for truly polymorphic callees
    /// (`rpython/jit/codewriter/call.py:103 graphs_from`).  An `impl
    /// Trait` parameter must therefore lower to `CallTarget::Method`,
    /// not `CallTarget::Indirect`.
    #[test]
    fn impl_trait_param_does_not_lower_to_indirect_call() {
        let parsed = crate::parse::parse_source(
            r#"
            pub trait Handler { fn run(&mut self); }
            pub fn call_via_impl(mut h: impl Handler) {
                h.run();
            }
            "#,
        );
        let program = build_semantic_program(&parsed);
        let func = program
            .functions
            .iter()
            .find(|f| f.graph.name == "call_via_impl")
            .expect("call_via_impl present");
        let saw_indirect = func
            .graph
            .block(func.graph.startblock)
            .operations
            .iter()
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Indirect { .. },
                        ..
                    }
                )
            });
        assert!(
            !saw_indirect,
            "impl Trait must not lower to CallTarget::Indirect, got {:?}",
            func.graph.block(func.graph.startblock).operations
        );
    }

    /// Issue #5 — receiver detection beyond simple bindings.  Field
    /// access (`self.handler.run()`), index (`handlers[i].run()`), and
    /// `Box<dyn T>`-returning calls (`make_boxed().run()`) must all
    /// reach `CallTarget::Indirect`.
    #[test]
    fn dyn_receiver_via_field_index_and_box_return() {
        let parsed = crate::parse::parse_source(
            r#"
            pub trait Handler { fn run(&mut self); }
            struct Owner { handler: Box<dyn Handler> }
            impl Owner {
                fn dispatch(&mut self) { self.handler.run(); }
            }
            fn list_dispatch(handlers: Vec<Box<dyn Handler>>, idx: usize) {
                handlers[idx].run();
            }
            fn make_boxed() -> Box<dyn Handler> { panic!() }
            fn ret_dispatch() { make_boxed().run(); }
            "#,
        );
        let program = build_semantic_program(&parsed);
        for fname in &["dispatch", "list_dispatch", "ret_dispatch"] {
            let func = program
                .functions
                .iter()
                .find(|f| f.graph.name == *fname)
                .unwrap_or_else(|| panic!("function {fname} present"));
            let saw_indirect = func
                .graph
                .block(func.graph.startblock)
                .operations
                .iter()
                .any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::Indirect { method_name, .. },
                            ..
                        } if method_name == "run"
                    )
                });
            assert!(
                saw_indirect,
                "expected CallTarget::Indirect{{method=run}} in {fname}, got {:?}",
                func.graph.block(func.graph.startblock).operations
            );
        }
    }

    #[test]
    fn value_return_routes_through_canonical_returnblock() {
        let parsed = crate::parse::parse_source(
            r#"
            fn returns_one() -> i64 { return 1; }
            "#,
        );
        let program = build_semantic_program(&parsed);
        let func = program
            .functions
            .iter()
            .find(|f| f.graph.name == "returns_one")
            .expect("returns_one present");
        let entry = func.graph.block(func.graph.startblock);
        // rpython/flowspace/model.py:171-180 Block is characterized by
        // exits + exitswitch; a non-void return is Link(
        // [return_value], graph.returnblock) with exitswitch=None.
        assert!(entry.exitswitch.is_none());
        assert_eq!(entry.exits.len(), 1);
        assert_eq!(entry.exits[0].prevblock, Some(func.graph.startblock));
        assert_eq!(entry.exits[0].target, func.graph.returnblock);
        assert_eq!(
            entry.exits[0].args,
            vec![crate::model::LinkArg::from(
                entry.operations[0].result.expect("const result"),
            )],
        );
    }

    #[test]
    fn void_return_routes_through_canonical_returnblock() {
        let parsed = crate::parse::parse_source(
            r#"
            fn returns_unit() { return; }
            "#,
        );
        let program = build_semantic_program(&parsed);
        let func = program
            .functions
            .iter()
            .find(|f| f.graph.name == "returns_unit")
            .expect("returns_unit present");
        let entry = func.graph.block(func.graph.startblock);
        // RPython `flowcontext.py` emits a fresh Variable on the
        // prevblock side for `return None`; the returnblock's own
        // inputarg stays distinct.
        let returnblock_arg = func.graph.block(func.graph.returnblock).inputargs[0];
        // Upstream `flowspace/model.py:171-180` keeps the void return shape
        // in Block.exits: a single Link([fresh_void], graph.returnblock)
        // with exitswitch=None.
        assert!(entry.exitswitch.is_none());
        assert_eq!(entry.exits.len(), 1);
        assert_eq!(entry.exits[0].prevblock, Some(func.graph.startblock));
        assert_eq!(entry.exits[0].target, func.graph.returnblock);
        assert_eq!(entry.exits[0].args.len(), 1);
        assert_ne!(
            entry.exits[0].args[0].as_value(),
            Some(returnblock_arg),
            "void return must allocate a fresh prevblock-side ValueId (`flowspace/model.py:114`), \
             not reuse the returnblock's own inputarg"
        );
    }
}

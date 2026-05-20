//! AST front-end: build semantic graphs from Rust source.
//!
//! RPython equivalent: flowspace/ — converts source to Block/Link/Variable/SpaceOperation.
//! This module lowers syn AST nodes into FunctionGraph ops with proper data flow (Variable linking).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use syn::{Item, ItemFn};

use crate::ParsedInterpreter;
use crate::flowspace::model::ConstValue;
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FrameState, FunctionGraph, ImmutableRank, Link,
    LinkArg, OpKind, UnknownKind, UnsupportedExprKind, UnsupportedLiteralKind, ValueType,
    exception_exitcase,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstGraphOptions;

/// Signal that lowering was halted due to an unsupported construct.
///
/// RPython `rpython/flowspace/flowcontext.py:258,417` raises `FlowingError`
/// when the abstract interpreter hits a bytecode it cannot model; that
/// error propagates all the way out of `build_flow_graph`, aborting the
/// current graph rather than silently continuing with a synthetic value.
///
/// Pyre's `Option<Variable>` return conflates "expression legitimately
/// produced no value" (e.g. `return` / `break`) with "lowering halted"
/// — making the latter an explicit `Err` variant restores the RPython
/// invariant that unsupported constructs stop the walk at once.  The
/// `Unknown` op is still emitted at the failure site so downstream
/// passes see evidence of the drop; the `Err` just guarantees no
/// synthesised SSA value follows it.
///
/// PyPy distinguishes two kinds of "stop this walk":
///
/// 1. `FlowingError` — unsupported opcode encountered, the whole graph
///    is invalid and `build_flow` re-raises upward
///    (`rpython/flowspace/objspace.py:38`,
///    `rpython/flowspace/flowcontext.py:417`).  This is the `Err` arm
///    here.
/// 2. `FlowSignal::Return` / `FlowSignal::Raise` / `FlowSignal::Break`
///    / `FlowSignal::Continue` — the current block is closed (return
///    to caller, raise into exceptblock, goto loop tail/header), but
///    sibling walks (the other arm of a conditional, arms of a match)
///    continue normally
///    (`rpython/flowspace/flowcontext.py:1253`
///    `Raise.nomoreblocks`).  These are signalled via
///    `Lowered { path_closed: true }` on the `Ok` arm — the caller
///    stops lowering into the closed block but keeps walking siblings.
#[derive(Debug, Clone)]
pub enum FlowingError {
    Unsupported { kind: UnknownKind },
}

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
    /// the graph handle even though the carrier flip made the
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

/// Legacy alias: callers that pre-date the `FlowingError` / `Lowered`
/// split in this file still reference `LoweringAbort`.  Keep the name
/// pointing at `FlowingError` until all in-crate consumers migrate.
pub type LoweringAbort = FlowingError;

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
    /// Module path of the defining file, as supplied to
    /// `parse_source_with_module` (e.g. `"pyframe"` for
    /// `pyre-interpreter/src/pyframe.rs`).  Empty when the caller did not
    /// supply a module path — top-level items remain at simple-name
    /// registration.
    ///
    /// Used by `lib.rs` registration so a free function's call sites that
    /// were qualified by `canonical_call_target:7494-7502` (single-segment
    /// bare call inside a non-empty module) can resolve through the
    /// `[module_path, name]` path, in addition to the bare-name and
    /// `crate::` alias paths.  Without the extra path the
    /// `#[majit_macros::elidable*]` / oopspec / loop-invariant hints
    /// registered against the bare name are silently dropped at every
    /// in-module call site.
    pub module_path: String,
    /// RPython: function-level hints set by GC transformer / decorators.
    /// "close_stack" → _gctransformer_hint_close_stack_
    /// "cannot_collect" → _gctransformer_hint_cannot_collect_
    /// "gc_effects" → random_effects_on_gcobjs
    /// "elidable" → _elidable_function_
    /// "loopinvariant" → _jit_loop_invariant_
    pub hints: Vec<String>,
    /// RPython `graph.access_directly` (flowspace attribute set by the
    /// annotator's `default_specialize` rewrite — see
    /// `description.rs:1333-1335` + `pygraph.rs:53-56`). Carried into
    /// `SemanticFunction` so `policy::look_inside_graph` can port the
    /// `policy.py:71-83` virtualizable safety gate without reaching back
    /// into the PyGraph layer.
    ///
    /// Today every SemanticFunction produced by `build_function_graph`
    /// defaults to `false` because the `front::ast` parser does not yet
    /// consult the annotator result; when the annotator-to-front bridge
    /// lands, the bridge assigns this field from
    /// `PyGraph.access_directly.get()` for the matching graph.
    pub access_directly: bool,
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
    /// Look up a field's type.  For array-typed fields like
    /// `Vec<Point>`, this returns the full type string `"Vec<Point>"`.
    /// Callers use `array_element_type_from_str` to extract `"Point"`.
    ///
    /// Resolution order: (1) exact registered key, then (2) canonical
    /// lexical resolution through `STRUCT_ORIGIN_REGISTRY` (PyPy
    /// `bookkeeper.getdesc(value)` analog) on the receiver leaf —
    /// registration dual-publishes bare + canonical so both spellings
    /// land at the same field list.  (3) crate-prefix-tolerant
    /// suffix-match shim absorbs `pyre_object::rangeobject::W_X` vs
    /// `rangeobject::W_X` divergence — orthogonal to lexical scope
    /// resolution, kept for test entries (`parse::parse_source`)
    /// that bypass `analyze_pipeline_from_parsed`'s
    /// `register_struct_origins`.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.lookup_fields(owner)?
            .iter()
            .find(|(name, _)| name == field_name)
            .map(|(_, ty)| ty.as_str())
    }

    /// Per-scope `field_type` lookup: route `owner` through the call
    /// site's `use_imports` + `module_prefix` first (PyPy
    /// `frame.f_globals` analog) so a bare receiver leaf lands at the
    /// canonical key before the program-wide bookkeeper fallback.
    pub fn field_type_in_scope(
        &self,
        owner: &str,
        field_name: &str,
        prefix: &str,
        use_imports: &HashMap<String, String>,
    ) -> Option<&str> {
        let canonical_owner = qualify_type_name_with_imports(owner, prefix, use_imports);
        self.field_type(&canonical_owner, field_name)
    }

    fn lookup_fields(&self, owner: &str) -> Option<&[(String, String)]> {
        if let Some(fields) = self.fields.get(owner) {
            return Some(fields.as_slice());
        }
        // Canonical lexical resolution: bare receiver leaves resolve
        // through `STRUCT_ORIGIN_REGISTRY` (PyPy `bookkeeper.getdesc`
        // analog).  Registration dual-publishes bare + canonical, so a
        // miss on the exact owner falls through canonical-leaf lookup
        // before the suffix-match shim below.
        let receiver_leaf = owner.rsplit("::").next().unwrap_or(owner);
        let canonical = majit_ir::descr::canonical_struct_name(receiver_leaf);
        if canonical != receiver_leaf
            && let Some(fields) = self.fields.get(&canonical)
        {
            return Some(fields.as_slice());
        }
        let key = self.unique_suffix_owner_key(owner)?;
        self.fields.get(key).map(Vec::as_slice)
    }

    fn unique_suffix_owner_key<'a>(&'a self, owner: &str) -> Option<&'a str> {
        let mut found: Option<&str> = None;
        for key in self.fields.keys() {
            let matches =
                is_path_suffix(owner, key.as_str()) || is_path_suffix(key.as_str(), owner);
            if !matches {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(key.as_str());
        }
        found
    }
}

fn is_path_suffix(longer: &str, shorter: &str) -> bool {
    if longer.len() <= shorter.len() || !longer.ends_with(shorter) {
        return false;
    }
    let prefix_len = longer.len() - shorter.len();
    longer[..prefix_len].ends_with("::")
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

pub fn build_semantic_program(parsed: &ParsedInterpreter) -> Result<SemanticProgram, FlowingError> {
    build_semantic_program_with_options(parsed, &AstGraphOptions::default())
}

pub fn build_semantic_program_from_parsed_files(
    parsed_files: &[ParsedInterpreter],
) -> Result<SemanticProgram, FlowingError> {
    build_semantic_program_from_parsed_files_with_options(parsed_files, &AstGraphOptions::default())
}

/// Pre-walk metadata produced by `collect_program_metadata_pub` —
/// the four registries that `build_function_graph` /
/// `build_function_graph_with_self_ty_pub` need before per-function
/// graph build can resolve typed call shapes.
pub struct ProgramMetadata {
    pub known_struct_names: std::collections::HashSet<String>,
    pub known_trait_names: std::collections::HashSet<String>,
    pub struct_fields: StructFieldRegistry,
    pub fn_return_types: HashMap<String, String>,
    /// Bare struct name → defining module path (use-import resolver
    /// support).  Populated when `collect_struct_names` walks per-file
    /// `ParsedInterpreter.module_path` non-empty: each top-level
    /// `Struct` registers as `struct_origins["Struct"] = module_path`.
    /// PyPy parity: `annotator.bookkeeper.getdesc(TYPE)` resolves the
    /// canonical defining-module path for every lltype reference;
    /// pyre carries names as strings so this map carries that
    /// resolution.  Empty when every parsed file was supplied via the
    /// bare `parse_source` entry — caller falls back to the
    /// dual-publish runtime convergence.
    pub struct_origins: HashMap<String, String>,
    /// Merged use-import table across all parsed files: each entry
    /// `(file_module_path, alias) → fully_qualified_path` mirrors the
    /// per-file `ParsedInterpreter.use_imports` populated by
    /// `parse::collect_use_imports`.  Keyed by `(module, alias)` rather
    /// than `alias` alone because the same alias `Foo` can resolve to
    /// different paths in different files (`use other_a::Foo` in one
    /// vs `use other_b::Foo` in another).
    pub use_imports: HashMap<(String, String), String>,
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

/// Qualify a bare type name with module prefix or, when the resolver
/// knows the canonical defining module, with the canonical prefix.
///
/// Resolves a per-source `use <path> as alias` table first, then the
/// program-wide `STRUCT_ORIGIN_REGISTRY` canonical-defining-module
/// table, then falls back to `prefix::bare`.
///
/// `bookkeeper.py:353-409 getdesc` resolves a bare identifier first in
/// the frame's `f_globals` (the file's own imports), then in the
/// program-wide scope summary; pyre's `STRUCT_ORIGIN_REGISTRY` plays
/// the role of the program-wide scope, while `use_imports` carries
/// the per-source `f_globals` slice.
///
/// `use_imports` is expected to be `GraphBuildContext.use_imports` —
/// each entry maps a local alias (`use other_mod::Foo as Q` →
/// `Q → other_mod::Foo`, plain `use other_mod::Foo` →
/// `Foo → other_mod::Foo`) to the fully-qualified path.  Pass
/// `&HashMap::new()` when the call site has no per-source scope
/// (parse-time registration, test fixtures, `lower_expr_into_graph`);
/// resolution then reduces to `STRUCT_ORIGIN_REGISTRY` + `prefix::bare`.
pub(crate) fn qualify_type_name_with_imports(
    bare: &str,
    prefix: &str,
    use_imports: &HashMap<String, String>,
) -> String {
    if bare.contains("::") {
        return bare.to_string();
    }
    if let Some(full) = use_imports.get(bare) {
        return full.clone();
    }
    if prefix.is_empty() {
        return bare.to_string();
    }
    let canonical = majit_ir::descr::canonical_struct_name(bare);
    if canonical != bare {
        return canonical;
    }
    format!("{}::{}", prefix, bare)
}

/// RPython: annotator whole-program type collection.
/// Recursively collects struct definitions, function return types, and impl
/// method return types from items, handling `mod` blocks with qualified paths.
/// The `prefix` carries the module path (e.g. "a::b") to produce exact callee
/// identities matching what `canonical_call_target` generates at call sites.
fn collect_types_from_items(
    items: &[Item],
    prefix: &str,
    use_imports: &HashMap<String, String>,
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
        use_imports,
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

/// RPython: pass 2 graph building with Item::Mod recursion.
/// Mirrors collect_types_from_items traversal so that module-internal
/// functions get proper SemanticFunction entries with qualified names.
///
/// RPython `flowspace/objspace.py:49` + `flowspace/flowcontext.py:417`
/// + `translator/translator.py:55` — `build_flow()` / `buildflowgraph()`
/// re-raise `FlowingError`, and the top-level translator observes the
/// unsupported construct as a hard failure.  This batch collector
/// propagates `FlowingError` the same way rather than silently dropping
/// a graph whose body hit an unsupported construct.
fn build_graphs_from_items(
    items: &[Item],
    prefix: &str,
    options: &AstGraphOptions,
    struct_fields: &StructFieldRegistry,
    fn_return_types: &HashMap<String, String>,
    use_imports: &HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
    functions: &mut Vec<SemanticFunction>,
) -> Result<(), FlowingError> {
    for item in items {
        match item {
            Item::Fn(func) => {
                // `rlib/jit.py:184-201 elidable_promote.decorator(func)`
                // installs two callable objects on module import
                // (closure-captured original + `exec`-built wrapper).
                // `synthesize_or_passthrough` mirrors that for both
                // free fns and impl methods so the lowering layer below
                // stays wrapper-blind, exactly like RPython's flow-graph
                // builder.
                for synth in synthesize_or_passthrough(func.clone(), None) {
                    let mut sf = build_function_graph(
                        &synth,
                        options,
                        None,
                        struct_fields,
                        fn_return_types,
                        prefix,
                        use_imports,
                        known_struct_names,
                        known_trait_names,
                    )?;
                    if !prefix.is_empty() {
                        sf.name = format!("{}::{}", prefix, sf.name);
                    }
                    functions.push(sf);
                }
            }
            Item::Impl(impl_block) => {
                // Qualify bare self type with module prefix (RPython: unique
                // type identity).  Route through the imports-aware variant
                // so this graph-build self_ty_root matches the
                // `parse::collect_trait_impls_from_items` /
                // `parse::collect_inherent_methods_from_items` registration
                // key when the receiver is a `use <path> as alias` form.
                let self_ty_root = type_root_ident(&impl_block.self_ty)
                    .map(|t| qualify_type_name_with_imports(&t, prefix, use_imports));
                for sub in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = sub {
                        let fake_fn = ItemFn {
                            attrs: method.attrs.clone(),
                            vis: syn::Visibility::Inherited,
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        // jit.py:184-201 — method-level `@elidable_promote`
                        // gets the same wrapper/orig synthesis as free
                        // fns; RPython decorators apply uniformly to any
                        // callable.  The qualified `self_ty_root` lets
                        // the wrapper's tail call hit the impl-method
                        // registration path built by
                        // `CallPath::for_impl_method`.
                        for synth in synthesize_or_passthrough(fake_fn, self_ty_root.as_deref()) {
                            let sf = build_function_graph(
                                &synth,
                                options,
                                self_ty_root.clone(),
                                struct_fields,
                                fn_return_types,
                                prefix,
                                use_imports,
                                known_struct_names,
                                known_trait_names,
                            )?;
                            functions.push(sf);
                        }
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
                        use_imports,
                        known_struct_names,
                        known_trait_names,
                        functions,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub fn build_semantic_program_with_options(
    parsed: &ParsedInterpreter,
    options: &AstGraphOptions,
) -> Result<SemanticProgram, FlowingError> {
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
        &parsed.use_imports,
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
        &parsed.use_imports,
        &known_struct_names,
        &known_trait_names,
        &mut functions,
    )?;

    Ok(SemanticProgram {
        functions,
        known_struct_names,
        known_trait_names,
        struct_fields,
        fn_return_types,
        immutable_fields,
    })
}

pub fn build_semantic_program_from_parsed_files_with_options(
    parsed_files: &[ParsedInterpreter],
    options: &AstGraphOptions,
) -> Result<SemanticProgram, FlowingError> {
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
            &parsed.use_imports,
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
        let functions_before = functions.len();
        build_graphs_from_items(
            &parsed.file.items,
            "",
            options,
            &struct_fields,
            &fn_return_types,
            &parsed.use_imports,
            &known_struct_names,
            &known_trait_names,
            &mut functions,
        )?;
        // Stamp the parsed file's module_path onto each new
        // SemanticFunction so the free-function hint registry in
        // `lib.rs` can add the `[module_path, name]` path alongside
        // the bare-name and `crate::` alias paths.  This pairs with
        // the single-segment qualification at
        // `canonical_call_target:7494-7502`: a bare call inside
        // `mod pyframe` resolves to `["pyframe", helper]` and must
        // find the same `#[majit_macros::elidable*]` hint that the
        // bare-name registration carries.
        if !parsed.module_path.is_empty() {
            for sf in &mut functions[functions_before..] {
                if sf.module_path.is_empty() {
                    sf.module_path = parsed.module_path.clone();
                }
            }
        }
    }
    Ok(SemanticProgram {
        functions,
        known_struct_names,
        known_trait_names,
        struct_fields,
        fn_return_types,
        immutable_fields,
    })
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
/// Closes the Cat 2.1 "adapter cross-block body Input" Skip family
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
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    let mut ctx = GraphBuildContext::new(
        &empty_registry,
        &empty_fn_ret,
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

pub fn build_function_graph_pub(func: &ItemFn) -> Result<SemanticFunction, FlowingError> {
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = HashMap::new();
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    let empty_use_imports = HashMap::new();
    build_function_graph(
        func,
        &AstGraphOptions::default(),
        None,
        &empty_registry,
        &empty_fn_ret,
        "",
        &empty_use_imports,
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
    use_imports: &HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> Result<SemanticFunction, FlowingError> {
    build_function_graph(
        func,
        &AstGraphOptions::default(),
        self_ty_root,
        struct_fields,
        fn_return_types,
        module_prefix,
        use_imports,
        known_struct_names,
        known_trait_names,
    )
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
    /// Module path prefix for qualifying bare type names.
    /// RPython: lltype identity is globally unique — bare "Foo" in mod "a"
    /// must resolve to "a::Foo" in struct_fields lookups.
    module_prefix: String,
    /// Per-source `use <path> as alias` table — `(alias → fully_qualified_path)`
    /// resolved from this source file's top-level `Item::Use` declarations.
    /// PyPy peer: `bookkeeper.getdesc(value)` walks the host Python lexical
    /// scope of the current frame's import resolutions; pyre carries names
    /// as strings, so this map is the per-graph slice of
    /// `CallControl.use_imports` aggregated by `analyze_pipeline_from_parsed`.
    /// Empty when the file has no use-aliases or the build entry bypassed
    /// the parsed-file plumbing (`parse::collect_function_graphs` tests).
    use_imports: HashMap<String, String>,
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
    #[allow(dead_code)]
    joinpoints: HashMap<i64, Vec<BlockId>>,
}

#[derive(Debug, Clone)]
struct LoopFrame {
    /// Block that `continue` jumps to.  For `while` / `for` this is
    /// the header; for `loop` this is the body entry (which also acts
    /// as the loop head).  The current header-phi name list is
    /// recomputed on demand from `continue_target.inputargs` at
    /// every close site (back-edge / `Expr::Continue`) via
    /// `header_phi_name_list(graph, continue_target)` — Cat 2-2
    /// Phase B α.1 (`audit_cat2_2_loop_header_phi_workl_fixpoint_*`).
    /// RPython parity: `flowspace/flowcontext.py:399-465 mergeblock`
    /// queries the merge target's current state at close time, not
    /// at frame push, so any phi that lazy-install added to the
    /// header during body walk is automatically threaded.
    continue_target: BlockId,
    /// Block that `break` jumps to — the loop's exit block.
    break_target: BlockId,
}

/// Names statically reachable as locals from inside a loop body —
/// produced by `loop_body_locals` ahead of the loop walk so the eager
/// phi allocator (Slice 5b.3) can pre-build the loop header's
/// inputarg list without running a `framestate.union` work-list to
/// fixpoint.
///
/// `read_names` collects names referenced as a single-ident
/// `Expr::Path` (a use) or as the LHS of a compound-assign
/// `Expr::Binary` (which both reads and rebinds the slot).
/// `rebound_names` collects names appearing as the LHS of a simple
/// `Expr::Assign`, the LHS of any compound-assign `Expr::Binary`, or
/// the binding of a `Stmt::Local` whose pattern surfaces idents.  The
/// scan recurses through `Block` (including `if` arms, `match` arms,
/// nested loop bodies, and `unsafe` blocks) but explicitly skips
/// `Expr::Closure` bodies, `Stmt::Item` definitions, and `Stmt::Macro`
/// invocations.
///
/// RPython parity: there is no single-line counterpart in
/// `flowspace/flowcontext.py` because RPython infers reads/writes from
/// `LOAD_FAST` / `STORE_FAST` bytecodes implicit in the per-bytecode
/// dispatch (`flowcontext.py:780-820`).  This pre-scan is the
/// static-AST analogue, computing the must-merge name set ahead of the
/// loop walk so the eager allocator can replace RPython's iterative
/// `mergeblock` widening at `flowcontext.py:430` with a single pass.
/// The None-kill at `framestate.py:110-111` is realised by
/// intersecting the pre-scan names with the pre-loop framestate inside
/// the allocator — body-only locals never enter the header phi list.
#[derive(Debug, Default, Clone)]
struct LoopBodyLocals {
    read_names: std::collections::HashSet<String>,
    rebound_names: std::collections::HashSet<String>,
}

/// Statically scan `body` and partition every locally-referenced name
/// into `read_names` and `rebound_names` for the eager loop-header
/// phi allocator.  See `LoopBodyLocals` for the exact contract.
fn loop_body_locals(body: &syn::Block) -> LoopBodyLocals {
    let mut state = LoopBodyLocals::default();
    visit_block_for_loop_locals(body, &mut state);
    state
}

fn visit_block_for_loop_locals(block: &syn::Block, state: &mut LoopBodyLocals) {
    for stmt in &block.stmts {
        visit_stmt_for_loop_locals(stmt, state);
    }
}

fn visit_stmt_for_loop_locals(stmt: &syn::Stmt, state: &mut LoopBodyLocals) {
    match stmt {
        syn::Stmt::Local(local) => {
            // `let pat [: ty] [= init];` — pattern bindings are
            // rebinds.  Inner-scope-only bindings (a `let` inside a
            // nested block whose name shadows nothing in the
            // pre-loop snapshot) are filtered by the allocator: a
            // name not present in the pre-loop framestate is not a
            // header phi candidate.
            collect_pat_idents(&local.pat, &mut state.rebound_names);
            if let Some(init) = &local.init {
                visit_expr_for_loop_locals(&init.expr, state);
                if let Some((_, diverge)) = &init.diverge {
                    visit_expr_for_loop_locals(diverge, state);
                }
            }
        }
        syn::Stmt::Expr(expr, _semi) => visit_expr_for_loop_locals(expr, state),
        syn::Stmt::Item(_) => {
            // `fn`, `struct`, `use`, etc. — item definitions live in
            // their own scope and do not refer to enclosing locals.
            // Slice 5b.2 contract: skip.
        }
        syn::Stmt::Macro(_) => {
            // Macro invocations cannot be statically introspected
            // for local references.  Conservatively skip; if a macro
            // mutates a local that later participates in the
            // back-edge, the `Link.__init__` arity check at build
            // time fails loud (Slice 5 risk #1).
        }
    }
}

fn visit_expr_for_loop_locals(expr: &syn::Expr, state: &mut LoopBodyLocals) {
    match expr {
        syn::Expr::Path(p) => {
            if let Some(ident) = path_as_single_ident(&p.path) {
                state.read_names.insert(ident.to_string());
            }
        }
        syn::Expr::Assign(a) => {
            // Simple LHS `name = rhs` is a pure rebind; complex LHS
            // (e.g. `obj.field = ...`, `arr[i] = ...`) reads the
            // base/index instead.
            if let syn::Expr::Path(p) = a.left.as_ref()
                && let Some(ident) = path_as_single_ident(&p.path)
            {
                state.rebound_names.insert(ident.to_string());
            } else {
                visit_expr_for_loop_locals(&a.left, state);
            }
            visit_expr_for_loop_locals(&a.right, state);
        }
        syn::Expr::Binary(b) => {
            // Compound assign (`a += b`, `a |= b`, …) lowers to
            // `Expr::Binary` with one of the `BinOp::*Assign(_)`
            // variants; a simple-ident LHS is BOTH read and rebound.
            if is_compound_assign(b.op) {
                if let syn::Expr::Path(p) = b.left.as_ref()
                    && let Some(ident) = path_as_single_ident(&p.path)
                {
                    let name = ident.to_string();
                    state.read_names.insert(name.clone());
                    state.rebound_names.insert(name);
                } else {
                    visit_expr_for_loop_locals(&b.left, state);
                }
            } else {
                visit_expr_for_loop_locals(&b.left, state);
            }
            visit_expr_for_loop_locals(&b.right, state);
        }
        syn::Expr::Block(b) => visit_block_for_loop_locals(&b.block, state),
        syn::Expr::Unsafe(u) => visit_block_for_loop_locals(&u.block, state),
        syn::Expr::Async(a) => visit_block_for_loop_locals(&a.block, state),
        syn::Expr::If(e) => {
            visit_expr_for_loop_locals(&e.cond, state);
            visit_block_for_loop_locals(&e.then_branch, state);
            if let Some((_, else_branch)) = &e.else_branch {
                visit_expr_for_loop_locals(else_branch, state);
            }
        }
        syn::Expr::Match(m) => {
            visit_expr_for_loop_locals(&m.expr, state);
            for arm in &m.arms {
                if let Some((_, guard)) = &arm.guard {
                    visit_expr_for_loop_locals(guard, state);
                }
                visit_expr_for_loop_locals(&arm.body, state);
            }
        }
        syn::Expr::While(e) => {
            visit_expr_for_loop_locals(&e.cond, state);
            visit_block_for_loop_locals(&e.body, state);
        }
        syn::Expr::Loop(e) => visit_block_for_loop_locals(&e.body, state),
        syn::Expr::ForLoop(e) => {
            // The for-loop's pat introduces a new inner-scope binding;
            // the allocator filters it out by intersecting against the
            // pre-loop framestate.  Visit iterable for reads + body
            // for any rebinds of *outer* locals.
            visit_expr_for_loop_locals(&e.expr, state);
            visit_block_for_loop_locals(&e.body, state);
        }
        syn::Expr::Let(l) => {
            // `if let Some(x) = expr { .. }` — pattern bindings are
            // inner-scope; only the scrutinee is a read of the
            // enclosing names.
            visit_expr_for_loop_locals(&l.expr, state);
        }
        syn::Expr::Closure(_) => {
            // Slice 5b.2 contract: skip closure body.  A closure
            // captures enclosing locals via a synthetic capture list;
            // those captures do not flow through the loop's
            // straight-line control path, so bindings inside the
            // closure must not influence the header phi set.  RPython
            // parity: a closure compiles to its own bytecode with
            // `LOAD_DEREF` / `STORE_DEREF`, distinct from the outer
            // `LOAD_FAST` / `STORE_FAST` sequence.
        }
        syn::Expr::Call(c) => {
            visit_expr_for_loop_locals(&c.func, state);
            for arg in &c.args {
                visit_expr_for_loop_locals(arg, state);
            }
        }
        syn::Expr::MethodCall(m) => {
            visit_expr_for_loop_locals(&m.receiver, state);
            for arg in &m.args {
                visit_expr_for_loop_locals(arg, state);
            }
        }
        syn::Expr::Field(f) => visit_expr_for_loop_locals(&f.base, state),
        syn::Expr::Index(i) => {
            visit_expr_for_loop_locals(&i.expr, state);
            visit_expr_for_loop_locals(&i.index, state);
        }
        syn::Expr::Reference(r) => visit_expr_for_loop_locals(&r.expr, state),
        syn::Expr::Unary(u) => visit_expr_for_loop_locals(&u.expr, state),
        syn::Expr::Paren(p) => visit_expr_for_loop_locals(&p.expr, state),
        syn::Expr::Try(t) => visit_expr_for_loop_locals(&t.expr, state),
        syn::Expr::TryBlock(t) => visit_block_for_loop_locals(&t.block, state),
        syn::Expr::Tuple(t) => {
            for elem in &t.elems {
                visit_expr_for_loop_locals(elem, state);
            }
        }
        syn::Expr::Cast(c) => visit_expr_for_loop_locals(&c.expr, state),
        syn::Expr::Array(a) => {
            for elem in &a.elems {
                visit_expr_for_loop_locals(elem, state);
            }
        }
        syn::Expr::Range(r) => {
            if let Some(s) = &r.start {
                visit_expr_for_loop_locals(s, state);
            }
            if let Some(e) = &r.end {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Return(r) => {
            if let Some(e) = &r.expr {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Break(b) => {
            if let Some(e) = &b.expr {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Yield(y) => {
            if let Some(e) = &y.expr {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Await(a) => visit_expr_for_loop_locals(&a.base, state),
        syn::Expr::Group(g) => visit_expr_for_loop_locals(&g.expr, state),
        syn::Expr::Struct(s) => {
            for field in &s.fields {
                visit_expr_for_loop_locals(&field.expr, state);
            }
            if let Some(rest) = &s.rest {
                visit_expr_for_loop_locals(rest, state);
            }
        }
        syn::Expr::Repeat(r) => {
            visit_expr_for_loop_locals(&r.expr, state);
            visit_expr_for_loop_locals(&r.len, state);
        }
        // Read-free leaves and constructs that cannot be statically
        // introspected for reads.  Macro args fall here (same
        // fail-loud rationale as `Stmt::Macro`).
        _ => {}
    }
}

fn collect_pat_idents(pat: &syn::Pat, out: &mut std::collections::HashSet<String>) {
    match pat {
        syn::Pat::Ident(i) => {
            out.insert(i.ident.to_string());
            if let Some((_, sub)) = &i.subpat {
                collect_pat_idents(sub, out);
            }
        }
        syn::Pat::Type(t) => collect_pat_idents(&t.pat, out),
        syn::Pat::Tuple(t) => {
            for elem in &t.elems {
                collect_pat_idents(elem, out);
            }
        }
        syn::Pat::TupleStruct(t) => {
            for elem in &t.elems {
                collect_pat_idents(elem, out);
            }
        }
        syn::Pat::Struct(s) => {
            for field in &s.fields {
                collect_pat_idents(&field.pat, out);
            }
        }
        syn::Pat::Or(o) => {
            for case in &o.cases {
                collect_pat_idents(case, out);
            }
        }
        syn::Pat::Reference(r) => collect_pat_idents(&r.pat, out),
        syn::Pat::Paren(p) => collect_pat_idents(&p.pat, out),
        syn::Pat::Slice(s) => {
            for elem in &s.elems {
                collect_pat_idents(elem, out);
            }
        }
        // Other variants (Lit, Path, Range, Rest, Wild, Macro, …)
        // bind no local names directly.
        _ => {}
    }
}

fn path_as_single_ident(path: &syn::Path) -> Option<&syn::Ident> {
    if path.leading_colon.is_none() && path.segments.len() == 1 {
        let seg = &path.segments[0];
        if matches!(seg.arguments, syn::PathArguments::None) {
            return Some(&seg.ident);
        }
    }
    None
}

fn is_compound_assign(op: syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_)
    )
}

/// Eagerly allocate the loop header's phi inputargs for every name in
/// `must_merge.read_names ∪ must_merge.rebound_names` that is also
/// present in `pre_loop_snapshot`.  The walk visits
/// `pre_loop_snapshot.entries` in order so the resulting header
/// inputarg slot order mirrors Stage A2's first-bind positional
/// invariant — RPython parity with `flowspace/framestate.py:14
/// _union`'s positional zip over `locals_w`.
///
/// Mirrors `flowcontext.py:430 mergeblock`'s "create phi for every
/// mergeable local, then let `simplify.transform_dead_op_vars` DCE
/// the unread ones".  Names rebound in the body but never read still
/// receive a transient header phi; `model::prune_dead_phis`
/// (`simplify.py:484-524` blanket DCE) removes the phi + matching
/// link.args slot when no graph-level reader appears.
///
/// For each surviving name this routine:
///   1. emits `OpKind::Input { name, ty: pre_loop_entry.value_type }`
///      at `header_entry`, allocating a fresh `Variable` (the phi var),
///   2. pushes that phi var onto `header_entry.inputargs`,
///   3. pushes the **pre-loop** var onto `pre_loop_block.exits[0].args`
///      so the forward-edge `Link.args` matches the new header arity,
///   4. updates `ctx.local_value_ids[name]` to point at the phi var
///      (with `header_entry` as the new defining block) so the body
///      walk reads the phi rather than the pre-loop var,
///   5. updates `ctx.local_value_types[name]` to the carried
///      `value_type`.
///
/// Names in `must_merge.read_names ∪ must_merge.rebound_names` that are
/// absent from `pre_loop_snapshot` are skipped — they are body-only locals
/// (e.g. the binding pattern of an inner `for` / `let` inside the body) that
/// have no pre-loop counterpart to merge with.  RPython parity:
/// `framestate.py:110-111` "if w1 is None or w2 is None: return None"
/// drops one-sided slots; here the pre-scan saw the binding but the
/// snapshot says it never existed pre-loop, so it cannot be a header
/// phi candidate.  A future body lowering pass that needs to read
/// such a name allocates it as a fresh `OpKind::Input` at its def
/// block as it does today.
///
/// The caller (Slice 5c.1+) is responsible for closing the back-edge
/// from `body_tail` to `header_entry` by fetching, per name in the
/// returned `Vec<String>`, the body's current `ctx.local_value_ids[name]`
/// and pushing those vids onto a `set_goto(body_tail, header_entry,
/// args)` call.  RPython parity for the back-edge close: the same
/// slot-by-slot mapping `framestate.py:92 getoutputargs` produces for
/// the closing predecessor's link.
///
/// Pre-conditions:
///   - `pre_loop_block` is already closed with a single goto-style
///     exit to `header_entry` (e.g. via `graph.set_goto(pre_loop_block,
///     header_entry, vec![])`).  This routine pushes onto that
///     existing exit's `args` rather than re-closing the block.
///   - `pre_loop_snapshot` was produced by
///     `ctx.getstate(graph, 0)` (or constructed from the same
///     ctx state) so its entries are in first-bind positional order.
///
/// Returns the ordered list of header-phi names.
fn allocate_loop_header_phis(
    graph: &mut FunctionGraph,
    ctx: &mut GraphBuildContext<'_>,
    pre_loop_block: BlockId,
    header_entry: BlockId,
    pre_loop_snapshot: &FrameState,
    must_merge: &LoopBodyLocals,
) -> Vec<String> {
    let mut header_phi_names = Vec::new();
    // Walk `pre_loop_snapshot.locals_w` per upstream
    // `framestate.py:19 self.locals_w` — pyre's `getstate` populates
    // the `Hlvalue` carrier in lockstep with `entries`, so the walk is
    // bit-identical to a `pre_loop_snapshot.entries` traversal while
    // keeping the read side in agreement with the upstream source of
    // truth.  Materialise the (slot_idx, vid) pairs up front so the
    // immutable `graph` borrow held by `locals_w_view` (derivation
    // fallback through `graph.variable`) releases before the mutable
    // `push_op` / `push_inputarg` / `block_mut` calls below.
    let pre_loop_pairs: Vec<(usize, crate::flowspace::model::Variable)> = pre_loop_snapshot
        .locals_w_view(graph)
        .iter()
        .enumerate()
        .filter_map(|(i, slot)| match slot {
            Some(crate::flowspace::model::Hlvalue::Variable(v)) => Some((i, v.clone())),
            _ => None,
        })
        .collect();
    for (slot_idx, entry_var) in pre_loop_pairs {
        let name = &ctx.local_first_bind_order[slot_idx];
        // `flowcontext.py:430 mergeblock` allocates a phi for every
        // mergeable local; `simplify.transform_dead_op_vars` then
        // DCEs the unread ones.  Pyre mirrors that: predicate is
        // `read_names ∪ rebound_names`, prune_dead_phis prunes the
        // transient `rebound-only-no-read` phi.
        if !must_merge.read_names.contains(name) && !must_merge.rebound_names.contains(name) {
            continue;
        }
        // Type lives on the op that produced `entry_var` (mirrors
        // upstream `Variable.concretetype` access pattern post-rtyper).
        let value_type = graph_value_type_var(graph, &entry_var).unwrap_or(ValueType::Unknown);
        let name = name.clone();
        let phi_var = graph
            .push_op_var(
                header_entry,
                OpKind::Input {
                    name: name.clone(),
                    ty: value_type.clone(),
                },
                true,
            )
            .expect("OpKind::Input always produces a result");
        graph.name_value_var(&phi_var, name.clone());
        graph.push_inputarg_var(header_entry, phi_var.clone());
        let entry_arg = LinkArg::Value(entry_var);
        graph.block_mut(pre_loop_block).exits[0]
            .args
            .push(entry_arg);
        ctx.bind_local_id_var(name.clone(), &phi_var, graph, header_entry);
        ctx.local_value_types.insert(name.clone(), value_type);
        header_phi_names.push(name);
    }
    header_phi_names
}

/// Walk `header.inputargs` and recover, in inputarg order, the local
/// name attached to each Variable.  Used by `Expr::While` / `Expr::Loop`
/// (Slice 5c.1+) to capture the FROZEN header-phi name list once the
/// loop header is fully populated — both by
/// `allocate_loop_header_phis`'s eager phis and by any cond-driven
/// cross-block lazy installs and carry-through Variables threaded in
/// via `ensure_variable_at_block` (Slice 4.2).  The returned list
/// drives the back-edge close and `Expr::Continue` per-name link-arg
/// threading; it must match `header.inputargs.len()`.
///
/// Reads `graph.value_name(vid)` directly so carry-through Variables
/// added to `inputargs` by `ensure_variable_at_block` (which does not
/// emit a paired `OpKind::Input { name, .. }` op) are recovered under
/// their original definition-site name — typically a function
/// parameter named at `entry`-block registration time.
fn header_phi_name_list(graph: &FunctionGraph, header: BlockId) -> Vec<String> {
    let header_block = graph.block(header);
    header_block
        .inputargs
        .iter()
        .filter_map(|iarg| {
            let slot = graph.slot_of(iarg)?;
            graph.value_name_at(slot).map(|s| s.to_string())
        })
        .collect()
}

/// Materialise a `Vec<Variable>` of per-name link args for a back-edge
/// or `continue` close: each name's `ctx.local_value_ids[name].0`
/// supplies the value the loop header should observe on this edge,
/// projected through `graph.must_variable` to the backing `Variable`.
/// Used at the closing predecessor of `Expr::While` / `Expr::Loop`'s
/// loop header.  RPython parity for the slot-by-slot mapping:
/// `flowspace/framestate.py:92 getoutputargs`.  Panics if any name is
/// absent from `ctx.local_value_ids` — `header_phi_names` is captured
/// from already-installed `OpKind::Input` ops and the eager allocator
/// (or the lazy installer that produced any extra phis) bound each
/// name into ctx, so a missing entry indicates a broken invariant
/// upstream of this call.
fn link_arg_vars_from_ctx(
    graph: &FunctionGraph,
    ctx: &GraphBuildContext<'_>,
    header_phi_names: &[String],
) -> Vec<crate::flowspace::model::Variable> {
    header_phi_names
        .iter()
        .map(|name| {
            let (var, _def_block) = ctx.local_var_of(name, graph).unwrap_or_else(|| {
                panic!(
                    "header phi name {:?} must have a current ctx binding at the closing \
                     predecessor",
                    name
                )
            });
            var
        })
        .collect()
}

impl<'a> GraphBuildContext<'a> {
    fn new(
        struct_fields: &'a StructFieldRegistry,
        fn_return_types: &'a HashMap<String, String>,
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
            module_prefix: module_prefix.to_string(),
            use_imports,
            known_struct_names,
            known_trait_names,
            loop_stack: Vec::new(),
            local_first_bind_order: Vec::new(),
            local_first_bind_seen: std::collections::HashSet::new(),
            value_stack: Vec::new(),
            last_exception: None,
            blockstack: Vec::new(),
            joinpoints: HashMap::new(),
        }
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
    /// `(vid, owner_block)` is the structurally honest pairing of the
    /// pyre `(ValueId, BlockId)` reuse-gate.
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
    /// Slice 4.2 production entry — used by `lower_if_expr`'s
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
                                // identity through ValueId, and Constants
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
                    if graph
                        .slot_of(&var)
                        .and_then(|s| graph.value_name_at(s))
                        .is_none()
                    {
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
    /// Pyre adaptation: `pendingblocks.append(newblock)` is omitted —
    /// the tree-recursive lowering visits every reachable block
    /// synchronously; the new block becomes the merge target that the
    /// caller lowers into directly.  Returns the BlockId of the merge
    /// target so the caller can continue.
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
    #[allow(dead_code)]
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
                // `flowcontext.py:433-436` — `make_next_block(:465-473)`
                // creates a fresh SpamBlock with `state.copy()` as its
                // framestate and unconditionally links the current block
                // to it.  Pyre's FrameState is Clone so `.clone()` plays
                // the role of `state.copy()`.
                let newstate = currentstate.clone();
                let newblock = graph.create_block_from_framestate(&newstate);
                graph.set_goto_from_framestate(currentblock, newblock, &currentstate, &newstate);
                self.joinpoints
                    .entry(next_offset)
                    .or_default()
                    .insert(0, newblock);
                newblock
            }
            Some((cand, newstate)) => {
                // `flowcontext.py:438` — `newstate.matches(block.framestate)`
                // is true when the union produced no fresh Variables, so
                // `block` already accepts `currentstate` shape-for-shape.
                let cand_fs = graph
                    .block(cand)
                    .framestate
                    .clone()
                    .expect("mergeblock: candidate must be a SpamBlock (framestate-bearing)");
                if newstate.matches(&cand_fs, graph) {
                    // `flowcontext.py:439-441` — direct link from
                    // current to the existing candidate.
                    graph.set_goto_from_framestate(currentblock, cand, &currentstate, &cand_fs);
                    cand
                } else {
                    // `flowcontext.py:443-463` — generalize: new
                    // SpamBlock, link current→new, retire old candidate.
                    let newblock = graph.create_block_from_framestate(&newstate);
                    // `:449-451` — `outputargs = currentstate.
                    // getoutputargs(newstate); currentblock.closeblock(
                    // Link(outputargs, newblock))`.
                    graph.set_goto_from_framestate(
                        currentblock,
                        newblock,
                        &currentstate,
                        &newstate,
                    );
                    // `:454-459` — dead-mark old candidate, clear body,
                    // replace exits with a forward to the new block.
                    // `outputargs = block.framestate.getoutputargs(
                    // newstate)` projects the old block's entry shape
                    // onto the new block's Variable slots.
                    let old_outputargs = cand_fs.getoutputargs(&newstate, graph);
                    {
                        let blk = graph.block_mut(cand);
                        blk.operations.clear();
                        blk.exitswitch = None;
                        blk.dead = true;
                    }
                    // Every `LinkArg::Value(var)` in `old_outputargs`
                    // must be defined at `cand`; backfill the inputargs/
                    // predecessor chain via `ensure_variable_at_block`
                    // before installing the link.  Mirrors the same
                    // backfill `set_goto_from_framestate` performs at
                    // the caller-driven close sites.
                    for arg in &old_outputargs {
                        if let LinkArg::Value(var) = arg {
                            graph.ensure_variable_at_block(cand, var);
                        }
                    }
                    let link = Link::new_mixed(old_outputargs, newblock, None);
                    graph.recloseblock(cand, vec![link]);
                    // `:460-462` — `candidates.remove(block); candidates.
                    // insert(0, newblock)`.
                    let candidates = self.joinpoints.entry(next_offset).or_default();
                    candidates.retain(|&c| c != cand);
                    candidates.insert(0, newblock);
                    newblock
                }
            }
        }
    }

    /// cfg(test) sibling of [`Self::bind_local_id`] that accepts the
    /// raw dense-slot index — lets test fixtures spell the local
    /// binding as `ctx.bind_local_id_at(name, N, block, &mut graph)`
    /// instead of allocating a `Variable` first.  If slot `idx` does
    /// not yet exist on `graph`, a placeholder `Variable` is minted
    /// via [`FunctionGraph::bind_variable_at`] so the carrier always
    /// holds a real handle (post-Slice 402 carrier flip).
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
    /// pre-carrier-flip signature.
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

/// Lazy cross-block local installer — Slice 2 of the Cat 2.1 epic.
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
/// **Stage B2 (final)**: the Slice 2 conservative fence is gone — the
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
    // widens to Unknown — Slice P2 retired the abort-on-disagreement
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
        // never inspects types).  Pyre's prior NEW-DEVIATION here
        // failed the install on concrete disagreement; that gate
        // was retired in slice P2 after a dry-run audit confirmed
        // zero fixture / production conflicts (cargo lib 2557/0/3
        // + check.py 14/14×2 PASS).
        let observed_type = graph_value_type_var(graph, &snap_var).unwrap_or(ValueType::Unknown);
        match (&shared_value_type, &observed_type) {
            (None, _) => shared_value_type = Some(observed_type.clone()),
            (Some(prior), new) if prior == new => {}
            (Some(ValueType::Unknown), _) => shared_value_type = Some(observed_type.clone()),
            (Some(_), ValueType::Unknown) => {}
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
    // (`let mut x: bool = ...`), `allocate_loop_header_phis` (loop
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
        // Phase 2 type-validation: retired in slice P2.  Phase 1's
        // wildcard fold (`(Some(_), _) => Unknown` arm) widens
        // disagreeing concrete kinds to Unknown so the freshly-
        // installed inputarg's `ty` is the upper bound observable
        // across this merge.  Per-link resolved-type validation here
        // was a NEW-DEVIATION duplicating that fold; the audit dry-
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

fn build_function_graph(
    func: &ItemFn,
    options: &AstGraphOptions,
    self_ty_root: Option<String>,
    struct_fields: &StructFieldRegistry,
    fn_return_types: &HashMap<String, String>,
    module_prefix: &str,
    use_imports: &HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> Result<SemanticFunction, FlowingError> {
    let fn_name = func.sig.ident.to_string();
    let previous = CURRENT_LOWERING_FN_NAME.with(|c| c.borrow_mut().replace(fn_name.clone()));
    let _restore_fn = LoweringFnNameGuard { previous };
    let mut graph = FunctionGraph::new(fn_name);
    let mut entry = graph.startblock;
    let mut ctx = GraphBuildContext::new(
        struct_fields,
        fn_return_types,
        module_prefix,
        use_imports.clone(),
        known_struct_names,
        known_trait_names,
    );
    ctx.generic_trait_roots =
        collect_generic_trait_roots(&func.sig.generics, module_prefix, known_trait_names);

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
            syn::FnArg::Receiver(recv) => {
                if let Some(self_ty_root) = &self_ty_root {
                    ctx.local_type_roots
                        .insert("self".to_string(), self_ty_root.clone());
                    ctx.local_type_strings
                        .insert("self".to_string(), self_ty_root.clone());
                }
                // `self`, `&self`, `&mut self` — all three correspond to
                // an `lltype.Ptr(<Self>)` register in RPython, so the
                // formal parameter always lands in the Ref class.
                let self_ty = classify_fn_arg_ty(&recv.ty);
                ctx.local_value_types
                    .insert("self".to_string(), self_ty.clone());
                if let Some(var) = graph.push_op_var(
                    entry,
                    OpKind::Input {
                        name: "self".to_string(),
                        ty: self_ty,
                    },
                    true,
                ) {
                    graph.name_value_var(&var, "self".to_string());
                    graph.push_inputarg_var(entry, var.clone());
                    // RPython `LOAD_FAST` parity: record the receiver
                    // binding so a body `Expr::Path` reference to
                    // `self` within the entry block reuses this
                    // Variable instead of emitting a fresh
                    // `OpKind::Input` — same treatment as typed
                    // parameters on the `FnArg::Typed` arm below
                    // (`flowspace/flowcontext.py:835`).
                    ctx.bind_local_id_var("self".to_string(), &var, &graph, entry);
                }
            }
            syn::FnArg::Typed(pat_type) => {
                let name = canonical_pat_name(&pat_type.pat);
                if let Some(type_root) = type_root_ident(&pat_type.ty) {
                    // Qualify bare type with module prefix for exact identity.
                    let qualified = qualify_type_name_with_imports(
                        &type_root,
                        &ctx.module_prefix,
                        &ctx.use_imports,
                    );
                    ctx.local_type_roots.insert(name.clone(), qualified);
                    if let Some(trait_root) = ctx.generic_trait_roots.get(&type_root) {
                        ctx.local_trait_bound_roots
                            .insert(name.clone(), trait_root.clone());
                    }
                }
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
                // RPython `rpython/jit/codewriter/support.py:getkind`
                // mapping: classify the Rust parameter type to one of
                // the three register classes so the annotator + rtyper
                // receive a non-`Unknown` seed. Upstream's rtyper
                // assigns a concretetype to every `Variable`, and
                // `assembler.py:write_insn` relies on every operand
                // having a coloring. Using `ValueType::Unknown` here
                // used to cascade into `build_value_kinds` dropping
                // the value, which produced the `(0, 'i')` fallback at
                // `lookup_reg_with_kind` — the source of the pyre-only
                // `getfield_gc_*/id>*` `_intbase` aliases.
                let arg_ty = classify_fn_arg_ty(&pat_type.ty);
                ctx.local_value_types.insert(name.clone(), arg_ty.clone());
                if let Some(var) = graph.push_op_var(
                    entry,
                    OpKind::Input {
                        name: name.clone(),
                        ty: arg_ty.clone(),
                    },
                    true,
                ) {
                    graph.name_value_var(&var, name.clone());
                    graph.push_inputarg_var(entry, var.clone());
                    // RPython `LOAD_FAST` parity: record the parameter
                    // binding so a body `Expr::Path` reference within
                    // the entry block reuses this Variable instead of
                    // emitting a fresh `OpKind::Input`
                    // (`flowspace/flowcontext.py:835`).
                    ctx.bind_local_id_var(name.clone(), &var, &graph, entry);
                }
            }
        }
    }

    // Lower function body.  RPython `flowspace/flowcontext.py` stops
    // abstract-interpreting the current graph on `FlowingError`
    // (unsupported opcode) — the exception propagates out of
    // `build_flow()` (`flowspace/objspace.py:38`) so the translator
    // observes the failure instead of receiving a partial graph.  A
    // path-closing `FlowSignal::Return` / `Raise` at the top level is
    // orderly termination: after `return x` there's nothing more to
    // walk but the graph is well-formed, so we break without
    // propagating.
    // Z4.B.0 tripwire: function-body entry is a simple-stmt boundary
    // (`flowcontext.py:413` first `handle_bytecode` call) — stack
    // depth 0 expected.  Today `value_stack` is never written; the
    // assert is a no-op safety net ahead of Z4.B.1's leaf-push.
    ctx.assert_stack_empty_at_stmt_boundary("build_function_graph body entry");
    let lowered = lower_stmt_list_with_tail_value(
        &mut graph,
        &mut entry,
        &func.block.stmts,
        options,
        &mut ctx,
    )?;
    // Function-body exit mirrors the entry invariant: every leaf push
    // must have a matching consumer pop somewhere in the body, so the
    // top-level stack must be empty again when the walk completes.
    ctx.assert_stack_empty_at_stmt_boundary("build_function_graph body exit");

    // Default terminator if none was set. RPython `RETURN_VALUE`
    // carries the evaluated tail expression into `graph.returnblock`;
    // only statement-only / empty bodies synthesize the void return
    // value.
    if !lowered.path_closed && graph.block(entry).is_open() {
        graph.set_return(entry, lowered.value_var(&graph));
    }

    // RPython: op.result.concretetype — module-qualified for exact type identity.
    let return_type = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => qualified_full_type_string_with_imports(
            ty,
            module_prefix,
            use_imports,
            known_struct_names,
            known_trait_names,
        ),
        syn::ReturnType::Default => Some("()".to_string()),
    };

    // RPython: function-level hints from decorators / GC transformer.
    // Scan #[jit_*] attributes to detect elidable, loopinvariant,
    // close_stack, cannot_collect, gc_effects.
    let hints = collect_jit_hints(&func.attrs, Some(&func.sig));

    // RPython `simplify_graph(graph)` (`simplify.py:1075-1081`) runs
    // a fixed list of passes; pyre runs the subset whose AST-graph
    // dependencies have landed:
    //
    //     transform_dead_op_vars         — `prune_dead_phis` (Z2.x)
    //     eliminate_empty_blocks         — `eliminate_empty_blocks`
    //     remove_assertion_errors        — pending
    //     remove_identical_vars_SSA      — pending
    //     constfold_exitswitch           — handled in optimizeopt
    //     remove_trivial_links           — pending
    //     SSA_to_SSI                     — pending
    //     coalesce_bool                  — pending
    //     transform_ovfcheck             — pending
    //     simplify_exceptions            — pending
    //     transform_xxxitem              — pending
    //     remove_dead_exceptions         — pending
    //
    // `transform_dead_op_vars`: backward dataflow over operation
    // operands + exitswitches + `Link.args`-as-dependencies.  Line-
    // by-line port of `simplify.transform_dead_op_vars_in_blocks(blocks,
    // graphs, translator=None)` (`simplify.py:422-524`).
    //
    // `eliminate_empty_blocks`: collapse empty-block forwarding
    // chains.  Line-by-line port of `simplify.py:52-69`.  No-op on
    // pyre's tree-recursive Match/If lowering today (no chain
    // emitted); becomes load-bearing once Z4's flowcontext-walker
    // rewrite materialises intermediate `SpamBlock`s per fold step.
    //
    // Convergence path for the remaining `pending` passes: port each
    // under its upstream name as the corresponding pyre IR construct
    // lands.
    crate::model::prune_dead_phis(&mut graph);
    crate::model::eliminate_empty_blocks(&mut graph);

    let graph = if module_prefix.is_empty() {
        graph
    } else {
        graph.with_source_module(module_prefix)
    };
    Ok(SemanticFunction {
        name: func.sig.ident.to_string(),
        graph,
        return_type,
        self_ty_root,
        hints,
        module_path: module_prefix.to_string(),
        access_directly: false,
    })
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
                // `call.py:292-299 getcalldescr` still runs the
                // `_canraise(op)` analysis on every elidable callsite,
                // so collapsing `elidable_cannot_raise` /
                // `elidable_or_memerror` to a single `"elidable"` hint
                // remains parity-correct — `getcalldescr` recovers the
                // `EF_ELIDABLE_*` 3-way from the per-op raise analysis
                // at `jit_codewriter/call.rs:2773-2782`.
                "elidable_cannot_raise" | "elidable_or_memerror" => {
                    hints.push("elidable".into());
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
    // be a NEW-DEVIATION (the hint would shadow no real upstream
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
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    let mut ctx = GraphBuildContext::new(
        &empty_registry,
        &empty_fn_ret,
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

    // Cat 2.1 Slice 2 / Stage A1: capture the locals frame as it
    // was when `*block` closed via `set_branch` so a later
    // cross-block read in the merge block can thread back
    // through either arm's `Link.args` even when the arm itself
    // rebinds nothing.  Stored on `Block.framestate` (per-block,
    // captured at close time) — both exits of one set_branch
    // share the same pre-branch snapshot, so the per-edge
    // duplication of Slice 2 collapses into a single field.
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
    // Cat 2.1 Slice 2: snapshot then-arm's locals state BEFORE
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
    // reused below for (a) Slice 4.2's migration path
    // (`create_block_from_framestate` + `set_goto_from_framestate`) and
    // (b) the legacy lean-merge-block ctx update (`None`-kill + lazy
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

    // Slice 4.2: when both arms are open and there is no value-phi to
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
    //   2. Loop-header arity contracts.  `allocate_loop_header_phis`
    //      (`ast.rs:2255`) populates header `inputargs` with NAMED
    //      `OpKind::Input` ops and the back-edge close at
    //      `Expr::Continue` (`ast.rs:6738-6741`) sends args derived
    //      from `header_phi_name_list` (named-only enumeration).
    //      `ensure_variable_at_block` adds carry-through Variables
    //      as unnamed inputargs — the back-edge then trips
    //      `set_goto`'s arity assert (`model.rs:3422-3433`) because
    //      its named-only args count is less than the header's
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
        // Slice 4.2 migration path: `create_block_from_framestate`
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
            // Cat 2.1 Slice 2 / Stage A1: cross-block read of a
            // single-segment local — try lazy install first (allocates
            // an inputarg in `*block` + threads `Link.args` back to
            // every predecessor whose closing site recorded a snapshot
            // in `Block.framestate`).  Falls back to the legacy naked
            // `OpKind::Input` emit when any predecessor lacks a
            // recorded snapshot, so non-if/else closing sites that
            // have not yet opted into Slice 2-6 wiring still build a
            // graph (just one whose cross-block reads remain pinned
            // by the cutover anchor for that flavour).
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
                            .filter(|ty| matches!(ty, ValueType::Int | ValueType::Ref))
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
                let pre_fork_local_vars: Vec<crate::flowspace::model::Variable> = merged_names
                    .iter()
                    .map(|name| pre_fork_locals[name].0.clone())
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
            // fail-loud arm, blocking Slice 10 of the rtyper cutover.
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
                // Body-input-retirement-epic.md Phase 2 brings the
                // same pattern to `Expr::If` / `Expr::Match` so all
                // fork/join shapes use the consistent
                // `[result, ...locals]` Link.args contract; the
                // single-fork forms (`!`-bool desugar) follow as a
                // sibling slice.
                let pre_fork_locals = ctx.local_value_ids.clone();
                let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
                merged_names.sort();
                let pre_fork_local_vars: Vec<crate::flowspace::model::Variable> = merged_names
                    .iter()
                    .map(|name| pre_fork_locals[name].0.clone())
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
            // an integer operand (Task #141 / Task #85 lock-in at
            // `pyre-jit-trace/src/jitcode_runtime.rs:1340`).
            if source_ty.as_ref() == Some(&ValueType::Ref) && result_ty == ValueType::Unsigned {
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
            // RPython has no `expr as T` syntax — numeric conversions go
            // through `int(v)` / `float(v)` / `bool(v)` builtin calls
            // (`rbuiltin.py:178-189 rtype_builtin_int/float/bool`), and
            // the rtyper's `Repr.rtype_int/float/bool` per-Repr methods
            // (rint.py:137-147, rfloat.py:48-58, rbool.py:55-70) emit
            // the low-level `cast_*_to_*` / `*_is_true` op as a side
            // effect of `inputargs(target_lltype)` coercion through
            // `pair_X_Y_convert_from_to`.  Pyre routes Rust's `as T`
            // through the same canonical chain — `Call { target:
            // FunctionPath { segments: vec!["int"/"float"/"bool"] } }`
            // resolves through `flowspace_adapter`'s HOST_ENV fallback
            // to the `__builtin__` class HostObject, annotates as
            // `SomeBuiltin(analyser_name="int")`, then rtypes through
            // `BuiltinFunctionRepr.rtype_simple_call →
            // BUILTIN_TYPER[...]`.
            if let Some(segments) = cast_builtin_name(source_ty.as_ref(), &result_ty) {
                let var = graph
                    .push_op_var(
                        *block,
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
                return Ok(Lowered::from_value_var(graph, &var));
            }
            // Identity / source-type-unknown casts: emit
            // `OpKind::UnaryOp { op: "same_as", result_ty, .. }` so
            // the target `ValueType` propagates through the graph
            // (graph_value_type reads producer op's result_ty).
            // RPython has no `expr as T` syntax, so identity coercion
            // is implicit at use sites via the rtyper's
            // `inputargs(target_lltype)` → `convertvar` →
            // `pair(SrcRepr, TgtRepr).convert_from_to` chain — but
            // pyre's surface DSL has `as T` which carries a target
            // type the downstream pipeline needs to see.  The
            // low-level `same_as` (rtyper.py:478-481) is the canonical
            // RPython op for type-preserving copy; the rtyper's
            // `"same_as"` arm dispatches to `rbuiltin::rtype_same_as`
            // which emits the matching LL op while preserving the
            // operand's lltype.  Backendopt's `removenoops::\
            // remove_same_as` collapses identity copies post-rtyper
            // when safe (removenoops.py:47-48).
            let var = graph
                .push_op_var(
                    *block,
                    OpKind::UnaryOp {
                        op: "same_as".to_string(),
                        operand: operand_var,
                        result_ty,
                    },
                    true,
                )
                .expect("OpKind::UnaryOp has has_result=true");
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
            let mut arm_entries: Vec<BlockId> = Vec::with_capacity(m.arms.len());
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
            for arm in &m.arms {
                let entry = graph.create_block();
                let mut tail = entry;
                let saved_locals = LocalBindingSnapshot::capture(ctx);
                bind_pattern_locals(&arm.pat, scrutinee_type_string.as_deref(), ctx);
                let arm_lowered_result = lower_expr(graph, &mut tail, &arm.body, options, ctx);
                // Slice 4a: snapshot this arm's exit framestate BEFORE
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
                arm_entries.push(entry);
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
            let (merge, merge_phi) = if all_open_arms_have_value {
                let (m_block, phi_args) = graph.create_block_with_arg_vars(1);
                (m_block, Some(phi_args[0].clone()))
            } else {
                (graph.create_block(), None)
            };

            let mut any_open = false;
            // Collect open-arm exit snapshots before `set_goto` closes
            // each arm, so the post-loop iterative-fold over 2-way
            // `FrameState::union` sees every reaching predecessor's
            // locals state.  Parity port of Expr::If's both-open-arm
            // merge generalised to N arms — see the union block below
            // for the per-slot semantics.
            //
            // Pair each framestate with the corresponding
            // `LocalBindingSnapshot` so the post-loop single-open-arm
            // branch can restore exactly that arm's ctx bindings
            // (per the carrier-comment above the `arm_tails` decl).
            let mut open_arm_snapshots: Vec<(FrameState, LocalBindingSnapshot)> = Vec::new();
            for (tail, result, exit_snapshot, exit_locals) in &arm_tails {
                if !graph.block(*tail).is_open() {
                    continue;
                }
                any_open = true;
                open_arm_snapshots.push((exit_snapshot.clone(), exit_locals.clone()));
                let goto_args: Vec<crate::flowspace::model::Variable> = if all_open_arms_have_value
                {
                    // Safe: the filter above guarantees every open arm's
                    // `result` is `Some`.
                    vec![result.clone().unwrap()]
                } else {
                    Vec::new()
                };
                graph.set_goto(*tail, merge, goto_args);
                // Slice 4a: stamp the arm tail's framestate so the
                // merge block's lazy installer can thread `Link.args`
                // back through this predecessor.  Mirrors the
                // `Expr::If` then/else stamp at the equivalent
                // `set_goto` site.
                graph.block_mut(*tail).framestate = Some(exit_snapshot.clone());
            }

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

            // Iterative-fold-driven merge when 2+ arms reach the
            // merge block.  Per-slot semantics mirror Expr::If's
            // both-open-arm merge, generalised to N arms by left-
            // folding `acc.union(arm)` against each open-arm
            // snapshot — direct port of `flowspace/flowcontext.py:
            // 430-436 mergeblock`'s repeated 2-way union over
            // arriving candidates.
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
            if open_arm_snapshots.len() >= 2 {
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
                // PRE-EXISTING-ADAPTATION (no SpamBlock /
                // recloseblock chain — fused into direct construction).
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
                let mut acc = open_arm_snapshots[0].0.clone();
                for (arm, _) in &open_arm_snapshots[1..] {
                    // Same `expect` rationale as the `Expr::If` site:
                    // AST-frontend `union` is total today (entries
                    // Variable-identity, vestigial empty/None for the
                    // other 4 projections).  Z4 walker activation
                    // makes the None branch reachable; this site
                    // then needs the upstream
                    // `flowcontext.py:431-436` candidate-loop
                    // fallback.
                    acc = acc.union(arm, graph).expect(
                        "AST frontend: union is total — entries domain has no UnionError, \
                         stack / last_exception / blocklist / next_offset are vestigial \
                         (framestate.py:78 None-return reachable only post-Z4 walker)",
                    );
                }
                let merged = acc;
                let first_arm = &open_arm_snapshots[0].0;
                // Type unification across arms is annotator-side per
                // upstream `framestate.py:union` (Hlvalue identity
                // only).  The prior carry-through retag block was a
                // NEW-DEVIATION dependent on the retired
                // `FrameStateEntry::value_type` field; it has been
                // removed in Path-Z Slice 2.3 (mirrors the If/else
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
                open_arm_snapshots[0].1.clone().restore(ctx);
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
            let header_entry = graph.create_block();
            let exit = graph.create_block();

            // Slice 5c.1 eager phi pre-allocation.  Capture the
            // pre-loop local-binding snapshot, close pre-loop's exit
            // to the header, then statically pre-scan the body for
            // its read/rebound names so `allocate_loop_header_phis`
            // can install header phis BEFORE the body walk.  This
            // replaces Slice 5a's lazy back-edge install (which blew
            // up with cycle / arity / kind-mismatch errors), in line
            // with RPython's work-list `mergeblock`+`union` fixpoint
            // semantics adapted for pyre's static AST.
            //
            // Framestate stamps on `pre_loop_block` (after allocator
            // pushes pre-loop link args) and `header_tail` (after
            // cond) keep cross-block lazy installs working for
            // forward (non-back-edge) reads — RPython parity for the
            // stamp shape:
            // `flowspace/flowcontext.py:407-408 record_block(block)`.
            let pre_loop_block = *block;
            let pre_loop_snapshot = ctx.getstate(graph, 0);
            graph.set_goto(pre_loop_block, header_entry, vec![]);

            let must_merge = loop_body_locals(&w.body);
            let _ = allocate_loop_header_phis(
                graph,
                ctx,
                pre_loop_block,
                header_entry,
                &pre_loop_snapshot,
                &must_merge,
            );
            graph.block_mut(pre_loop_block).framestate = Some(pre_loop_snapshot);

            // Capture the ctx state AFTER `allocate_loop_header_phis`
            // has rebound the must_merge names to the header phis but
            // BEFORE the body walk introduces body-local bindings.
            // Restored at `*block = exit` below so body-only locals
            // (e.g. inner `let z = ...`, the `for` pattern variable)
            // drop out of `ctx.local_value_ids` once the loop closes.
            // RPython parity: `flowspace/flowcontext.py:407 setstate`
            // resets `self.locals_w` to the joined frame's slots at
            // every block entry — post-loop reads see only the slots
            // that flow through the header, not the body's transient
            // rebinds.  Pyre's flat `local_value_ids` HashMap doesn't
            // model scope, so without this restore body-local bindings
            // leak past the loop close and surface as orphan
            // Variables in any framestate captured at the post-loop
            // block (e.g. `set_goto_from_framestate` threading every
            // slot via `getoutputargs`).
            let post_eager_phi_locals = LocalBindingSnapshot::capture(ctx);

            // Header: evaluate condition, branch to body or exit.
            // `lower_expr(&mut header_tail, ...)` may rewire to a
            // sub-merge; the cond-branch attaches to header_tail so
            // the branch lives at the header's actual end.
            //
            // RPython `flowspace/flowcontext.py:91,107,364`: unsupported
            // cond raises `FlowingError`.  We propagate that via `?` —
            // no fake cond, no fallback goto-exit.  The exit block we
            // pre-created above becomes dead; simplify prunes it.
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

            // Body → back to header_entry (entry, not tail —
            // header_entry is the back-edge target).  Each stmt may
            // close its path (inner `return` / `break` / `panic!`); on
            // closure we stop walking the body and the back-edge is
            // skipped via the `is_open` check below.  The loop frame
            // makes `break` / `continue` in the body route to exit /
            // header.  Cat 2-2 Phase B α.1: header-phi name list is
            // computed on-demand at each close site (back-edge below,
            // `Expr::Continue`) via `header_phi_name_list(graph, ...)`
            // so any lazy-install addition to `header_entry.inputargs`
            // during the body walk is automatically threaded.
            ctx.loop_stack.push(LoopFrame {
                continue_target: header_entry,
                break_target: exit,
            });
            let mut body_tail = body_entry;
            for stmt in &w.body.stmts {
                let closed = lower_stmt(graph, &mut body_tail, stmt, options, ctx)?;
                if closed {
                    break;
                }
            }
            ctx.loop_stack.pop();
            if graph.block(body_tail).is_open() {
                // Each phi name's current ctx binding — body's new
                // write on rebind, or a body-side inputarg installed
                // by the lazy cross-block reader on first read, or the
                // header phi itself for read-only names — supplies the
                // back-edge arg.  RPython parity:
                // `flowspace/framestate.py:92 getoutputargs` produces
                // the same slot-by-slot mapping for the closing
                // predecessor link.
                let body_tail_snapshot = ctx.getstate(graph, 0);
                let header_phi_names = header_phi_name_list(graph, header_entry);
                let back_edge_vars = link_arg_vars_from_ctx(graph, ctx, &header_phi_names);
                graph.set_goto(body_tail, header_entry, back_edge_vars);
                // Audit Cat 2-1: stamp the body-tail's framestate
                // so the post-loop lazy installer can thread reads
                // of pre-loop locals through this back-edge.  Same
                // role as the break / continue stamps; the cycle
                // hazard cited by the prior cycle-breaker comment
                // is now closed structurally by the lazy installer's
                // push-first refactor (`lazy_install_local_at_current_block`
                // installs the outer phi's inputarg slot before
                // recursing into predecessors, so a back-edge
                // recursion finds it via the same-block graph-state
                // idempotency check).
                graph.block_mut(body_tail).framestate = Some(body_tail_snapshot);
            }

            *block = exit;
            post_eager_phi_locals.restore(ctx);
            Ok(Lowered::no_value())
        }
        syn::Expr::Loop(l) => {
            // Slice 5c.2 eager phi pre-allocation.  `Expr::Loop` has
            // no cond block — `body_entry` IS the loop head and the
            // `continue` target.  Otherwise the shape mirrors
            // `Expr::While` (5c.1): pre-loop snapshot capture, body
            // pre-scan, eager phi install at the head before the body
            // walk, frozen header_phi_names captured for the LoopFrame
            // and back-edge close.  RPython parity: same as the
            // `Expr::While` justification above.
            let body_entry = graph.create_block();
            let exit = graph.create_block();

            let pre_loop_block = *block;
            let pre_loop_snapshot = ctx.getstate(graph, 0);
            graph.set_goto(pre_loop_block, body_entry, vec![]);

            let must_merge = loop_body_locals(&l.body);
            let _ = allocate_loop_header_phis(
                graph,
                ctx,
                pre_loop_block,
                body_entry,
                &pre_loop_snapshot,
                &must_merge,
            );
            graph.block_mut(pre_loop_block).framestate = Some(pre_loop_snapshot);

            // Body-local scope cleanup — see the matching
            // `Expr::While` capture/restore comment above.
            let post_eager_phi_locals = LocalBindingSnapshot::capture(ctx);

            // Cat 2-2 Phase B α.1: header-phi name list recomputed on
            // demand at the back-edge close below, so any lazy install
            // that adds an inputarg to `body_entry` during the body
            // walk is automatically threaded.
            ctx.loop_stack.push(LoopFrame {
                continue_target: body_entry,
                break_target: exit,
            });
            let mut body_tail = body_entry;
            for stmt in &l.body.stmts {
                let closed = lower_stmt(graph, &mut body_tail, stmt, options, ctx)?;
                if closed {
                    break;
                }
            }
            ctx.loop_stack.pop();
            if graph.block(body_tail).is_open() {
                // Audit Cat 2-1: stamp body-tail's framestate (see
                // the matching `Expr::While` body-tail stamp for the
                // cycle-safety rationale).
                let body_tail_snapshot = ctx.getstate(graph, 0);
                let header_phi_names = header_phi_name_list(graph, body_entry);
                let back_edge_vars = link_arg_vars_from_ctx(graph, ctx, &header_phi_names);
                graph.set_goto(body_tail, body_entry, back_edge_vars);
                graph.block_mut(body_tail).framestate = Some(body_tail_snapshot);
            }

            *block = exit;
            post_eager_phi_locals.restore(ctx);
            Ok(Lowered::no_value())
        }
        syn::Expr::ForLoop(f) => {
            // RPython `for` lowers to the iterator protocol: `GET_ITER`
            // on the iterable, then a `FOR_ITER` at the header whose
            // true arm binds the next item into the body and whose
            // false arm falls through (`rpython/flowspace/
            // flowcontext.py:782,787,1378`).  Pyre has NO `Iter` /
            // `Next` op yet (Slice 6 port).  The shape below is
            // deliberately NOT claiming op-level equivalence with
            // upstream's iter/next — it emits a SINGLE `Unknown`
            // marker tagged `ForLoop` at the header that stands for
            // the whole iterator protocol, and walks the iterable
            // sub-expression for its side effects so the
            // `build_flow`-visible part of the construct is complete
            // even when the loop ops themselves are stubbed.
            //
            // Slice 5c.3 eager phi pre-allocation: applied identically
            // to `Expr::Loop` / `Expr::While`.  The iterable is
            // single-evaluation (RPython `flowcontext.py:1378
            // GET_ITER` evaluates it once before the loop), so its
            // result vid is bound in `pre_loop_block` and reads of
            // it inside the header are forward edges covered by lazy
            // install — no special-casing needed.
            let iterable_pre_var =
                get_value_var!(lower_expr(graph, block, &f.expr, options, ctx)?, graph);
            ctx.pushvid_var(&iterable_pre_var);
            let iterable = ctx.popvid_var(graph);
            let _ = iterable;

            let header_entry = graph.create_block();
            let body_entry = graph.create_block();
            let exit = graph.create_block();

            let pre_loop_block = *block;
            let pre_loop_snapshot = ctx.getstate(graph, 0);
            graph.set_goto(pre_loop_block, header_entry, vec![]);

            let must_merge = loop_body_locals(&f.body);
            let _ = allocate_loop_header_phis(
                graph,
                ctx,
                pre_loop_block,
                header_entry,
                &pre_loop_snapshot,
                &must_merge,
            );
            graph.block_mut(pre_loop_block).framestate = Some(pre_loop_snapshot);

            // Body-local scope cleanup — see the matching
            // `Expr::While` capture/restore comment above.  The
            // `f.pat` loop variable is body-only by `loop_body_locals`'s
            // own filter (`ast.rs:2031-2037`), so this restore is what
            // drops `entry` / `(k, v)` / etc. from `local_value_ids`
            // once the loop closes — without it, a downstream
            // `set_goto_from_framestate` over the post-loop block's
            // framestate threads the orphan body-local vid through
            // every `getoutputargs` slot and trips
            // `ensure_variable_at_block`'s pred-chain reachability
            // assert.
            let post_eager_phi_locals = LocalBindingSnapshot::capture(ctx);

            // Single iterator-protocol placeholder, NOT two separate
            // iter/next markers.  The branch shape is required to
            // make `exit` reachable from the normal control-flow
            // fallthrough (without it, loops without `break` would
            // leave every statement after the `for` unreachable).
            let for_cond_var = graph.push_op_var(
                header_entry,
                OpKind::Abort {
                    kind: UnknownKind::UnsupportedExpr {
                        variant: UnsupportedExprKind::ForLoop,
                    },
                },
                true,
            );
            // Stamp header_entry.framestate before the cond-branch
            // close — mirrors `Expr::While`.  Reads inside the body
            // or post-loop exit recurse back to the header and find
            // its exit-time snapshot (which already includes the
            // eager phis bound to ctx).
            let header_branch_snapshot = ctx.getstate(graph, 0);
            if let Some(cond_var) = for_cond_var {
                graph.set_branch(header_entry, cond_var, body_entry, vec![], exit, vec![]);
            } else {
                graph.set_goto(header_entry, body_entry, vec![]);
            }
            graph.block_mut(header_entry).framestate = Some(header_branch_snapshot);

            // Cat 2-2 Phase B α.1: header-phi name list recomputed on
            // demand at the back-edge close below.
            ctx.loop_stack.push(LoopFrame {
                continue_target: header_entry,
                break_target: exit,
            });
            let mut body_tail = body_entry;
            for stmt in &f.body.stmts {
                let closed = lower_stmt(graph, &mut body_tail, stmt, options, ctx)?;
                if closed {
                    break;
                }
            }
            ctx.loop_stack.pop();
            if graph.block(body_tail).is_open() {
                // Audit Cat 2-1: stamp body-tail's framestate (see
                // the matching `Expr::While` body-tail stamp for the
                // cycle-safety rationale).
                let body_tail_snapshot = ctx.getstate(graph, 0);
                let header_phi_names = header_phi_name_list(graph, header_entry);
                let back_edge_vars = link_arg_vars_from_ctx(graph, ctx, &header_phi_names);
                graph.set_goto(body_tail, header_entry, back_edge_vars);
                graph.block_mut(body_tail).framestate = Some(body_tail_snapshot);
            }

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
                    // `flowcontext.py:438` close the predecessor via
                    // `currentstate.getoutputargs(newstate)`.  The
                    // post-loop block's entry framestate, when present,
                    // is the merge result for this Link's target —
                    // RPython `flowcontext.py:399-465 mergeblock` reads
                    // `newstate.mergeable` to decide which positions
                    // contribute args.  Read it from the break target
                    // when set; fall back to the empty `FrameState`
                    // otherwise (today's lazy-installer pipeline does
                    // not eagerly populate the post-loop block's
                    // framestate at break time, so the target state is
                    // typically `None` here and getoutputargs over an
                    // empty target.mergeable contributes zero args —
                    // the previous explicit `vec![]` shape).  Any
                    // later-installed exit inputarg is accompanied by a
                    // corresponding lazy-installed arg push onto this
                    // exit's link by the lazy installer's predecessor
                    // walk; Z4 walker convergence retires that
                    // adaptation in favour of the eager-merge protocol.
                    let pre_break_snapshot = ctx.getstate(graph, 0);
                    let target_state = graph
                        .block(frame.break_target)
                        .framestate
                        .clone()
                        .unwrap_or_default();
                    graph.set_goto_from_framestate(
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
            if let Some(frame) = ctx.loop_stack.last().cloned() {
                if graph.block(*block).is_open() {
                    // `continue` jumps to the loop's continue_target —
                    // for `while` / `loop` (Slice 5c.1+) this is the
                    // header with its eager-phi inputargs.  Thread
                    // per-name args from `ctx.local_value_ids[name]`
                    // using the header's CURRENT inputarg name list
                    // (Cat 2-2 Phase B α.1: recomputed on demand from
                    // `frame.continue_target.inputargs` at close time
                    // so any lazy install that added an inputarg
                    // during body walk is automatically threaded).
                    // RPython parity for the slot-by-slot mapping:
                    // `flowspace/framestate.py:92 getoutputargs`.
                    //
                    // Audit Cat 2-1: stamp the continue source's
                    // framestate so the post-loop lazy installer can
                    // thread reads of pre-loop locals (NOT in
                    // `must_merge`) back through this back-edge —
                    // same role as the break source's framestate
                    // stamp at line 3872.  RPython parity:
                    // `flowspace/flowcontext.py:399-465 mergeblock`
                    // requires every closing predecessor of the
                    // merge target to carry a FrameState so
                    // `getoutputargs` can resolve the target's
                    // inputargs slot-by-slot.  The earlier "stamping
                    // would overflow the stack" cycle hazard is now
                    // closed structurally by the lazy installer's
                    // Phase-2 push-first refactor: the back-edge's
                    // snap_vid for a header-phi name points at the
                    // header's already-installed phi inputarg, so
                    // the recursive install at the header short-
                    // circuits via the same-block graph-state
                    // idempotency check on `block.inputargs`.
                    let pre_continue_snapshot = ctx.getstate(graph, 0);
                    let header_phi_names = header_phi_name_list(graph, frame.continue_target);
                    let arg_vars = link_arg_vars_from_ctx(graph, ctx, &header_phi_names);
                    graph.set_goto(*block, frame.continue_target, arg_vars);
                    graph.block_mut(*block).framestate = Some(pre_continue_snapshot);
                }
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
        // attempt to walk the body in-place was a NEW-DEVIATION (it
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
            // (Slice 5 port), so the construct lowers to a single
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
            let inner_pre_var =
                get_value_var!(lower_expr(graph, block, &t.expr, options, ctx)?, graph);
            ctx.pushvid_var(&inner_pre_var);
            let inner_var = ctx.popvid_var(graph);
            if let Some(ok_ty) = ok_ty {
                retag_result_value_type(graph, &inner_var, ok_ty);
            }
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
            *block = continuation;
            Ok(Lowered::from_value_var(graph, &continuation_arg))
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
/// callable-resolved Path (via the M2.c `classdef_impl_types`
/// registry, `description.py:407-519` parity); the classifier
/// can then route through the resolved callable and the shortlist
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
                // `unsafe fn` bodies to graph analysis (multi-session;
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
                return CallTarget::synthetic_transparent_ctor(
                    segments
                        .last()
                        .expect("transparent ctor path is non-empty")
                        .clone(),
                );
            }
            if segments.len() == 1 && !ctx.module_prefix.is_empty() {
                let mut qualified = ctx
                    .module_prefix
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
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    matches!(
        path.as_slice(),
        // Result/Option transparent wrappers — elided at jtransform.
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
            // Pyre-side `Class::Variant` ctors — classdesc-routed,
            // not elided.
            | ["LoopResult", "Done"]
            | ["LoopResult", "ContinueRunningNormally"]
            | ["JitAction", "Return"]
            | ["JitAction", "Continue"]
            | ["StepResult", "Continue"]
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
        ConstValue::Int(_) => Some(ValueType::Int),
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
        | ConstValue::HostObject(_) => Some(ValueType::Ref),
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
        'r' => Some(ValueType::Ref),
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
        // PRE-EXISTING-ADAPTATION (Task #345): the upstream surface is
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
        (Some(ValueType::Ref), ValueType::Int) => Some(&[
            "rpython",
            "rtyper",
            "lltypesystem",
            "lltype",
            "cast_ptr_to_int",
        ]),
        (Some(ValueType::Int), ValueType::Ref) => Some(&[
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

    let method_name = method.to_string();
    let receiver_leaf = receiver_root.rsplit("::").next().unwrap_or(receiver_root);
    // Rust imports can make the call-site owner path shorter or longer
    // than the impl key. Use the leaf owner only when it is unambiguous.
    let mut matches = ctx.fn_return_types.iter().filter_map(|(key, ret)| {
        let (owner, candidate_method) = key.rsplit_once("::")?;
        if candidate_method == method_name.as_str()
            && owner.rsplit("::").next().unwrap_or(owner) == receiver_leaf
        {
            Some(ret)
        } else {
            None
        }
    });
    let first = matches.next()?;
    if matches.next().is_none() {
        Some(first)
    } else {
        None
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
pub(crate) fn nolength_from_array_type_id(array_type_id: Option<&str>) -> bool {
    let Some(s) = array_type_id else {
        return false;
    };
    let mut inner = s.trim();
    loop {
        let stripped = inner
            .strip_prefix("*const ")
            .or_else(|| inner.strip_prefix("*mut "))
            .or_else(|| inner.strip_prefix("&mut "))
            .or_else(|| inner.strip_prefix('&'));
        match stripped {
            Some(rest) => inner = rest.trim_start(),
            None => break,
        }
    }
    // `[T]` / `[T; N]` are unambiguous headerless item runs.
    if inner.starts_with('[') && inner.ends_with(']') {
        return true;
    }
    // Length-prefixed wrappers carry `<` (generic) or `(` (paren-style
    // lltype spelling such as `Ptr(GcArray(...))`).  Keep the PyPy
    // default `False` for those — a pointer to a wrapper still
    // dereferences a length header.
    if inner.contains('<') || inner.contains('(') {
        return false;
    }
    // Bare identifier pointee (`*const i64`, `*const Point`) means the
    // pointer addresses items[0] of a primitive / struct item type.
    // A bare identifier with NO pointer prefix is a value-type binding
    // (e.g. an `array_type_id` directly naming a struct that contains
    // an embedded array); preserve the PyPy default `False` for that.
    s.trim() != inner
}

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
                None => return ValueType::Ref,
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
                return ValueType::Ref;
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
                _ => ValueType::Ref,
            }
        }
        // `&T` / `&mut T` — pointer → Ref (lltype.Ptr in RPython).
        syn::Type::Reference(_) => ValueType::Ref,
        // `*const T` / `*mut T` — raw pointer, same class as Ref.  pyre
        // often stores GC objects as `*mut PyObject`; classify as Ref
        // so field/array bases reach the canonical `/rd>X` encoding
        // rather than the pyre-only `*_intbase` aliases.
        syn::Type::Ptr(_) => ValueType::Ref,
        syn::Type::Paren(paren) => classify_fn_arg_ty(&paren.elem),
        syn::Type::Group(group) => classify_fn_arg_ty(&group.elem),
        // `dyn Trait` — GC pointer to a trait object.
        syn::Type::TraitObject(_) => ValueType::Ref,
        // Tuple/array/slice: treat as Ref (bulk data, not a register
        // primitive).  RPython `lltype.Array` + `lltype.Struct` both
        // flatten to `lltype.Ptr` at the call-site boundary.
        syn::Type::Tuple(_) | syn::Type::Array(_) | syn::Type::Slice(_) => ValueType::Ref,
        // `fn(T) -> T`, `impl Trait`, never — no runtime
        // representation reaches the SSA level; default to Ref for
        // safe-by-default classification.
        _ => ValueType::Ref,
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
        _ => ValueType::Ref,
    }
}

pub(crate) fn transparent_result_ok_type(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    for prefix in ["Result<", "std::result::Result<", "core::result::Result<"] {
        let Some(inner) = trimmed
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };
        let ok_type = first_top_level_generic_arg(inner).map(str::trim)?;
        if ok_type == "()" {
            return None;
        }
        return Some(ok_type);
    }
    None
}

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
            // + `PyreCallRegistry` entry (multi-session epic).
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
    fn struct_field_registry_does_not_fall_back_to_ambiguous_leaf_names() {
        let parsed = crate::parse::parse_source(
            r#"
            mod a {
                pub struct Foo { pub x: i64 }
                pub fn read(foo: Foo) -> i64 { foo.x }
                pub fn destructure(foo: Foo) -> i64 {
                    let Foo { x } = foo;
                    x
                }
            }
            mod b {
                pub struct Foo { pub x: f64 }
                pub fn read(foo: Foo) -> f64 { foo.x }
                pub fn destructure(foo: Foo) -> f64 {
                    let Foo { x } = foo;
                    x
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");

        assert_eq!(program.struct_fields.field_type("a::Foo", "x"), Some("i64"));
        assert_eq!(program.struct_fields.field_type("b::Foo", "x"), Some("f64"));
        assert_eq!(
            program.struct_fields.field_type("Foo", "x"),
            None,
            "bare Foo is ambiguous and must not pick the first registered module"
        );

        let field_read = |func_name: &str| {
            let graph = &program
                .functions
                .iter()
                .find(|func| func.name == func_name)
                .unwrap_or_else(|| panic!("{func_name} graph"))
                .graph;
            graph
                .blocks
                .iter()
                .flat_map(|block| block.operations.iter())
                .find_map(|op| match &op.kind {
                    OpKind::FieldRead { field, ty, .. } if field.name == "x" => {
                        Some((field.owner_root.clone(), ty.clone()))
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{func_name} FieldRead"))
        };

        assert_eq!(
            field_read("a::read"),
            (Some("a::Foo".to_string()), ValueType::Int)
        );
        assert_eq!(
            field_read("b::read"),
            (Some("b::Foo".to_string()), ValueType::Float)
        );

        let destructured_input_ty = |func_name: &str| {
            let graph = &program
                .functions
                .iter()
                .find(|func| func.name == func_name)
                .unwrap_or_else(|| panic!("{func_name} graph"))
                .graph;
            graph.blocks.iter().find_map(|block| {
                block.operations.iter().find_map(|op| match &op.kind {
                    OpKind::Input { name, ty } if name == "x" => Some(ty.clone()),
                    _ => None,
                })
            })
        };

        assert_eq!(
            destructured_input_ty("a::destructure"),
            Some(ValueType::Int)
        );
        assert_eq!(
            destructured_input_ty("b::destructure"),
            Some(ValueType::Float)
        );
    }

    #[test]
    fn type_string_helpers_preserve_outer_slice_and_as_ref_wrapper_identity() {
        assert_eq!(
            extract_element_type_from_str("[std::rc::Rc<majit_ir::RdVirtualInfo>]"),
            Some("std::rc::Rc<majit_ir::RdVirtualInfo>".to_string())
        );
        assert_eq!(
            extract_element_type_from_str("&[FieldDescrInfo]"),
            Some("FieldDescrInfo".to_string())
        );
        assert_eq!(
            method_as_ref_return_type("&std::rc::Rc<majit_ir::RdVirtualInfo>"),
            Some("&majit_ir::RdVirtualInfo".to_string())
        );
        assert_eq!(
            method_as_ref_return_type("Vec<FieldDescrInfo>"),
            Some("&[FieldDescrInfo]".to_string())
        );
        assert_eq!(
            method_as_ref_return_type("Option<FieldDescrInfo>"),
            Some("Option<&FieldDescrInfo>".to_string())
        );
    }

    #[test]
    fn binds_slice_get_result_from_enum_variant_tuple_destructure() {
        let ir = crate::parse::parse_source(
            r#"
            enum Type {
                Ref,
                Float,
                Int,
            }
            struct FieldDescrInfo {
                field_type: Type,
                field_size: usize,
            }
            enum RdVirtualInfo {
                VirtualInfo {
                    fielddescrs: Vec<FieldDescrInfo>,
                    descr_size: usize,
                },
                Empty,
            }
        "#,
        );
        let eval = crate::parse::parse_source(
            r#"
            fn read_field_size(
                rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
                vidx: usize,
                i: usize,
            ) -> i64 {
                let Some(virtuals) = rd_virtuals else {
                    return 0;
                };
                let Some(entry) = virtuals.get(vidx) else {
                    return 0;
                };
                let (fielddescrs, _descr_size) = match entry.as_ref() {
                    majit_ir::RdVirtualInfo::VirtualInfo {
                        fielddescrs,
                        descr_size,
                    } => (fielddescrs.as_slice(), *descr_size),
                    _ => return 0,
                };
                let Some(descr) = fielddescrs.get(i) else {
                    return 0;
                };
                match descr.field_size {
                    1 => 1,
                    _ => 8,
                }
            }
        "#,
        );
        let program =
            build_semantic_program_from_parsed_files(&[ir, eval]).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|func| func.name == "read_field_size")
            .expect("read_field_size graph")
            .graph;
        let field_size_read = graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .find_map(|op| match &op.kind {
                OpKind::FieldRead { field, ty, .. } if field.name == "field_size" => {
                    Some((field.owner_root.clone(), ty.clone()))
                }
                _ => None,
            })
            .expect("field_size FieldRead");
        assert_eq!(
            field_size_read,
            (Some("FieldDescrInfo".to_string()), ValueType::Int)
        );
    }

    #[test]
    fn binds_option_some_inner_type_in_match_arm() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Meta { tracing_call_depth: Option<u32> }
            fn call_depth() -> u32 { 0 }
            fn example(meta: Meta) -> bool {
                let tracing_depth = meta.tracing_call_depth;
                if let Some(depth) = tracing_depth {
                    call_depth() == depth
                } else {
                    false
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|func| func.name == "example")
            .expect("example graph")
            .graph;
        assert!(
            graph.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Input { name, ty }
                            if name == "depth" && *ty == ValueType::Int
                    )
                })
            }),
            "expected Some(depth) to bind depth as Int; graph:\n{}",
            graph.dump()
        );
    }

    #[test]
    fn match_three_arm_rebind_routes_through_iterative_fold_merge() {
        // Audit Cat 2-3 fix: a 3-arm match whose arms each rebind a
        // pre-existing local must produce a single merge-block phi
        // that unifies all three arm vids — exercising the iterative
        // left-fold over 2-way `FrameState::union` (vid disagreement
        // at any fold step allocates a fresh phi) plus eager phi
        // install at union time.  RPython parity:
        // `flowspace/flowcontext.py:430-436 mergeblock` repeatedly
        // 2-way-unions arriving candidates against the running
        // state; pyre's static AST shape applies the same fold over
        // the open-arm exit snapshots in a single Expr::Match
        // lowering visit.
        //
        // Without the iterative fold, the merge would carry no ctx
        // update at union time and the post-merge `x` read would
        // depend on the lazy installer firing post-hoc — the audit's
        // exact "PyPy-style merge state not reflected in ctx" gap.
        let parsed = crate::parse::parse_source(
            r#"
            fn example(tag: i64) -> i64 {
                let mut x = 0;
                match tag {
                    1 => { x = 10; }
                    2 => { x = 20; }
                    _ => { x = 30; }
                }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|func| func.name == "example")
            .expect("example graph")
            .graph;
        // The merge block must own an `x` phi inputarg installed
        // eagerly by the iterative fold's fresh-phi step (the
        // disagreeing-vid branch of 2-way `union`).
        let phi_block = graph.blocks.iter().find(|block| {
            block.operations.iter().any(|op| {
                op.result
                    .as_ref()
                    .is_some_and(|r| block.inputargs.contains(r))
                    && matches!(
                        &op.kind,
                        OpKind::Input { name, .. } if name == "x"
                    )
            })
        });
        assert!(
            phi_block.is_some(),
            "expected 3-arm match merge to allocate an `x` phi inputarg via the iterative fold; graph:\n{}",
            graph.dump()
        );
        // All three arms must thread their per-arm rebind value as
        // `Link.args` into the merge — each predecessor edge must
        // carry ≥1 arg (the rebound `x`).  ≥3 predecessors covers
        // the n-way generalisation; the ≥1 per-edge contract pins
        // that the eager install threaded through every arm rather
        // than only a subset.
        let phi_block_id = phi_block.unwrap().id;
        let pred_arg_count: Vec<usize> = graph
            .blocks
            .iter()
            .flat_map(|b| {
                b.exits.iter().filter_map(move |exit| {
                    if exit.target == phi_block_id {
                        Some(exit.args.len())
                    } else {
                        None
                    }
                })
            })
            .collect();
        assert!(
            pred_arg_count.len() >= 3 && pred_arg_count.iter().all(|&n| n >= 1),
            "expected ≥3 predecessor links (3-arm match) with ≥1 Link.args each; got {:?}\ngraph:\n{}",
            pred_arg_count,
            graph.dump()
        );
    }

    #[test]
    fn match_arm_rebind_threads_phi_inputarg_through_lazy_installer() {
        // Slice 4 / Stage D parity test: a pre-match local rebound on
        // every match arm must resolve to a merge-block phi inputarg
        // when read after the match, with both predecessor links
        // carrying the arm-specific rebind value.  RPython parity:
        // `flowspace/framestate.py:113-114 union` returns a fresh
        // `Variable` when both incoming `locals_w` slots are
        // `Variable`s, and `flowspace/flowcontext.py:449
        // currentstate.getoutputargs(newstate)` produces the
        // predecessor-side `Link.args`.
        //
        // Without Slice 4a's per-arm `Block.framestate` stamp, the
        // lazy installer's predecessor walk would find no snapshot
        // on the arm tails and fall back to a naked `OpKind::Input`
        // emit — the resulting graph would still type-check but the
        // arm-specific rebinds would not flow into the merge.
        let parsed = crate::parse::parse_source(
            r#"
            fn example(cond: bool) -> i64 {
                let mut x = 0;
                match cond {
                    true => { x = 10; }
                    false => { x = 20; }
                }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|func| func.name == "example")
            .expect("example graph")
            .graph;
        // Some block must carry `OpKind::Input { name: "x" }` as an
        // inputarg — that's the merge-block phi the lazy installer
        // allocates when the post-match `x` read fires.
        let phi_block = graph.blocks.iter().find(|block| {
            block.operations.iter().any(|op| {
                op.result
                    .as_ref()
                    .is_some_and(|r| block.inputargs.contains(r))
                    && matches!(
                        &op.kind,
                        OpKind::Input { name, .. } if name == "x"
                    )
            })
        });
        assert!(
            phi_block.is_some(),
            "expected match-arm-rebind merge to allocate an `x` phi inputarg; graph:\n{}",
            graph.dump()
        );
        // Each predecessor edge into the phi block must carry one
        // extra `Link.args` value — the per-arm rebind threaded back
        // by the lazy installer.
        let phi_block_id = phi_block.unwrap().id;
        let pred_arg_count: Vec<usize> = graph
            .blocks
            .iter()
            .flat_map(|b| {
                b.exits.iter().filter_map(move |exit| {
                    if exit.target == phi_block_id {
                        Some(exit.args.len())
                    } else {
                        None
                    }
                })
            })
            .collect();
        assert!(
            pred_arg_count.len() >= 2 && pred_arg_count.iter().all(|&n| n >= 1),
            "expected ≥2 predecessor links with ≥1 Link.args each; got {:?}\ngraph:\n{}",
            pred_arg_count,
            graph.dump()
        );
    }

    #[test]
    fn while_loop_back_edge_threads_modified_local_through_header_phi() {
        // Slice 5a parity test: a while loop whose body rebinds a
        // pre-loop local must produce a header inputarg phi for that
        // local with link args from BOTH the pre-loop block (forward
        // edge, supplying the pre-loop value) AND the body tail
        // (back-edge, supplying the body's rebound value).  RPython
        // parity: `flowspace/flowcontext.py:438-451 mergeblock` opens
        // a fresh `SpamBlock(newstate)` whose `framestate.locals_w`
        // slot for a body-modified name becomes a `Variable`; the
        // closing forward-edge link's `getoutputargs` carries the
        // pre-loop value and each subsequent merge (back-edge in
        // pyre's single-pass) carries the body's value.
        let parsed = crate::parse::parse_source(
            r#"
            fn example() -> i64 {
                let mut i: i64 = 0;
                while i < 10 {
                    i = i + 1;
                }
                i
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|func| func.name == "example")
            .expect("example graph")
            .graph;
        // Find the header block — the one whose inputarg is
        // `Input { name: "i" }` and that has TWO predecessor edges
        // (forward + back-edge).
        let mut header_id: Option<BlockId> = None;
        for block in &graph.blocks {
            if block.inputargs.is_empty() {
                continue;
            }
            let has_i_input = block.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::Input { name, .. } if name == "i")
                    && op
                        .result
                        .as_ref()
                        .is_some_and(|r| block.inputargs.contains(r))
            });
            if !has_i_input {
                continue;
            }
            let pred_count = graph
                .blocks
                .iter()
                .flat_map(|b| b.exits.iter().filter(|e| e.target == block.id).map(|_| ()))
                .count();
            if pred_count >= 2 {
                header_id = Some(block.id);
                break;
            }
        }
        let header_id = header_id.unwrap_or_else(|| {
            panic!(
                "expected header block with `i` inputarg + ≥2 preds; graph:\n{}",
                graph.dump()
            )
        });
        // Each predecessor edge into the header must carry exactly
        // one Link.arg (the per-iteration value of `i`).
        let pred_arg_counts: Vec<usize> = graph
            .blocks
            .iter()
            .flat_map(|b| {
                b.exits.iter().filter_map(move |exit| {
                    if exit.target == header_id {
                        Some(exit.args.len())
                    } else {
                        None
                    }
                })
            })
            .collect();
        assert!(
            pred_arg_counts.iter().all(|&n| n == 1),
            "every predecessor link into the header must carry 1 arg \
             (the `i` value); got {:?}; graph:\n{}",
            pred_arg_counts,
            graph.dump()
        );
    }

    #[test]
    fn binds_result_err_inner_type_in_if_let_arm() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(result: Result<i64, f64>) -> f64 {
                if let Err(err) = result {
                    err
                } else {
                    0.0
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;

        assert!(
            graph.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Input { name, ty }
                            if name == "err" && *ty == ValueType::Float
                    )
                })
            }),
            "expected Err(err) to bind err as Float; graph:\n{}",
            graph.dump()
        );
    }

    fn count_field_reads(graph: &FunctionGraph, field_name: &str) -> usize {
        let mut n = 0;
        for bid in 0..graph.blocks.len() {
            for op in &graph.blocks[bid].operations {
                if let OpKind::FieldRead { field, .. } = &op.kind {
                    if field.name == field_name {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    /// Block tail expression must be lowered exactly once.
    ///
    /// RPython flow-space invariant: every source expression is walked
    /// once. Before the `lower_stmt_list_with_tail_value` refactor, a
    /// block's tail was lowered via `lower_stmt` (which dispatches to
    /// `lower_expr`) AND a second explicit `lower_expr` call, emitting
    /// the op twice.
    #[test]
    fn block_tail_expression_lowered_once() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            fn read_once(s: S) -> i64 {
                let y = { s.x };
                y
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        assert_eq!(
            count_field_reads(graph, "x"),
            1,
            "block tail `s.x` must produce exactly one FieldRead"
        );
    }

    /// `unsafe { .. }` lowers through the same single-walk path as a
    /// plain block — `unsafe` is a type-system marker, not a runtime
    /// wrapper, so the tail expression is walked once.
    #[test]
    fn unsafe_tail_expression_lowered_once() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            fn read_once(s: S) -> i64 {
                let y = unsafe { s.x };
                y
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        assert_eq!(
            count_field_reads(graph, "x"),
            1,
            "unsafe tail `s.x` must produce exactly one FieldRead"
        );
    }

    /// `if` then-branch tail expression is walked once. Counts
    /// FieldReads of `s.x` across every block in the graph so the
    /// assertion is independent of how the then/else blocks are laid
    /// out.
    #[test]
    fn if_then_tail_expression_lowered_once() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            fn read_once(s: S, c: bool) -> i64 {
                if c { s.x } else { 0 }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        assert_eq!(
            count_field_reads(graph, "x"),
            1,
            "if-then tail `s.x` must produce exactly one FieldRead"
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
    fn lowers_cast_deref_field_access_with_typed_float_fieldread() {
        let parsed = crate::parse::parse_source(
            r#"
            struct PyObject { ob_type: i64, w_class: i64 }
            struct W_FloatObject { ob_header: PyObject, floatval: f64 }
            type PyObjectRef = *mut PyObject;

            unsafe fn w_float_get_value(obj: PyObjectRef) -> f64 {
                (*(obj as *const W_FloatObject)).floatval
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, ty, .. }
                    if field.name == "floatval" && *ty == ValueType::Float
            )),
            "expected typed float FieldRead for 'floatval', got {:?}",
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
    fn lowers_unregistered_ok_call_to_synthetic_transparent_ctor() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_example(x: i64) -> i64 {
                Ok(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target == &CallTarget::synthetic_transparent_ctor("Ok")
            )),
            "expected synthetic transparent Ok ctor, got {:?}",
            ops
        );
    }

    #[test]
    fn registered_function_named_ok_does_not_lower_to_synthetic_ctor() {
        let parsed = crate::parse::parse_source(
            r#"
            fn Ok(x: i64) -> i64 { x }
            fn call_example(x: i64) -> i64 {
                Ok(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let func = program
            .functions
            .iter()
            .find(|func| func.name == "call_example")
            .expect("call_example present");
        let ops = &func.graph.block(func.graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.last().map(String::as_str) == Some("Ok")
            )),
            "expected registered Ok function path, got {:?}",
            ops
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )),
            "registered Ok must not be treated as a synthetic ctor: {:?}",
            ops
        );
    }

    #[test]
    fn err_and_some_calls_lower_to_synthetic_transparent_ctors() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_err(x: i64) -> i64 {
                Err(x)
            }
            fn call_some(x: i64) -> i64 {
                Some(x)
            }
            fn call_qualified_ok(x: i64) -> i64 {
                Result::Ok(x)
            }
            fn call_std_some(x: i64) -> i64 {
                std::option::Option::Some(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        for func_name in [
            "call_err",
            "call_some",
            "call_qualified_ok",
            "call_std_some",
        ] {
            let func = program
                .functions
                .iter()
                .find(|func| func.name == func_name)
                .expect("function present");
            let ops = &func.graph.block(func.graph.startblock).operations;
            assert!(
                ops.iter().any(|op| matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { .. },
                        ..
                    }
                )),
                "{func_name} must lower to a synthetic transparent ctor: {:?}",
                ops
            );
        }
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
    fn match_literals_emit_switch_exitcases_for_all_arms() {
        let parsed = crate::parse::parse_source(
            r#"
            fn pick(x: i64) -> i64 {
                match x {
                    0 => 11,
                    1 => 22,
                    _ => 33,
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let entry = graph.block(graph.startblock);

        assert!(
            matches!(entry.exitswitch, Some(crate::model::ExitSwitch::Value(_))),
            "primitive match should switch on the scrutinee, got {:?}",
            entry.exitswitch,
        );
        let exitcases: Vec<_> = entry
            .exits
            .iter()
            .map(|link| link.exitcase.clone())
            .collect();
        assert_eq!(
            exitcases,
            vec![
                Some(ExitCase::Const(ConstValue::Int(0))),
                Some(ExitCase::Const(ConstValue::Int(1))),
                Some(ExitCase::Const(ConstValue::byte_str("default"))),
            ],
        );
        let llexitcases: Vec<_> = entry
            .exits
            .iter()
            .map(|link| link.llexitcase.clone())
            .collect();
        assert_eq!(
            llexitcases,
            vec![Some(ConstValue::Int(0)), Some(ConstValue::Int(1)), None],
        );
    }

    #[test]
    fn match_negative_int_literals_are_classified_as_switch() {
        // RPython `flatten.py:269` reads `link.llexitcase` as Signed,
        // so `match x { -1 => ... }` lands in the switch path the same
        // way `match x { 1 => ... }` does.  syn 2.x represents `-1` as
        // `Pat::Lit(ExprLit { lit: Lit::Int })` whose token text
        // includes the `-`; `LitInt::base10_parse::<i64>` returns
        // `Ok(-1)` for that token.  This guard catches a future
        // regression where `classify_switch_pattern`'s `Pat::Lit` arm
        // narrows to non-negative literals only.
        let parsed = crate::parse::parse_source(
            r#"
            fn pick(x: i64) -> i64 {
                match x {
                    -1 => 11,
                    -2 | -3 => 22,
                    5 => 33,
                    _ => 44,
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let entry = graph.block(graph.startblock);

        assert!(
            matches!(entry.exitswitch, Some(crate::model::ExitSwitch::Value(_))),
            "negative int literals must remain on the switch path, got {:?}",
            entry.exitswitch,
        );
        let exitcases: Vec<_> = entry
            .exits
            .iter()
            .map(|link| link.exitcase.clone())
            .collect();
        assert_eq!(
            exitcases,
            vec![
                Some(ExitCase::Const(ConstValue::Int(-1))),
                Some(ExitCase::Const(ConstValue::Int(-2))),
                Some(ExitCase::Const(ConstValue::Int(-3))),
                Some(ExitCase::Const(ConstValue::Int(5))),
                Some(ExitCase::Const(ConstValue::byte_str("default"))),
            ],
        );
    }

    #[test]
    fn bool_match_wildcard_emits_true_false_exitcases() {
        let parsed = crate::parse::parse_source(
            r#"
            fn pick(flag: bool) -> i64 {
                match flag {
                    true => 11,
                    _ => 22,
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let entry = graph.block(graph.startblock);

        assert!(
            matches!(entry.exitswitch, Some(crate::model::ExitSwitch::Value(_))),
            "bool match should branch on the scrutinee, got {:?}",
            entry.exitswitch,
        );
        let exitcases: Vec<_> = entry
            .exits
            .iter()
            .map(|link| link.exitcase.clone())
            .collect();
        assert_eq!(
            exitcases,
            vec![Some(ExitCase::Bool(true)), Some(ExitCase::Bool(false))],
        );
        let llexitcases: Vec<_> = entry
            .exits
            .iter()
            .map(|link| link.llexitcase.clone())
            .collect();
        assert_eq!(
            llexitcases,
            vec![Some(ConstValue::Bool(true)), Some(ConstValue::Bool(false))],
        );
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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

    #[test]
    fn unary_not_on_int_param_lowers_to_invert() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64) -> i64 {
                !x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        assert!(
            graph
                .block(graph.startblock)
                .operations
                .iter()
                .any(|op| matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, .. } if op == "invert"
                )),
            "expected `!i64` to lower as RPython UNARY_INVERT"
        );
    }

    #[test]
    fn unary_not_on_bool_param_lowers_to_bool_branch() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(flag: bool) -> bool {
                !flag
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let start = graph.block(graph.startblock);
        assert!(
            start.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, result_ty, .. }
                    if op == "bool" && *result_ty == ValueType::Bool
            )),
            "expected `!bool` to lower through RPython UNARY_NOT"
        );
        assert!(
            start.exitswitch.is_some(),
            "UNARY_NOT must branch on bool(x)"
        );
    }

    /// `!x` on a statically unclassified operand fail-louds, mirroring
    /// `flowcontext.py:194,535-538`'s strict UNARY_NOT vs UNARY_INVERT
    /// dispatch and the `build_flow.rs:4404-4416` fail-loud peer.
    /// The `expr_unary_not_operand_kind` classifier
    /// (`front/ast.rs:5582`) covers the production patterns surfaced
    /// in `pyre-{object,interpreter,jit}/src/` plus
    /// `majit-ir/src/resoperation.rs`; an opaque operand with no type
    /// information remains a hard error.
    #[test]
    fn unary_not_on_unknown_operand_fail_louds() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Opaque;
            fn example(x: Opaque) -> bool {
                !x
            }
        "#,
        );
        let err =
            build_semantic_program(&parsed).expect_err("`!x` on opaque operand must fail-loud");
        assert!(
            format!("{err:?}").contains("UnaryNotUnknownOperand"),
            "expected UnaryNotUnknownOperand variant, got {err:?}",
        );
    }

    /// `!std::ptr::eq(a, b)` — stdlib free-function path call whose
    /// `bool` return is known by shortlist (parity with the
    /// `is_null`/`is_some`/... method-call shortlist; pyre's walker
    /// has no stdlib visibility, so RPython
    /// `bookkeeper.getdesc(value)` host-stdlib lookup is mirrored
    /// statically here).  Must lower through the UNARY_NOT
    /// `op.bool` + branch shape, not pop out as Unknown.
    #[test]
    fn unary_not_on_stdlib_ptr_eq_lowers_to_bool_branch() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(a: *const u8, b: *const u8) -> bool {
                !std::ptr::eq(a, b)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let start = graph.block(graph.startblock);
        assert!(
            start.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, result_ty, .. }
                    if op == "bool" && *result_ty == ValueType::Bool
            )),
            "expected `!std::ptr::eq(...)` to lower through RPython UNARY_NOT"
        );
    }

    /// Multi-segment user path call (e.g. `crate::predicate(...)`,
    /// `pyre_object::is_exception(...)`) — the analyser registers
    /// free functions under their file-local prefix
    /// (`build_semantic_program_from_parsed_files_with_options`
    /// passes `prefix=""` per file at front/ast.rs:751-780), so a
    /// multi-segment crate-relative lookup misses the keyed
    /// `lookup_function_return_type`.  The last-segment fallback in
    /// `expr_unary_not_operand_kind`'s `Expr::Call` arm recovers
    /// the bool return so `!` classifies correctly.
    #[test]
    fn unary_not_on_multi_segment_user_path_classifies_via_last_segment() {
        let parsed = crate::parse::parse_source(
            r#"
            pub fn predicate(x: i64) -> bool { x == 0 }
            fn example(x: i64) -> bool {
                !crate::predicate(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let start = example.graph.block(example.graph.startblock);
        assert!(
            start.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, result_ty, .. }
                    if op == "bool" && *result_ty == ValueType::Bool
            )),
            "expected `!crate::predicate(...)` to lower through RPython UNARY_NOT"
        );
    }

    /// `let local = if cond { call_a() } else { call_b() };
    /// if !local { ... }` — `expression_type_string` unifies the two
    /// arms' tail-expression types and stamps `local`'s recorded
    /// type, so the subsequent `!local` classifies via the bool
    /// `Path` arm rather than falling through.  RPython parity:
    /// `annotator/model.py` `unionof(s_then, s_else)` resolves the
    /// merged binding's annotation; pyre handles the narrow case
    /// where both arms agree on a single primitive type string.
    #[test]
    fn unary_not_on_let_bound_if_else_classifies_via_arm_unification() {
        let parsed = crate::parse::parse_source(
            r#"
            pub fn pick_a(x: i64) -> bool { x == 0 }
            pub fn pick_b(x: i64) -> bool { x != 0 }
            fn example(cond: bool, x: i64) -> bool {
                let local = if cond { pick_a(x) } else { pick_b(x) };
                !local
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!local` (bound from if-else) to lower through RPython UNARY_NOT"
        );
    }

    /// Companion to the if-else case for `match` arms.  Each match
    /// arm returns a known-bool call; `expression_type_string`'s
    /// `Expr::Match` arm unifies them so the let-bound `flag`
    /// records `bool`, classifying the downstream `!flag`.
    #[test]
    fn unary_not_on_let_bound_match_classifies_via_arm_unification() {
        let parsed = crate::parse::parse_source(
            r#"
            pub fn pick_a(x: i64) -> bool { x == 0 }
            pub fn pick_b(x: i64) -> bool { x != 0 }
            pub fn pick_c(x: i64) -> bool { x > 0 }
            fn example(tag: i64, x: i64) -> bool {
                let flag = match tag {
                    0 => pick_a(x),
                    1 => pick_b(x),
                    _ => pick_c(x),
                };
                !flag
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!flag` (bound from match) to lower through RPython UNARY_NOT"
        );
    }

    /// `!driver.is_tracing()` — cross-crate method on a receiver
    /// whose owner type (`WarmState` in
    /// `majit-metainterp/src/warmstate.rs:143`) is outside the
    /// analyser source set.  `lookup_method_return_type` returns
    /// None because `fn_return_types` has no entry for the
    /// cross-crate owner; the method-name shortlist in
    /// `expr_unary_not_operand_kind`'s `Expr::Call` /
    /// `Expr::MethodCall` arm substitutes for the missing
    /// whole-program annotator visibility.  RPython parity:
    /// `bookkeeper.getdesc(receiver).find_method`
    /// (`unaryop.py:206-213`) resolves cross-module methods by
    /// host-identity; pyre's static shortlist mirrors that.
    #[test]
    fn unary_not_on_cross_crate_method_classifies_via_method_shortlist() {
        let parsed = crate::parse::parse_source(
            r#"
            type Driver = i64;
            // No `impl` block — the receiver type's `is_tracing`
            // method lives in a crate not visible to the analyser.
            // The shortlist mirrors RPython's host-stdlib resolution.
            fn example(driver: &Driver) -> bool {
                !driver.is_tracing()
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!driver.is_tracing()` to lower through RPython UNARY_NOT"
        );
    }

    /// `!pyre_object::is_exception(obj)` — cross-crate predicate
    /// path that the analyser source set
    /// (`generated::PYRE_JIT_GRAPH_SOURCES`) does not visit.  The
    /// shortlist in `expr_unary_not_operand_kind`'s `Expr::Call` arm
    /// substitutes for the missing whole-program visibility, the
    /// way RPython's `bookkeeper.getdesc(value)` would resolve the
    /// helper by host-identity.  Pinned to bool so the surface
    /// `!is_exception(...)` lowers through UNARY_NOT.
    #[test]
    fn unary_not_on_pyre_object_predicate_classifies_via_crate_shortlist() {
        let parsed = crate::parse::parse_source(
            r#"
            type Obj = i64;
            fn example(obj: Obj) -> bool {
                !pyre_object::is_exception(obj)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!pyre_object::is_exception(...)` to lower through RPython UNARY_NOT"
        );
    }

    /// `!crate::is_function(obj)` — pyre-interpreter predicate whose
    /// defining module (`function.rs`) is outside
    /// `generated::PYRE_JIT_GRAPH_SOURCES`.  This is the crate-local
    /// version of the cross-crate predicate shortlist above: RPython
    /// resolves the host callable by object identity, while pyre's
    /// source-only classifier needs the explicit bool predicate entry.
    #[test]
    fn unary_not_on_crate_is_function_classifies_via_predicate_shortlist() {
        for predicate in ["crate::is_function", "crate::is_function_with_fixed_code"] {
            let parsed = crate::parse::parse_source(&format!(
                r#"
            type Obj = i64;
            fn example(obj: Obj) -> bool {{
                !{predicate}(obj)
            }}
        "#
            ));
            let program = build_semantic_program(&parsed).expect("source must lower");
            let example = program
                .functions
                .iter()
                .find(|f| f.graph.name == "example")
                .expect("example graph must be present");
            let saw_bool = example
                .graph
                .iter_blocks()
                .flat_map(|b| b.operations.iter())
                .any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::UnaryOp { op, result_ty, .. }
                            if op == "bool" && *result_ty == ValueType::Bool
                    )
                });
            assert!(
                saw_bool,
                "expected `!{predicate}(...)` to lower through RPython UNARY_NOT"
            );
        }
    }

    /// `let v = call_returning_result()?; if !v { ... }` —
    /// `Expr::Try` projects the success arm's `T` out of
    /// `Result<T, E>` so the let-binding records the inner type.
    /// RPython has no peer for `?`; the closest is
    /// `flowcontext.py:194-198` exception-channel join, which would
    /// surface as a separate `SomeBool` annotation on the success
    /// arm via the rtyper's POP_BLOCK / END_FINALLY shape.  Pyre's
    /// surface lowers `?` at the front-end via the analyser's
    /// type-string carrier, mirroring the unwrapping that
    /// `method_as_ref_return_type` does for `Rc`/`Arc`/`Box`.
    #[test]
    fn unary_not_on_let_bound_try_classifies_via_result_inner() {
        let parsed = crate::parse::parse_source(
            r#"
            pub fn maybe_pred(x: i64) -> Result<bool, i64> { Ok(x == 0) }
            fn example(x: i64) -> Result<bool, i64> {
                let flag = maybe_pred(x)?;
                Ok(!flag)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!flag` (let-bound from `expr?`) to lower through RPython UNARY_NOT"
        );
    }

    /// `!frame.is_root` — struct field access whose declared type
    /// is in `ctx.struct_fields`.  Mirror of the Call/MethodCall
    /// shortcut: the classifier projects `Expr::Field` through
    /// `expression_type_string → field_type_string_from_expr` and
    /// classifies the resulting type string.  RPython parity:
    /// `SomeInstance.find_attribute` (`annotator/model.py:430+`)
    /// resolves the field's annotation; pyre uses the static
    /// struct-field registry as the equivalent lookup table.
    #[test]
    fn unary_not_on_struct_field_classifies_via_field_type_registry() {
        let parsed = crate::parse::parse_source(
            r#"
            pub struct Frame { pub is_root: bool, pub depth: i64 }
            fn example(frame: &Frame) -> bool {
                !frame.is_root
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!frame.is_root` to lower through RPython UNARY_NOT"
        );
    }

    /// `!unsafe { f(...) }` — the unsafe block is a transparent
    /// wrapper for the classifier (mirror of the Paren / Group
    /// arms).  The inner call's bool return must propagate so `!`
    /// resolves to UNARY_NOT rather than falling through to the
    /// Unknown arm flagged by `TODO(receiver-typed-dispatch)`.
    #[test]
    fn unary_not_on_unsafe_block_unwraps_inner() {
        let parsed = crate::parse::parse_source(
            r#"
            unsafe fn predicate(x: i64) -> bool { x == 0 }
            fn example(x: i64) -> bool {
                !unsafe { predicate(x) }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let start = example.graph.block(example.graph.startblock);
        assert!(
            start.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, result_ty, .. }
                    if op == "bool" && *result_ty == ValueType::Bool
            )),
            "expected `!unsafe {{ ... }}` to lower through RPython UNARY_NOT"
        );
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
        let program = build_semantic_program(&parsed).expect("source must lower");

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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
        let program = build_semantic_program(&parsed).expect("source must lower");
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
            vec![crate::model::LinkArg::Value(
                entry.operations[0].result.clone().expect("const result"),
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
        let program = build_semantic_program(&parsed).expect("source must lower");
        let func = program
            .functions
            .iter()
            .find(|f| f.graph.name == "returns_unit")
            .expect("returns_unit present");
        let entry = func.graph.block(func.graph.startblock);
        // RPython `flowcontext.py` emits a fresh Variable on the
        // prevblock side for `return None`; the returnblock's own
        // inputarg stays distinct.
        let returnblock_arg = func.graph.block(func.graph.returnblock).inputargs[0].clone();
        // Upstream `flowspace/model.py:171-180` keeps the void return shape
        // in Block.exits: a single Link([fresh_void], graph.returnblock)
        // with exitswitch=None.
        assert!(entry.exitswitch.is_none());
        assert_eq!(entry.exits.len(), 1);
        assert_eq!(entry.exits[0].prevblock, Some(func.graph.startblock));
        assert_eq!(entry.exits[0].target, func.graph.returnblock);
        assert_eq!(entry.exits[0].args.len(), 1);
        assert_ne!(
            entry.exits[0].args[0].as_variable(),
            Some(&returnblock_arg),
            "void return must allocate a fresh prevblock-side Variable (`flowspace/model.py:114`), \
             not reuse the returnblock's own inputarg"
        );
    }

    // ── FrameState — Cat 2.1 cross-block locals threading scaffold ──
    //
    // Slice 1: data-type + pure-function tests only.  The
    // capture/install methods are exercised through Slices 2-6 when
    // wired into the lowering path; here we pin the storage-order
    // contract (Stage A2: first-bind positional, mirroring RPython
    // `co_varnames` slot order) + the rebind detection contract that
    // later slices depend on.

    fn frame_entry(
        _name: &str,
        vid: usize,
        _ty: ValueType,
        graph: &mut crate::model::FunctionGraph,
    ) -> Option<crate::flowspace::model::Variable> {
        // `_name` and `_ty` are retained as positional readability cues
        // for tests — the slot index in `FrameState.entries` (set by
        // the surrounding fixture's vec! position) is the structural
        // identity, and types live on the op that defines `vid` per
        // upstream `Variable.concretetype`.  Materializes a Variable at
        // slot `vid` on `graph` if not already present so subsequent
        // `slot_of(v)` reads round-trip back to `vid`.  Always returns
        // the canonical `value_variables[vid]` placeholder so repeated
        // calls for the same `vid` yield identical `Variable::id()`s —
        // FrameState merge semantics rely on identity equality of
        // entries when both predecessors agree.
        if graph.variable_at(vid).is_none() {
            let v = crate::flowspace::model::Variable::new();
            graph.bind_variable_at(vid, v);
        }
        graph.variable_at(vid).cloned()
    }

    fn entry_slots(
        entries: &[Option<crate::flowspace::model::Variable>],
        graph: &crate::model::FunctionGraph,
    ) -> Vec<Option<usize>> {
        entries
            .iter()
            .map(|e| e.as_ref().and_then(|v| graph.slot_of(v)))
            .collect()
    }

    #[test]
    fn locals_frame_iter_preserves_storage_order() {
        // RPython `flowcontext.py:835 LOAD_FAST` reads `frame.locals_w`
        // by slot index.  Pyre stores entries densely at graph-wide
        // first-bind slot positions (`co_varnames` slot order parity),
        // so `iter()` must walk that order verbatim, yielding only
        // bound (non-None) slots — every predecessor link feeding a
        // merge block sees the same slot order, so `Link.args[i]` lines
        // up with `inputargs[i]` at the successor.
        let mut graph = FunctionGraph::new("iter_slots_demo");
        let frame = FrameState {
            entries: vec![
                frame_entry("c", 3, ValueType::Int, &mut graph),
                frame_entry("a", 1, ValueType::Int, &mut graph),
                frame_entry("b", 2, ValueType::Ref, &mut graph),
            ],
            ..Default::default()
        };
        let walked: Vec<usize> = frame.iter_slots(&graph).map(|(_, slot)| slot).collect();
        assert_eq!(
            walked,
            vec![3, 1, 2],
            "iter must walk entries in storage (first-bind) order; \
             alphabetisation would break slot-position parity at merges"
        );
    }

    #[test]
    fn locals_frame_iter_yields_storage_order() {
        // The `iter` API is what successor blocks consume to allocate
        // matching `inputargs`; slot order must be stable so the i-th
        // `Link.args` entry from a predecessor lines up with the i-th
        // `inputargs` slot at the successor.  Pyre stores entries
        // densely at graph-wide first-bind slot positions, so iteration
        // walks the stored order verbatim, skipping unbound (None)
        // slots.
        let mut graph = FunctionGraph::new("iter_slots_demo");
        let frame = FrameState {
            entries: vec![
                frame_entry("b", 2, ValueType::Ref, &mut graph),
                frame_entry("a", 1, ValueType::Int, &mut graph),
            ],
            ..Default::default()
        };
        // `iter_slots` now yields `(slot_idx, value_slot)` — callers
        // translate `slot_idx` to a name via `local_first_bind_order`.
        // The fixture above sets up entries [b=slot2, a=slot1] at slots
        // [0, 1] so the iter walks the same positions; verifying the
        // (slot, value_slot) pairs is the direct equivalent of the
        // pre-Slice-2.2 (name, vid) check.
        let collected: Vec<(usize, usize)> = frame.iter_slots(&graph).collect();
        assert_eq!(
            collected,
            vec![(0, 2), (1, 1)],
            "iter must walk stored (first-bind positional) order"
        );
    }

    #[test]
    fn frame_state_union_carries_through_when_predecessors_agree() {
        // Both predecessors bound `x` to the same Variable — successor
        // need not allocate a phi.  RPython `framestate.py:108
        // if w1 == w2: return w1`.
        let mut graph = crate::model::FunctionGraph::new("test");
        let a = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        let b = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        graph.set_next_value(100);
        let merged = a.union(&b, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        assert_eq!(entry_slots(&merged.entries, &graph), vec![Some(7)]);
        // No fresh allocation: `next_value` cursor stays at 100.
        assert_eq!(graph.next_value(), 100);
    }

    #[test]
    fn frame_state_union_needs_phi_on_value_id_disagreement() {
        // Predecessors disagree on `x`'s Variable — successor must
        // allocate a fresh phi inputarg.  RPython `framestate.py:
        // 113-114 return Variable()`.
        let mut graph = crate::model::FunctionGraph::new("test");
        let a = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        let b = FrameState {
            entries: vec![frame_entry("x", 8, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        graph.set_next_value(100);
        let merged = a.union(&b, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        assert_eq!(entry_slots(&merged.entries, &graph), vec![Some(100)]);
        // Fresh allocation consumed exactly one vid.
        assert_eq!(graph.next_value(), 101);
    }

    #[test]
    fn frame_state_union_kills_one_sided_slots() {
        // RPython `framestate.py:110-111` None-kill: a slot present in
        // only one predecessor is dropped from the merged state.  Pyre
        // realises this via dense positional entries — graph-wide
        // first-bind order assigns slots [survivor=0, only_a=1,
        // only_b=2]; predecessor A bound only_a but never only_b, B
        // bound only_b but never only_a, so each one-sided slot
        // collapses to None-kill at union.  The merged state preserves
        // slot positions: [Some(survivor), None, None].
        let mut graph = crate::model::FunctionGraph::new("test");
        let a = FrameState {
            entries: vec![
                frame_entry("survivor", 1, ValueType::Int, &mut graph),
                frame_entry("only_a", 2, ValueType::Int, &mut graph),
                None,
            ],
            ..Default::default()
        };
        let b = FrameState {
            entries: vec![
                frame_entry("survivor", 1, ValueType::Int, &mut graph),
                None,
                frame_entry("only_b", 3, ValueType::Int, &mut graph),
            ],
            ..Default::default()
        };
        graph.set_next_value(100);
        let merged = a.union(&b, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        assert_eq!(
            merged.entries.len(),
            3,
            "merged state must preserve positional length = max(len)"
        );
        // Slot 0 is the shared survivor; slots 1 and 2 are the
        // one-sided locals that must collapse via None-kill.  The
        // positional shape is the structural identity (mirrors
        // upstream `framestate.py:locals_w` slot-index convention).
        assert!(
            merged.entries[0].is_some(),
            "shared survivor slot stays bound"
        );
        assert!(
            merged.entries[1].is_none() && merged.entries[2].is_none(),
            "one-sided slots must be killed (None-kill semantics)"
        );
    }

    #[test]
    fn frame_state_union_pads_shorter_side_with_implicit_none() {
        // RPython `framestate.py:14 _union` zips equal-length
        // `locals_w`.  Pyre's graph-wide first-bind order is
        // append-only, so a predecessor whose snapshot was taken before
        // a later name was first bound has a shorter `entries` Vec; the
        // shorter side is treated as `None` for the missing tail slots
        // (None-kill at union).  The merged state's length matches the
        // wider side (`max(len)`).
        let mut graph = crate::model::FunctionGraph::new("test");
        let early = FrameState {
            entries: vec![frame_entry("x", 1, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        let late = FrameState {
            entries: vec![
                frame_entry("x", 1, ValueType::Int, &mut graph),
                frame_entry("y_added_later", 2, ValueType::Int, &mut graph),
            ],
            ..Default::default()
        };
        graph.set_next_value(100);
        let merged = early.union(&late, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        assert_eq!(
            merged.entries.len(),
            2,
            "merged state must extend to wider predecessor's length"
        );
        // Slot 0 is the shared `x`; slot 1 is the late-added local
        // missing on the shorter predecessor's snapshot, collapsing
        // via None-kill on the position-extended union.
        assert!(merged.entries[0].is_some(), "shared slot stays bound");
        assert!(merged.entries[1].is_none(), "padded slot stays as None");
    }

    // Path-Z Slice 2.3 retired the `UnionError::TypeMismatch`
    // variant: per-slot type unification belongs on the annotator side
    // (`Variable.annotation`, `framestate.py:union` is Hlvalue-identity
    // only).  Two prior tests exercising that failure surface
    // (`frame_state_union_concrete_kind_disagreement_returns_type_mismatch`
    // and `frame_state_union_does_not_advance_vid_counter_on_late_type_mismatch`)
    // are removed — the deviation they pinned no longer exists.

    #[test]
    fn frame_state_union_carries_through_same_vid() {
        // RPython `flowspace/framestate.py:108-109 if w1 == w2: return
        // w1`: matching Hlvalue identity carries through unchanged.
        // Type unification is annotator-side post-Path-Z-Slice-2.3, so
        // this fixture only verifies the vid-identity carry-through;
        // any per-arm type metadata flows on `Variable.concretetype`
        // at the rtyper layer.
        let mut graph = crate::model::FunctionGraph::new("test");
        let inferred = FrameState {
            entries: vec![frame_entry("x", 1, ValueType::Unknown, &mut graph)],
            ..Default::default()
        };
        let annotated = FrameState {
            entries: vec![frame_entry("x", 1, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        graph.set_next_value(100);
        let merged = inferred.union(&annotated, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        assert_eq!(entry_slots(&merged.entries, &graph), vec![Some(1)]);
        // Symmetric.
        let merged_swap = annotated.union(&inferred, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        assert_eq!(entry_slots(&merged_swap.entries, &graph), vec![Some(1)]);
    }

    #[test]
    fn frame_state_union_three_arm_fold_carries_through_when_all_agree() {
        // Three-arm iterative fold mirroring upstream `flowspace/
        // flowcontext.py:430-436 mergeblock`'s repeated 2-way union
        // against each arriving candidate.  All three arms bind `x`
        // to the same vid → every fold step carries the vid through
        // (`framestate.py:108 if w1 == w2: return w1`).  No fresh
        // allocation along the chain.
        let mut graph = crate::model::FunctionGraph::new("test");
        let a = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        let b = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        let c = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        graph.set_next_value(100);
        let acc = a.union(&b, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)").union(&c, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        assert_eq!(entry_slots(&acc.entries, &graph), vec![Some(7)]);
        assert_eq!(graph.next_value(), 100, "no fresh allocation expected");
    }

    #[test]
    fn frame_state_union_fold_allocates_fresh_vid_at_disagreement_step() {
        // Iterative fold: a + b agree on vid=7, c brings vid=8 — the
        // SECOND fold step (`acc.union(c)`) is where vids disagree, so
        // the fresh vid is allocated there (mirrors upstream's
        // `framestate.py:113-114 return Variable()` triggered by the
        // generalising `mergeblock` step that introduces a new
        // SpamBlock).
        let mut graph = crate::model::FunctionGraph::new("test");
        let a = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        let b = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        let c = FrameState {
            entries: vec![frame_entry("x", 8, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        graph.set_next_value(100);
        let acc = a.union(&b, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        assert_eq!(
            graph.next_value(),
            100,
            "carry-through step must not allocate"
        );
        let acc = acc.union(&c, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        assert_eq!(entry_slots(&acc.entries, &graph), vec![Some(100)]);
        assert_eq!(
            graph.next_value(),
            101,
            "disagreement step allocates exactly one vid"
        );
    }

    #[test]
    fn frame_state_union_fold_kills_slot_if_any_arm_unbound() {
        // `framestate.py:110-111` None-kill propagates through the
        // iterative fold: once any fold step encounters a side with
        // `None` at slot `i`, the merged slot is None for the rest of
        // the chain.  Here slot 1 (`only_ab`) is bound on a + b but
        // unbound on c — the second fold step collapses it to None.
        let mut graph = crate::model::FunctionGraph::new("test");
        let a = FrameState {
            entries: vec![
                frame_entry("survivor", 1, ValueType::Int, &mut graph),
                frame_entry("only_ab", 2, ValueType::Int, &mut graph),
            ],
            ..Default::default()
        };
        let b = FrameState {
            entries: vec![
                frame_entry("survivor", 1, ValueType::Int, &mut graph),
                frame_entry("only_ab", 2, ValueType::Int, &mut graph),
            ],
            ..Default::default()
        };
        let c = FrameState {
            entries: vec![frame_entry("survivor", 1, ValueType::Int, &mut graph)],
            ..Default::default()
        };
        graph.set_next_value(100);
        let acc = a.union(&b, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)").union(&c, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        assert_eq!(acc.entries.len(), 2);
        assert!(
            acc.entries[0].is_some(),
            "shared survivor slot stays bound across the fold chain"
        );
        assert!(
            acc.entries[1].is_none(),
            "any-None slot collapses to None-kill across the fold chain"
        );
    }

    #[test]
    fn frame_state_union_preserves_graph_wide_slot_order() {
        // Both predecessors share the same graph-wide first-bind order
        // [c=0, a=1, b=2] (the order they were first bound anywhere in
        // the function).  Their dense `entries` vectors line up
        // positionally — slot 0 = c, slot 1 = a, slot 2 = b — so the
        // positional zip merges each slot with its name-mate.  The
        // merged state preserves this slot order.
        let mut graph = crate::model::FunctionGraph::new("test");
        let a = FrameState {
            entries: vec![
                frame_entry("c", 3, ValueType::Int, &mut graph),
                frame_entry("a", 1, ValueType::Int, &mut graph),
                frame_entry("b", 2, ValueType::Int, &mut graph),
            ],
            ..Default::default()
        };
        let b = FrameState {
            entries: vec![
                frame_entry("c", 3, ValueType::Int, &mut graph),
                frame_entry("a", 1, ValueType::Int, &mut graph),
                frame_entry("b", 2, ValueType::Int, &mut graph),
            ],
            ..Default::default()
        };
        graph.set_next_value(100);
        let merged = a.union(&b, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        // The fixture sets up slots [c=vid3, a=vid1, b=vid2] in that
        // graph-wide first-bind order; the union must preserve those
        // positional vids.  Slot 0 = c (vid 3), slot 1 = a (vid 1),
        // slot 2 = b (vid 2) — direct port of upstream's positional
        // `_union(locals_w_self, locals_w_other)` zip.
        assert_eq!(
            entry_slots(&merged.entries, &graph),
            vec![Some(3), Some(1), Some(2)],
            "merged slot order must follow graph-wide first-bind order"
        );
    }

    #[test]
    fn frame_state_getoutputargs_walks_target_in_slot_order() {
        // RPython `framestate.py:92 getoutputargs` walks the target
        // (merged) state's slot order and picks the corresponding
        // self-side Variable at each position.  Pyre's analogue is a
        // direct positional zip — `target.entries[i]` lines up with
        // `self.entries[i]` because both share the graph-wide
        // first-bind slot order.
        let mut graph = crate::model::FunctionGraph::new("test");
        let pred = FrameState {
            entries: vec![
                frame_entry("a", 10, ValueType::Int, &mut graph),
                frame_entry("b", 20, ValueType::Int, &mut graph),
                frame_entry("c", 30, ValueType::Int, &mut graph),
            ],
            ..Default::default()
        };
        let other = FrameState {
            entries: vec![
                frame_entry("a", 10, ValueType::Int, &mut graph),
                frame_entry("b", 99, ValueType::Int, &mut graph),
                frame_entry("c", 30, ValueType::Int, &mut graph),
            ],
            ..Default::default()
        };
        graph.set_next_value(100);
        let merged = pred.union(&other, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        let link_args = pred.getoutputargs(&merged, &graph);
        let link_arg_slots: Vec<usize> = link_args
            .iter()
            .map(|a| {
                a.slot_in(&graph)
                    .expect("locals projection is Variable-only")
            })
            .collect();
        assert_eq!(
            link_arg_slots,
            vec![10, 20, 30],
            "getoutputargs must yield self's slot indices in target slot order"
        );
    }

    #[test]
    fn frame_state_getoutputargs_skips_none_killed_slots_positionally() {
        // After None-kill at union, `merged.entries[i]` is `None` at the
        // killed slot; `getoutputargs` walks positionally and skips
        // those, preserving the source ValueIds at surviving slots
        // without a name lookup.
        let mut graph = crate::model::FunctionGraph::new("test");
        let a = FrameState {
            entries: vec![
                frame_entry("survivor", 11, ValueType::Int, &mut graph),
                frame_entry("only_a", 22, ValueType::Int, &mut graph),
                None,
            ],
            ..Default::default()
        };
        let b = FrameState {
            entries: vec![
                frame_entry("survivor", 11, ValueType::Int, &mut graph),
                None,
                frame_entry("only_b", 33, ValueType::Int, &mut graph),
            ],
            ..Default::default()
        };
        graph.set_next_value(100);
        let merged = a.union(&b, &mut graph).expect("test invariant: AST frontend union is total — entries domain has no UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)");
        let link_args = a.getoutputargs(&merged, &graph);
        let link_arg_slots: Vec<usize> = link_args
            .iter()
            .map(|a| {
                a.slot_in(&graph)
                    .expect("locals projection is Variable-only")
            })
            .collect();
        assert_eq!(
            link_arg_slots,
            vec![11],
            "only surviving slots emit link args; None-killed slots are skipped"
        );
    }

    #[test]
    fn loop_body_locals_excludes_closure_captures() {
        // Slice 5b.2 contract: the static pre-scan reaches every
        // straight-line statement in the loop body (including nested
        // blocks, `if` / `match` / nested loops, and `unsafe` blocks)
        // but does NOT descend into `Expr::Closure` bodies.  A name
        // referenced only inside a closure must not appear in either
        // `read_names` or `rebound_names`, because the closure's
        // capture is not part of the outer loop's straight-line
        // control flow that drives header phi allocation.
        let body: syn::Block = syn::parse_quote! {{
            let local_let = 0;
            outer_assign = 1;
            outer_compound += 2;
            let _read_via_path = outer_read;
            let _capture = || {
                closure_only_read;
                closure_only_assigned = 7;
            };
        }};
        let result = loop_body_locals(&body);

        // `let local_let = 0` and `let _read_via_path = outer_read`
        // both rebind their pattern names; `_capture` is also a
        // pattern name on a closure-bound `let`.
        assert!(result.rebound_names.contains("local_let"));
        assert!(result.rebound_names.contains("_read_via_path"));
        assert!(result.rebound_names.contains("_capture"));
        // Simple `outer_assign = 1` rebinds `outer_assign`.
        assert!(result.rebound_names.contains("outer_assign"));
        // Compound `outer_compound += 2` is BOTH read and rebound.
        assert!(result.read_names.contains("outer_compound"));
        assert!(result.rebound_names.contains("outer_compound"));
        // `outer_read` appears as the RHS of a `let` init — read.
        assert!(result.read_names.contains("outer_read"));

        // Closure body must be fully invisible to the pre-scan.
        assert!(
            !result.read_names.contains("closure_only_read"),
            "closure body reads must be excluded from outer loop pre-scan"
        );
        assert!(
            !result.rebound_names.contains("closure_only_assigned"),
            "closure body rebinds must be excluded from outer loop pre-scan"
        );
    }

    #[test]
    fn allocate_loop_header_phis_eager_install_and_pre_loop_link_args() {
        // Slice 5b.3 contract: the eager allocator must (a) emit one
        // `OpKind::Input` per surviving name at `header_entry` and
        // append its phi vid to `header_entry.inputargs`, (b) push the
        // pre-loop vid onto `pre_loop_block.exits[0].args` so the
        // forward-edge `Link.args` arity matches the new header arity,
        // and (c) rewire `ctx.local_value_ids[name]` to point at the
        // freshly-allocated phi vid (with `header_entry` as the new
        // defining block) so subsequent body lowering reads the phi.
        //
        // The walk filters by intersecting the must-merge set with
        // `pre_loop_snapshot.entries`: a pre-loop name not referenced
        // in the body is skipped (no header phi), and a body-only
        // name (referenced but absent from the snapshot) is killed
        // per RPython `framestate.py:110-111` None-kill semantics.
        let mut graph = FunctionGraph::new("loop_header_phi_demo");
        let pre_loop_block = graph.startblock;
        let header_entry = graph.create_block();

        // Seed pre-loop bindings for `x` and `y` by emitting
        // `OpKind::Input` at the start block; this approximates the
        // pre-loop state Slice 5c.1 will hand to the allocator.
        let pre_x = graph
            .push_op_var(
                pre_loop_block,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .expect("OpKind::Input must produce a Variable");
        let pre_y = graph
            .push_op_var(
                pre_loop_block,
                OpKind::Input {
                    name: "y".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .expect("OpKind::Input must produce a Variable");
        // Close pre-loop block with the empty-args goto Slice 5c.1
        // installs before calling the allocator.
        graph.set_goto(pre_loop_block, header_entry, vec![]);

        let empty_registry = StructFieldRegistry::default();
        let empty_fn_ret = HashMap::new();
        let empty_names = std::collections::HashSet::new();
        let empty_trait_names = std::collections::HashSet::new();
        let mut ctx = GraphBuildContext::new(
            &empty_registry,
            &empty_fn_ret,
            "",
            HashMap::new(),
            &empty_names,
            &empty_trait_names,
        );
        ctx.bind_local_id_var("x".into(), &pre_x, &graph, pre_loop_block);
        ctx.local_value_types.insert("x".into(), ValueType::Int);
        ctx.bind_local_id_var("y".into(), &pre_y, &graph, pre_loop_block);
        ctx.local_value_types.insert("y".into(), ValueType::Int);

        // `pre_loop_snapshot` is produced by ctx in the real lowering;
        // mirror that here so the allocator walks the same first-bind
        // positional order Slice 5c.1 will feed it.
        let pre_loop_snapshot = ctx.getstate(&graph, 0);
        assert_eq!(pre_loop_snapshot.entries.len(), 2);

        // `x` is read inside the body, `z` is body-only (rebound
        // without a pre-loop counterpart).  `y` is in the pre-loop
        // snapshot but never referenced in the body.
        let must_merge = LoopBodyLocals {
            read_names: ["x".to_string()].into_iter().collect(),
            rebound_names: ["z".to_string()].into_iter().collect(),
        };

        let header_phi_names = allocate_loop_header_phis(
            &mut graph,
            &mut ctx,
            pre_loop_block,
            header_entry,
            &pre_loop_snapshot,
            &must_merge,
        );

        // Only `x` survives — `y` filtered out (not referenced), `z`
        // filtered out (None-killed: not in pre-loop snapshot).
        assert_eq!(header_phi_names, vec!["x".to_string()]);

        let header_inputarg_vars = graph.block(header_entry).inputargs.clone();
        assert_eq!(header_inputarg_vars.len(), 1);
        let phi_var = header_inputarg_vars[0].clone();
        let header = graph.block(header_entry);
        assert_eq!(header.operations.len(), 1);
        let phi_op = &header.operations[0];
        match &phi_op.kind {
            OpKind::Input { name, ty } => {
                assert_eq!(name, "x");
                assert_eq!(*ty, ValueType::Int);
            }
            other => panic!("expected OpKind::Input, got {:?}", other),
        }
        assert_eq!(phi_op.result.as_ref(), Some(&phi_var));

        let pre_exit = &graph.block(pre_loop_block).exits[0];
        assert_eq!(
            pre_exit.args,
            vec![LinkArg::Value(pre_x.clone())],
            "forward-edge link arg for `x` must carry the pre-loop Variable"
        );

        let (current_x_var, current_x_block) = ctx.local_value_ids["x"].clone();
        let phi_slot = graph.slot_of(&phi_var).expect("phi inputarg registered");
        let current_x_slot = graph
            .slot_of(&current_x_var)
            .expect("ctx.local_value_ids[x] backing Variable must be registered");
        assert_eq!(
            current_x_slot, phi_slot,
            "ctx.local_value_ids[x] must point at the header phi"
        );
        assert_eq!(
            current_x_block, header_entry,
            "ctx.local_value_ids[x].defining_block must be the header"
        );

        // `y` was not referenced, so its ctx binding still points at
        // the pre-loop slot; no header phi was allocated for it.
        let (current_y_var, _) = ctx.local_value_ids["y"].clone();
        let pre_y_slot = graph.slot_of(&pre_y).expect("pre_y registered");
        let current_y_slot = graph
            .slot_of(&current_y_var)
            .expect("ctx.local_value_ids[y] backing Variable must be registered");
        assert_eq!(current_y_slot, pre_y_slot);
    }

    /// Slice 5d nested-pattern coverage #1: nested while loops where
    /// the inner loop's `continue` re-enters the inner header.  The
    /// outer header's phi for the outer counter must remain
    /// independent of the inner control flow — every link landing on
    /// the outer header must carry exactly the names the outer
    /// header's `LoopFrame::header_phi_names` recorded, and every
    /// link landing on the inner header must carry the inner's.
    #[test]
    fn nested_loops_per_header_phi_arity_consistent() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example() -> i64 {
                let mut i: i64 = 0;
                let mut j: i64 = 0;
                while i < 10 {
                    j = 0;
                    while j < 5 {
                        j = j + 1;
                        if j == 3 { continue; }
                    }
                    i = i + 1;
                }
                i + j
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // Every block whose `inputargs` is non-empty corresponds to a
        // loop header.  For each such header, every predecessor edge
        // must carry exactly `inputargs.len()` link args.
        for header in &graph.blocks {
            if header.inputargs.is_empty() {
                continue;
            }
            let arity = header.inputargs.len();
            for pred in &graph.blocks {
                for exit in &pred.exits {
                    if exit.target == header.id {
                        assert_eq!(
                            exit.args.len(),
                            arity,
                            "predecessor {:?}→header {:?}: link.args len {} ≠ \
                             header.inputargs len {}",
                            pred.id,
                            header.id,
                            exit.args.len(),
                            arity,
                        );
                    }
                }
            }
        }
    }

    /// A name rebound in the loop body but never read (neither in the
    /// body nor post-loop) gets a transient loop-header phi during
    /// lowering, then `prune_dead_phis` removes that phi and its matching
    /// link-arg before downstream passes see the graph.  This mirrors
    /// `flowcontext.py:430 mergeblock` + `simplify.transform_dead_op_vars`.
    #[test]
    fn rebound_only_no_post_read_skips_header_phi() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(n: i64) -> i64 {
                let mut x: i64 = 0;
                let mut count: i64 = 0;
                while count < n {
                    count = count + 1;
                    x = 5;
                }
                count
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // The single loop header is the block with non-empty inputargs
        // that is not the start / return / except.
        let header = graph
            .blocks
            .iter()
            .find(|b| {
                !b.inputargs.is_empty()
                    && b.id != graph.startblock
                    && b.id != graph.returnblock
                    && b.id != graph.exceptblock
            })
            .expect("loop header must exist");
        let header_phi_names: Vec<&str> = header
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::Input { name, .. }
                    if header.inputargs.contains(op.result.as_ref().unwrap()) =>
                {
                    Some(name.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(
            header_phi_names.contains(&"count"),
            "`count` is read in the body, so its header phi must exist; got {:?}",
            header_phi_names,
        );
        assert!(
            !header_phi_names.contains(&"x"),
            "`x` is rebound-only-no-read; prune_dead_phis must remove the \
             transient header phi before the final graph. Got {:?}",
            header_phi_names,
        );
    }

    /// A name rebound in the body that is read post-loop (but not
    /// inside the body) must still get a header phi.
    /// `flowcontext.py:430 mergeblock`'s `union` of pre-loop and
    /// body-tail framestates reports `NeedsPhi` (different vids),
    /// creating the phi at union time.  Pyre's read∪rebound
    /// allocator emits the phi at the header eagerly; the post-loop
    /// reader of `x` resolves to the eager phi vid.
    #[test]
    fn rebound_only_with_post_loop_read_lazy_installs_header_phi() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(n: i64) -> i64 {
                let mut x: i64 = 0;
                let mut count: i64 = 0;
                while count < n {
                    count = count + 1;
                    x = 5;
                }
                x + count
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        let header = graph
            .blocks
            .iter()
            .find(|b| {
                !b.inputargs.is_empty()
                    && b.id != graph.startblock
                    && b.id != graph.returnblock
                    && b.id != graph.exceptblock
            })
            .expect("loop header must exist");
        let header_phi_names: Vec<&str> = header
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::Input { name, .. }
                    if header.inputargs.contains(op.result.as_ref().unwrap()) =>
                {
                    Some(name.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(
            header_phi_names.contains(&"count"),
            "`count` is read in the body; got {:?}",
            header_phi_names,
        );
        assert!(
            header_phi_names.contains(&"x"),
            "`x` is rebound in the body and read post-loop; the eager \
             allocator must emit a header phi for it. Got {:?}",
            header_phi_names,
        );
    }

    /// Nested while loops where the inner loop reads outer-loop
    /// locals.  `loop_body_locals`'s recursive AST visit propagates
    /// the inner loop's reads to the outer body's `read_names`, so
    /// both loops allocate header phis for shared locals
    /// consistently.  Positive-path probe — the existing
    /// `nested_loops_per_header_phi_arity_consistent` covers the
    /// arity invariant; this one pins the
    /// `read_names ∪ rebound_names` allocation scope.
    #[test]
    fn nested_while_inner_reads_outer_local_threads_header_phi() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(n: i64) -> i64 {
                let mut outer: i64 = 0;
                let mut inner: i64 = 0;
                while outer < n {
                    inner = 0;
                    while inner < outer {
                        inner = inner + 1;
                    }
                    outer = outer + 1;
                }
                outer + inner
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // Two non-special blocks should carry inputargs (the two loop
        // headers).  Both must include `inner` and `outer` per
        // `read_names` (inner loop's `inner < outer` reads outer; outer
        // loop's `while outer < n` reads outer; both reads propagate).
        // A "loop header" is any block whose inputargs include a phi
        // for `outer` AND has a back-edge predecessor (a predecessor
        // whose block id is greater — back-edges close to lower-id
        // blocks).  At least two such blocks must exist (outer and
        // inner loop headers).
        let mut loop_headers_with_outer = 0;
        for header in &graph.blocks {
            let has_outer_phi = header.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::Input { name, .. } if name == "outer")
                    && op
                        .result
                        .as_ref()
                        .map(|r| header.inputargs.contains(r))
                        .unwrap_or(false)
            });
            if !has_outer_phi {
                continue;
            }
            let has_back_edge = graph.blocks.iter().any(|pred| {
                pred.id.0 > header.id.0 && pred.exits.iter().any(|e| e.target == header.id)
            });
            if has_back_edge {
                loop_headers_with_outer += 1;
            }
        }
        assert!(
            loop_headers_with_outer >= 2,
            "expected at least two loop headers carrying `outer` phi; got {}",
            loop_headers_with_outer,
        );
    }

    /// Cat 2-2 Phase A.2 probe: closure inside the loop body must NOT
    /// influence header phi allocation.  `visit_expr_for_loop_locals`
    /// explicitly skips `Expr::Closure`; a name read only inside a
    /// closure body should not appear in `read_names` and therefore
    /// should not get a header phi.  RPython parity: a closure
    /// compiles to its own bytecode with `LOAD_DEREF` / `STORE_DEREF`,
    /// distinct from the outer `LOAD_FAST` / `STORE_FAST` sequence, so
    /// captures do not flow through the loop's straight-line control
    /// path.
    ///
    /// Positive-path version: reads of a captured-only name inside a
    /// closure body do NOT count as read on the loop's header.  We
    /// pin the negative — closure-only-captured name is absent from
    /// `read_names` and therefore not a phi candidate.
    #[test]
    fn closure_captured_name_excluded_from_loop_header_phi() {
        // `loop_body_locals` is the unit under test; build a body
        // directly without going through `build_semantic_program` so
        // the probe stays orthogonal to the rest of the pipeline.
        let body: syn::Block = syn::parse_quote! {
            {
                let _f = || {
                    captured_only
                };
                straight_local = 1;
            }
        };
        let result = loop_body_locals(&body);
        assert!(
            !result.read_names.contains("captured_only"),
            "`captured_only` read only inside the closure body must not \
             appear in read_names; got {:?}",
            result.read_names,
        );
        assert!(
            result.rebound_names.contains("straight_local"),
            "`straight_local` rebound on the body's straight path must \
             appear in rebound_names; got {:?}",
            result.rebound_names,
        );
    }

    /// Cat 2-2 Phase A.2 probe: Rust lexical shadowing — the audit's
    /// primary Phase B motivator.  `let x: i64 = count + 100` inside
    /// a loop body shadows the pre-loop `let x: i64 = 7` for the
    /// duration of the body's scope, but pyre's `loop_body_locals`
    /// AST scan collapses both into the single name `"x"`.  Under
    /// the current shape, the loop header allocates a phi for `x`
    /// (because the body rebinds + reads `x`), and the back-edge
    /// close threads the inner `x`'s vid through the same slot as
    /// the outer pre-loop binding.  The post-loop read of outer `x`
    /// then resolves to the merged loop-header phi instead of the
    /// pre-loop `ConstInt(7)`.
    ///
    /// Marked `#[ignore]` because the current shape DOES NOT match
    /// Rust semantics: the test documents the gap until Phase B
    /// (back-edge worklist + scope-aware tracking) closes it.
    /// Removing `#[ignore]` is the regression-anchor handshake when
    /// Phase B lands.
    ///
    /// The Phase B post-fix expectation: NO loop-header phi for `x`
    /// exists in the final graph, because the inner `let x` lives in
    /// its own lexical scope that ends at the body block boundary
    /// and therefore has no back-edge.
    #[test]
    #[ignore = "Cat 2-2 Phase B: lexical shadowing not yet supported \
                — current AST scan collapses outer/inner `x` into one \
                slot, so the loop header allocates a phi for `x`."]
    fn lexical_shadowing_in_loop_body_does_not_emit_x_header_phi() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(n: i64) -> i64 {
                let x: i64 = 7;
                let mut count: i64 = 0;
                while count < n {
                    let x: i64 = count + 100;
                    count = count + x;
                }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // Phase B post-fix expectation: no block carries a phi named
        // `x`.  Today the loop header carries one because the AST
        // scan collapses both `let x` bindings into the same name.
        let any_x_phi = graph.blocks.iter().any(|b| {
            b.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::Input { name, .. } if name == "x")
                    && op
                        .result
                        .as_ref()
                        .map(|r| b.inputargs.contains(r))
                        .unwrap_or(false)
            })
        });
        assert!(
            !any_x_phi,
            "Phase B post-fix: inner `let x` is a lexically-scoped \
             shadow that must not produce a loop-header phi for `x`; \
             currently the AST scan collapses outer/inner into one slot.",
        );
    }

    /// Slice 5d nested-pattern coverage #2: `break` from a deeply
    /// nested `if` arm whose framestate has been mutated by a sibling
    /// arm's `LocalBindingSnapshot::restore`.  The lowering must close
    /// the break source's block with a goto to the loop's exit and the
    /// resulting graph must build without panic — `break` to a
    /// loop-exit block does not need header-phi args (exit's
    /// inputargs are determined by post-loop reads, threaded by the
    /// lazy cross-block installer).
    #[test]
    fn break_from_nested_if_arm_lowers_without_panic() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example() -> i64 {
                let mut x: i64 = 0;
                while x < 100 {
                    if x > 50 {
                        if x % 2 == 0 {
                            break;
                        } else {
                            x = x + 1;
                        }
                    } else {
                        x = x + 2;
                    }
                }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // Sanity: header-arity invariant from #1 must still hold even
        // with a nested break.
        for header in &graph.blocks {
            if header.inputargs.is_empty() {
                continue;
            }
            let arity = header.inputargs.len();
            for pred in &graph.blocks {
                for exit in &pred.exits {
                    if exit.target == header.id {
                        assert_eq!(exit.args.len(), arity);
                    }
                }
            }
        }
    }

    /// Slice 5d nested-pattern coverage #3 (PyPy-parity revision):
    /// a `loop` with NO `break` whose body contains only side-effect-
    /// free ops.  After `prune_dead_phis` (RPython
    /// `transform_dead_op_vars`, `simplify.py:422-524`) DCEs the
    /// unobservable body, the loop header becomes an empty block whose
    /// only exit is its own back-edge.  `eliminate_empty_blocks`
    /// (`simplify.py:64`) detects this as `"the graph contains an
    /// empty infinite loop"` and asserts — upstream marks such a graph
    /// as a broken program (no observable termination, no observable
    /// effects).  Pyre matches the assert.
    #[test]
    #[should_panic(expected = "the graph contains an empty infinite loop")]
    fn loop_with_no_break_panics_on_empty_infinite_loop_per_simplify_assert() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example() -> i64 {
                let mut x: i64 = 0;
                loop { x = x + 1; }
            }
        "#,
        );
        let _program = build_semantic_program(&parsed).expect("source must lower");
    }

    /// Slice 5d nested-pattern coverage #4: a `loop` where the only
    /// exit is `break` — body_tail is closed by the break (or the
    /// `is_open` check skips an empty back-edge).  Exit is reachable
    /// post-loop.  Smoke test for the back-edge `is_open` guard at
    /// `Expr::Loop`'s tail close in `front/ast.rs::lower_expr`.
    #[test]
    fn loop_with_only_break_lowers_without_panic() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example() -> i64 {
                let mut x: i64 = 0;
                loop {
                    x = x + 1;
                    if x > 10 { break; }
                }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // Header-arity invariant must still hold.
        for header in &graph.blocks {
            if header.inputargs.is_empty() {
                continue;
            }
            let arity = header.inputargs.len();
            for pred in &graph.blocks {
                for exit in &pred.exits {
                    if exit.target == header.id {
                        assert_eq!(exit.args.len(), arity);
                    }
                }
            }
        }
    }

    /// Audit Cat 2-1 probe: a `while` loop whose body contains
    /// `continue`, with a pre-loop local read after the loop that the
    /// body never touches.  Lazy installer must thread the post-loop
    /// read all the way back to the pre-loop binding via the loop
    /// header — and through each header predecessor (pre-loop edge,
    /// body-tail back-edge, AND continue back-edge).  RPython parity:
    /// `flowspace/flowcontext.py:399-465 mergeblock` requires every
    /// closing predecessor of the merge target to carry a FrameState
    /// so `getoutputargs` can resolve the target's inputargs slot-by-
    /// slot.  Pre-fix, body_tail and continue source both have
    /// `framestate = None` (intentional cycle breaker), so the
    /// installer's `framestate.as_ref()?` short-circuit aborts at
    /// the header's predecessor walk and the post-loop read drops to
    /// the naked-`Input` fallback (disconnected from pre-loop's vid).
    #[test]
    fn while_with_continue_threads_post_loop_read_through_header_phi() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(n: i64) -> i64 {
                let y: i64 = 7;
                let mut count: i64 = 0;
                while count < n {
                    if count % 2 == 0 {
                        count = count + 1;
                        continue;
                    }
                    count = count + 1;
                }
                y
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|func| func.name == "example")
            .expect("example graph")
            .graph;
        // Find the return block: ends with `Return` and reads `y`.
        let returnblock_id = graph.returnblock;
        // The closing predecessor of the returnblock is the post-loop
        // exit block that produces the return value (`y`).  In the
        // canonical Slice-2 lazy-install path, the post-loop block
        // owns `y` as an inputarg threaded back through the loop
        // header phi via every header predecessor (pre-loop, body-
        // tail, continue source).  In the pre-fix path, the
        // installer short-circuits at the header predecessor walk
        // because body_tail and continue source both have
        // `framestate = None`, and the post-loop block emits a naked
        // `OpKind::Input { name: "y" }` op as a synthetic disconnected
        // Variable.
        //
        // Probe: walk the returnblock's predecessor link.  Its single
        // arg is the post-loop block's `y` value.  Verify that vid is
        // an inputarg of the post-loop block (i.e. lazy install
        // succeeded).  Pre-fix, the vid would be the result of a
        // standalone `OpKind::Input` op NOT in inputargs.
        let pred_link: Option<(BlockId, crate::flowspace::model::Variable)> =
            graph.blocks.iter().find_map(|b| {
                b.exits.iter().find_map(|exit| {
                    if exit.target == returnblock_id {
                        let arg_var = exit.args[0].as_variable()?;
                        Some((b.id, arg_var.clone()))
                    } else {
                        None
                    }
                })
            });
        let (post_loop_id, ret_value_var) =
            pred_link.expect("returnblock must have one closing predecessor");
        let post_loop_block = graph.block(post_loop_id);
        let is_inputarg = post_loop_block.inputargs.contains(&ret_value_var);
        assert!(
            is_inputarg,
            "post-loop block must own `y` as an inputarg threaded back to \
             pre-loop; got naked-`Input` fallback (var {:?} not in \
             inputargs {:?}). graph:\n{}",
            ret_value_var,
            post_loop_block.inputargs,
            graph.dump()
        );
        // Audit Cat 2-1 also stamps continue + body_tail framestate
        // on the SOURCE blocks of every back-edge link into the loop
        // header.  Confirm directly that every header predecessor
        // has a stamped framestate; pre-fix at least one of them
        // (the continue source AND the body_tail) would be `None`,
        // and the residual fallback at the post-loop block was the
        // observable consequence.
        let header_id = graph
            .blocks
            .iter()
            .find_map(|b| {
                let owns_count_phi = b.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::Input { name, .. } if name == "count")
                        && op.result.as_ref().is_some_and(|r| b.inputargs.contains(r))
                });
                let pred_count = graph
                    .blocks
                    .iter()
                    .flat_map(|p| p.exits.iter().filter(|e| e.target == b.id).map(|_| ()))
                    .count();
                (owns_count_phi && pred_count >= 2).then_some(b.id)
            })
            .expect("loop header with `count` phi + ≥2 preds");
        let header_pred_ids: Vec<BlockId> = graph
            .blocks
            .iter()
            .filter_map(|b| {
                b.exits
                    .iter()
                    .any(|e| e.target == header_id)
                    .then_some(b.id)
            })
            .collect();
        for pred_id in &header_pred_ids {
            let pred = graph.block(*pred_id);
            assert!(
                pred.framestate.is_some(),
                "header predecessor block {:?} must have framestate \
                 stamped (continue + body_tail Cat 2-1 stamps); \
                 graph:\n{}",
                pred_id,
                graph.dump()
            );
        }
    }

    /// `rlib/jit.py:184-201 elidable_promote.decorator(func)` produces
    /// **two** function objects after module import: the closure-
    /// captured `func` (which carries `_elidable_function_`) and the
    /// `exec`-built `result` (which does not).  Pyre's parser-level
    /// synthesizer is the line-by-line mirror: a single
    /// `#[elidable_promote] fn foo(...)` source item must produce two
    /// `SemanticFunction`s — `_orig_foo_unlikely_name` carrying the
    /// `elidable` hint, and the user-facing wrapper `foo` whose body
    /// is `hint_promote_or_string(arg); …; _orig_foo_unlikely_name(args)`.
    /// `rlib/jit.py:191` evaluates `args[int(i)]` in
    /// `elidable_promote.decorator`, which raises `IndexError` when
    /// the literal points past the function's argument list.  Pyre
    /// mirrors the fail-loud behaviour through
    /// `synthesize_elidable_promote_pair`'s index bounds check.
    #[test]
    #[should_panic(expected = "out of range")]
    fn elidable_promote_panics_on_out_of_range_promote_index() {
        let parsed = crate::parse::parse_source(
            r#"
            #[elidable_promote(promote_args = "0,5")]
            pub fn foo(x: i64) -> i64 { x }
        "#,
        );
        let _ = build_semantic_program(&parsed);
    }

    /// `rlib/jit.py:189-191 promote_args.split(",")` propagates
    /// `ValueError` on a non-integer piece via `int(i)`.  Pyre matches
    /// that by panicking on a `usize::parse` failure rather than
    /// silently dropping the malformed literal.
    #[test]
    #[should_panic(expected = "elidable_promote")]
    fn elidable_promote_panics_on_malformed_promote_args() {
        let parsed = crate::parse::parse_source(
            r#"
            #[elidable_promote(promote_args = "0,not_a_number")]
            pub fn foo(x: i64) -> i64 { x }
        "#,
        );
        let _ = build_semantic_program(&parsed);
    }

    /// `rlib/jit.py:184-201 elidable_promote.decorator(func)` applies
    /// uniformly to any callable — module-level fn, instance method,
    /// classmethod.  Pyre's `synthesize_or_passthrough` must produce
    /// the (`_orig_<NAME>_unlikely_name`, wrapper) pair for impl
    /// methods too, not just `Item::Fn`s, otherwise method callers
    /// silently drop the elidable flag.
    #[test]
    fn elidable_promote_on_inherent_method_synthesizes_pair() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            impl S {
                #[elidable_promote(promote_args = "all")]
                pub fn double(&self, n: i64) -> i64 { n + n }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        // Two SemanticFunctions emitted from the decorated method —
        // the synthesized orig and the wrapper.
        let names: Vec<_> = program
            .functions
            .iter()
            .map(|sf| sf.name.as_str())
            .collect();
        assert!(
            names.contains(&"_orig_double_unlikely_name"),
            "expected synthesized orig in {names:?}"
        );
        assert!(
            names.contains(&"double"),
            "expected wrapper retaining user-facing name in {names:?}"
        );
        // jit.py:185 — elidable flag lives on the orig only.
        let orig = program
            .functions
            .iter()
            .find(|sf| sf.name == "_orig_double_unlikely_name")
            .expect("orig graph");
        assert!(orig.hints.iter().any(|h| h == "elidable"));
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "double")
            .expect("wrapper graph");
        assert!(!wrapper.hints.iter().any(|h| h == "elidable"));
    }

    /// `lib.rs:531-537` registers inherent-method graphs under
    /// `CallPath::for_impl_method(impl_type, name)` which produces
    /// `[<impl_type segments...>, name]`.  The synthesized wrapper's
    /// tail call must therefore be a type-qualified path so the IR
    /// `Call`-target segments match the registered callee.  A bare
    /// `_orig_<name>_unlikely_name(self, args)` would resolve to a
    /// non-existent free function and silently drop the elidable
    /// binding bound to the orig graph.
    #[test]
    fn elidable_promote_wrapper_tail_call_is_type_qualified_for_inherent_method() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            impl S {
                #[elidable_promote(promote_args = "all")]
                pub fn double(&self, n: i64) -> i64 { n + n }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "double")
            .expect("wrapper graph");
        let ops = &wrapper.graph.block(wrapper.graph.startblock).operations;
        let target = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::Call { target, .. }
                    if target.path_segments().and_then(|s| s.last().copied())
                        == Some("_orig_double_unlikely_name") =>
                {
                    Some(target.clone())
                }
                _ => None,
            })
            .expect("wrapper must tail-call _orig_double_unlikely_name");
        let segments = target
            .path_segments()
            .expect("function-path tail call must carry path segments");
        assert_eq!(
            segments,
            vec!["S", "_orig_double_unlikely_name"],
            "wrapper tail call must be type-qualified \
             [\"S\", \"_orig_double_unlikely_name\"] to match \
             CallPath::for_impl_method; got {segments:?}"
        );
    }

    #[test]
    fn elidable_promote_synthesizes_orig_and_wrapper_graphs() {
        let parsed = crate::parse::parse_source(
            r#"
            #[elidable_promote(promote_args = "all")]
            pub fn foo(x: i64, y: i64) -> i64 {
                x + y
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        assert_eq!(
            program.functions.len(),
            2,
            "expected exactly two graphs after synthesis, got {:?}",
            program
                .functions
                .iter()
                .map(|sf| sf.name.as_str())
                .collect::<Vec<_>>()
        );

        // jit.py:185 elidable(func) — the renamed original carries the
        // `elidable` hint that downstream `mark_elidable` reads.
        let orig = program
            .functions
            .iter()
            .find(|sf| sf.name == "_orig_foo_unlikely_name")
            .expect("orig graph");
        assert!(
            orig.hints.iter().any(|h| h == "elidable"),
            "orig hints must include 'elidable', got {:?}",
            orig.hints
        );

        // jit.py:198-201 — wrapper `result` has no `_elidable_function_`.
        // After Slice C retired the `collect_jit_hints` fallback the
        // wrapper's hints are empty unless the user explicitly added
        // another decorator alongside `#[elidable_promote]`.
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "foo")
            .expect("wrapper graph");
        assert!(
            !wrapper.hints.iter().any(|h| h == "elidable"),
            "wrapper must not carry the 'elidable' hint — RPython places \
             `_elidable_function_` only on the orig (jit.py:185), not \
             on the wrapper (jit.py:198-201).  Got hints: {:?}",
            wrapper.hints
        );
        let ops = &wrapper.graph.block(wrapper.graph.startblock).operations;
        // jit.py:192-194 — one `hint(arg, promote=True, promote_string=
        // True)` per selected arg, which pyre's synthesiser emits as
        // `hint_promote_or_string(arg)`.  With promote_args="all" and
        // two args, exactly two of these calls.
        let hint_count = ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target, .. } if target.path_segments().and_then(|s| s.last().copied()) == Some("hint_promote_or_string")
                )
            })
            .count();
        assert_eq!(
            hint_count, 2,
            "wrapper must emit hint_promote_or_string per arg; ops:\n{ops:#?}"
        );
        // jit.py:195 — tail call to the renamed original.
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target.path_segments().and_then(|s| s.last().copied()) == Some("_orig_foo_unlikely_name")
            )),
            "wrapper must tail-call _orig_foo_unlikely_name; ops:\n{ops:#?}"
        );
    }

    /// `synthesize_elidable_promote_pair` always emits the dual hint
    /// `hint_promote_or_string`, mirroring RPython jit.py:191-194
    /// (`hint(arg, promote=True, promote_string=True)`).  The str /
    /// unicode / plain dispatch happens at jtransform time
    /// (`jit_codewriter/jtransform.py:599-606`), not at synthesis
    /// time.  This test guards against regressing to a synth-time
    /// classifier that pre-commits the str vs unicode choice.
    #[test]
    fn elidable_promote_routes_str_arg_to_hint_promote_or_string() {
        let parsed = crate::parse::parse_source(
            r#"
            #[elidable_promote(promote_args = "all")]
            pub fn lookup(s: &str) -> i64 { s.len() as i64 }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "lookup")
            .expect("wrapper graph");
        let ops = &wrapper.graph.block(wrapper.graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target.path_segments().and_then(|s| s.last().copied()) == Some("hint_promote_or_string")
            )),
            "wrapper must emit hint_promote_or_string for &str arg; ops:\n{ops:#?}"
        );
    }

    /// Byte-string-like args (`&[u8]`, `Vec<u8>`, `Box<[u8]>`) also
    /// get the dual hint `hint_promote_or_string`.  Pyre's
    /// `PromoteOrString` rewrite arm falls through to plain
    /// `ref_guard_value` (`jit_codewriter/jtransform.py:603-606 else
    /// branch + :608-614`) because pyre lacks a `Ptr(rstr.STR)` GC
    /// layout to satisfy the `if op.args[0].concretetype ==
    /// lltype.Ptr(rstr.STR)` test at `jtransform.py:601`.
    #[test]
    fn elidable_promote_routes_u8_slice_to_hint_promote_or_string() {
        let parsed = crate::parse::parse_source(
            r#"
            #[elidable_promote(promote_args = "all")]
            pub fn count_bytes(b: &[u8]) -> i64 { b.len() as i64 }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "count_bytes")
            .expect("wrapper graph");
        let ops = &wrapper.graph.block(wrapper.graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target.path_segments().and_then(|s| s.last().copied()) == Some("hint_promote_or_string")
            )),
            "wrapper must emit hint_promote_or_string for &[u8] arg; ops:\n{ops:#?}"
        );
    }

    /// Opaque pointers (`PyObjectRef`, `*const ()` aliases, generic
    /// `Ref`-classed types) fall through to `hint_promote_or_string`,
    /// the rewrite-time-dispatched dual hint whose `PromoteOrString`
    /// arm defaults to plain promote.  This matches today's
    /// `_get_immutable_code(func: PyObjectRef)` site at
    /// `pyre-interpreter::function.rs:344`.
    #[test]
    fn elidable_promote_routes_pyobject_ref_to_hint_promote_or_string() {
        let parsed = crate::parse::parse_source(
            r#"
            type PyObjectRef = *const ();
            #[elidable_promote(promote_args = "all")]
            pub unsafe fn fetch(p: PyObjectRef) -> i64 { 0 }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "fetch")
            .expect("wrapper graph");
        let ops = &wrapper.graph.block(wrapper.graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target.path_segments().and_then(|s| s.last().copied()) == Some("hint_promote_or_string")
            )),
            "wrapper must emit hint_promote_or_string for opaque pointer arg; ops:\n{ops:#?}"
        );
    }

    /// `rlib/jit.py:186, 191` — `_get_args(func)` reads `co_varnames`
    /// raw, so `self` is at index 0 and `promote_args='all'` covers it.
    /// Pyre can't shadow `self` with `let self = ...`, so the
    /// synthesizer routes the receiver through a fresh
    /// `__self_promoted` local and rewrites the tail call accordingly.
    /// The wrapper graph must emit one `hint_promote_or_string` per
    /// argument including the receiver (jit.py:192-194 dual hint).
    #[test]
    fn elidable_promote_promotes_self_receiver_for_inherent_method() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            impl S {
                #[elidable_promote(promote_args = "all")]
                pub fn double(&self, n: i64) -> i64 { n + n }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "double")
            .expect("wrapper graph");
        let ops = &wrapper.graph.block(wrapper.graph.startblock).operations;
        let hint_count = ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target, .. } if target.path_segments().and_then(|s| s.last().copied()) == Some("hint_promote_or_string")
                )
            })
            .count();
        assert_eq!(
            hint_count, 2,
            "wrapper must emit hint_promote_or_string for self and n \
             (2 total); ops:\n{ops:#?}"
        );
    }

    // ── `collect_jit_hints` argname-threading parity tests ──────────

    #[test]
    fn collect_jit_hints_emits_oopspec_argnames_for_free_fn() {
        // `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]`
        // — for a free function `fn foo(x, y, z)`, co_varnames is
        // `('x', 'y', 'z')`.
        let item: syn::ItemFn = syn::parse_quote! {
            #[oopspec("foo(x, y)")]
            fn foo(x: i64, y: i64) -> i64 { 0 }
        };
        let hints = super::collect_jit_hints_with_sig(&item.attrs, &item.sig);
        assert!(hints.contains(&"oopspec:foo(x, y)".to_string()));
        assert!(hints.contains(&"oopspec_argnames:x,y".to_string()));
    }

    #[test]
    fn collect_jit_hints_includes_self_as_first_argname_for_method() {
        // `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]`
        // — for a bound method `def foo(self, x): ...`, `co_varnames[:2]`
        // is `('self', 'x')`.  Pyre maps `FnArg::Receiver` to "self"
        // for strict parity.
        let item: syn::ImplItemFn = syn::parse_quote! {
            #[oopspec("foo(self, x)")]
            fn foo(&self, x: i64) -> i64 { 0 }
        };
        let hints = super::collect_jit_hints_with_sig(&item.attrs, &item.sig);
        assert!(hints.contains(&"oopspec_argnames:self,x".to_string()));
    }

    #[test]
    fn collect_jit_hints_omits_argname_hint_when_no_oopspec() {
        // No `#[oopspec(...)]` → no `oopspec_argnames:` hint emitted,
        // even when the signature has positional params.
        let item: syn::ItemFn = syn::parse_quote! {
            fn foo(x: i64, y: i64) -> i64 { 0 }
        };
        let hints = super::collect_jit_hints_with_sig(&item.attrs, &item.sig);
        assert!(hints.iter().all(|h| !h.starts_with("oopspec_argnames:")));
    }

    // ----------------------------------------------------------------
    // Slice Z4.A: flow-space value-stack helper coverage.
    //
    // The helper API surface is ported from `flowspace/flowcontext.py:
    // 317-345` ahead of any consumer (slice Z4.B+ converts `lower_expr`
    // leaves to push/pop).  The tests below pin the upstream semantics:
    // LIFO push/pop, `peekvalue(0)` = top, `popvalues(0)` = empty,
    // `popvalues(n)` returns oldest-first, `dropvaluesuntil` shrinks.
    // Run against an isolated `GraphBuildContext` instance — no graph
    // wiring required because the value stack lives entirely on ctx.

    fn z4a_test_ctx<'a>(
        struct_fields: &'a StructFieldRegistry,
        fn_return_types: &'a HashMap<String, String>,
        known_struct_names: &'a std::collections::HashSet<String>,
        known_trait_names: &'a std::collections::HashSet<String>,
    ) -> GraphBuildContext<'a> {
        GraphBuildContext::new(
            struct_fields,
            fn_return_types,
            "",
            HashMap::new(),
            known_struct_names,
            known_trait_names,
        )
    }

    /// Test-only push helper — mints a fresh `Variable`, binds it to
    /// `vid` via `bind_variable_at`, and pushes the
    /// `StackElem::Value(Hlvalue::Variable(...))` cell.  Mirrors how
    /// production op-emission would push the SpaceOperation result
    /// once leaves migrate to `value_stack` (Z4.B.1+).
    fn z4a_push_vid(ctx: &mut GraphBuildContext, graph: &mut FunctionGraph, vid: usize) {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Hlvalue, Variable};
        let v = Variable::new();
        graph.bind_variable_at(vid, v.clone());
        ctx.pushvalue(StackElem::Value(Hlvalue::Variable(v)));
    }

    /// Bridge a `StackElem::Value(Hlvalue::Variable)` cell back to
    /// its registered slot index via `slot_of`, panicking on a
    /// non-Variable shape (Constant / FlowSignal arms unused in
    /// current tests) or an unregistered Variable.
    fn z4a_cell_to_vid(
        graph: &mut FunctionGraph,
        elem: &crate::flowspace::framestate::StackElem,
    ) -> usize {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::Hlvalue;
        match elem {
            StackElem::Value(Hlvalue::Variable(v)) => graph
                .slot_of(v)
                .expect("Variable cell must be registered on graph"),
            other => panic!("expected Variable cell, got {other:?}"),
        }
    }

    #[test]
    fn z4a_pushvalue_popvalue_round_trip_matches_flowcontext_lifo() {
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a_demo");
        assert_eq!(ctx.stackdepth(), 0);
        z4a_push_vid(&mut ctx, &mut graph, 7);
        z4a_push_vid(&mut ctx, &mut graph, 11);
        assert_eq!(ctx.stackdepth(), 2);
        // `popvalue` returns the topmost cell — LIFO per
        // `flowcontext.py:325 self.stack.pop()`.
        let top = ctx.popvalue();
        assert_eq!(z4a_cell_to_vid(&mut graph, &top), 11);
        let bottom = ctx.popvalue();
        assert_eq!(z4a_cell_to_vid(&mut graph, &bottom), 7);
        assert_eq!(ctx.stackdepth(), 0);
    }

    /// Production-side `pushvid` / `popvid` round-trip — the graph
    /// slot index threads through `value_stack` as
    /// `StackElem::Value(Hlvalue::Variable(graph.variable_at(slot)))` and
    /// recovers back through `graph.slot_of`.  Z4.B.1+ leaf
    /// migrations consume this pair; the test pins the contract before
    /// any production caller relies on it.
    #[test]
    fn pushvid_popvid_round_trip_through_value_stack() {
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("pushvid_round_trip");
        let var_a = graph.alloc_value_var();
        let var_b = graph.alloc_value_var();
        let vid_a = graph
            .slot_of(&var_a)
            .expect("freshly minted Variable has slot");
        let vid_b = graph
            .slot_of(&var_b)
            .expect("freshly minted Variable has slot");
        ctx.pushvid(&graph, vid_a);
        ctx.pushvid(&graph, vid_b);
        assert_eq!(ctx.stackdepth(), 2);
        // LIFO: top is `vid_b`.
        assert_eq!(ctx.popvid(&graph), vid_b);
        assert_eq!(ctx.popvid(&graph), vid_a);
        assert_eq!(ctx.stackdepth(), 0);
    }

    /// `pushvid` panics when the vid was minted outside the canonical
    /// allocation path (no backing Variable at `graph.value_variables`).
    #[test]
    #[should_panic(expected = "pushvid: slot")]
    fn pushvid_panics_when_vid_has_no_backing_variable() {
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let graph = FunctionGraph::new("pushvid_unbacked");
        // slot 42 was never allocated — graph.variable(42) returns None.
        ctx.pushvid(&graph, 42);
    }

    #[test]
    fn z4a_peekvalue_index_zero_is_top_per_flowcontext_tilde_indexing() {
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a_demo");
        z4a_push_vid(&mut ctx, &mut graph, 1);
        z4a_push_vid(&mut ctx, &mut graph, 2);
        z4a_push_vid(&mut ctx, &mut graph, 3);
        // `peekvalue(0)` is the top per upstream's `~0 == -1` indexing
        // (`flowcontext.py:329`).
        let top0 = ctx.peekvalue(0).clone();
        let top1 = ctx.peekvalue(1).clone();
        let top2 = ctx.peekvalue(2).clone();
        assert_eq!(z4a_cell_to_vid(&mut graph, &top0), 3);
        assert_eq!(z4a_cell_to_vid(&mut graph, &top1), 2);
        assert_eq!(z4a_cell_to_vid(&mut graph, &top2), 1);
        // peekvalue is non-destructive.
        assert_eq!(ctx.stackdepth(), 3);
    }

    #[test]
    fn z4a_settopvalue_overwrites_at_index_from_top() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Hlvalue, Variable};
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a_demo");
        z4a_push_vid(&mut ctx, &mut graph, 1);
        z4a_push_vid(&mut ctx, &mut graph, 2);
        z4a_push_vid(&mut ctx, &mut graph, 3);
        // Replace index_from_top=1 with a fresh cell paired to slot 99.
        let v99 = Variable::new();
        graph.bind_variable_at(99, v99.clone());
        ctx.settopvalue(StackElem::Value(Hlvalue::Variable(v99)), 1);
        let top0 = ctx.peekvalue(0).clone();
        let top1 = ctx.peekvalue(1).clone();
        let top2 = ctx.peekvalue(2).clone();
        assert_eq!(z4a_cell_to_vid(&mut graph, &top0), 3);
        assert_eq!(z4a_cell_to_vid(&mut graph, &top1), 99);
        assert_eq!(z4a_cell_to_vid(&mut graph, &top2), 1);
    }

    #[test]
    fn z4a_popvalues_zero_returns_empty_without_touching_stack() {
        // `flowcontext.py:337-338`: `if n == 0: return []` short-circuits
        // before the slice — verify the stack is untouched.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a_demo");
        z4a_push_vid(&mut ctx, &mut graph, 1);
        let popped = ctx.popvalues(0);
        assert!(popped.is_empty());
        assert_eq!(ctx.stackdepth(), 1);
    }

    #[test]
    fn z4a_popvalues_n_returns_oldest_first_in_stack_order() {
        // `flowcontext.py:339-340`: `values_w = self.stack[-n:]; del
        // self.stack[-n:]` — slice preserves stack order so element 0
        // is the OLDEST of the popped cells (deepest of the n).
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a_demo");
        z4a_push_vid(&mut ctx, &mut graph, 10);
        z4a_push_vid(&mut ctx, &mut graph, 20);
        z4a_push_vid(&mut ctx, &mut graph, 30);
        z4a_push_vid(&mut ctx, &mut graph, 40);
        let popped = ctx.popvalues(3);
        let popped_vids: Vec<usize> = popped
            .iter()
            .map(|e| z4a_cell_to_vid(&mut graph, e))
            .collect();
        assert_eq!(popped_vids, vec![20, 30, 40]);
        assert_eq!(ctx.stackdepth(), 1);
        let bottom = ctx.peekvalue(0).clone();
        assert_eq!(z4a_cell_to_vid(&mut graph, &bottom), 10);
    }

    #[test]
    fn z4a_dropvaluesuntil_truncates_to_finaldepth() {
        // `flowcontext.py:343-344 dropvaluesuntil(self, finaldepth)` —
        // shrink the stack to `finaldepth` cells.  Used by
        // `FrameBlock.cleanupstack` (`flowcontext.py:1335-1336`) on
        // SETUP_* unwinds.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a_demo");
        for v in 1..=5 {
            z4a_push_vid(&mut ctx, &mut graph, v);
        }
        ctx.dropvaluesuntil(2);
        assert_eq!(ctx.stackdepth(), 2);
        let top0 = ctx.peekvalue(0).clone();
        let top1 = ctx.peekvalue(1).clone();
        assert_eq!(z4a_cell_to_vid(&mut graph, &top0), 2);
        assert_eq!(z4a_cell_to_vid(&mut graph, &top1), 1);
    }

    #[test]
    fn z4a_popvalues_n_exceeds_stack_returns_all_and_clears() {
        // `flowcontext.py:339-340 self.stack[-n:]` is Python negative-
        // slice — when `n > len(self.stack)`, the slice silently
        // clamps to the whole stack and the subsequent `del` clears
        // it.  Mirror the upstream tolerance: no assert, no panic.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a_demo");
        z4a_push_vid(&mut ctx, &mut graph, 1);
        z4a_push_vid(&mut ctx, &mut graph, 2);
        let popped = ctx.popvalues(5);
        let popped_vids: Vec<usize> = popped
            .iter()
            .map(|e| z4a_cell_to_vid(&mut graph, e))
            .collect();
        assert_eq!(popped_vids, vec![1, 2]);
        assert_eq!(ctx.stackdepth(), 0);
    }

    #[test]
    fn z4a_dropvaluesuntil_finaldepth_exceeds_stack_is_noop() {
        // `flowcontext.py:343-344 del self.stack[finaldepth:]` is a
        // Python slice — when `finaldepth >= len(self.stack)` the
        // slice past the end is empty and `del` is a no-op.  Pyre
        // mirrors via `Vec::truncate` which leaves the stack
        // unchanged on overflow inputs.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a_demo");
        z4a_push_vid(&mut ctx, &mut graph, 1);
        z4a_push_vid(&mut ctx, &mut graph, 2);
        ctx.dropvaluesuntil(10);
        assert_eq!(ctx.stackdepth(), 2);
        let top0 = ctx.peekvalue(0).clone();
        let top1 = ctx.peekvalue(1).clone();
        assert_eq!(z4a_cell_to_vid(&mut graph, &top0), 2);
        assert_eq!(z4a_cell_to_vid(&mut graph, &top1), 1);
    }

    #[test]
    #[should_panic(expected = "popvalue: empty stack")]
    fn z4a_popvalue_panics_on_empty_stack_per_flowcontext_indexerror() {
        // Upstream `flowcontext.py:325 self.stack.pop()` raises
        // `IndexError` on an empty stack; pyre raises via `expect`.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let _ = ctx.popvalue();
    }

    // ----------------------------------------------------------------
    // Slice Z4.A.2 — `getstate` / `setstate` skeleton coverage.
    //
    // `getstate(next_offset)` constructs a `FrameState` from the
    // current ctx fields per `flowcontext.py:346-348`; `setstate`
    // restores ctx fields from a `FrameState` per `:350-356`.  The
    // tests below pin upstream's positional-vs-name shape: locals
    // round-trip through `local_first_bind_order`, the four other
    // projections (stack / last_exception / blocklist / next_offset)
    // thread directly between `FrameState` and ctx fields with the
    // stack staying vestigial-empty until Z2.5 bridges Hlvalue
    // identity at Z4.last.

    #[test]
    fn z4a2_getstate_constructs_frame_state_from_locals_and_blockstack() {
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a2_getstate_demo");
        ctx.bind_local_id_at("x".into(), 7, BlockId(0), &mut graph);
        ctx.bind_local_id_at("y".into(), 11, BlockId(0), &mut graph);
        // last_exception + blockstack stay at defaults (None, []);
        // value_stack stays empty so the stack projection round-trips
        // as an empty Vec.
        let state = ctx.getstate(&graph, 42);
        assert_eq!(entry_slots(&state.entries, &graph), vec![Some(7), Some(11)]);
        assert!(
            state.stack.is_empty(),
            "empty value_stack must project to empty FrameState.stack"
        );
        assert!(state.last_exception.is_none());
        assert!(state.blocklist.is_empty());
        assert_eq!(state.next_offset, 42);
    }

    #[test]
    fn z4a2_getstate_copies_value_stack_into_frame_state_stack() {
        // Z4.A.5 parity: `flowcontext.py:347 self.stack[:]` — pyre's
        // value_stack and FrameState.stack share the Vec<StackElem>
        // carrier so getstate clones verbatim.  When the walker
        // populates value_stack with Hlvalue::Variable cells, the
        // snapshot retains the same cell identities.
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::Hlvalue;
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a2_demo");
        z4a_push_vid(&mut ctx, &mut graph, 101);
        z4a_push_vid(&mut ctx, &mut graph, 202);
        let state = ctx.getstate(&graph, 0);
        assert_eq!(state.stack.len(), 2);
        let StackElem::Value(Hlvalue::Variable(v0)) = &state.stack[0] else {
            panic!("expected Variable cell at stack[0]");
        };
        let StackElem::Value(Hlvalue::Variable(v1)) = &state.stack[1] else {
            panic!("expected Variable cell at stack[1]");
        };
        assert_eq!(graph.slot_of(v0).expect("v0 registered"), 101);
        assert_eq!(graph.slot_of(v1).expect("v1 registered"), 202);
    }

    #[test]
    fn z4a2_setstate_restores_locals_and_value_stack_from_captured_frame_state() {
        // Round-trip: getstate → mutate → setstate restores.  Locals
        // project back through `local_first_bind_order`; value_stack
        // restores from `state.stack.clone()` per the Z4.A.5
        // direct-copy port of `flowcontext.py:352 self.stack =
        // state.stack[:]`.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a2_demo");
        ctx.bind_local_id_at("x".into(), 1, BlockId(0), &mut graph);
        ctx.bind_local_id_at("y".into(), 2, BlockId(0), &mut graph);
        z4a_push_vid(&mut ctx, &mut graph, 50);
        let captured = ctx.getstate(&graph, 7);
        // Mutate ctx after capture: rebind x, push a different cell.
        ctx.bind_local_id_at("x".into(), 99, BlockId(1), &mut graph);
        z4a_push_vid(&mut ctx, &mut graph, 123);
        assert_eq!(ctx.stackdepth(), 2);
        ctx.setstate(&captured, &graph);
        // Locals back to captured shape.
        assert_eq!(
            ctx.local_value_ids
                .get("x")
                .map(|(var, _)| graph.slot_of(var).expect("backing Variable registered")),
            Some(1),
            "x must be restored to captured slot"
        );
        assert_eq!(
            ctx.local_value_ids
                .get("y")
                .map(|(var, _)| graph.slot_of(var).expect("backing Variable registered")),
            Some(2),
            "y must be restored to captured slot"
        );
        // value_stack restored to single-cell captured shape carrying
        // slot 50.
        assert_eq!(ctx.stackdepth(), 1);
        let top = ctx.peekvalue(0).clone();
        assert_eq!(z4a_cell_to_vid(&mut graph, &top), 50);
    }

    #[test]
    fn z4b0_assert_stack_empty_at_stmt_boundary_passes_when_value_stack_is_empty() {
        // Z4.B.0 tripwire — today `value_stack` is never written by
        // production code so this assert is a no-op safety net.  The
        // test pins the helper's no-fire behaviour against an empty
        // stack so a future regression (push without matching pop in
        // Z4.B.1+) is caught at the next `lower_stmt` entry rather
        // than silently leaking depth across statements.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        // Must not panic.
        ctx.assert_stack_empty_at_stmt_boundary("z4b0_test");
    }

    #[test]
    #[should_panic(expected = "stmt-boundary stack imbalance")]
    fn z4b0_assert_stack_empty_at_stmt_boundary_panics_when_value_stack_carries_residue() {
        // Inverse direction — once `value_stack` carries cells across
        // a stmt boundary, the tripwire fires with the
        // `flowcontext.py:413` invariant message.  Z4.B.1+ relies on
        // this firing immediately so the offending Stmt is flagged
        // rather than the imbalance propagating into later joins.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4b0_demo");
        z4a_push_vid(&mut ctx, &mut graph, 42);
        ctx.assert_stack_empty_at_stmt_boundary("z4b0_negative_test");
    }

    #[test]
    fn setstate_normalizes_raise_implicit_signals_on_value_stack() {
        // `flowcontext.py:358-362 _normalize_raise_signals` — every
        // RaiseImplicit cell on the post-setstate stack downgrades to
        // a plain Raise carrying the same `w_exc`.  Verifies the AST
        // setstate path invokes the helper so a future Z4.H wire-in
        // doesn't need to chase the call site retroactively.
        use crate::flowspace::flowcontext::FlowSignal;
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Constant, FSException, Hlvalue};
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let graph = FunctionGraph::new("setstate_normalize_demo");
        let exc = FSException::new(
            Hlvalue::Constant(Constant::new(ConstValue::Int(101))),
            Hlvalue::Constant(Constant::new(ConstValue::None)),
        );
        let captured = FrameState {
            entries: Vec::new(),
            stack: vec![StackElem::Signal(FlowSignal::RaiseImplicit {
                w_exc: exc.clone(),
            })],
            ..Default::default()
        };
        ctx.setstate(&captured, &graph);
        assert_eq!(ctx.value_stack.len(), 1);
        match &ctx.value_stack[0] {
            StackElem::Signal(FlowSignal::Raise { w_exc }) => {
                assert_eq!(w_exc, &exc, "Raise must carry the same w_exc");
            }
            other => panic!("expected Raise after normalization, got {other:?}"),
        }
    }

    #[test]
    fn z4a2_setstate_drops_local_when_state_entry_is_none_killed() {
        // None-killed slots collapse the binding so post-merge reads
        // surface as undefined-local.  Mirrors the per-slot
        // `framestate.py:110-111 if w1 or w2 is None: return None`
        // semantic on the consumer side.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = FunctionGraph::new("z4a2_setstate_drops_demo");
        ctx.bind_local_id_at("x".into(), 1, BlockId(0), &mut graph);
        ctx.bind_local_id_at("y".into(), 2, BlockId(0), &mut graph);
        // Build a FrameState whose y slot is None-killed.
        let killed_state = FrameState {
            entries: FrameState::entries_from_slots(&[Some(1), None], &mut graph),
            ..Default::default()
        };
        ctx.setstate(&killed_state, &graph);
        assert!(ctx.local_value_ids.contains_key("x"));
        assert!(
            !ctx.local_value_ids.contains_key("y"),
            "None-killed slot must remove the binding from ctx"
        );
    }

    // ------------------------------------------------------------------
    // mergeblock helper (Slice 4.1 scaffold).
    //
    // Tests cover the three control-flow arms of
    // `flowspace/flowcontext.py:424-463 mergeblock`:
    //   1. No candidate at `next_offset` (or none accept union) — fresh
    //      SpamBlock candidate is registered (`make_next_block`).
    //   2. A candidate's union with current succeeds AND `matches` the
    //      candidate's framestate — direct link, candidate list
    //      unchanged.
    //   3. A candidate's union with current succeeds but does NOT match
    //      — fresh SpamBlock created with the merged state, old
    //      candidate retired via `recloseblock`, candidate list swaps
    //      the old block for the new.
    //
    // Tree-recursive lowering omits the `pendingblocks.append(newblock)`
    // step from upstream; the merge target block is returned to the
    // caller for synchronous continued lowering.

    /// Build a single-slot FrameState whose only local is `vid`.
    /// Variables backing the entries must already be registered with
    /// `graph` so `create_block_from_framestate` / `getvariables` /
    /// `set_goto_from_framestate` round-trip correctly.
    fn mergeblock_test_state_single(
        graph: &mut crate::model::FunctionGraph,
        next_offset: i64,
    ) -> FrameState {
        let var = graph.alloc_value_var();
        let slot = graph
            .slot_of(&var)
            .expect("alloc_value_var registered the slot");
        FrameState {
            entries: FrameState::entries_from_slots(&[Some(slot)], graph),
            locals_w: Vec::new(),
            stack: Vec::new(),
            last_exception: None,
            blocklist: Vec::new(),
            next_offset,
        }
    }

    #[test]
    fn mergeblock_first_arrival_creates_new_spam_block_candidate() {
        // `flowcontext.py:433-436` — candidates empty, fall through to
        // make_next_block: fresh SpamBlock with `currentstate.copy()` as
        // its framestate, link current→new, insert at head.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = crate::model::FunctionGraph::new("test");
        let currentstate = mergeblock_test_state_single(&mut graph, 10);
        let current = graph.create_block_from_framestate(&currentstate);

        let merge = ctx.mergeblock(&mut graph, current, currentstate.clone());

        assert_eq!(
            ctx.joinpoints.get(&10).cloned().unwrap_or_default(),
            vec![merge],
            "candidate list at next_offset=10 must contain only the new SpamBlock"
        );
        assert_ne!(merge, current, "mergeblock must create a NEW block");
        assert!(
            graph.block(merge).framestate.is_some(),
            "make_next_block: merge target is a SpamBlock (framestate-bearing)"
        );
        let exits = &graph.block(current).exits;
        assert_eq!(exits.len(), 1, "current closed with a single goto");
        assert_eq!(exits[0].target, merge, "exit targets the new merge block");
    }

    #[test]
    fn mergeblock_matching_candidate_takes_direct_link() {
        // `flowcontext.py:438-441` — union succeeds, `matches` returns
        // true (both predecessors carry Variables at the same slot
        // positions modulo identity), so currentblock is linked
        // directly to the existing candidate.  No new SpamBlock, no
        // candidate-list mutation.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = crate::model::FunctionGraph::new("test");

        // Predecessor #1.
        let state1 = mergeblock_test_state_single(&mut graph, 20);
        let current1 = graph.create_block_from_framestate(&state1);
        let cand = ctx.mergeblock(&mut graph, current1, state1);

        // Predecessor #2 — different Variable in the same slot, same
        // next_offset.  Per `framestate.py:113-114`, the per-cell union
        // mints a fresh Variable; `matches` accepts the pair as both-
        // Variables, so we take the direct-link arm.
        let state2 = mergeblock_test_state_single(&mut graph, 20);
        let current2 = graph.create_block_from_framestate(&state2);
        let merge = ctx.mergeblock(&mut graph, current2, state2);

        assert_eq!(merge, cand, "matches arm returns the existing candidate");
        assert_eq!(
            ctx.joinpoints.get(&20).cloned().unwrap_or_default(),
            vec![cand],
            "candidate list unchanged on matches-direct-link arm"
        );
        assert_eq!(
            graph.block(current1).exits[0].target,
            cand,
            "first predecessor still targets the original candidate"
        );
        assert_eq!(
            graph.block(current2).exits[0].target,
            cand,
            "second predecessor takes the direct link to the same candidate"
        );
        assert!(
            !graph.block(cand).dead,
            "matches-direct-link arm does not retire the candidate"
        );
    }

    #[test]
    fn mergeblock_disagreeing_candidate_generalizes_and_retires_old() {
        // `flowcontext.py:443-463` — second predecessor's union with
        // the candidate produces a state that does NOT match the
        // candidate (a slot present-as-Variable in the candidate
        // collapses to None-kill when unioned with a one-sided
        // arrival).  Build a fresh SpamBlock for the merged state,
        // link current→new, retire the old candidate via
        // `recloseblock` to forward to the new block, and swap the
        // candidate-list entry.
        let registry = StructFieldRegistry::default();
        let fn_ret = HashMap::new();
        let names = std::collections::HashSet::new();
        let trait_names = std::collections::HashSet::new();
        let mut ctx = z4a_test_ctx(&registry, &fn_ret, &names, &trait_names);
        let mut graph = crate::model::FunctionGraph::new("test");

        // Predecessor #1 binds a fresh slot.
        let state1 = mergeblock_test_state_single(&mut graph, 30);
        let current1 = graph.create_block_from_framestate(&state1);
        let cand = ctx.mergeblock(&mut graph, current1, state1);

        // Predecessor #2 leaves slot 0 unbound.  Per
        // `framestate.py:110-111`, the per-cell union returns
        // Ok(None) — the merged state has [None] at slot 0.
        // `matches` rejects (Variable vs None) → take the generalize
        // arm.
        let state2 = FrameState {
            entries: vec![None],
            locals_w: Vec::new(),
            stack: Vec::new(),
            last_exception: None,
            blocklist: Vec::new(),
            next_offset: 30,
        };
        let current2 = graph.create_block_from_framestate(&state2);
        let merge = ctx.mergeblock(&mut graph, current2, state2);

        assert_ne!(merge, cand, "generalize arm allocates a fresh SpamBlock");
        assert_eq!(
            ctx.joinpoints.get(&30).cloned().unwrap_or_default(),
            vec![merge],
            "candidate list swaps the old block for the new (insert at head, remove old)"
        );
        // Old candidate retired: dead-marked, ops cleared, exits
        // forwarded to the new block.
        let cand_block = graph.block(cand);
        assert!(cand_block.dead, "retired candidate must be dead-marked");
        assert!(
            cand_block.operations.is_empty(),
            "retired candidate's operations must be cleared"
        );
        assert!(
            cand_block.exitswitch.is_none(),
            "retired candidate's exitswitch must be cleared"
        );
        assert_eq!(
            cand_block.exits.len(),
            1,
            "retired candidate forwards via a single reclosed Link"
        );
        assert_eq!(
            cand_block.exits[0].target, merge,
            "retired candidate forwards to the new merged SpamBlock"
        );
        assert_eq!(
            graph.block(current2).exits[0].target,
            merge,
            "current2 links directly to the new merge block"
        );
    }
}
